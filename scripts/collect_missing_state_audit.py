#!/usr/bin/env python3
"""Inventory and collect simulator truth for the missing-state audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from robosuite.utils.transform_utils import mat2quat

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path

from scripts import audit_local_candidate_counterfactual as base
from scripts.evaluate_c0_all import policy_observation, refresh_policy_observation, trusted_init_states


PROTOCOL_VERSION = "local-visual-counterfactual-missing-state-audit-v1"
CONTACT_NAMES = ("robot_object", "object_environment", "gripper_object", "grasp_helper")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def validate_protocol(path: Path, mode: str, partition: str | None) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected missing-state protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_STATE_INVENTORY_OR_REPLAY":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise ValueError("protocol must be committed before simulator access")
    frozen = protocol["frozen_upstream"]
    if subprocess.check_output(
        ["git", "cat-file", "-t", frozen["primary_tag"]], cwd=ROOT, text=True
    ).strip() != "tag":
        raise ValueError("primary freeze tag is not annotated")
    if subprocess.check_output(
        ["git", "rev-parse", f"{frozen['primary_tag']}^{{}}"], cwd=ROOT, text=True
    ).strip() != frozen["primary_commit"]:
        raise ValueError("primary freeze tag moved")
    for item in frozen["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen upstream changed: {item['path']}")
    for key in ("collector", "feature_module"):
        item = protocol["implementation"][key]
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"implementation changed after preregistration: {item['path']}")
    if mode == "collect":
        if partition not in {"calibration", "evaluation"}:
            raise ValueError("collection partition must be calibration or evaluation")
        inventory = ROOT / protocol["outputs"]["state_inventory"]
        if not inventory.exists() or not last_commit(inventory):
            raise RuntimeError("state inventory must exist and be committed before branch collection")
        if partition == "evaluation":
            freeze = ROOT / protocol["outputs"]["calibration_freeze"]
            if not freeze.exists() or not last_commit(freeze):
                raise RuntimeError("calibration freeze must be committed before evaluation replay")
            if subprocess.check_output(
                ["git", "status", "--porcelain", "--", str(freeze.relative_to(ROOT))],
                cwd=ROOT,
                text=True,
            ).strip():
                raise RuntimeError("calibration freeze has uncommitted changes")
    return protocol, commit


def describe_array(name: str, value: Any, source: str, observation: bool, simulator: bool, oracle: bool) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "semantic_source": source,
        "observation_accessible": observation,
        "simulator_internal_only": simulator,
        "future_oracle": oracle,
    }


def relevant_objects(inner: Any) -> list[str]:
    declared = list(inner.parsed_problem.get("obj_of_interest", []))
    names = [name for name in declared if name in inner.objects_dict]
    if not names:
        names = list(inner.objects_dict.keys())
    if not names:
        raise RuntimeError("environment exposes no manipulable/relevant objects")
    return names


def object_schema(inner: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in relevant_objects(inner):
        model = inner.objects_dict[name]
        result[name] = {
            "category": getattr(model, "category_name", None),
            "root_body": model.root_body,
            "body_id": int(inner.obj_body_id[name]),
            "contact_geoms": list(model.contact_geoms),
        }
    return result


def contact_state(inner: Any, schema: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    robot_geoms = set(inner.robots[0].robot_model.contact_geoms)
    gripper_geoms = set(inner.robots[0].gripper.contact_geoms)
    all_object_geoms = {geom for record in schema.values() for geom in record["contact_geoms"]}
    names = list(schema)
    per_object = np.zeros((len(names), len(CONTACT_NAMES)), dtype=np.bool_)
    counts = np.zeros(len(CONTACT_NAMES), dtype=np.int64)
    pairs: list[list[str]] = []
    for contact_index in range(int(inner.sim.data.ncon)):
        contact = inner.sim.data.contact[contact_index]
        first = inner.sim.model.geom_id2name(int(contact.geom1))
        second = inner.sim.model.geom_id2name(int(contact.geom2))
        if first is None or second is None:
            continue
        pairs.append([first, second])
        pair = {first, second}
        for object_index, name in enumerate(names):
            object_geoms = set(schema[name]["contact_geoms"])
            touches_object = bool(pair & object_geoms)
            if not touches_object:
                continue
            other = second if first in object_geoms else first
            if other in robot_geoms:
                per_object[object_index, 0] = True
                counts[0] += 1
            if other in gripper_geoms:
                per_object[object_index, 2] = True
                counts[2] += 1
            if other not in robot_geoms and other not in all_object_geoms:
                per_object[object_index, 1] = True
                counts[1] += 1
    for object_index, name in enumerate(names):
        model = inner.objects_dict[name]
        grasp = bool(inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=model))
        per_object[object_index, 3] = grasp
        counts[3] += int(grasp)
    return per_object, counts, pairs


def extract_state(env: Any, max_objects: int) -> dict[str, np.ndarray]:
    inner = env.env
    schema = object_schema(inner)
    names = list(schema)
    ee_site = int(inner.robots[0].eef_site_id)
    ee_pos = np.asarray(inner.sim.data.site_xpos[ee_site], dtype=np.float64).copy()
    ee_rotmat = np.asarray(inner.sim.data.site_xmat[ee_site], dtype=np.float64).reshape(3, 3).copy()
    ee_quat = np.asarray(mat2quat(ee_rotmat), dtype=np.float64)
    object_pos = np.zeros((max_objects, 3), dtype=np.float64)
    object_quat = np.zeros((max_objects, 4), dtype=np.float64)
    object_rotmat = np.zeros((max_objects, 3, 3), dtype=np.float64)
    object_mask = np.zeros(max_objects, dtype=np.bool_)
    for index, name in enumerate(names):
        body_id = schema[name]["body_id"]
        object_pos[index] = np.asarray(inner.sim.data.body_xpos[body_id], dtype=np.float64)
        quat_wxyz = np.asarray(inner.sim.data.body_xquat[body_id], dtype=np.float64)
        object_quat[index] = quat_wxyz
        w, x, y, z = quat_wxyz
        object_rotmat[index] = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        object_mask[index] = True
    per_object, counts, _ = contact_state(inner, schema)
    contact_padded = np.zeros((max_objects, len(CONTACT_NAMES)), dtype=np.bool_)
    contact_padded[: len(names)] = per_object
    return {
        "ee_pos": ee_pos,
        "ee_quat_xyzw": ee_quat,
        "ee_rotmat": ee_rotmat,
        "object_pos": object_pos,
        "object_quat_wxyz": object_quat,
        "object_rotmat": object_rotmat,
        "object_mask": object_mask,
        "contact_per_object": contact_padded,
        "contact_any": np.any(contact_padded, axis=0),
        "contact_counts": counts,
    }


def inventory(protocol: dict[str, Any], preregistered_commit: str) -> dict[str, Any]:
    suite = benchmark.get_benchmark_dict()[protocol["scope"]["suite"]]()
    controller = json.loads((ROOT / protocol["frozen_upstream"]["controller_protocol"]).read_text())[
        "partitions"
    ]["controller_development"]
    tasks = []
    for task_id in protocol["scope"]["task_ids"]:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = trusted_init_states(suite, task_id)[31]
        seed = int(controller["task_seeds"][str(task_id)][1])
        env, _ = base.create_episode_env(bddl, seed, init_state)
        try:
            observation = refresh_policy_observation(env)
            inner = env.env
            schema = object_schema(inner)
            _, _, pairs = contact_state(inner, schema)
            observation_fields = [
                describe_array(
                    name,
                    observation[name],
                    "LIBERO/robosuite observation",
                    True,
                    False,
                    False,
                )
                for name in sorted(observation)
            ]
            simulator_fields = [
                describe_array("sim.data.body_xpos", inner.sim.data.body_xpos, "MuJoCo body world positions", False, True, True),
                describe_array("sim.data.body_xquat", inner.sim.data.body_xquat, "MuJoCo body world quaternions (wxyz)", False, True, True),
                describe_array("sim.data.site_xpos", inner.sim.data.site_xpos, "MuJoCo site world positions", False, True, True),
                describe_array("sim.data.site_xmat", inner.sim.data.site_xmat, "MuJoCo site world rotation matrices", False, True, True),
                describe_array("sim.data.qpos", inner.sim.data.qpos, "MuJoCo generalized positions", False, True, True),
                describe_array("sim.data.qvel", inner.sim.data.qvel, "MuJoCo generalized velocities", False, True, True),
                describe_array("sim.data.contact[:ncon]", inner.sim.data.contact[: int(inner.sim.data.ncon)], "MuJoCo active contacts", False, True, True),
            ]
            tasks.append(
                {
                    "task_id": task_id,
                    "task_language": task.language,
                    "bddl": str(bddl),
                    "observation_fields": observation_fields,
                    "robot_state_keys": [row["name"] for row in observation_fields if row["name"].startswith("robot0_")],
                    "object_state_keys": [row["name"] for row in observation_fields if not row["name"].startswith("robot0_") and row["name"] not in {"agentview_image", "object-state"}],
                    "simulator_fields": simulator_fields,
                    "parsed_obj_of_interest": list(inner.parsed_problem.get("obj_of_interest", [])),
                    "all_objects": list(inner.objects_dict.keys()),
                    "fixtures": list(inner.fixtures_dict.keys()),
                    "selected_relevant_objects": list(schema),
                    "object_schema": schema,
                    "eef": {
                        "site_id": int(inner.robots[0].eef_site_id),
                        "site_name": inner.sim.model.site_id2name(int(inner.robots[0].eef_site_id)),
                        "observation_quaternion_key_present": "robot0_eef_quat" in observation,
                        "simulator_rotation_matrix_present": True,
                    },
                    "contact": {
                        "sim_ncon": int(inner.sim.data.ncon),
                        "initial_active_geom_pairs": pairs,
                        "robot_contact_geoms": list(inner.robots[0].robot_model.contact_geoms),
                        "gripper_contact_geoms": list(inner.robots[0].gripper.contact_geoms),
                        "check_grasp_callable": callable(getattr(inner, "_check_grasp", None)),
                        "classification_names": list(CONTACT_NAMES),
                    },
                }
            )
        finally:
            env.close()
    return {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "protocol_sha256": sha256(ROOT / "configs/missing_state_audit_protocol.json"),
        "preregistered_commit": preregistered_commit,
        "inventory_scope": "task0-9, official init31, controller-development seed position1; reset/set-init only",
        "tasks": tasks,
        "availability": {
            "ee_orientation": all(task["eef"]["simulator_rotation_matrix_present"] for task in tasks),
            "object_pose": all(bool(task["selected_relevant_objects"]) for task in tasks),
            "contact_state": all(task["contact"]["check_grasp_callable"] for task in tasks),
            "relative_geometry": True,
        },
        "future_oracle_warning": "EE/object/contact future state is used only for mechanism diagnosis and is not a deployable input.",
        "forbidden_activity_audit": {
            "candidate_execution": False,
            "evaluation_init_36_39_access": False,
            "init_40_49_access": False,
            "reward_done_success_read": False,
            "model_or_controller_change": False,
        },
    }


def collect_partition(protocol: dict[str, Any], partition: str, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    source_path = ROOT / protocol["partitions"][partition]["source_sidecar"]
    source = np.load(source_path, allow_pickle=False)
    policy = base.C1V2RerankPolicy(
        checkpoint=ROOT / protocol["frozen_upstream"]["c0_v2_checkpoint"],
        device=str(device),
        config=ROOT / protocol["frozen_upstream"]["c1_v2_config"],
    )
    suite = benchmark.get_benchmark_dict()[protocol["scope"]["suite"]]()
    inventory_data = json.loads((ROOT / protocol["outputs"]["state_inventory"]).read_text())
    max_objects = max(len(task["selected_relevant_objects"]) for task in inventory_data["tasks"])
    records = []
    collected: dict[str, list[np.ndarray]] = {}
    for branch_index in range(source["task_id"].shape[0]):
        task_id = int(source["task_id"][branch_index])
        init_index = int(source["init_state_index"][branch_index])
        seed = int(source["seed"][branch_index])
        prefix_length = int(source["prefix_lengths"][branch_index])
        prefix = source["prefix_actions"][branch_index, :prefix_length].astype(np.float32)
        candidates = source["candidates"][branch_index].astype(np.float32)
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = trusted_init_states(suite, task_id)[init_index]
        envs = []
        try:
            observations, states = [], []
            for _ in range(3):
                env, observation, sim_state = base.replay_prefix(bddl, seed, init_state, prefix)
                envs.append(env)
                observations.append(observation)
                states.append(sim_state)
            identity_latents = base.encode_images(policy, [obs["agentview_image"] for obs in observations])
            audit = base.prefix_audit(protocol, prefix, observations, states, identity_latents)
            if not audit["pass"]:
                raise base.CounterfactualReplayError("missing-state prefix identity failed", audit)
            context = policy._actor_context(policy_observation(observations[0]), task.language)
            plan_seed = policy.candidate_seed + seed * 1_000_003 + 8
            regenerated = policy._interface(context["actor_normalized"], plan_seed)["candidates"].detach().float().cpu().numpy()
            if not np.array_equal(regenerated, candidates):
                raise RuntimeError("regenerated frozen candidate differs from source candidate")
            schemas = [object_schema(env.env) for env in envs]
            if any(schema != schemas[0] for schema in schemas[1:]):
                raise RuntimeError("branch object schema mismatch")
            names = list(schemas[0])
            branch_arrays = {
                "ee_pos": np.empty((3, 8, 3), dtype=np.float64),
                "ee_quat_xyzw": np.empty((3, 8, 4), dtype=np.float64),
                "ee_rotmat": np.empty((3, 8, 3, 3), dtype=np.float64),
                "object_pos": np.empty((3, 8, max_objects, 3), dtype=np.float64),
                "object_quat_wxyz": np.empty((3, 8, max_objects, 4), dtype=np.float64),
                "object_rotmat": np.empty((3, 8, max_objects, 3, 3), dtype=np.float64),
                "contact_per_object": np.empty((3, 8, max_objects, 4), dtype=np.bool_),
                "contact_any": np.empty((3, 8, 4), dtype=np.bool_),
                "contact_counts": np.empty((3, 8, 4), dtype=np.int64),
            }
            object_mask = np.zeros(max_objects, dtype=np.bool_)
            object_mask[: len(names)] = True
            future_rgb_exact = True
            future_proprio_max = 0.0
            for horizon in range(8):
                for candidate_index, env in enumerate(envs):
                    env.step(candidates[candidate_index, horizon])
                    observation = refresh_policy_observation(env)
                    future_rgb_exact = future_rgb_exact and np.array_equal(
                        np.asarray(observation["agentview_image"], dtype=np.uint8),
                        source["real_rgb"][branch_index, candidate_index, horizon],
                    )
                    actual_p = base.proprio_from_observation(observation).astype(np.float64)
                    expected_p = source["real_proprio"][branch_index, candidate_index, horizon].astype(np.float64)
                    future_proprio_max = max(future_proprio_max, float(np.max(np.abs(actual_p - expected_p))))
                    snapshot = extract_state(env, max_objects)
                    for key in branch_arrays:
                        branch_arrays[key][candidate_index, horizon] = snapshot[key]
            if not future_rgb_exact or future_proprio_max > 0.0:
                raise RuntimeError(
                    f"replayed future differs from frozen branch: rgb={future_rgb_exact}, proprio={future_proprio_max}"
                )
            for key, value in branch_arrays.items():
                collected.setdefault(key, []).append(value)
            collected.setdefault("object_mask", []).append(object_mask)
            padded_names = np.full(max_objects, "", dtype=f"<U{max(max(map(len, names)), 1)}")
            padded_names[: len(names)] = names
            collected.setdefault("object_names", []).append(padded_names)
            record = {
                "branch_index": branch_index,
                "task_id": task_id,
                "init_state_index": init_index,
                "seed": seed,
                "planning_index": 8,
                "prefix_steps": prefix_length,
                "candidate_sha256": base.array_sha256(candidates),
                "relevant_objects": names,
                "prefix_audit": audit,
                "future_rgb_exact": future_rgb_exact,
                "future_proprio_max_abs_difference": future_proprio_max,
            }
            records.append(record)
            print(json.dumps({"partition": partition, "branch": branch_index + 1, "task": task_id, "init": init_index}), flush=True)
        finally:
            for env in envs:
                env.close()
    arrays = {key: np.stack(values) for key, values in collected.items()}
    for name in ("task_id", "init_state_index", "seed", "planning_index"):
        arrays[name] = np.asarray(source[name])
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != "object_names"):
        raise FloatingPointError("collected state contains NaN/Inf")
    return records, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inventory", "collect"), required=True)
    parser.add_argument("--partition", choices=("calibration", "evaluation"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/missing_state_audit_protocol.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol, commit = validate_protocol(args.protocol, args.mode, args.partition)
    if args.mode == "inventory":
        output = ROOT / protocol["outputs"]["state_inventory"]
        if output.exists():
            raise FileExistsError(output)
        result = inventory(protocol, commit)
        atomic_json(output, result)
        print(json.dumps({"status": "COMPLETE", "inventory": str(output.relative_to(ROOT))}))
        return 0
    partition = str(args.partition)
    sidecar = ROOT / protocol["partitions"][partition]["state_sidecar"]
    manifest = ROOT / protocol["partitions"][partition]["state_manifest"]
    if sidecar.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite missing-state collection")
    records, arrays = collect_partition(protocol, partition, torch.device(args.device))
    atomic_npz(sidecar, arrays)
    result = {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "partition": partition,
        "protocol_sha256": sha256(args.protocol),
        "preregistered_commit": commit,
        "source_sidecar": protocol["partitions"][partition]["source_sidecar"],
        "source_sidecar_sha256": sha256(ROOT / protocol["partitions"][partition]["source_sidecar"]),
        "branch_identities": records,
        "artifact": {
            "path": str(sidecar.relative_to(ROOT)),
            "sha256": sha256(sidecar),
            "arrays": {key: list(value.shape) for key, value in arrays.items()},
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "mid_episode_restore": False,
            "reward_done_success_read": False,
            "controller_or_model_change": False,
            "init_40_49_access": False,
        },
    }
    atomic_json(manifest, result)
    print(json.dumps({"status": "COMPLETE", "partition": partition, "sidecar_sha256": result["artifact"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
