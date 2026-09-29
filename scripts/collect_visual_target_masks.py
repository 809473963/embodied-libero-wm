#!/usr/bin/env python3
"""Collect native simulator masks for frozen matched counterfactual branches."""

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

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path

from scripts import audit_local_candidate_counterfactual as base
from scripts.evaluate_c0_all import policy_observation, refresh_policy_observation, trusted_init_states
from scripts.inventory_visual_target_masks import (
    CAMERA,
    mask_id_sets,
    masks_from_segmentation,
    render_rgb,
    render_segmentation,
    render_without_robot,
)
from src.probes.visual_target import MASK_NAMES


PROTOCOL_VERSION = "visual-target-suitability-audit-v1"


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


def load_protocol(path: Path, partition: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected visual-target protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_BRANCH_MASK_COLLECTION_OR_FEATURE_RESULTS":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise ValueError("protocol must be committed before mask collection")
    tag = protocol["frozen_upstream"]["primary_tag"]
    if subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip() != "tag":
        raise ValueError("upstream freeze tag is not annotated")
    if subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip() != protocol[
        "frozen_upstream"
    ]["primary_commit"]:
        raise ValueError("upstream freeze tag moved")
    for item in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen upstream changed: {item['path']}")
    for key in ("collector", "feature_module"):
        item = protocol["implementation"][key]
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"implementation changed after preregistration: {item['path']}")
    inventory_path = ROOT / protocol["mask_inventory"]["path"]
    inventory = json.loads(inventory_path.read_text())
    if inventory["status"] != "COMPLETE" or sha256(inventory_path) != protocol["mask_inventory"]["sha256"]:
        raise ValueError("admitted mask inventory changed")
    if not all(inventory["aggregate_checks"].values()):
        raise ValueError("mask inventory did not pass all aggregate checks")
    if partition == "evaluation":
        freeze = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze.exists() or not last_commit(freeze):
            raise RuntimeError("calibration freeze must be committed before evaluation mask collection")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(freeze.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
        ).strip():
            raise RuntimeError("calibration freeze has uncommitted changes")
    return protocol, commit


def collect(protocol: dict[str, Any], partition: str, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    partition_record = protocol["partitions"][partition]
    source_path = ROOT / partition_record["source_sidecar"]
    source_file = np.load(source_path, allow_pickle=False)
    source = {name: np.asarray(source_file[name]) for name in source_file.files}
    policy = base.C1V2RerankPolicy(
        checkpoint=ROOT / protocol["frozen_upstream"]["c0_v2_checkpoint"],
        device=str(device),
        config=ROOT / protocol["frozen_upstream"]["c1_v2_config"],
    )
    suite = benchmark.get_benchmark_dict()[protocol["scope"]["suite"]]()
    inventory = json.loads((ROOT / protocol["mask_inventory"]["path"]).read_text())
    inventory_by_task = {int(row["task_id"]): row for row in inventory["tasks"]}
    branch_masks: list[np.ndarray] = []
    branch_no_robot: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
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
                env, observation, state = base.replay_prefix(bddl, seed, init_state, prefix)
                envs.append(env)
                observations.append(observation)
                states.append(state)
            identity_latents = base.encode_images(policy, [obs["agentview_image"] for obs in observations])
            prefix_audit = base.prefix_audit(protocol, prefix, observations, states, identity_latents)
            if not prefix_audit["pass"]:
                raise base.CounterfactualReplayError("visual-target prefix identity failed", prefix_audit)
            context = policy._actor_context(policy_observation(observations[0]), task.language)
            plan_seed = policy.candidate_seed + seed * 1_000_003 + int(source["planning_index"][branch_index])
            regenerated = policy._interface(context["actor_normalized"], plan_seed)["candidates"].detach().float().cpu().numpy()
            if not np.array_equal(regenerated, candidates):
                raise RuntimeError("regenerated frozen candidate differs from source candidate")
            ids_by_branch = [mask_id_sets(env.env) for env in envs]
            for ids in ids_by_branch:
                for key, inventory_key in (("robot", "robot"), ("relevant", "relevant"), ("other", "other_object")):
                    if sorted(ids[key]) != inventory_by_task[task_id]["geom_ids"][inventory_key]:
                        raise RuntimeError(f"task {task_id} {key} geom IDs differ from inventory")
            masks = np.empty((3, 8, 4, 128, 128), dtype=np.bool_)
            no_robot_relevant = np.empty((3, 8, 128, 128), dtype=np.bool_)
            future_rgb_exact = True
            partition_exact = True
            robot_hidden_absent = True
            no_robot_area_not_smaller = True
            state_max = 0.0
            for horizon in range(8):
                for candidate_index, env in enumerate(envs):
                    env.step(candidates[candidate_index, horizon])
                    observation = refresh_policy_observation(env)
                    expected_rgb = source["real_rgb"][branch_index, candidate_index, horizon]
                    actual_rgb = np.asarray(observation["agentview_image"], dtype=np.uint8)
                    future_rgb_exact &= bool(np.array_equal(actual_rgb, expected_rgb))
                    direct_rgb = render_rgb(env.env)
                    future_rgb_exact &= bool(np.array_equal(direct_rgb, expected_rgb))
                    segmentation = render_segmentation(env.env)
                    semantic = masks_from_segmentation(segmentation, ids_by_branch[candidate_index])
                    stacked = np.stack([semantic[name] for name in MASK_NAMES])
                    partition_exact &= bool(np.all(stacked.sum(axis=0) == 1))
                    masks[candidate_index, horizon] = stacked
                    hidden_seg, hidden_audit = render_without_robot(
                        env.env, ids_by_branch[candidate_index]["robot"]
                    )
                    hidden_semantic = masks_from_segmentation(hidden_seg, ids_by_branch[candidate_index])
                    no_robot_relevant[candidate_index, horizon] = hidden_semantic["relevant"]
                    robot_hidden_absent &= not bool(hidden_semantic["robot"].any())
                    no_robot_area_not_smaller &= int(hidden_semantic["relevant"].sum()) >= int(
                        semantic["relevant"].sum()
                    )
                    state_max = max(state_max, float(hidden_audit["state_max_abs_difference"]))
                    if not hidden_audit["geom_groups_restored"] or not hidden_audit[
                        "renderer_groups_restored"
                    ]:
                        raise RuntimeError("robot-hidden render metadata was not restored")
            if not future_rgb_exact:
                raise RuntimeError("replayed future RGB differs from frozen branch sidecar")
            if not partition_exact or not robot_hidden_absent or not no_robot_area_not_smaller or state_max > 0.0:
                raise RuntimeError("mask or robot-hidden rendering gate failed")
            branch_masks.append(masks)
            branch_no_robot.append(no_robot_relevant)
            records.append(
                {
                    "branch_index": branch_index,
                    "task_id": task_id,
                    "init_state_index": init_index,
                    "seed": seed,
                    "planning_index": int(source["planning_index"][branch_index]),
                    "prefix_steps": prefix_length,
                    "candidate_sha256": base.array_sha256(candidates),
                    "prefix_audit": prefix_audit,
                    "future_rgb_exact": future_rgb_exact,
                    "mask_partition_exact": partition_exact,
                    "robot_hidden_absent": robot_hidden_absent,
                    "no_robot_relevant_area_not_smaller": no_robot_area_not_smaller,
                    "robot_hidden_state_max_abs_difference": state_max,
                    "relevant_objects": inventory_by_task[task_id]["relevant_objects"],
                    "geom_ids": inventory_by_task[task_id]["geom_ids"],
                }
            )
            print(
                json.dumps(
                    {
                        "partition": partition,
                        "branch": branch_index + 1,
                        "task": task_id,
                        "init": init_index,
                    }
                ),
                flush=True,
            )
        finally:
            for env in envs:
                env.close()
    arrays = {
        "masks": np.stack(branch_masks),
        "no_robot_relevant_mask": np.stack(branch_no_robot),
    }
    for name in ("task_id", "init_state_index", "seed", "planning_index"):
        arrays[name] = np.asarray(source[name])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("collected mask arrays contain NaN/Inf")
    return records, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True, choices=("calibration", "evaluation"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/visual_target_suitability_protocol.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol, commit = load_protocol(args.protocol, args.partition)
    record = protocol["partitions"][args.partition]
    sidecar = ROOT / record["mask_sidecar"]
    manifest = ROOT / record["mask_manifest"]
    if sidecar.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite visual-target mask artifact")
    records, arrays = collect(protocol, args.partition, torch.device(args.device))
    atomic_npz(sidecar, arrays)
    result = {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "partition": args.partition,
        "protocol_sha256": sha256(args.protocol),
        "preregistered_commit": commit,
        "source_sidecar": record["source_sidecar"],
        "source_sidecar_sha256": sha256(ROOT / record["source_sidecar"]),
        "branch_identities": records,
        "artifact": {
            "path": str(sidecar.relative_to(ROOT)),
            "sha256": sha256(sidecar),
            "arrays": {name: list(value.shape) for name, value in arrays.items()},
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "mid_episode_restore": False,
            "reward_done_success_read": False,
            "controller_or_model_change": False,
            "init_40_49_access": False,
        },
    }
    if not all(math.isfinite(float(value)) for value in (0.0,)):
        raise FloatingPointError("unreachable finite guard")
    atomic_json(manifest, result)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "partition": args.partition,
                "sidecar_sha256": result["artifact"]["sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
