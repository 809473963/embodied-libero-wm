#!/usr/bin/env python3
"""Inventory the camera and collect missing per-object natural masks.

Only calibration init31 is used for camera inventory.  Branch collection
replays the already frozen candidates and writes per-object semantic masks;
it never reads reward, done, success, or task outcomes.
"""

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
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_camera_transform_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path  # noqa: E402

from scripts import audit_local_candidate_counterfactual as base  # noqa: E402
from scripts.evaluate_c0_all import refresh_policy_observation, trusted_init_states  # noqa: E402
from scripts.inventory_visual_target_masks import (  # noqa: E402
    mask_id_sets,
    masks_from_segmentation,
    render_rgb,
    render_segmentation,
)
from src.probes.image_space_relevance import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    point_to_mask_distance,
    project_world_points,
)
from src.probes.visual_target import MASK_NAMES  # noqa: E402


PROTOCOL_VERSION = "natural-image-space-interaction-target-audit-v1"


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


def load_protocol(path: Path, phase: str, partition: str | None) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected image-space protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_CAMERA_INVENTORY_OR_TARGET_RESULTS":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise RuntimeError("protocol must be committed before camera or branch access")
    tag = protocol["frozen_upstream"]["primary_tag"]
    tag_type = subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip()
    tag_commit = subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip()
    if tag_type != "tag" or tag_commit != protocol["frozen_upstream"]["primary_commit"]:
        raise RuntimeError("upstream annotated tag changed")
    for item in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen upstream changed: {item['path']}")
    for item in protocol["implementation"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"implementation changed after preregistration: {item['path']}")
    inventory_path = ROOT / protocol["outputs"]["camera_inventory"]
    if phase == "inventory":
        if inventory_path.exists():
            raise FileExistsError(inventory_path)
    else:
        if not inventory_path.exists() or not last_commit(inventory_path):
            raise RuntimeError("camera inventory must be committed before branch collection")
        inventory = json.loads(inventory_path.read_text())
        if inventory["status"] != "CAMERA_PROJECTION_VALIDATED":
            raise RuntimeError("camera projection gate did not pass")
        if partition not in {"calibration", "development_evaluation"}:
            raise ValueError("invalid collection partition")
        if partition == "development_evaluation":
            freeze = ROOT / protocol["outputs"]["calibration_freeze"]
            if not freeze.exists() or not last_commit(freeze):
                raise RuntimeError("calibration freeze must be committed before evaluation replay")
    return protocol, commit


def _camera_record(inner: Any, camera: str) -> dict[str, Any]:
    sim = inner.sim
    camera_id = int(sim.model.camera_name2id(camera))
    intrinsic = get_camera_intrinsic_matrix(sim, camera, IMAGE_HEIGHT, IMAGE_WIDTH)
    extrinsic = get_camera_extrinsic_matrix(sim, camera)
    transform = get_camera_transform_matrix(sim, camera, IMAGE_HEIGHT, IMAGE_WIDTH)
    return {
        "camera_id": camera_id,
        "fovy_degrees": float(sim.model.cam_fovy[camera_id]),
        "camera_position_world": np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64).tolist(),
        "camera_rotation_world": np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).tolist(),
        "intrinsic": np.asarray(intrinsic, dtype=np.float64).tolist(),
        "camera_pose_extrinsic": np.asarray(extrinsic, dtype=np.float64).tolist(),
        "world_to_pixel": np.asarray(transform, dtype=np.float64).tolist(),
    }


def inventory_camera(protocol: dict[str, Any], protocol_commit: str) -> dict[str, Any]:
    scope = protocol["scope"]
    camera = protocol["camera_projection"]["camera_name"]
    suite = benchmark.get_benchmark_dict()[scope["suite"]]()
    task_records = []
    convention_distances = {"direct_render": [], "vertical_flip_to_direct_render": []}
    for task_id in scope["task_ids"]:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = trusted_init_states(suite, task_id)[31]
        seed = 30000 + 100 * task_id + 1
        env, _ = base.create_episode_env(bddl, seed, init_state)
        try:
            observation = refresh_policy_observation(env)
            inner = env.env
            direct_rgb = render_rgb(inner)
            segmentation = render_segmentation(inner)
            ids = mask_id_sets(inner)
            semantic = masks_from_segmentation(segmentation, ids)
            camera_record = _camera_record(inner, camera)
            ee_site_id = int(inner.robots[0].eef_site_id)
            ee_world = np.asarray(inner.sim.data.site_xpos[ee_site_id], dtype=np.float64)
            convention_records = {}
            for convention in convention_distances:
                projected, depth = project_world_points(
                    ee_world,
                    np.asarray(camera_record["world_to_pixel"], dtype=np.float64),
                    convention,
                )
                distance = point_to_mask_distance(projected, semantic["robot"])
                if distance is None:
                    raise RuntimeError("robot mask is empty during projection inventory")
                convention_distances[convention].append(distance)
                convention_records[convention] = {
                    "projected_uv": projected.tolist(),
                    "depth": float(depth),
                    "distance_to_robot_mask_pixels": distance,
                    "inside_image": bool(
                        0.0 <= projected[0] < IMAGE_WIDTH and 0.0 <= projected[1] < IMAGE_HEIGHT
                    ),
                }
            site_mask = (segmentation[..., 0] == 6) & (segmentation[..., 1] == ee_site_id)
            selected_site_distances = {}
            for convention in convention_distances:
                projected = np.asarray(convention_records[convention]["projected_uv"])
                selected_site_distances[convention] = point_to_mask_distance(projected, site_mask)
            task_records.append(
                {
                    "task_id": task_id,
                    "init_state_index": 31,
                    "seed": seed,
                    "camera_name": camera,
                    "image_width": IMAGE_WIDTH,
                    "image_height": IMAGE_HEIGHT,
                    "pixel_origin": "top-left of direct upright agentview RGB; u rightward, v downward",
                    "projected_coordinate_order": "(u, v)",
                    "world_axis_convention": "MuJoCo world xyz",
                    "camera_axis_correction": "robosuite get_camera_extrinsic_matrix diag(1,-1,-1,1)",
                    "rgb_matches_observation_bitwise": bool(
                        np.array_equal(direct_rgb, np.asarray(observation["agentview_image"], dtype=np.uint8))
                    ),
                    "eef_site_id": ee_site_id,
                    "eef_site_name": inner.sim.model.site_id2name(ee_site_id),
                    "eef_world_position": ee_world.tolist(),
                    "site_segmentation_visible_pixels": int(site_mask.sum()),
                    "site_distance_pixels": selected_site_distances,
                    "projection_candidates": convention_records,
                    **camera_record,
                }
            )
        finally:
            env.close()
        print(json.dumps({"phase": "camera_inventory", "task": task_id}), flush=True)

    summaries = {
        name: {
            "median_distance_to_robot_mask_pixels": float(np.median(values)),
            "p95_distance_to_robot_mask_pixels": float(np.quantile(values, 0.95)),
            "max_distance_to_robot_mask_pixels": float(np.max(values)),
        }
        for name, values in convention_distances.items()
    }
    selected = min(
        summaries,
        key=lambda name: (
            summaries[name]["median_distance_to_robot_mask_pixels"],
            name != "direct_render",
        ),
    )
    gate = protocol["camera_projection"]["validation_gate"]
    selected_summary = summaries[selected]
    alternate = next(name for name in summaries if name != selected)
    median_margin = (
        summaries[alternate]["median_distance_to_robot_mask_pixels"]
        - selected_summary["median_distance_to_robot_mask_pixels"]
    )
    positive_depth = all(
        row["projection_candidates"][selected]["depth"] > 0.0 for row in task_records
    )
    inside = all(
        row["projection_candidates"][selected]["inside_image"] for row in task_records
    )
    passed = bool(
        all(row["rgb_matches_observation_bitwise"] for row in task_records)
        and all(row["eef_site_name"] == "gripper0_grip_site" for row in task_records)
        and positive_depth
        and inside
        and selected_summary["median_distance_to_robot_mask_pixels"]
        <= float(gate["maximum_median_robot_mask_distance_pixels"])
        and selected_summary["p95_distance_to_robot_mask_pixels"]
        <= float(gate["maximum_p95_robot_mask_distance_pixels"])
        and median_margin >= float(gate["minimum_median_margin_over_alternative_pixels"])
    )
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "date": protocol["date"],
        "status": "CAMERA_PROJECTION_VALIDATED" if passed else "CAMERA_PROJECTION_BLOCK",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/natural_image_space_interaction_protocol.json"),
        "camera_name": camera,
        "selected_projection_convention": selected,
        "candidate_conventions": summaries,
        "selected_median_margin_pixels": median_margin,
        "projection_contract": {
            "world_to_pixel": "robosuite intrinsic @ inverse(corrected camera pose)",
            "output": "unclipped floating (u,v), top-left origin, u right, v down",
            "normalization": "u / image_width; v / image_height",
        },
        "gate": {**gate, "pass": passed},
        "tasks": task_records,
        "forbidden_activity_audit": {
            "candidate_execution": False,
            "evaluation_init36_39_access": False,
            "init40_49_access": False,
            "reward_done_success_or_task_outcome_read": False,
            "model_or_controller_change": False,
        },
    }


def _per_object_masks(segmentation: np.ndarray, ids: dict[str, Any], names: list[str]) -> np.ndarray:
    is_geom = segmentation[..., 0] == 5
    geom = segmentation[..., 1]
    result = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
    for index, name in enumerate(names[:2]):
        result[index] = is_geom & np.isin(geom, list(ids["per_object"][name]))
    return result


def collect_partition(
    protocol: dict[str, Any],
    partition: str,
    protocol_commit: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record = protocol["partitions"][partition]
    source_file = np.load(ROOT / record["branch_sidecar"], allow_pickle=False)
    source = {name: np.asarray(source_file[name]) for name in source_file.files}
    frozen_mask_file = np.load(ROOT / record["frozen_masks"], allow_pickle=False)
    frozen_masks = {name: np.asarray(frozen_mask_file[name]) for name in frozen_mask_file.files}
    state_file = np.load(ROOT / record["state_sidecar"], allow_pickle=False)
    state = {name: np.asarray(state_file[name]) for name in state_file.files}
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        if not np.array_equal(source[key], frozen_masks[key]) or not np.array_equal(source[key], state[key]):
            raise RuntimeError(f"frozen partition identity mismatch: {key}")
    suite = benchmark.get_benchmark_dict()[protocol["scope"]["suite"]]()
    inventory = json.loads((ROOT / protocol["outputs"]["camera_inventory"]).read_text())
    inventory_by_task = {int(row["task_id"]): row for row in inventory["tasks"]}
    all_masks = []
    records = []
    for branch in range(source["task_id"].shape[0]):
        task_id = int(source["task_id"][branch])
        init_index = int(source["init_state_index"][branch])
        seed = int(source["seed"][branch])
        prefix_length = int(source["prefix_lengths"][branch])
        prefix = np.asarray(source["prefix_actions"][branch, :prefix_length], dtype=np.float32)
        candidates = np.asarray(source["candidates"][branch], dtype=np.float32)
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = trusted_init_states(suite, task_id)[init_index]
        envs = []
        try:
            observations, simulator_states = [], []
            for _ in range(3):
                env, observation, sim_state = base.replay_prefix(bddl, seed, init_state, prefix)
                envs.append(env)
                observations.append(observation)
                simulator_states.append(sim_state)
            reference_rgb = np.asarray(observations[0]["agentview_image"], dtype=np.uint8)
            if not all(
                np.array_equal(reference_rgb, np.asarray(obs["agentview_image"], dtype=np.uint8))
                for obs in observations[1:]
            ):
                raise RuntimeError("fresh branch planning RGB identity failed")
            if not all(np.array_equal(simulator_states[0], value) for value in simulator_states[1:]):
                raise RuntimeError("fresh branch simulator-state identity failed")
            per_candidate_ids = [mask_id_sets(env.env) for env in envs]
            object_names = [str(name) for name in state["object_names"][branch] if str(name)]
            if object_names != list(envs[0].env.parsed_problem.get("obj_of_interest", [])):
                raise RuntimeError("relevant-object identity changed")
            masks = np.zeros((3, 8, 2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
            rgb_exact = True
            merged_exact = True
            state_exact = True
            camera_static = True
            for horizon in range(8):
                for candidate, env in enumerate(envs):
                    env.step(candidates[candidate, horizon])
                    observation = refresh_policy_observation(env)
                    expected_rgb = np.asarray(source["real_rgb"][branch, candidate, horizon])
                    actual_rgb = np.asarray(observation["agentview_image"], dtype=np.uint8)
                    rgb_exact &= bool(np.array_equal(actual_rgb, expected_rgb))
                    rgb_exact &= bool(np.array_equal(render_rgb(env.env), expected_rgb))
                    segmentation = render_segmentation(env.env)
                    ids = per_candidate_ids[candidate]
                    semantic = masks_from_segmentation(segmentation, ids)
                    stacked = np.stack([semantic[name] for name in MASK_NAMES])
                    merged_exact &= bool(
                        np.array_equal(stacked, frozen_masks["masks"][branch, candidate, horizon])
                    )
                    per_object = _per_object_masks(segmentation, ids, object_names)
                    union = np.any(per_object[np.asarray(state["object_mask"][branch], dtype=bool)], axis=0)
                    merged_exact &= bool(np.array_equal(union, semantic["relevant"]))
                    masks[candidate, horizon] = per_object
                    ee_site = int(env.env.robots[0].eef_site_id)
                    actual_ee = np.asarray(env.env.sim.data.site_xpos[ee_site], dtype=np.float64)
                    expected_ee = np.asarray(state["ee_pos"][branch, candidate, horizon], dtype=np.float64)
                    state_exact &= bool(np.max(np.abs(actual_ee - expected_ee)) <= 1.0e-12)
                    current_camera = _camera_record(env.env, protocol["camera_projection"]["camera_name"])
                    camera_static &= bool(
                        np.max(
                            np.abs(
                                np.asarray(current_camera["world_to_pixel"])
                                - np.asarray(inventory_by_task[task_id]["world_to_pixel"])
                            )
                        )
                        <= 1.0e-12
                    )
            if not rgb_exact or not merged_exact or not state_exact or not camera_static:
                raise RuntimeError("frozen replay/per-object mask gate failed")
            all_masks.append(masks)
            records.append(
                {
                    "branch_index": branch,
                    "task_id": task_id,
                    "init_state_index": init_index,
                    "seed": seed,
                    "planning_index": int(source["planning_index"][branch]),
                    "prefix_steps": prefix_length,
                    "candidate_sha256": base.array_sha256(candidates),
                    "relevant_objects": object_names,
                    "planning_rgb_identical_across_fresh_branches": True,
                    "simulator_state_identical_across_fresh_branches": True,
                    "future_rgb_matches_frozen": rgb_exact,
                    "merged_masks_match_frozen": merged_exact,
                    "ee_world_state_matches_frozen": state_exact,
                    "camera_transform_static": camera_static,
                }
            )
        finally:
            for env in envs:
                env.close()
        print(
            json.dumps(
                {
                    "phase": "per_object_mask_collection",
                    "partition": partition,
                    "branch": branch + 1,
                    "task": task_id,
                    "init": init_index,
                }
            ),
            flush=True,
        )
    arrays = {"per_object_masks": np.stack(all_masks)}
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        arrays[key] = np.asarray(source[key])
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "partition": partition,
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/natural_image_space_interaction_protocol.json"),
        "reason_for_replay": "frozen semantic sidecars contain only merged relevant-object masks; object-wise natural masks and verified camera projection were absent",
        "branch_identities": records,
        "forbidden_activity_audit": {
            "masked_clip_inference": False,
            "reward_done_success_or_task_outcome_read": False,
            "model_or_controller_change": False,
            "init40_49_access": False,
        },
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("per-object sidecar contains NaN/Inf")
    return manifest, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("inventory", "collect"))
    parser.add_argument("--partition", choices=("calibration", "development_evaluation"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/natural_image_space_interaction_protocol.json",
    )
    args = parser.parse_args()
    protocol, commit = load_protocol(args.protocol, args.phase, args.partition)
    if args.phase == "inventory":
        result = inventory_camera(protocol, commit)
        output = ROOT / protocol["outputs"]["camera_inventory"]
        atomic_json(output, result)
        print(json.dumps({"status": result["status"], "sha256": sha256(output)}, indent=2))
        return 0 if result["status"] == "CAMERA_PROJECTION_VALIDATED" else 2
    if args.partition is None:
        raise ValueError("--partition is required for collection")
    record = protocol["partitions"][args.partition]
    sidecar = ROOT / record["per_object_mask_sidecar"]
    manifest_path = ROOT / record["per_object_mask_manifest"]
    if sidecar.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite image-space collection outputs")
    manifest, arrays = collect_partition(protocol, args.partition, commit)
    atomic_npz(sidecar, arrays)
    manifest["artifact"] = {
        "path": str(sidecar.relative_to(ROOT)),
        "sha256": sha256(sidecar),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "sha256": manifest["artifact"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

