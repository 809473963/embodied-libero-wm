#!/usr/bin/env python3
"""Resolve the LIBERO agentview world-to-upright-pixel contract."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from robosuite import macros
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

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
from src.probes.camera_projection_contract import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    absolute_alignment_pass,
    independent_projection,
    official_float_projection,
    point_to_mask_distance,
    summarize,
    to_upright,
)


PROTOCOL_VERSION = "camera-projection-contract-resolution-v1"


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


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def load_protocol(path: Path, phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected camera contract protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_LANDMARK_INVENTORY_OR_QUANTITATIVE_COMPARISON":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise RuntimeError("protocol must be committed")
    tag = protocol["frozen_upstream"]["primary_tag"]
    if subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip() != "tag":
        raise RuntimeError("upstream tag is not annotated")
    if subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip() != protocol["frozen_upstream"]["primary_commit"]:
        raise RuntimeError("upstream tag moved")
    for item in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"upstream file changed: {item['path']}")
    for item in protocol["implementation"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"implementation changed: {item['path']}")
    inventory_path = ROOT / protocol["outputs"]["landmark_inventory"]
    camera_inventory_path = ROOT / protocol["outputs"]["camera_inventory"]
    result_path = ROOT / protocol["outputs"]["diagnosis"]
    contract_path = ROOT / protocol["outputs"]["frozen_contract"]
    if phase == "inventory":
        if inventory_path.exists() or camera_inventory_path.exists() or result_path.exists() or contract_path.exists():
            raise FileExistsError("camera contract output already exists")
    else:
        if not inventory_path.exists() or not last_commit(inventory_path):
            raise RuntimeError("landmark inventory must be committed before comparison")
        inventory = json.loads(inventory_path.read_text())
        if inventory["status"] != "LANDMARK_INVENTORY_FROZEN":
            raise RuntimeError("landmark inventory did not freeze")
        if not inventory["primary_landmarks"]:
            raise RuntimeError("LANDMARK_VALIDATION_INSUFFICIENT")
        if result_path.exists() or contract_path.exists():
            raise FileExistsError("camera comparison output already exists")
    return protocol, commit


def create_env(protocol: dict[str, Any], task_id: int, init_index: int) -> tuple[Any, Any]:
    suite = benchmark.get_benchmark_dict()[protocol["scope"]["suite"]]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    seed = int(protocol["scope"]["seed_base"] + 100 * task_id + init_index)
    env, observation = base.create_episode_env(bddl, seed, trusted_init_states(suite, task_id)[init_index])
    return env, observation


def camera_record(inner: Any, camera: str) -> dict[str, Any]:
    sim = inner.sim
    camera_id = int(sim.model.camera_name2id(camera))
    return {
        "camera_name": camera,
        "camera_id": camera_id,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "fovy_degrees": float(sim.model.cam_fovy[camera_id]),
        "camera_position_world": np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64).tolist(),
        "camera_rotation_world": np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).tolist(),
        "intrinsic": np.asarray(get_camera_intrinsic_matrix(sim, camera, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64).tolist(),
        "corrected_camera_pose_world": np.asarray(get_camera_extrinsic_matrix(sim, camera), dtype=np.float64).tolist(),
        "world_to_raw_pixel": np.asarray(get_camera_transform_matrix(sim, camera, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64).tolist(),
    }


def source_trace(protocol: dict[str, Any]) -> dict[str, Any]:
    camera_utils = Path(inspect.getsourcefile(get_camera_transform_matrix) or "")
    robot_env_source = Path(inspect.getsourcefile(refresh_policy_observation) or "")
    robosuite_robot_env = ROOT / ".venv/lib/python3.10/site-packages/robosuite/environments/robot_env.py"
    mjcf_utils = ROOT / ".venv/lib/python3.10/site-packages/robosuite/utils/mjcf_utils.py"
    macros_source = ROOT / ".venv/lib/python3.10/site-packages/robosuite/macros.py"
    paths = [camera_utils, robosuite_robot_env, mjcf_utils, macros_source, robot_env_source]
    return {
        "source_files": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "robosuite_image_convention": macros.IMAGE_CONVENTION,
        "robosuite_image_convention_slice": int(IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]),
        "camera_axis_correction": "diag(1,-1,-1,1) applied on the right of the MuJoCo camera pose",
        "utility_transform": "K4 @ inverse(corrected_camera_pose_world)",
        "utility_project": "homogeneous transform, divide by z, round, return clipped (row=v, column=u)",
        "direct_render_pipeline": "sim.render result is returned unchanged because IMAGE_CONVENTION='opengl' maps to slice +1",
        "coordinate_resolution": "direct RGB remains unchanged; only raw projected v is converted by v_upright=H-1-v_raw",
        "final_formula": {
            "camera_coordinates": "[x_c,y_c,z_c] = diag(1,-1,-1) @ R_wc^T @ (X_world-C)",
            "u_upright": "fx*x_c/z_c + cx",
            "v_upright": "H - 1 - (fy*y_c/z_c + cy)",
            "depth": "z_c, required positive"
        },
        "protocol_commit": last_commit(ROOT / "configs/camera_projection_contract_protocol.json"),
        "old_four_pixel_robot_mask_margin_reused": False,
    }


def inventory(protocol: dict[str, Any], protocol_commit: str) -> dict[str, Any]:
    camera = protocol["scope"]["camera_name"]
    tasks = []
    site_visibility: dict[str, list[dict[str, Any]]] = {}
    camera_reference = None
    for task_id in protocol["scope"]["task_ids"]:
        env, observation = create_env(protocol, task_id, 31)
        try:
            inner = env.env
            observation = refresh_policy_observation(env)
            direct = render_rgb(inner)
            segmentation = render_segmentation(inner)
            record = camera_record(inner, camera)
            if camera_reference is None:
                camera_reference = record
            site_rows = []
            for site_id in range(int(inner.sim.model.nsite)):
                name = inner.sim.model.site_id2name(site_id)
                mask = (segmentation[..., 0] == 6) & (segmentation[..., 1] == site_id)
                pixels = int(mask.sum())
                row = {
                    "name": name,
                    "mujoco_type": "site",
                    "id": site_id,
                    "world_position_source": "sim.data.site_xpos[site_id]",
                    "world_position_finite": bool(np.isfinite(inner.sim.data.site_xpos[site_id]).all()),
                    "visible": pixels > 0,
                    "visible_pixels": pixels,
                    "site_size": np.asarray(inner.sim.model.site_size[site_id], dtype=np.float64).tolist(),
                    "site_type": int(inner.sim.model.site_type[site_id]),
                    "site_group": int(inner.sim.model.site_group[site_id]),
                }
                site_rows.append(row)
                site_visibility.setdefault(str(name), []).append({"task_id": task_id, **row})
            tasks.append({
                "task_id": task_id,
                "init_state_index": 31,
                "rgb_shape": list(direct.shape),
                "rgb_dtype": str(direct.dtype),
                "direct_rgb_equals_observation_bitwise": bool(np.array_equal(direct, np.asarray(observation["agentview_image"], dtype=np.uint8))),
                "direct_rgb_observation_max_abs_diff": int(np.max(np.abs(direct.astype(np.int16)-np.asarray(observation["agentview_image"], dtype=np.int16)))),
                "camera": record,
                "sites": site_rows,
            })
        finally:
            env.close()
        print(json.dumps({"phase": "landmark_inventory", "task": task_id}), flush=True)

    criteria = protocol["landmarks"]["primary_inclusion"]
    candidates = []
    for name, rows in sorted(site_visibility.items()):
        visible_all = len(rows) == 10 and all(row["visible"] for row in rows)
        compact = visible_all and all(criteria["minimum_visible_pixels"] <= row["visible_pixels"] <= criteria["maximum_visible_pixels"] for row in rows)
        allowed_name = name in protocol["landmarks"]["primary_name_allowlist"]
        include = bool(visible_all and compact and allowed_name and all(row["world_position_finite"] for row in rows))
        candidates.append({
            "name": name,
            "mujoco_type": "site",
            "id": int(rows[0]["id"]),
            "visible_task_count": int(sum(row["visible"] for row in rows)),
            "visible_pixel_range": [int(min(row["visible_pixels"] for row in rows)), int(max(row["visible_pixels"] for row in rows))],
            "primary": include,
            "reason": "explicit semantic allowlist; finite site_xpos; visible and compact in all ten tasks" if include else "excluded by frozen name, visibility, or compactness criterion",
        })
    primary = [row for row in candidates if row["primary"]]
    status = "LANDMARK_INVENTORY_FROZEN" if primary else "LANDMARK_VALIDATION_INSUFFICIENT"
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "date": protocol["date"],
        "status": status,
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/camera_projection_contract_protocol.json"),
        "source_coordinate_trace": source_trace(protocol),
        "camera_reference": camera_reference,
        "tasks": tasks,
        "landmarks": candidates,
        "primary_landmarks": primary,
        "quantitative_projection_comparison_performed": False,
        "forbidden_activity_audit": {
            "candidate_execution": False,
            "init36_39_access": False,
            "init30_35_access": False,
            "init40_49_access": False,
            "reward_done_success_or_task_outcome_read": False,
            "image_space_target_or_representation_analysis": False,
            "model_controller_or_upstream_change": False
        }
    }


def metric_groups(rows: list[dict[str, Any]], conventions: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"pooled": {}, "per_task": {}, "per_landmark": {}, "per_init": {}}
    for convention in conventions:
        result["pooled"][convention] = summarize([row["errors"][convention] for row in rows])
    for group_name, key in (("per_task", "task_id"), ("per_landmark", "landmark_name"), ("per_init", "init_state_index")):
        values = sorted({row[key] for row in rows}, key=str)
        for value in values:
            selected = [row for row in rows if row[key] == value]
            result[group_name][str(value)] = {convention: summarize([row["errors"][convention] for row in selected]) for convention in conventions}
    return result


def evaluate(protocol: dict[str, Any], protocol_commit: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    inventory_path = ROOT / protocol["outputs"]["landmark_inventory"]
    frozen_inventory = json.loads(inventory_path.read_text())
    primary = frozen_inventory["primary_landmarks"]
    conventions = protocol["conventions"]
    camera_name = protocol["scope"]["camera_name"]
    rows = []
    rgb_rows = []
    robot_mask_rows = []
    object_rows = []
    max_source_difference = 0.0
    max_depth_difference = 0.0
    camera_static_difference = 0.0
    for init_index in protocol["scope"]["init_state_indices"]:
        for task_id in protocol["scope"]["task_ids"]:
            env, observation = create_env(protocol, task_id, init_index)
            try:
                inner = env.env
                observation = refresh_policy_observation(env)
                direct = render_rgb(inner)
                segmentation = render_segmentation(inner)
                ids = mask_id_sets(inner)
                semantic = masks_from_segmentation(segmentation, ids)
                camera = camera_record(inner, camera_name)
                reference = frozen_inventory["camera_reference"]
                camera_static_difference = max(camera_static_difference, float(np.max(np.abs(np.asarray(camera["world_to_raw_pixel"])-np.asarray(reference["world_to_raw_pixel"])))))
                rgb_rows.append({
                    "task_id": task_id,
                    "init_state_index": init_index,
                    "shape": list(direct.shape),
                    "dtype": str(direct.dtype),
                    "bitwise_equal": bool(np.array_equal(direct, np.asarray(observation["agentview_image"], dtype=np.uint8))),
                    "max_abs_difference": int(np.max(np.abs(direct.astype(np.int16)-np.asarray(observation["agentview_image"], dtype=np.int16)))),
                })
                transform = np.asarray(camera["world_to_raw_pixel"], dtype=np.float64)
                for landmark in primary:
                    site_id = int(landmark["id"])
                    point = np.asarray(inner.sim.data.site_xpos[site_id], dtype=np.float64)
                    mask = (segmentation[..., 0] == 6) & (segmentation[..., 1] == site_id)
                    raw, depth = official_float_projection(point, transform)
                    independent_raw, independent_upright, independent_depth = independent_projection(
                        point,
                        np.asarray(camera["camera_position_world"]),
                        np.asarray(camera["camera_rotation_world"]),
                        float(camera["fovy_degrees"]),
                    )
                    max_source_difference = max(max_source_difference, float(np.max(np.abs(raw-independent_raw))))
                    max_depth_difference = max(max_depth_difference, abs(float(depth-independent_depth)))
                    official_integer = project_points_from_world_to_camera(point, transform, IMAGE_HEIGHT, IMAGE_WIDTH)
                    errors = {}
                    pixels = {}
                    for convention in conventions:
                        pixel = to_upright(raw, convention)
                        distance = point_to_mask_distance(pixel, mask)
                        if distance is None:
                            raise RuntimeError("primary landmark is not visible")
                        pixels[convention] = pixel.tolist()
                        errors[convention] = distance
                    selected = protocol["selected_convention"]
                    selected_pixel = np.asarray(pixels[selected])
                    rows.append({
                        "task_id": task_id,
                        "init_state_index": init_index,
                        "landmark_name": landmark["name"],
                        "landmark_id": site_id,
                        "site_visible_pixels": int(mask.sum()),
                        "world_position": point.tolist(),
                        "raw_utility_float_uv": raw.tolist(),
                        "official_rounded_clipped_row_col": np.asarray(official_integer).tolist(),
                        "independent_raw_uv": independent_raw.tolist(),
                        "independent_upright_uv": independent_upright.tolist(),
                        "camera_depth": float(depth),
                        "independent_depth": float(independent_depth),
                        "pixels": pixels,
                        "errors": errors,
                        "selected_in_frame": bool(0 <= selected_pixel[0] < IMAGE_WIDTH and 0 <= selected_pixel[1] < IMAGE_HEIGHT),
                        "finite": bool(np.isfinite(point).all() and np.isfinite(raw).all() and np.isfinite(independent_raw).all()),
                    })

                eef_id = int(inner.robots[0].eef_site_id)
                eef_point = np.asarray(inner.sim.data.site_xpos[eef_id], dtype=np.float64)
                raw, _ = official_float_projection(eef_point, transform)
                robot_mask_rows.append({convention: point_to_mask_distance(to_upright(raw, convention), semantic["robot"]) for convention in conventions})

                for name in inner.parsed_problem.get("obj_of_interest", []):
                    if name not in inner.objects_dict:
                        continue
                    body_id = int(inner.obj_body_id[name])
                    center = np.asarray(inner.sim.data.body_xpos[body_id], dtype=np.float64)
                    raw, depth = official_float_projection(center, transform)
                    pixel = to_upright(raw, protocol["selected_convention"])
                    mask = (segmentation[..., 0] == 5) & np.isin(segmentation[..., 1], list(ids["per_object"][name]))
                    distance = point_to_mask_distance(pixel, mask)
                    yx = np.argwhere(mask)
                    inside_bbox = False
                    if yx.size:
                        ymin, xmin = yx.min(axis=0); ymax, xmax = yx.max(axis=0)
                        inside_bbox = bool(xmin <= pixel[0] <= xmax and ymin <= pixel[1] <= ymax)
                    object_rows.append({"task_id": task_id, "init_state_index": init_index, "object_name": name, "body_id": body_id, "visible_pixels": int(mask.sum()), "depth": float(depth), "projected_uv": pixel.tolist(), "inside_mask": bool(distance == 0.0), "distance_to_mask_pixels": distance, "inside_visible_bbox": inside_bbox})
            finally:
                env.close()
            print(json.dumps({"phase": "projection_evaluation", "task": task_id, "init": init_index}), flush=True)

    metrics = metric_groups(rows, conventions)
    selected = protocol["selected_convention"]
    alternate = next(name for name in conventions if name != selected)
    thresholds = protocol["gates"]["absolute_landmark_alignment"]
    selected_pass = absolute_alignment_pass(metrics["pooled"][selected], thresholds)
    selected_group_pass = all(
        absolute_alignment_pass(by_convention[selected], thresholds)
        for group in (metrics["per_task"], metrics["per_landmark"], metrics["per_init"])
        for by_convention in group.values()
    )
    alternate_pass = absolute_alignment_pass(metrics["pooled"][alternate], thresholds)
    source_pass = bool(max_source_difference <= protocol["gates"]["maximum_source_vs_independent_pixel_difference"] and max_depth_difference <= protocol["gates"]["maximum_source_vs_independent_depth_difference"])
    rgb_pass = all(row["bitwise_equal"] and row["max_abs_difference"] == 0 for row in rgb_rows)
    validity_pass = all(row["finite"] and row["camera_depth"] > 0 and row["selected_in_frame"] for row in rows)
    consistency_pass = bool(len(rows) == len(primary)*40 and selected_group_pass)
    source_trace_pass = bool(frozen_inventory["source_coordinate_trace"]["coordinate_resolution"] and camera_static_difference <= protocol["gates"]["maximum_camera_matrix_drift"])
    gates = {
        "source_coordinate_trace_complete": source_trace_pass,
        "direct_rgb_orientation_verified": rgb_pass,
        "source_vs_independent_projection": source_pass,
        "selected_absolute_alignment": selected_pass,
        "alternative_rejected_by_same_absolute_gate": not alternate_pass,
        "multi_task_state_landmark_consistency": consistency_pass,
        "finite_positive_depth_in_frame": validity_pass,
    }
    if all(gates.values()):
        verdict = "CAMERA_PROJECTION_CONTRACT_VALIDATED"
        authorization = "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART"
    elif selected_pass and alternate_pass:
        verdict = "PROJECTION_CONVENTION_AMBIGUOUS"
        authorization = "BLOCK_IMAGE_SPACE_TARGET_AUDIT_RESTART"
    elif not source_pass or not source_trace_pass:
        verdict = "CAMERA_SOURCE_CONVENTION_INCONSISTENT"
        authorization = "BLOCK_IMAGE_SPACE_TARGET_AUDIT_RESTART"
    elif not primary or not consistency_pass:
        verdict = "LANDMARK_VALIDATION_INSUFFICIENT"
        authorization = "BLOCK_IMAGE_SPACE_TARGET_AUDIT_RESTART"
    else:
        verdict = "DATA_OR_IMPLEMENTATION_BLOCK"
        authorization = "BLOCK_IMAGE_SPACE_TARGET_AUDIT_RESTART"

    robot_history = {convention: summarize([float(row[convention]) for row in robot_mask_rows if row[convention] is not None]) for convention in conventions}
    object_visible = [row for row in object_rows if row["distance_to_mask_pixels"] is not None]
    object_check = {
        "role": "secondary only; body centers need not equal visible-mask centroids",
        "n": len(object_visible),
        "inside_mask_rate": float(np.mean([row["inside_mask"] for row in object_visible])) if object_visible else None,
        "inside_visible_bbox_rate": float(np.mean([row["inside_visible_bbox"] for row in object_visible])) if object_visible else None,
        "distance_to_own_mask": summarize([row["distance_to_mask_pixels"] for row in object_visible]) if object_visible else None,
    }
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "date": protocol["date"],
        "status": "COMPLETE",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/camera_projection_contract_protocol.json"),
        "landmark_inventory_sha256": sha256(inventory_path),
        "old_block_preserved": {"verdict": "DATA_OR_IMPLEMENTATION_BLOCK", "blocking_condition": "CAMERA_PROJECTION_BLOCK", "old_robot_mask_margin_threshold_pixels": 4.0, "old_result_changed": False},
        "data_boundary": {"init_state_indices": protocol["scope"]["init_state_indices"], "tasks": protocol["scope"]["task_ids"], "planning_observation_only": True, "candidate_execution": False, "init30_35_access": False, "init36_39_access": False, "init40_49_access": False},
        "coordinate_trace": frozen_inventory["source_coordinate_trace"],
        "camera": frozen_inventory["camera_reference"],
        "primary_landmarks": primary,
        "primary_rows": rows,
        "primary_metrics": metrics,
        "rgb_orientation": {"rows": rgb_rows, "all_bitwise_equal": rgb_pass, "maximum_absolute_difference": max(row["max_abs_difference"] for row in rgb_rows), "interpretation": "direct sim.render RGB already equals upright observation; image array is not flipped"},
        "source_vs_independent": {"maximum_raw_pixel_difference": max_source_difference, "maximum_depth_difference": max_depth_difference, "maximum_camera_matrix_drift": camera_static_difference, "pass": source_pass},
        "alternative_conventions": conventions,
        "selected_convention": selected,
        "historical_robot_mask_reproduction": robot_history,
        "relevant_object_center_secondary": object_check,
        "geom_footprint_secondary": {"status": "GEOM_FOOTPRINT_CHECK_NOT_IMPLEMENTED", "reason": "a new mesh/primitive rasterizer would add implementation risk and is not needed when native site segmentation supplies a tight primary reference"},
        "gates": gates,
        "decision": {"final_verdict": verdict, "authorization": authorization, "spatial_target_design_authorized": False, "world_model_reranker_controller_authorized": False},
        "nan_inf_depth_in_frame_audit": "pass" if validity_pass else "fail",
        "forbidden_activity_audit": {"candidate_execution": False, "reward_done_success_or_task_outcome_read": False, "init30_35_access": False, "init36_39_access": False, "init40_49_access": False, "image_space_target_construction": False, "global_patch_ridge_pca_or_shuffle": False, "model_reranker_controller_change": False, "upstream_overwrite": False},
    }
    contract = None
    if verdict == "CAMERA_PROJECTION_CONTRACT_VALIDATED":
        contract = {
            "schema_version": 1,
            "status": verdict,
            "source_result_sha256_pending": True,
            "camera_name": camera_name,
            "camera_id": frozen_inventory["camera_reference"]["camera_id"],
            "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "intrinsic": frozen_inventory["camera_reference"]["intrinsic"],
            "fovy_degrees": frozen_inventory["camera_reference"]["fovy_degrees"],
            "camera_position_world": frozen_inventory["camera_reference"]["camera_position_world"],
            "camera_rotation_world": frozen_inventory["camera_reference"]["camera_rotation_world"],
            "corrected_camera_pose_world": frozen_inventory["camera_reference"]["corrected_camera_pose_world"],
            "world_to_raw_pixel": frozen_inventory["camera_reference"]["world_to_raw_pixel"],
            "selected_convention": selected,
            "world_to_upright_pixel_formula": frozen_inventory["source_coordinate_trace"]["final_formula"],
            "primary_landmark_thresholds": thresholds,
            "primary_landmark_metrics": metrics,
            "source_vs_independent": result["source_vs_independent"],
            "landmark_inventory_sha256": result["landmark_inventory_sha256"],
            "protocol_sha256": result["protocol_sha256"],
            "authorization": "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART",
            "downstream_constraint": "all future 2D image-space target construction must read this frozen contract; it does not authorize target redesign, model, reranker, or controller work"
        }
    return result, contract


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("inventory", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/camera_projection_contract_protocol.json")
    args = parser.parse_args()
    protocol, commit = load_protocol(args.protocol, args.phase)
    if args.phase == "inventory":
        value = inventory(protocol, commit)
        atomic_json(ROOT / protocol["outputs"]["landmark_inventory"], value)
        camera_value = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "date": protocol["date"],
            "status": "SOURCE_CAMERA_INVENTORY_COMPLETE",
            "protocol_commit": commit,
            "protocol_sha256": sha256(ROOT / "configs/camera_projection_contract_protocol.json"),
            "source_coordinate_trace": value["source_coordinate_trace"],
            "camera_reference": value["camera_reference"],
            "task_runtime_camera_records": [
                {"task_id": row["task_id"], "init_state_index": row["init_state_index"], "camera": row["camera"], "rgb_shape": row["rgb_shape"], "rgb_dtype": row["rgb_dtype"], "direct_rgb_equals_observation_bitwise": row["direct_rgb_equals_observation_bitwise"], "direct_rgb_observation_max_abs_diff": row["direct_rgb_observation_max_abs_diff"]}
                for row in value["tasks"]
            ],
            "quantitative_projection_comparison_performed": False,
        }
        atomic_json(ROOT / protocol["outputs"]["camera_inventory"], camera_value)
        print(json.dumps({"status": value["status"], "primary": [row["name"] for row in value["primary_landmarks"]]}, indent=2))
        return 0 if value["status"] == "LANDMARK_INVENTORY_FROZEN" else 2
    result, contract = evaluate(protocol, commit)
    result_path = ROOT / protocol["outputs"]["diagnosis"]
    atomic_json(result_path, result)
    if contract is not None:
        contract["source_result_sha256"] = sha256(result_path)
        del contract["source_result_sha256_pending"]
        atomic_json(ROOT / protocol["outputs"]["frozen_contract"], contract)
    print(json.dumps(result["decision"], indent=2))
    return 0 if result["decision"]["final_verdict"] == "CAMERA_PROJECTION_CONTRACT_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
