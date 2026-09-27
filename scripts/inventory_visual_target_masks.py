#!/usr/bin/env python3
"""Inventory native MuJoCo segmentation for the visual-target audit.

This stage inspects one calibration init state per LIBERO Spatial task.  It
does not execute a candidate, read an outcome, or access the held-out branch
partition.  The resulting inventory is used to preregister the later mask
collector against the APIs and ID mappings that are actually present.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path

from scripts import audit_local_candidate_counterfactual as base
from scripts.evaluate_c0_all import refresh_policy_observation, trusted_init_states
from scripts.collect_missing_state_audit import relevant_objects


OUTPUT = ROOT / "results/visual_target_suitability/mask_inventory_v1_1.json"
CAMERA = "agentview"
HEIGHT = 128
WIDTH = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def existing_geom_ids(sim: Any, names: Iterable[str]) -> set[int]:
    result: set[int] = set()
    for name in names:
        try:
            value = int(sim.model.geom_name2id(str(name)))
        except Exception:
            continue
        if value >= 0:
            result.add(value)
    return result


def model_geom_ids(sim: Any, model: Any) -> set[int]:
    names: list[str] = []
    for attribute in ("visual_geoms", "contact_geoms"):
        names.extend(list(getattr(model, attribute, []) or []))
    return existing_geom_ids(sim, names)


def mask_id_sets(inner: Any) -> dict[str, set[int]]:
    sim = inner.sim
    robot = inner.robots[0]
    robot_ids = model_geom_ids(sim, robot.robot_model)
    robot_ids |= model_geom_ids(sim, robot.gripper)
    relevant_names = relevant_objects(inner)
    relevant_ids: set[int] = set()
    other_ids: set[int] = set()
    per_object: dict[str, set[int]] = {}
    for name, model in inner.objects_dict.items():
        ids = model_geom_ids(sim, model)
        per_object[name] = ids
        if name in relevant_names:
            relevant_ids |= ids
        else:
            other_ids |= ids
    overlaps = {
        "robot_relevant": sorted(robot_ids & relevant_ids),
        "robot_other": sorted(robot_ids & other_ids),
        "relevant_other": sorted(relevant_ids & other_ids),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"mask geom groups overlap: {overlaps}")
    return {
        "robot": robot_ids,
        "relevant": relevant_ids,
        "other": other_ids,
        "per_object": per_object,  # type: ignore[dict-item]
    }


def render_segmentation(inner: Any) -> np.ndarray:
    raw = inner.sim.render(
        camera_name=CAMERA,
        width=WIDTH,
        height=HEIGHT,
        depth=False,
        segmentation=True,
    )
    # This installed robosuite uses the direct OpenCV-style sensor convention.
    # Inventory v1 proved direct sim.render RGB is bitwise identical to the
    # stored upright agentview; applying [::-1] would invert the mask.
    result = np.asarray(raw, dtype=np.int32)
    if result.shape != (HEIGHT, WIDTH, 2):
        raise RuntimeError(f"unexpected segmentation shape {result.shape}")
    return result


def render_rgb(inner: Any) -> np.ndarray:
    raw = inner.sim.render(
        camera_name=CAMERA,
        width=WIDTH,
        height=HEIGHT,
        depth=False,
        segmentation=False,
    )
    return np.asarray(raw, dtype=np.uint8)


def masks_from_segmentation(segmentation: np.ndarray, ids: dict[str, set[int]]) -> dict[str, np.ndarray]:
    geom = segmentation[..., 1]
    robot = np.isin(geom, list(ids["robot"]))
    relevant = np.isin(geom, list(ids["relevant"]))
    other = np.isin(geom, list(ids["other"]))
    background = ~(robot | relevant | other)
    return {
        "robot": robot,
        "relevant": relevant,
        "other": other,
        "background": background,
    }


def render_without_robot(inner: Any, robot_ids: set[int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Render an element mask with robot visual geoms in a disabled group.

    MuJoCo exposes six renderer geom groups.  We temporarily move the robot
    geoms to an otherwise unused group, disable it in the offscreen renderer,
    render, and restore both model and renderer metadata in a finally block.
    Physics state is never changed or restored.
    """

    sim = inner.sim
    if sim._render_context_offscreen is None:
        render_segmentation(inner)
    context = sim._render_context_offscreen
    original_groups = np.asarray(sim.model.geom_group, dtype=np.int32).copy()
    original_vopt = np.asarray(context.vopt.geomgroup, dtype=np.uint8).copy()
    used = set(int(value) for index, value in enumerate(original_groups) if index not in robot_ids)
    candidates = [group for group in range(len(original_vopt)) if group not in used]
    if not candidates:
        raise RuntimeError("no unused MuJoCo geom group is available for robot-only hiding")
    hidden_group = int(candidates[-1])
    state_before = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    try:
        sim.model.geom_group[list(sorted(robot_ids))] = hidden_group
        context.vopt.geomgroup[hidden_group] = 0
        segmentation = render_segmentation(inner)
    finally:
        sim.model.geom_group[:] = original_groups
        context.vopt.geomgroup[:] = original_vopt
    state_after = np.asarray(sim.get_state().flatten(), dtype=np.float64)
    return segmentation, {
        "hidden_group": hidden_group,
        "nonrobot_groups_before": sorted(used),
        "state_max_abs_difference": float(np.max(np.abs(state_before - state_after))),
        "geom_groups_restored": bool(np.array_equal(sim.model.geom_group, original_groups)),
        "renderer_groups_restored": bool(np.array_equal(context.vopt.geomgroup, original_vopt)),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("mask inventory requires a clean worktree")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    tasks: list[dict[str, Any]] = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = trusted_init_states(suite, task_id)[31]
        seed = 30000 + 100 * task_id + 1
        env, observation = base.create_episode_env(bddl, seed, init_state)
        try:
            observation = refresh_policy_observation(env)
            inner = env.env
            segmentation = render_segmentation(inner)
            direct_rgb = render_rgb(inner)
            image = np.asarray(observation["agentview_image"], dtype=np.uint8)
            ids = mask_id_sets(inner)
            masks = masks_from_segmentation(segmentation, ids)
            no_robot_seg, hide_audit = render_without_robot(inner, ids["robot"])
            no_robot_masks = masks_from_segmentation(no_robot_seg, ids)
            normal_relevant = int(masks["relevant"].sum())
            revealed_relevant = int(no_robot_masks["relevant"].sum())
            unique_geom_ids = sorted(int(value) for value in np.unique(segmentation[..., 1]))
            valid_unique = [value for value in unique_geom_ids if 0 <= value < int(inner.sim.model.ngeom)]
            unknown_visible = [
                value
                for value in valid_unique
                if value not in ids["robot"] | ids["relevant"] | ids["other"]
            ]
            partition_sum = sum(mask.astype(np.int8) for mask in masks.values())
            tasks.append(
                {
                    "task_id": task_id,
                    "task_language": task.language,
                    "init_state_index": 31,
                    "seed": seed,
                    "camera": CAMERA,
                    "rgb_matches_observation_bitwise": bool(np.array_equal(direct_rgb, image)),
                    "segmentation_shape": list(segmentation.shape),
                    "segmentation_dtype": str(segmentation.dtype),
                    "segmentation_type_ids": sorted(int(value) for value in np.unique(segmentation[..., 0])),
                    "segmentation_geom_ids": unique_geom_ids,
                    "segmentation_valid_geom_ids": valid_unique,
                    "segmentation_invalid_or_background_ids": [value for value in unique_geom_ids if value not in valid_unique],
                    "ngeom": int(inner.sim.model.ngeom),
                    "relevant_objects": relevant_objects(inner),
                    "all_objects": list(inner.objects_dict),
                    "geom_ids": {
                        "robot": sorted(ids["robot"]),
                        "relevant": sorted(ids["relevant"]),
                        "other_object": sorted(ids["other"]),
                    },
                    "geom_names": {
                        key: [inner.sim.model.geom_id2name(value) for value in sorted(ids[key])]
                        for key in ("robot", "relevant", "other")
                    },
                    "visible_pixels": {key: int(value.sum()) for key, value in masks.items()},
                    "partition_exact": bool(np.all(partition_sum == 1)),
                    "unknown_visible_geom_ids_classified_as_background": unknown_visible,
                    "robot_hidden_render": {
                        **hide_audit,
                        "normal_relevant_visible_pixels": normal_relevant,
                        "without_robot_relevant_visible_pixels": revealed_relevant,
                        "revealed_relevant_pixels": max(revealed_relevant - normal_relevant, 0),
                        "no_robot_area_not_smaller": revealed_relevant >= normal_relevant,
                    },
                }
            )
        finally:
            env.close()
        print(json.dumps({"task": task_id, "inventory": "complete"}), flush=True)
    result = {
        "schema_version": 1,
        "experiment_version": "visual-target-suitability-audit-mask-inventory-v1.1",
        "status": "COMPLETE",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "task0-9, calibration init31 only; reset/set-init and render only",
        "implementation": {
            "element_segmentation": "direct sim.render(..., segmentation=True); no vertical flip because direct RGB is bitwise identical to upright agentview in this installed robosuite",
            "robot_mask": "robot_model and gripper visual/contact geom IDs",
            "relevant_object_mask": "parsed_problem['obj_of_interest'] object visual/contact geom IDs",
            "other_object_mask": "remaining objects_dict visual/contact geom IDs",
            "background_mask": "complement, including fixtures/table/empty image",
            "occlusion_probe": "temporary robot geom-group hiding for segmentation render; exact metadata restoration; no physics-state mutation",
        },
        "tasks": tasks,
        "aggregate_checks": {
            "rgb_orientation_matches_all_tasks": all(row["rgb_matches_observation_bitwise"] for row in tasks),
            "mask_partition_exact_all_tasks": all(row["partition_exact"] for row in tasks),
            "robot_hide_state_unchanged_all_tasks": all(row["robot_hidden_render"]["state_max_abs_difference"] == 0.0 for row in tasks),
            "robot_hide_metadata_restored_all_tasks": all(
                row["robot_hidden_render"]["geom_groups_restored"]
                and row["robot_hidden_render"]["renderer_groups_restored"]
                for row in tasks
            ),
            "robot_hide_relevant_area_not_smaller_all_tasks": all(
                row["robot_hidden_render"]["no_robot_area_not_smaller"] for row in tasks
            ),
        },
        "forbidden_activity_audit": {
            "candidate_execution": False,
            "held_out_init_36_39_access": False,
            "init_40_49_access": False,
            "reward_done_success_read": False,
            "model_or_controller_change": False,
        },
        "correction_history": {
            "blocked_v1": "results/visual_target_suitability/mask_inventory.json",
            "change": "Removed one erroneous vertical flip after the v1 RGB orientation gate failed; no candidate or held-out branch was accessed.",
        },
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": "COMPLETE", "output": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
