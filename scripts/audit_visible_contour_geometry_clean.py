#!/usr/bin/env python3
"""Repeat the fixed visible-mask geometry audit after removing renderer speckles.

The first audit is retained unchanged.  This diagnostic uses the largest
4-connected component of each *already collected* physical-instance mask.
It neither reads a new scene nor uses a position, depth, pose, or RGB label to
choose that component.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_visible_contour_geometry import (  # frozen first-attempt primitives
    fit_shape,
    perturb,
)
from scripts.collect_visible_correspondence import NAMES
from scripts.diagnose_multi_object_perception import inputs, read_npz, stats
from scripts.train_unified_object_state_cv import EDGES, evaluate
from src.probes.candidate_utility_oracle import atomic_json, sha256

OUT = ROOT / "results/visible_correspondence_clean_component_audit"
SOURCE = ROOT / "results/visible_correspondence_audit"


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep one deterministic observed component; never consult scene state."""
    labels_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=4
    )
    if labels_count <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = component_stats[1:, cv2.CC_STAT_AREA]
    # np.argmax resolves equal areas to the smallest scan-order label.
    label = 1 + int(np.argmax(areas))
    return labels == label


def component_summary(raw: np.ndarray, clean: np.ndarray) -> dict[str, int]:
    count, _, stats_table, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=4)
    return {
        "raw_pixels": int(raw.sum()),
        "kept_pixels": int(clean.sum()),
        "discarded_pixels": int(raw.sum() - clean.sum()),
        "component_count": int(count - 1),
        "largest_component_pixels": int(stats_table[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0,
    }


def main() -> None:
    cfg = json.loads((ROOT / "configs/visible_contour_geometry_clean_protocol.json").read_text())
    for path, expected in cfg["frozen_files"].items():
        assert sha256(ROOT / path) == expected, path
    if (OUT / "runtime.json").exists():
        raise RuntimeError("Existing clean-component attempt retained")
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = {"status": "RUNNING", "completed": 0, "training": False, "started_epoch": time.time()}
    atomic_json(OUT / "runtime.json", runtime)
    try:
        dataset, nodes, _ = inputs()
        masks_file = read_npz(SOURCE / "visible_masks.npz")
        cases = read_npz(SOURCE / "geometry_cases.npz")
        hulls = read_npz(SOURCE / "known_mesh_hulls.npz")
        indices = cases["indices"]
        roots = nodes["roots"]
        truth = nodes["positions"]
        raw_masks = np.unpackbits(masks_file["packed_masks"][indices], axis=-1)[..., :128].astype(bool)
        clean_masks = np.zeros_like(raw_masks)
        for frame in range(len(indices)):
            for obj in range(5):
                clean_masks[frame, obj] = largest_component(raw_masks[frame, obj])
        component_rows = [[component_summary(raw_masks[i, o], clean_masks[i, o]) for o in range(5)] for i in range(len(indices))]

        camera = json.loads((SOURCE / "collection_manifest.json").read_text())["camera"]
        K = np.asarray(camera["intrinsic"], dtype=float)
        pose = np.asarray(camera["corrected_camera_pose_world"], dtype=float)
        rotations = np.zeros((3, 5, 3, 3), dtype=float)
        for fold in range(3):
            fit = dataset["fold"] != fold
            weights = np.asarray([1 / (roots[fit] == root).sum() for root in roots[fit]])
            for obj in range(5):
                quaternions_xyzw = nodes["object_pose_velocity"][fit, obj][:, [4, 5, 6, 3]]
                rotations[fold, obj] = Rotation.from_quat(quaternions_xyzw).mean(weights=weights).as_matrix()

        variants = ["bbox_native", "contour_native", "contour_erode1", "contour_dilate1", "contour_res64"]
        results: dict[str, object] = {}
        arrays: dict[str, np.ndarray] = {}
        cache: dict[tuple[int, int, str, bytes], object] = {}
        old_shared = read_npz(ROOT / "results/unified_object_state_cv/oof_shared_ensemble.npz")
        old_independent = read_npz(ROOT / "results/unified_object_state_cv/oof_independent_ensemble.npz")
        old_mean = read_npz(ROOT / "results/unified_object_state_cv/oof_mean_position.npz")
        for variant in variants:
            predicted = np.full((len(indices), 5, 3), np.nan)
            residual = np.full((len(indices), 5), np.nan)
            valid = np.zeros((len(indices), 5), dtype=bool)
            boxes = np.full((len(indices), 5, 4), np.nan)
            mode = "bbox" if variant.startswith("bbox") else "contour"
            perturbation = variant.split("_")[1]
            for row, index in enumerate(indices):
                fold = int(dataset["fold"][index])
                for obj in range(5):
                    mask = perturb(clean_masks[row, obj], perturbation)
                    key = (fold, obj, mode, mask.tobytes())
                    if key not in cache:
                        cache[key] = fit_shape(mask, hulls[f"object{obj}"], rotations[fold, obj], K, pose, mode)
                    solution = cache[key]
                    if solution is not None:
                        predicted[row, obj] = solution["xyz"]
                        residual[row, obj] = solution["residual_rms"]
                        valid[row, obj] = bool(solution["solver_success"] and solution["positive_depth"])
                        boxes[row, obj] = solution["projected_bbox"]
                runtime["completed"] += 1
                if row % 20 == 0:
                    runtime["variant"] = variant
                    atomic_json(OUT / "runtime.json", runtime)
            error = np.linalg.norm(predicted - truth[indices, :5], axis=2) * 1000
            complete = valid.all(axis=1)
            hybrid = old_shared["nodes"][indices].copy()
            object_nodes = hybrid[:, :5].copy()
            object_nodes[valid] = predicted[valid]
            hybrid[:, :5] = object_nodes
            edges = np.stack([hybrid[:, left] - hybrid[:, right] for left, right in EDGES], axis=1)
            results[variant] = {
                "valid_counts": valid.sum(axis=0).tolist(),
                "complete_frames": int(complete.sum()),
                "per_object_valid_only_mm": {
                    NAMES[obj]: stats(error[valid[:, obj], obj], roots[indices][valid[:, obj]]) for obj in range(5)
                },
                "all_rows_with_original_model_fallback": evaluate(hybrid, edges, truth[indices], nodes["relations"][indices], roots[indices]),
                "residual_rms_valid_only": [
                    float(residual[valid[:, obj], obj].mean()) if valid[:, obj].any() else None for obj in range(5)
                ],
            }
            arrays[variant] = predicted
            arrays[variant + "_valid"] = valid
            arrays[variant + "_bbox"] = boxes
        np.savez_compressed(OUT / "predictions.npz", indices=indices, clean_masks=clean_masks, rotations=rotations, **arrays)
        atomic_json(OUT / "component_accounting.json", {
            "rule": "For each observed instance raster, retain its largest 4-connected component; ties choose the smallest scan-order label.",
            "selection_inputs": "mask pixels only; no RGB, GT state, depth, pose, or predicted position.",
            "frames": int(len(indices)), "objects": NAMES, "per_frame_object": component_rows,
        })
        atomic_json(OUT / "evaluation.json", {
            "variants": results,
            "baselines_same104frames": {
                name: evaluate(item["nodes"][indices], item["edges"][indices], truth[indices], nodes["relations"][indices], roots[indices])
                for name, item in (("shared", old_shared), ("independent", old_independent), ("mean", old_mean))
            },
            "interpretation": "A renderer-artifact robustness diagnostic. It remains an oracle because the component instance mask and physical identity are supplied. It is not a trained detector or a deployment result.",
            "known_limit": "The existing calibrated-camera/CAD chain is separately audited here; low residual does not establish correct depth or natural-image recoverability.",
        })
        runtime.update(status="COMPLETE", finished_epoch=time.time())
        atomic_json(OUT / "runtime.json", runtime)
    except Exception:
        runtime.update(status="BLOCKED", error=traceback.format_exc())
        atomic_json(OUT / "runtime.json", runtime)
        raise


if __name__ == "__main__":
    main()
