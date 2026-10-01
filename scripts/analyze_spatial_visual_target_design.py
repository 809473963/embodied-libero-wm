#!/usr/bin/env python3
"""Calibrate and evaluate the preregistered Spatial Visual Target Design Study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probes.missing_state import (
    fit_ridge,
    predict_ridge,
    ridge_from_arrays,
    ridge_to_arrays,
    scalar_metrics,
    select_ridge_alpha,
)
from src.probes.spatial_relevance import (
    PAIR_NAMES,
    fit_randomized_pca,
    pca_from_arrays,
    pca_to_arrays,
    safe_correlation,
    safe_spearman,
)
from src.probes.spatial_target_design import (
    footprint_from_object_masks,
    generic_pair_summary,
    metadata_from_derived,
    patch_global_residual_state,
    pca_patch_state,
    predict_proxy,
    proxy_from_arrays,
    proxy_to_arrays,
    recovery_metrics,
    reshape_patch_grid,
    spatial_pair_summary,
    fit_proxy,
    temporal_and_candidate_distance,
)


PROTOCOL_VERSION = "spatial-visual-target-design-study-v1"
PROTOCOL_PATH = ROOT / "configs/spatial_visual_target_design_protocol.json"
RAW_TARGETS = ("object_mask_difference", "visible_area_difference", "ee_image_displacement")
RESIDUAL_TARGETS = ("object_mask_residual", "visible_area_residual")
ALL_TARGETS = RAW_TARGETS + RESIDUAL_TARGETS
UNIQUE_REPRESENTATIONS = (
    "global",
    "full_patch_reference",
    "generic_spatial",
    "global_plus_generic",
    "footprint_spatial",
    "global_plus_footprint",
    "proxy_spatial",
    "global_plus_proxy",
)


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def load_protocol(phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected spatial-target protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_CALIBRATION_OR_EVALUATION_RESULTS":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(PROTOCOL_PATH)
    if not commit:
        raise RuntimeError("protocol must be committed")
    tag = protocol["frozen_upstream"]["tag"]
    if subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip() != "tag":
        raise RuntimeError("upstream tag is not annotated")
    if subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip() != protocol["frozen_upstream"]["commit"]:
        raise RuntimeError("upstream tag moved")
    for record in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen upstream changed: {record['path']}")
    for record in protocol["implementation"].values():
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"implementation changed: {record['path']}")
    outputs = protocol["outputs"]
    if phase == "calibrate":
        for name in ("calibration_weights", "calibration_derived", "calibration_freeze", "diagnosis"):
            if (ROOT / outputs[name]).exists():
                raise FileExistsError(outputs[name])
    else:
        freeze_path = ROOT / outputs["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("calibration freeze must be committed")
        freeze = json.loads(freeze_path.read_text())
        if freeze["status"] != "FROZEN_BEFORE_INIT36_39_TARGET_DESIGN_EVALUATION":
            raise RuntimeError("calibration freeze status changed")
        if freeze["protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise RuntimeError("calibration freeze references another protocol")
        for record in freeze["artifacts"].values():
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"calibration artifact changed: {record['path']}")
    return protocol, commit


def load_partition(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    record = protocol["partitions"][name]
    features = load_npz(ROOT / record["visual_features"])
    masks = load_npz(ROOT / record["per_object_masks"])
    derived = load_npz(ROOT / record["image_space_derived"])
    expected_inits = set(record["init_state_indices"])
    if set(np.unique(derived["metadata__init_state"]).tolist()) != expected_inits:
        raise ValueError("partition init-state boundary changed")
    if not np.array_equal(masks["init_state_index"], np.asarray(record["init_state_indices"] * 10).reshape(10, 4).T.reshape(-1)):
        # Collector order is not semantically relevant; exact set and branch identities are.
        if set(np.unique(masks["init_state_index"]).tolist()) != expected_inits:
            raise ValueError("mask init-state boundary changed")
    global_feature = np.asarray(features["global_variant_features"], dtype=np.float64)[..., 0, :]
    patch = np.asarray(features["patch_features"], dtype=np.float64)
    if global_feature.shape != (40, 3, 8, 512) or patch.shape != (40, 3, 8, 49, 512):
        raise ValueError("frozen natural feature shape changed")
    if np.asarray(masks["per_object_masks"]).shape != (40, 3, 8, 2, 128, 128):
        raise ValueError("per-object mask shape changed")
    return {
        "features": features,
        "masks": masks,
        "derived": derived,
        "metadata": metadata_from_derived(derived),
        "global": global_feature,
        "patch": patch,
    }


def image_init_states(partition: dict[str, Any]) -> np.ndarray:
    return np.repeat(np.asarray(partition["masks"]["init_state_index"]), 3 * 8)


def flatten_images(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    return source.reshape((40 * 3 * 8,) + source.shape[3:])


def oof_prediction(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    prediction = np.empty_like(np.asarray(y, dtype=np.float64).reshape(-1))
    for held_out in sorted(np.unique(groups).tolist()):
        train = groups != held_out
        valid = ~train
        fit = fit_ridge(np.asarray(x)[train], np.asarray(y)[train], alpha)
        prediction[valid] = predict_ridge(np.asarray(x)[valid], fit).reshape(-1)
    return prediction


def cv_increment(global_features: np.ndarray, extra: np.ndarray, target: np.ndarray, groups: np.ndarray, grid: list[float]) -> dict[str, Any]:
    global_alpha, _ = select_ridge_alpha(global_features, target, groups, grid)
    child_features = np.concatenate((global_features, extra), axis=1)
    child_alpha, _ = select_ridge_alpha(child_features, target, groups, grid)
    parent = oof_prediction(global_features, target, groups, global_alpha)
    child = oof_prediction(child_features, target, groups, child_alpha)
    parent_rmse = float(np.sqrt(np.mean(np.square(parent - target))))
    child_rmse = float(np.sqrt(np.mean(np.square(child - target))))
    return {
        "global_alpha": global_alpha,
        "child_alpha": child_alpha,
        "global_oof_rmse": parent_rmse,
        "child_oof_rmse": child_rmse,
        "relative_improvement": (parent_rmse - child_rmse) / max(parent_rmse, 1.0e-12),
    }


def calibration_selection(protocol: dict[str, Any], partition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata = partition["metadata"]
    derived = partition["derived"]
    global_summary = np.asarray(derived["representation__global_summary"], dtype=np.float64)
    groups = np.asarray(metadata["init_state"])
    grid = [float(value) for value in protocol["probes"]["ridge_grid"]]
    targets = {
        name: np.asarray(derived[f"target__{name}"], dtype=np.float64)
        for name in RAW_TARGETS
    }

    patch_images = flatten_images(partition["patch"])
    pca_fit, _ = fit_randomized_pca(
        patch_images.reshape(-1, 512),
        max(protocol["t2_generic"]["pca_ranks"]),
        int(protocol["t2_generic"]["pca_seed"]),
    )
    generic_states: dict[str, np.ndarray] = {}
    generic_features: dict[str, np.ndarray] = {}
    pca_all = pca_patch_state(partition["patch"], pca_fit)
    for rank in protocol["t2_generic"]["pca_ranks"]:
        name = f"pca_rank_{rank}"
        state = reshape_patch_grid(pca_all[..., : int(rank)])
        generic_states[name] = state
        generic_features[name] = generic_pair_summary(state, metadata)[0]
    residual = reshape_patch_grid(patch_global_residual_state(partition["patch"], partition["global"]))
    generic_states["patch_global_residual"] = residual
    generic_features["patch_global_residual"] = generic_pair_summary(residual, metadata)[0]
    generic_cv = {}
    for name, feature in generic_features.items():
        generic_cv[name] = {
            target: cv_increment(global_summary, feature, targets[target], groups, grid)
            for target in ("object_mask_difference", "visible_area_difference")
        }
        generic_cv[name]["minimum_object_improvement"] = min(
            generic_cv[name][target]["relative_improvement"]
            for target in ("object_mask_difference", "visible_area_difference")
        )
        generic_cv[name]["dimensions"] = int(np.prod(generic_states[name].shape[3:]))
    best_generic = max(generic_cv, key=lambda name: (generic_cv[name]["minimum_object_improvement"], -generic_cv[name]["dimensions"], name))
    best_value = generic_cv[best_generic]["minimum_object_improvement"]
    eligible_generic = [
        name for name in generic_cv
        if generic_cv[name]["minimum_object_improvement"] >= best_value - float(protocol["t2_generic"]["tie_tolerance_percentage_points"])
    ]
    selected_generic = min(eligible_generic, key=lambda name: (generic_cv[name]["dimensions"], name))

    occupancy_by_resolution: dict[str, np.ndarray] = {}
    area_by_resolution: dict[str, np.ndarray] = {}
    footprint_features: dict[str, np.ndarray] = {}
    resolution_cv: dict[str, Any] = {}
    for resolution in protocol["footprint"]["resolutions"]:
        name = str(resolution)
        occupancy, area = footprint_from_object_masks(partition["masks"]["per_object_masks"], int(resolution))
        occupancy_by_resolution[name] = occupancy
        area_by_resolution[name] = area
        feature, _ = spatial_pair_summary(occupancy, area, metadata)
        footprint_features[name] = feature
        resolution_cv[name] = {
            target: cv_increment(global_summary, feature, targets[target], groups, grid)
            for target in ("object_mask_difference", "visible_area_difference")
        }
        resolution_cv[name]["minimum_object_improvement"] = min(
            resolution_cv[name][target]["relative_improvement"]
            for target in ("object_mask_difference", "visible_area_difference")
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_groups = image_init_states(partition)
    patch_flat = flatten_images(partition["patch"])
    proxy_cv: dict[str, Any] = {}
    proxy_oof: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for resolution in protocol["footprint"]["resolutions"]:
        occ_flat = flatten_images(occupancy_by_resolution[str(resolution)])
        area_flat = flatten_images(area_by_resolution[str(resolution)])
        baseline_map = occ_flat.mean(axis=0)
        baseline_area = float(area_flat.mean())
        for hidden in protocol["t4_proxy"]["hidden_widths"]:
            key = f"r{resolution}_h{hidden}"
            oof_occ = np.empty_like(occ_flat)
            oof_area = np.empty_like(area_flat)
            fold_records = []
            for fold, held_out in enumerate(sorted(np.unique(image_groups).tolist())):
                train = image_groups != held_out
                valid = ~train
                fit, train_record = fit_proxy(
                    patch_flat[train], occ_flat[train], area_flat[train], int(hidden), int(resolution),
                    int(protocol["t4_proxy"]["cv_epochs"]), float(protocol["t4_proxy"]["learning_rate"]),
                    float(protocol["t4_proxy"]["weight_decay"]), int(protocol["t4_proxy"]["seed"]) + fold,
                    device=device,
                )
                oof_occ[valid], oof_area[valid] = predict_proxy(patch_flat[valid], fit, device=device)
                fold_records.append({"held_out_init": int(held_out), **train_record})
            metrics = recovery_metrics(oof_occ, occ_flat, oof_area, area_flat, 0.5, baseline_map, baseline_area)
            normalized_loss = (
                metrics["occupancy_mse"] / max(float(np.mean(np.square(occ_flat - baseline_map))), 1.0e-12)
                + metrics["area_mae"] / max(float(np.mean(np.abs(area_flat - baseline_area))), 1.0e-12)
            )
            proxy_cv[key] = {"resolution": int(resolution), "hidden": int(hidden), "normalized_loss": normalized_loss, "metrics_at_0_5": metrics, "folds": fold_records}
            proxy_oof[key] = (oof_occ, oof_area)

    best_loss = min(record["normalized_loss"] for record in proxy_cv.values())
    eligible_proxy = [
        name for name, record in proxy_cv.items()
        if record["normalized_loss"] <= best_loss * (1.0 + float(protocol["t4_proxy"]["selection_tolerance_fraction"]))
    ]
    selected_proxy = min(eligible_proxy, key=lambda name: (proxy_cv[name]["resolution"] ** 2, proxy_cv[name]["hidden"], name))
    selected_resolution = int(proxy_cv[selected_proxy]["resolution"])
    # The single target resolution is constrained to an oracle-valid resolution.
    oracle_valid = [
        int(name) for name, record in resolution_cv.items()
        if record["minimum_object_improvement"] >= float(protocol["gates"]["minimum_primary_increment"])
    ]
    if oracle_valid and selected_resolution not in oracle_valid:
        allowed_proxy = [name for name in proxy_cv if proxy_cv[name]["resolution"] in oracle_valid]
        selected_proxy = min(allowed_proxy, key=lambda name: (proxy_cv[name]["normalized_loss"], proxy_cv[name]["resolution"] ** 2, proxy_cv[name]["hidden"]))
        selected_resolution = int(proxy_cv[selected_proxy]["resolution"])

    oof_occ, oof_area = proxy_oof[selected_proxy]
    occ_true = flatten_images(occupancy_by_resolution[str(selected_resolution)])
    area_true = flatten_images(area_by_resolution[str(selected_resolution)])
    threshold_records = []
    for threshold in protocol["t4_proxy"]["threshold_grid"]:
        metrics = recovery_metrics(oof_occ, occ_true, oof_area, area_true, float(threshold), occ_true.mean(axis=0), float(area_true.mean()))
        threshold_records.append({"threshold": float(threshold), "iou": metrics["iou"], "balanced_accuracy": metrics["balanced_accuracy"]})
    selected_threshold = max(threshold_records, key=lambda row: (row["iou"], row["balanced_accuracy"], -row["threshold"]))["threshold"]

    final_proxy, train_record = fit_proxy(
        patch_flat, occ_true, area_true, int(proxy_cv[selected_proxy]["hidden"]), selected_resolution,
        int(protocol["t4_proxy"]["final_epochs"]), float(protocol["t4_proxy"]["learning_rate"]),
        float(protocol["t4_proxy"]["weight_decay"]), int(protocol["t4_proxy"]["seed"]) + 100,
        device=device,
    )
    selection = {
        "generic_candidates": generic_cv,
        "selected_generic": selected_generic,
        "footprint_resolution_candidates": resolution_cv,
        "proxy_candidates": proxy_cv,
        "selected_proxy": selected_proxy,
        "selected_resolution": selected_resolution,
        "selected_hidden_width": int(proxy_cv[selected_proxy]["hidden"]),
        "threshold_candidates": threshold_records,
        "selected_threshold": float(selected_threshold),
        "final_proxy_training": train_record,
        "device": device,
    }
    weights = {}
    weights.update(pca_to_arrays("generic_pca", pca_fit))
    weights.update(proxy_to_arrays("natural_proxy", final_proxy))
    weights["selected_threshold"] = np.asarray(selected_threshold, dtype=np.float64)
    weights["baseline_occupancy"] = occ_true.mean(axis=0)
    weights["baseline_area"] = np.asarray(area_true.mean(), dtype=np.float64)
    selected_arrays = {
        "generic_state": generic_states[selected_generic],
        "footprint_occupancy": occupancy_by_resolution[str(selected_resolution)],
        "footprint_area": area_by_resolution[str(selected_resolution)],
        "proxy_oof_occupancy": oof_occ.reshape(40, 3, 8, selected_resolution, selected_resolution),
        "proxy_oof_area": oof_area.reshape(40, 3, 8),
    }
    return selection, weights, selected_arrays


def build_representations(
    partition: dict[str, Any],
    weights: dict[str, np.ndarray],
    selected: dict[str, Any],
    phase: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata = partition["metadata"]
    derived = partition["derived"]
    global_summary = np.asarray(derived["representation__global_summary"], dtype=np.float64)
    full_patch = np.asarray(derived["representation__patch_summary"], dtype=np.float64)
    pca_fit = pca_from_arrays("generic_pca", weights)
    selected_generic = selected["selected_generic"]
    if selected_generic == "patch_global_residual":
        generic_state = reshape_patch_grid(patch_global_residual_state(partition["patch"], partition["global"]))
    else:
        rank = int(selected_generic.rsplit("_", 1)[1])
        generic_state = reshape_patch_grid(pca_patch_state(partition["patch"], pca_fit)[..., :rank])
    resolution = int(selected["selected_resolution"])
    footprint_occupancy, footprint_area = footprint_from_object_masks(partition["masks"]["per_object_masks"], resolution)
    proxy_fit = proxy_from_arrays("natural_proxy", weights)
    proxy_occ_flat, proxy_area_flat = predict_proxy(flatten_images(partition["patch"]), proxy_fit, device="cuda" if torch.cuda.is_available() else "cpu")
    proxy_occupancy = proxy_occ_flat.reshape(40, 3, 8, resolution, resolution)
    proxy_area = proxy_area_flat.reshape(40, 3, 8)
    generic_summary, generic_distance = generic_pair_summary(generic_state, metadata)
    footprint_summary, footprint_distance = spatial_pair_summary(footprint_occupancy, footprint_area, metadata)
    proxy_summary, proxy_distance = spatial_pair_summary(proxy_occupancy, proxy_area, metadata)
    representations = {
        "global": global_summary,
        "full_patch_reference": full_patch,
        "generic_spatial": generic_summary,
        "global_plus_generic": np.concatenate((global_summary, generic_summary), axis=1),
        "footprint_spatial": footprint_summary,
        "global_plus_footprint": np.concatenate((global_summary, footprint_summary), axis=1),
        "proxy_spatial": proxy_summary,
        "global_plus_proxy": np.concatenate((global_summary, proxy_summary), axis=1),
    }
    states = {
        "generic": generic_state,
        "footprint": footprint_occupancy,
        "footprint_area": footprint_area,
        "proxy": proxy_occupancy,
        "proxy_area": proxy_area,
    }
    distances = {"generic": generic_distance, "footprint": footprint_distance, "proxy": proxy_distance}
    for name, value in representations.items():
        if value.shape[0] != 960 or not np.isfinite(value).all():
            raise ValueError(f"invalid representation {name}: {value.shape}")
    if phase == "calibration":
        proxy_oof_occupancy = selected.get("proxy_oof_occupancy")
        proxy_oof_area = selected.get("proxy_oof_area")
        if proxy_oof_occupancy is not None:
            proxy_summary, proxy_distance = spatial_pair_summary(proxy_oof_occupancy, proxy_oof_area, metadata)
            representations["proxy_spatial"] = proxy_summary
            representations["global_plus_proxy"] = np.concatenate((global_summary, proxy_summary), axis=1)
            distances["proxy"] = proxy_distance
            states["proxy"] = proxy_oof_occupancy
            states["proxy_area"] = proxy_oof_area
    return representations, states, distances


def construct_targets(partition: dict[str, Any], weights: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    derived = partition["derived"]
    target = {name: np.asarray(derived[f"target__{name}"], dtype=np.float64) for name in RAW_TARGETS}
    if weights is not None:
        ee = target["ee_image_displacement"][:, None]
        for raw, residual in (("object_mask_difference", "object_mask_residual"), ("visible_area_difference", "visible_area_residual")):
            expected = predict_ridge(ee, ridge_from_arrays(f"ee_control__{raw}", weights)).reshape(-1)
            target[residual] = target[raw] - expected
    return target


def calibrate(protocol: dict[str, Any], protocol_commit: str) -> None:
    partition = load_partition(protocol, "calibration")
    selection, weights, selected_arrays = calibration_selection(protocol, partition)
    selected_for_build = {**selection, **selected_arrays}
    representations, states, distances = build_representations(partition, weights, selected_for_build, "calibration")
    targets = construct_targets(partition)
    grid = [float(value) for value in protocol["probes"]["ridge_grid"]]
    groups = partition["metadata"]["init_state"]
    ee = targets["ee_image_displacement"][:, None]
    ee_models = {}
    for raw, residual in (("object_mask_difference", "object_mask_residual"), ("visible_area_difference", "visible_area_residual")):
        alpha, cv = select_ridge_alpha(ee, targets[raw], groups, grid)
        fit = fit_ridge(ee, targets[raw], alpha)
        weights.update(ridge_to_arrays(f"ee_control__{raw}", fit))
        targets[residual] = targets[raw] - predict_ridge(ee, fit).reshape(-1)
        ee_models[raw] = {"alpha": alpha, "cv": cv}
    models = {}
    for representation, feature in representations.items():
        models[representation] = {}
        for target_name, target in targets.items():
            alpha, cv = select_ridge_alpha(feature, target, groups, grid)
            fit = fit_ridge(feature, target, alpha)
            weights.update(ridge_to_arrays(f"ridge__{representation}__{target_name}", fit))
            models[representation][target_name] = {"alpha": alpha, "dimensions": int(feature.shape[1]), "cv": cv}
    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    derived_path = ROOT / protocol["outputs"]["calibration_derived"]
    atomic_npz(weights_path, weights)
    arrays = {
        **{f"metadata__{name}": value for name, value in partition["metadata"].items()},
        **{f"target__{name}": value.astype(np.float32) for name, value in targets.items()},
        **{f"representation__{name}": value.astype(np.float32) for name, value in representations.items()},
        **{f"distance__{name}": value.astype(np.float32) for name, value in distances.items()},
        "proxy_oof_occupancy": np.asarray(selected_arrays["proxy_oof_occupancy"], dtype=np.float32),
        "proxy_oof_area": np.asarray(selected_arrays["proxy_oof_area"], dtype=np.float32),
    }
    atomic_npz(derived_path, arrays)
    freeze = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_INIT36_39_TARGET_DESIGN_EVALUATION",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "partition": {"init_state_indices": [31, 32, 33, 34], "branch_points": 40, "pair_rows": 960, "evaluation_numeric_arrays_read": False},
        "selection": selection,
        "ee_control_models": ee_models,
        "probe_models": models,
        "artifacts": {
            "weights": {"path": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path)},
            "derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)},
        },
        "forbidden_activity_audit": {
            "evaluation_numeric_arrays_read": False,
            "simulator_run": False,
            "world_model_or_encoder_training": False,
            "candidate_reranker_or_controller_change": False,
            "reward_done_success_or_task_outcome_read": False,
            "init40_49_access": False,
        },
    }
    atomic_json(ROOT / protocol["outputs"]["calibration_freeze"], freeze)
    print(json.dumps({"status": freeze["status"], "selection": {key: selection[key] for key in ("selected_generic", "selected_resolution", "selected_hidden_width", "selected_threshold")}}, indent=2))


def bootstrap_indices(metadata: dict[str, np.ndarray], replicates: int, seed: int) -> list[np.ndarray]:
    branch = metadata["branch"]
    task = metadata["task"]
    rng = np.random.default_rng(seed)
    branches_by_task = {task_id: np.unique(branch[task == task_id]) for task_id in range(10)}
    rows = {int(value): np.flatnonzero(branch == value) for value in np.unique(branch)}
    draws = []
    for _ in range(replicates):
        selected = []
        for task_id in range(10):
            values = branches_by_task[task_id]
            sampled = rng.choice(values, size=values.size, replace=True)
            selected.extend(rows[int(value)] for value in sampled)
        draws.append(np.concatenate(selected))
    return draws


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_metric(draws: list[np.ndarray], function: Callable[[np.ndarray], float]) -> list[float]:
    return interval([float(function(draw)) for draw in draws])


def rmse(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if indices is not None:
        pred, truth = pred[indices], truth[indices]
    return float(np.sqrt(np.mean(np.square(pred - truth))))


def improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    parent_rmse = rmse(parent, target, indices)
    return (parent_rmse - rmse(child, target, indices)) / max(parent_rmse, 1.0e-12)


def cut_improvements(parent: np.ndarray, child: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {"overall": improvement(parent, child, target), "task": [], "horizon": [], "pair": []}
    for field, values in (("task", range(10)), ("horizon", range(1, 9)), ("pair", range(3))):
        for value in values:
            indices = np.flatnonzero(metadata[field] == value)
            label: Any = PAIR_NAMES[value] if field == "pair" else value
            result[field].append({field: label, "relative_rmse_improvement": improvement(parent, child, target, indices)})
    return result


def derange(values: np.ndarray, metadata: dict[str, np.ndarray], seed: int) -> np.ndarray:
    source = np.asarray(values)
    output = source.copy()
    rng = np.random.default_rng(seed)
    for task in range(10):
        for horizon in range(1, 9):
            for pair in range(3):
                indices = np.flatnonzero((metadata["task"] == task) & (metadata["horizon"] == horizon) & (metadata["pair"] == pair))
                ordered = indices[np.argsort(metadata["branch"][indices], kind="stable")]
                output[ordered] = source[np.roll(ordered, int(rng.integers(1, ordered.size)))]
    return output


def correlation_record(value: np.ndarray, target: np.ndarray, draws: list[np.ndarray]) -> dict[str, Any]:
    record = safe_correlation(value, target)
    record["spearman_ci95"] = bootstrap_metric(draws, lambda index: safe_spearman(value[index], target[index]))
    return record


def evaluate(protocol: dict[str, Any], protocol_commit: str) -> None:
    freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    weights = load_npz(ROOT / protocol["outputs"]["calibration_weights"])
    partition = load_partition(protocol, "development_evaluation")
    selection = freeze["selection"]
    representations, states, distances = build_representations(partition, weights, selection, "evaluation")
    targets = construct_targets(partition, weights)
    metadata = partition["metadata"]
    draws = bootstrap_indices(metadata, int(protocol["bootstrap"]["replicates"]), int(protocol["bootstrap"]["seed"]))
    correlation_draws = draws[: int(protocol["bootstrap"]["correlation_replicates"])]

    predictions = {name: {} for name in representations}
    probe_metrics = {name: {} for name in representations}
    for representation, feature in representations.items():
        for target_name, target in targets.items():
            prediction = predict_ridge(feature, ridge_from_arrays(f"ridge__{representation}__{target_name}", weights)).reshape(-1)
            predictions[representation][target_name] = prediction
            probe_metrics[representation][target_name] = scalar_metrics(prediction, target)

    design_mapping = {
        "D1_low_dim_generic": ("generic_spatial", "global_plus_generic"),
        "D2_object_footprint": ("footprint_spatial", "global_plus_footprint"),
        "D3_natural_proxy": ("proxy_spatial", "global_plus_proxy"),
        "D4_hybrid_global_proxy": ("proxy_spatial", "global_plus_proxy"),
        "D5_global_object_auxiliary": ("footprint_spatial", "global_plus_footprint"),
    }
    incremental = {}
    shuffles = {}
    design_gates = {}
    shuffled_extra = {
        "generic_spatial": derange(representations["generic_spatial"], metadata, int(protocol["shuffle"]["seed"])),
        "footprint_spatial": derange(representations["footprint_spatial"], metadata, int(protocol["shuffle"]["seed"]) + 1),
        "proxy_spatial": derange(representations["proxy_spatial"], metadata, int(protocol["shuffle"]["seed"]) + 2),
    }
    for design, (spatial_name, hybrid_name) in design_mapping.items():
        incremental[design] = {}
        shuffles[design] = {}
        for target_name, target in targets.items():
            parent = predictions["global"][target_name]
            child = predictions[hybrid_name][target_name]
            cuts = cut_improvements(parent, child, target, metadata)
            ci = bootstrap_metric(draws, lambda index, p=parent, c=child, t=target: improvement(p, c, t, index))
            incremental[design][target_name] = {**cuts, "ci95": ci}
            fit = ridge_from_arrays(f"ridge__{hybrid_name}__{target_name}", weights)
            alternate_features = np.concatenate((representations["global"], shuffled_extra[spatial_name]), axis=1)
            alternate = predict_ridge(alternate_features, fit).reshape(-1)
            original_rmse = rmse(child, target)
            degradation = (rmse(alternate, target) - original_rmse) / max(original_rmse, 1.0e-12)
            degradation_ci = bootstrap_metric(
                draws,
                lambda index, original=child, changed=alternate, truth=target: (rmse(changed, truth, index) - rmse(original, truth, index)) / max(rmse(original, truth, index), 1.0e-12),
            )
            shuffles[design][target_name] = {
                "original_rmse": original_rmse,
                "shuffled_rmse": rmse(alternate, target),
                "relative_degradation": degradation,
                "ci95": degradation_ci,
                "passes": bool(degradation > 0.0 and degradation_ci[0] > 0.0),
            }

    # Candidate and temporal movement use the same target coordinates, so their ratio removes scale.
    candidate_distances = {}
    temporal_distances = {}
    candidate_distances["D0_global"] , temporal_distances["D0_global"] = temporal_and_candidate_distance(partition["global"], "cosine")
    candidate_distances["T1_full_patch_reference"], temporal_distances["T1_full_patch_reference"] = temporal_and_candidate_distance(partition["patch"], "cosine")
    generic_candidate, generic_temporal = temporal_and_candidate_distance(states["generic"], "rmse")
    footprint_vector = np.concatenate((states["footprint"].reshape(40, 3, 8, -1), states["footprint_area"][..., None]), axis=-1)
    proxy_vector = np.concatenate((states["proxy"].reshape(40, 3, 8, -1), states["proxy_area"][..., None]), axis=-1)
    footprint_candidate, footprint_temporal = temporal_and_candidate_distance(footprint_vector, "rmse")
    proxy_candidate, proxy_temporal = temporal_and_candidate_distance(proxy_vector, "rmse")
    candidate_distances.update({"D1_low_dim_generic": generic_candidate, "D2_object_footprint": footprint_candidate, "D3_natural_proxy": proxy_candidate})
    temporal_distances.update({"D1_low_dim_generic": generic_temporal, "D2_object_footprint": footprint_temporal, "D3_natural_proxy": proxy_temporal})
    global_scale = float(np.mean(temporal_distances["D0_global"]))
    for design, spatial in (("D4_hybrid_global_proxy", "D3_natural_proxy"), ("D5_global_object_auxiliary", "D2_object_footprint")):
        spatial_scale = float(np.mean(temporal_distances[spatial]))
        candidate_distances[design] = np.sqrt(np.square(candidate_distances["D0_global"] / max(global_scale, 1.0e-12)) + np.square(candidate_distances[spatial] / max(spatial_scale, 1.0e-12))) / np.sqrt(2.0)
        temporal_count = min(temporal_distances["D0_global"].size, temporal_distances[spatial].size)
        temporal_distances[design] = np.sqrt(np.square(temporal_distances["D0_global"][:temporal_count] / max(global_scale, 1.0e-12)) + np.square(temporal_distances[spatial][:temporal_count] / max(spatial_scale, 1.0e-12))) / np.sqrt(2.0)

    separation = {}
    for name in candidate_distances:
        value = candidate_distances[name]
        ratio = float(np.mean(value) / max(float(np.mean(temporal_distances[name])), 1.0e-12))
        correlations = {
            target: correlation_record(value, targets[target], correlation_draws)
            for target in ("object_mask_difference", "visible_area_difference", "ee_image_displacement")
        }
        object_correlation_pass = any(
            correlations[target]["spearman"] is not None
            and correlations[target]["spearman"] >= float(protocol["gates"]["minimum_separation_object_spearman"])
            and correlations[target]["spearman_ci95"][0] > 0.0
            for target in ("object_mask_difference", "visible_area_difference")
        )
        separation[name] = {
            "candidate_mean": float(np.mean(value)),
            "ordinary_temporal_mean": float(np.mean(temporal_distances[name])),
            "candidate_to_temporal_ratio": ratio,
            "correlations": correlations,
            "not_scale_artifact": bool(ratio >= float(protocol["gates"]["minimum_candidate_temporal_ratio"]) and object_correlation_pass),
        }

    resolution = int(selection["selected_resolution"])
    true_occ, true_area = footprint_from_object_masks(partition["masks"]["per_object_masks"], resolution)
    recovery = recovery_metrics(
        flatten_images(states["proxy"]), flatten_images(true_occ), flatten_images(states["proxy_area"]), flatten_images(true_area),
        float(np.asarray(weights["selected_threshold"])), np.asarray(weights["baseline_occupancy"]), float(np.asarray(weights["baseline_area"])),
    )
    recovery_gate = protocol["gates"]["natural_recovery"]
    recovery_pass = bool(
        recovery["occupancy_mse_improvement_over_constant"] >= recovery_gate["minimum_occupancy_mse_improvement"]
        and recovery["iou"] >= recovery_gate["minimum_iou"]
        and recovery["balanced_accuracy"] >= recovery_gate["minimum_balanced_accuracy"]
        and recovery["positive_cell_recall"] >= recovery_gate["minimum_positive_recall"]
        and recovery["area_mae_improvement_over_constant"] >= recovery_gate["minimum_area_mae_improvement"]
    )

    for design, (_, hybrid_name) in design_mapping.items():
        primary_targets = []
        for target_name in ("object_mask_difference", "visible_area_difference"):
            record = incremental[design][target_name]
            tasks = sum(row["relative_rmse_improvement"] > 0.0 for row in record["task"])
            horizons = sum(row["relative_rmse_improvement"] > 0.0 for row in record["horizon"])
            pairs = sum(row["relative_rmse_improvement"] > 0.0 for row in record["pair"])
            if (
                record["overall"] >= float(protocol["gates"]["minimum_primary_increment"])
                and record["ci95"][0] > 0.0
                and shuffles[design][target_name]["passes"]
                and tasks >= int(protocol["gates"]["minimum_positive_tasks"])
                and horizons >= int(protocol["gates"]["minimum_positive_horizons"])
                and pairs == 3
            ):
                primary_targets.append(target_name)
        residual_targets = []
        for target_name in RESIDUAL_TARGETS:
            record = incremental[design][target_name]
            if record["overall"] >= float(protocol["gates"]["minimum_residual_increment"]) and record["ci95"][0] > 0.0:
                residual_targets.append(target_name)
        deployment_compatible = design in ("D1_low_dim_generic", "D3_natural_proxy", "D4_hybrid_global_proxy")
        if design in ("D3_natural_proxy", "D4_hybrid_global_proxy"):
            deployment_compatible = deployment_compatible and recovery_pass
        dimension = {
            "D1_low_dim_generic": int(np.prod(states["generic"].shape[3:])),
            "D2_object_footprint": resolution * resolution + 1,
            "D3_natural_proxy": resolution * resolution + 1,
            "D4_hybrid_global_proxy": 512 + resolution * resolution + 1,
            "D5_global_object_auxiliary": 512 + resolution * resolution + 1,
        }[design]
        design_gates[design] = {
            "primary_passing_targets": primary_targets,
            "residual_passing_targets": residual_targets,
            "separation_not_scale_artifact": separation[design]["not_scale_artifact"],
            "deployment_compatible": deployment_compatible,
            "dimension": dimension,
            "dimension_reasonable": dimension <= int(protocol["gates"]["maximum_total_dimension"]),
        }
        design_gates[design]["passes"] = bool(
            primary_targets and residual_targets and design_gates[design]["separation_not_scale_artifact"]
            and design_gates[design]["dimension_reasonable"]
            and deployment_compatible
        )

    labels = []
    if design_gates["D2_object_footprint"]["primary_passing_targets"]:
        labels.extend(["OBJECT_FOOTPRINT_TARGET_VALID", "ORACLE_SPATIAL_TARGET_VALID"])
    if design_gates["D1_low_dim_generic"]["passes"]:
        labels.append("LOW_DIM_SPATIAL_MAP_VALID")
    if recovery_pass and design_gates["D3_natural_proxy"]["primary_passing_targets"]:
        labels.append("NATURAL_SPATIAL_PROXY_VALID")
    elif "ORACLE_SPATIAL_TARGET_VALID" in labels:
        labels.append("NATURAL_SPATIAL_EXTRACTION_BLOCK")
    if design_gates["D4_hybrid_global_proxy"]["passes"]:
        labels.append("HYBRID_GLOBAL_SPATIAL_TARGET_VALID")
    if design_gates["D5_global_object_auxiliary"]["primary_passing_targets"] and design_gates["D5_global_object_auxiliary"]["residual_passing_targets"]:
        labels.append("OBJECT_AUXILIARY_TARGET_VALID")
    if any(design_gates[name]["residual_passing_targets"] for name in design_gates):
        labels.append("OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED")

    selected_design = None
    if design_gates["D4_hybrid_global_proxy"]["passes"]:
        selected_design = "D4_hybrid_global_proxy"
    elif design_gates["D1_low_dim_generic"]["passes"]:
        selected_design = "D1_low_dim_generic"
    elif design_gates["D3_natural_proxy"]["passes"]:
        selected_design = "D3_natural_proxy"
    if selected_design is None:
        labels.append("SPATIAL_TARGET_DESIGN_UNRESOLVED")
        authorization = "BLOCK_SPATIAL_WM_PROTOTYPE"
    else:
        authorization = "GO_SPATIAL_WM_PROTOTYPE"

    dimensions = {
        "T0_global_clip": {"dimensions": 512, "bytes_float32": 2048},
        "T1_full_patch_reference": {"dimensions": 49 * 512, "bytes_float32": 49 * 512 * 4},
        "T2_selected_generic": {"dimensions": int(np.prod(states["generic"].shape[3:])), "bytes_float32": int(np.prod(states["generic"].shape[3:])) * 4},
        "T3_object_footprint": {"dimensions": resolution * resolution + 1, "bytes_float32": (resolution * resolution + 1) * 4},
        "T4_natural_proxy": {"dimensions": resolution * resolution + 1, "bytes_float32": (resolution * resolution + 1) * 4},
        "T5_global_plus_proxy": {"dimensions": 512 + resolution * resolution + 1, "bytes_float32": (512 + resolution * resolution + 1) * 4},
        "T6_global_plus_footprint_auxiliary": {"dimensions": 512 + resolution * resolution + 1, "bytes_float32": (512 + resolution * resolution + 1) * 4},
    }

    derived_path = ROOT / protocol["outputs"]["evaluation_derived"]
    arrays = {
        **{f"metadata__{name}": value for name, value in metadata.items()},
        **{f"target__{name}": value.astype(np.float32) for name, value in targets.items()},
        **{f"representation__{name}": value.astype(np.float32) for name, value in representations.items()},
        **{f"prediction__{rep}__{target}": predictions[rep][target].astype(np.float32) for rep in predictions for target in predictions[rep]},
        **{f"candidate_distance__{name}": value.astype(np.float32) for name, value in candidate_distances.items()},
        **{f"temporal_distance__{name}": value.astype(np.float32) for name, value in temporal_distances.items()},
        "proxy_occupancy": states["proxy"].astype(np.float32),
        "proxy_area": states["proxy_area"].astype(np.float32),
        "footprint_occupancy": true_occ.astype(np.float32),
        "footprint_area": true_area.astype(np.float32),
    }
    atomic_npz(derived_path, arrays)
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "date": protocol["date"],
        "status": "COMPLETE",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "calibration_freeze_sha256": sha256(freeze_path),
        "data_boundary": {"calibration_init_states": [31, 32, 33, 34], "development_evaluation_init_states": [36, 37, 38, 39], "development_evaluation_is_blind_confirmation": False, "historical_reference_only": [30, 35], "init40_49_accessed": False, "branch_points": 40, "pair_rows": 960},
        "candidate_definitions": protocol["candidate_definitions"],
        "calibration_selection": selection,
        "dimensions": dimensions,
        "probe_metrics": probe_metrics,
        "incremental": incremental,
        "shuffle_controls": shuffles,
        "candidate_separation": separation,
        "natural_proxy_recovery": {**recovery, "passes": recovery_pass},
        "design_gates": design_gates,
        "decision": {
            "mechanism_labels": labels,
            "selected_design": selected_design,
            "next_stage_authorization": authorization,
            "controller_authorization": False,
            "prototype_scope_if_go": "seed11 single-model small spatial World Model prototype only",
        },
        "candidate_headroom_reference": protocol["candidate_headroom_reference"],
        "artifacts": {"evaluation_derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)}},
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "simulator_run": False,
            "world_model_or_encoder_training": False,
            "candidate_or_reranker_change": False,
            "controller_or_success_run": False,
            "reward_done_success_or_task_outcome_read": False,
            "init40_49_access": False,
            "seed23_47_run": False,
            "upstream_overwrite": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    atomic_json(ROOT / protocol["outputs"]["diagnosis"], result)
    print(json.dumps(result["decision"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol, commit = load_protocol(args.phase)
    if args.phase == "calibrate":
        calibrate(protocol, commit)
    else:
        evaluate(protocol, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
