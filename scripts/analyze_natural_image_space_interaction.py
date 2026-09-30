#!/usr/bin/env python3
"""Calibrate and evaluate the natural image-space interaction target audit."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probes.image_space_relevance import (  # noqa: E402
    PAIR_NAMES,
    SCALAR_REPRESENTATIONS,
    TARGET_FAMILIES,
    TARGET_NAMES,
    build_image_space_dataset,
    grouped_horizon_masks,
    point_to_mask_distance,
    project_world_points,
)
from src.probes.spatial_relevance import (  # noqa: E402
    cut_metrics,
    fit_logistic,
    fit_ridge,
    logistic_from_arrays,
    logistic_to_arrays,
    predict_logistic,
    predict_ridge,
    ridge_from_arrays,
    ridge_to_arrays,
    safe_correlation,
    safe_spearman,
    select_ridge_alpha,
)


PROTOCOL_VERSION = "natural-image-space-interaction-target-audit-v1"
REPRESENTATIONS = (
    "global_summary",
    "patch_summary",
    "global_plus_patch",
    "global_plus_patch_plus_strict",
    "global_pca16",
    "patch_pca16",
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def load_protocol(path: Path, phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected image-space protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_CAMERA_INVENTORY_OR_TARGET_RESULTS":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise RuntimeError("protocol must be committed")
    tag = protocol["frozen_upstream"]["primary_tag"]
    if subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip() != "tag":
        raise RuntimeError("upstream freeze tag is not annotated")
    if subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip() != protocol["frozen_upstream"]["primary_commit"]:
        raise RuntimeError("upstream freeze tag moved")
    for item in protocol["implementation"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"implementation changed: {item['path']}")
    for item in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {item['path']}")
    inventory = ROOT / protocol["outputs"]["camera_inventory"]
    if not inventory.exists() or not last_commit(inventory):
        raise RuntimeError("camera inventory must be committed")
    camera = json.loads(inventory.read_text())
    if camera["status"] != "CAMERA_PROJECTION_VALIDATED":
        raise RuntimeError("CAMERA_PROJECTION_BLOCK")
    if phase == "calibrate":
        for key in ("calibration_weights", "calibration_derived", "calibration_freeze", "diagnosis"):
            if (ROOT / protocol["outputs"][key]).exists():
                raise FileExistsError(protocol["outputs"][key])
    else:
        freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("calibration freeze must be committed")
        freeze = json.loads(freeze_path.read_text())
        if freeze["status"] != "FROZEN_BEFORE_INIT36_39_IMAGE_SPACE_EVALUATION":
            raise ValueError("calibration freeze changed")
        if freeze["protocol_sha256"] != sha256(path):
            raise ValueError("calibration freeze references another protocol")
        for item in freeze["artifacts"].values():
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"calibration artifact changed: {item['path']}")
    return protocol, commit


def load_partition(protocol: dict[str, Any], partition: str) -> dict[str, Any]:
    record = protocol["partitions"][partition]
    spatial = _load_npz(ROOT / record["natural_spatial_derived"])
    feature = _load_npz(ROOT / record["visual_features"])
    masks = _load_npz(ROOT / record["frozen_masks"])
    per_object = _load_npz(ROOT / record["per_object_mask_sidecar"])
    state = _load_npz(ROOT / record["state_sidecar"])
    camera = json.loads((ROOT / protocol["outputs"]["camera_inventory"]).read_text())
    for name in ("branch", "task", "init_state", "horizon", "pair"):
        if spatial[f"metadata__{name}"].shape != (960,):
            raise ValueError("frozen row identity changed")
    combined_masks = dict(masks)
    combined_masks["patch_coverage"] = feature["patch_coverage"]
    dataset = build_image_space_dataset(
        spatial,
        combined_masks,
        per_object,
        state,
        camera,
        float(protocol["image_space_targets"]["strict_patch_sigma_pixels"]),
        int(protocol["image_space_targets"]["adjacency_radius_pixels"]),
        float(protocol["image_space_targets"]["near_contact_pixels"]),
    )
    expected_inits = set(record["init_state_indices"])
    if set(np.unique(state["init_state_index"]).tolist()) != expected_inits:
        raise ValueError("partition init-state boundary changed")
    return {"dataset": dataset, "spatial": spatial, "state": state, "masks": combined_masks}


def build_representations(spatial: dict[str, np.ndarray], dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(spatial[f"representation__{name}"], dtype=np.float64)
        for name in ("global_summary", "patch_summary", "global_plus_patch", "global_pca16", "patch_pca16")
    }
    strict = np.asarray(dataset["diagnostics"]["strict_interaction_patch_mean"], dtype=np.float64)[:, None]
    result["global_plus_patch_plus_strict"] = np.concatenate((result["global_plus_patch"], strict), axis=1)
    dimensions = {
        "global_summary": 10,
        "patch_summary": 10,
        "global_plus_patch": 20,
        "global_plus_patch_plus_strict": 21,
        "global_pca16": 16,
        "patch_pca16": 16,
    }
    for name, value in result.items():
        if value.shape != (960, dimensions[name]) or not np.isfinite(value).all():
            raise ValueError(f"representation changed: {name} {value.shape}")
    return result


def bootstrap_indices(metadata: dict[str, np.ndarray], replicates: int, seed: int) -> list[np.ndarray]:
    branch = metadata["branch"]
    task = metadata["task"]
    rng = np.random.default_rng(seed)
    branches_by_task = {task_id: np.unique(branch[task == task_id]) for task_id in range(10)}
    rows = {int(value): np.flatnonzero(branch == value) for value in np.unique(branch)}
    result = []
    for _ in range(replicates):
        selected = []
        for task_id in range(10):
            values = branches_by_task[task_id]
            sampled = rng.choice(values, size=values.size, replace=True)
            selected.extend(rows[int(value)] for value in sampled)
        result.append(np.concatenate(selected))
    return result


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_metric(draws: list[np.ndarray], function: Callable[[np.ndarray], float]) -> list[float]:
    return interval([float(function(index)) for index in draws])


def rmse(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is not None:
        prediction, target = prediction[indices], target[indices]
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def relative_improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    base = rmse(parent, target, indices)
    return (base - rmse(child, target, indices)) / max(base, 1.0e-12)


def deranged_group(values: np.ndarray, metadata: dict[str, np.ndarray], seed: int) -> np.ndarray:
    source = np.asarray(values)
    result = source.copy()
    rng = np.random.default_rng(seed)
    for task in range(10):
        for horizon in range(1, 9):
            for pair in range(3):
                indices = np.flatnonzero(
                    (metadata["task"] == task)
                    & (metadata["horizon"] == horizon)
                    & (metadata["pair"] == pair)
                )
                ordered = indices[np.argsort(metadata["branch"][indices], kind="stable")]
                if ordered.size < 2:
                    raise RuntimeError("shuffle cell too small")
                result[ordered] = source[np.roll(ordered, int(rng.integers(1, ordered.size)))]
    return result


def correlation_cuts(value: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, Any]:
    def metric(mask: np.ndarray) -> dict[str, Any]:
        use = mask & valid
        return safe_correlation(value[use], target[use])

    result: dict[str, Any] = {"overall": metric(np.ones(valid.size, dtype=bool)), "horizon": [], "task": [], "pair": []}
    for field, values in (("horizon", range(1, 9)), ("task", range(10)), ("pair", range(3))):
        for item in values:
            label: Any = PAIR_NAMES[item] if field == "pair" else item
            result[field].append({field: label, **metric(metadata[field] == item)})
    return result


def cut_improvements(parent: np.ndarray, child: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, Any]:
    def value(mask: np.ndarray) -> float:
        use = np.flatnonzero(mask & valid)
        return relative_improvement(parent, child, target, use)

    result: dict[str, Any] = {"overall": value(np.ones(valid.size, dtype=bool)), "horizon": [], "task": [], "pair": [], "horizon_groups": []}
    for field, values in (("horizon", range(1, 9)), ("task", range(10)), ("pair", range(3))):
        for item in values:
            label: Any = PAIR_NAMES[item] if field == "pair" else item
            result[field].append({field: label, "relative_rmse_improvement": value(metadata[field] == item)})
    for name, mask in grouped_horizon_masks(metadata).items():
        result["horizon_groups"].append({"group": name, "relative_rmse_improvement": value(mask)})
    return result


def projection_validation(dataset: dict[str, Any], masks: dict[str, np.ndarray], protocol: dict[str, Any]) -> dict[str, Any]:
    projected = np.asarray(dataset["projected_ee"])
    depth = np.asarray(dataset["projection_depth"])
    robot = np.asarray(masks["masks"], dtype=bool)[..., 0, :, :]
    distances = []
    inside = []
    for branch in range(40):
        for candidate in range(3):
            for horizon in range(8):
                point = projected[branch, candidate, horizon]
                distance = point_to_mask_distance(point, robot[branch, candidate, horizon])
                if distance is None:
                    raise RuntimeError("robot mask empty during projection validation")
                distances.append(distance)
                inside.append(0 <= point[0] < 128 and 0 <= point[1] < 128)
    gate = protocol["camera_projection"]["full_partition_gate"]
    record = {
        "n": len(distances),
        "median_distance_pixels": float(np.median(distances)),
        "p95_distance_pixels": float(np.quantile(distances, 0.95)),
        "max_distance_pixels": float(np.max(distances)),
        "positive_depth_fraction": float(np.mean(depth > 0)),
        "inside_image_fraction": float(np.mean(inside)),
    }
    record["pass"] = bool(
        record["median_distance_pixels"] <= gate["maximum_median_robot_mask_distance_pixels"]
        and record["p95_distance_pixels"] <= gate["maximum_p95_robot_mask_distance_pixels"]
        and record["positive_depth_fraction"] >= gate["minimum_positive_depth_fraction"]
        and record["inside_image_fraction"] >= gate["minimum_inside_image_fraction"]
    )
    return record


def save_derived(path: Path, dataset: dict[str, Any], representations: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]] | None = None) -> None:
    arrays: dict[str, np.ndarray] = {}
    arrays.update({f"metadata__{name}": value for name, value in dataset["metadata"].items()})
    arrays.update({f"target__{name}": value.astype(np.float32) for name, value in dataset["targets"].items()})
    arrays.update({f"valid__{name}": value for name, value in dataset["valid"].items()})
    arrays.update({f"representation__{name}": value.astype(np.float32) for name, value in representations.items()})
    arrays.update({f"diagnostic__{name}": np.asarray(value) for name, value in dataset["diagnostics"].items()})
    arrays["projected_ee"] = np.asarray(dataset["projected_ee"], dtype=np.float32)
    arrays["projection_depth"] = np.asarray(dataset["projection_depth"], dtype=np.float32)
    if predictions:
        for representation, records in predictions.items():
            for target, value in records.items():
                arrays[f"prediction__{representation}__{target}"] = value.astype(np.float32)
    atomic_npz(path, arrays)


def calibrate(protocol: dict[str, Any], protocol_commit: str) -> None:
    loaded = load_partition(protocol, "calibration")
    dataset, spatial = loaded["dataset"], loaded["spatial"]
    projection = projection_validation(dataset, loaded["masks"], protocol)
    if not projection["pass"]:
        raise RuntimeError("CAMERA_PROJECTION_BLOCK")
    representations = build_representations(spatial, dataset)
    grid = [float(value) for value in protocol["probes"]["ridge_grid"]]
    groups = dataset["metadata"]["init_state"]
    weights: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}
    for representation, features in representations.items():
        models[representation] = {}
        for target_name in TARGET_NAMES:
            valid = dataset["valid"][target_name]
            alpha, cv = select_ridge_alpha(features[valid], dataset["targets"][target_name][valid], groups[valid], grid)
            fit = fit_ridge(features[valid], dataset["targets"][target_name][valid], alpha)
            weights.update(ridge_to_arrays(f"ridge__{representation}__{target_name}", fit))
            models[representation][target_name] = {"dimensions": int(features.shape[1]), "selected_alpha": alpha, "cv": cv, "calibration_rows": int(valid.sum())}
    for representation in ("global_summary", "patch_summary", "global_plus_patch", "global_plus_patch_plus_strict"):
        fit = fit_logistic(representations[representation], dataset["diagnostics"]["near_contact_either"].astype(int), float(protocol["near_contact_probe"]["alpha"]))
        weights.update(logistic_to_arrays(f"near_contact__{representation}", fit))
    target_correlations = {}
    for first in TARGET_NAMES:
        target_correlations[first] = {}
        for second in TARGET_NAMES:
            use = dataset["valid"][first] & dataset["valid"][second]
            target_correlations[first][second] = safe_correlation(
                dataset["targets"][first][use], dataset["targets"][second][use]
            )
    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    derived_path = ROOT / protocol["outputs"]["calibration_derived"]
    atomic_npz(weights_path, weights)
    save_derived(derived_path, dataset, representations)
    freeze = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_INIT36_39_IMAGE_SPACE_EVALUATION",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/natural_image_space_interaction_protocol.json"),
        "calibration_partition": {"init_state_indices": [31, 32, 33, 34], "branch_points": 40, "rows": 960, "evaluation_partition_read": False},
        "projection_validation": projection,
        "models": models,
        "calibration_target_target_correlations": target_correlations,
        "target_valid_rows": {name: int(value.sum()) for name, value in dataset["valid"].items()},
        "artifacts": {
            "weights": {"path": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path)},
            "derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)},
        },
        "forbidden_activity_audit": {"evaluation_partition_read": False, "init40_49_access": False, "reward_done_success_or_task_outcome_read": False, "model_controller_or_encoder_training": False},
    }
    atomic_json(ROOT / protocol["outputs"]["calibration_freeze"], freeze)
    print(json.dumps({"status": freeze["status"], "projection": projection}, indent=2))


def _masked_indices(draw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return draw[valid[draw]]


def evaluate(protocol: dict[str, Any], protocol_commit: str) -> None:
    freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    weights = _load_npz(ROOT / protocol["outputs"]["calibration_weights"])
    loaded = load_partition(protocol, "development_evaluation")
    dataset, spatial = loaded["dataset"], loaded["spatial"]
    projection = projection_validation(dataset, loaded["masks"], protocol)
    if not projection["pass"]:
        raise RuntimeError("CAMERA_PROJECTION_BLOCK")
    representations = build_representations(spatial, dataset)
    metadata = dataset["metadata"]
    draws = bootstrap_indices(metadata, int(protocol["bootstrap"]["replicates"]), int(protocol["bootstrap"]["seed"]))
    correlation_draws = draws[: int(protocol["bootstrap"]["correlation_replicates"])]

    scalar_values = {
        "global_distance": np.asarray(spatial["diagnostic__global_distance"], dtype=np.float64),
        "patch_mean": np.asarray(spatial["diagnostic__patch_mean"], dtype=np.float64),
        "patch_top5_mean": np.asarray(spatial["diagnostic__patch_top5_mean"], dtype=np.float64),
        "patch_top10_mean": np.asarray(spatial["diagnostic__patch_top10_mean"], dtype=np.float64),
        "robot_patch_mean": np.asarray(spatial["diagnostic__robot_patch_mean"], dtype=np.float64),
        "relevant_patch_mean": np.asarray(spatial["diagnostic__relevant_patch_mean"], dtype=np.float64),
        "strict_interaction_patch_mean": np.asarray(dataset["diagnostics"]["strict_interaction_patch_mean"], dtype=np.float64),
        "background_patch_mean": np.asarray(spatial["diagnostic__background_patch_mean"], dtype=np.float64),
    }
    raw_correlations: dict[str, Any] = {}
    delta_rho: dict[str, Any] = {}
    for target_name in TARGET_NAMES:
        target, valid = dataset["targets"][target_name], dataset["valid"][target_name]
        raw_correlations[target_name], delta_rho[target_name] = {}, {}
        global_rho = safe_spearman(scalar_values["global_distance"][valid], target[valid])
        for name, value in scalar_values.items():
            record = correlation_cuts(value, target, metadata, valid)
            record["overall"]["spearman_ci95"] = bootstrap_metric(
                correlation_draws,
                lambda draw, v=value, t=target, ok=valid: safe_spearman(v[_masked_indices(draw, ok)], t[_masked_indices(draw, ok)]),
            )
            raw_correlations[target_name][name] = record
            if name != "global_distance":
                point = safe_spearman(value[valid], target[valid]) - global_rho
                ci = bootstrap_metric(
                    correlation_draws,
                    lambda draw, v=value, t=target, g=scalar_values["global_distance"], ok=valid: safe_spearman(v[_masked_indices(draw, ok)], t[_masked_indices(draw, ok)]) - safe_spearman(g[_masked_indices(draw, ok)], t[_masked_indices(draw, ok)]),
                )
                delta_rho[target_name][name] = {"delta_rho": point, "ci95": ci}

    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in REPRESENTATIONS}
    probe_metrics: dict[str, Any] = {name: {} for name in REPRESENTATIONS}
    for representation, features in representations.items():
        for target_name in TARGET_NAMES:
            prediction = predict_ridge(features, ridge_from_arrays(f"ridge__{representation}__{target_name}", weights)).reshape(-1)
            predictions[representation][target_name] = prediction
            valid = dataset["valid"][target_name]
            probe_metrics[representation][target_name] = cut_metrics(prediction[valid], dataset["targets"][target_name][valid], {name: value[valid] for name, value in metadata.items()})

    shuffled_patch = deranged_group(representations["patch_summary"], metadata, int(protocol["shuffle"]["seed"]))
    relevant_patch = representations["patch_summary"].copy()
    relevant_patch[:, 5:6] = deranged_group(relevant_patch[:, 5:6], metadata, int(protocol["shuffle"]["seed"]) + 1)
    strict = representations["global_plus_patch_plus_strict"][:, 20:21]
    shuffled_strict = deranged_group(strict, metadata, int(protocol["shuffle"]["seed"]) + 2)
    incremental: dict[str, Any] = {}
    shuffles: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    gate = protocol["gates"]
    for target_name in TARGET_NAMES:
        target, valid = dataset["targets"][target_name], dataset["valid"][target_name]
        parent, child = predictions["global_summary"][target_name], predictions["global_plus_patch"][target_name]
        cuts = cut_improvements(parent, child, target, metadata, valid)
        ci = bootstrap_metric(draws, lambda draw, p=parent, c=child, t=target, ok=valid: relative_improvement(p, c, t, _masked_indices(draw, ok)))
        incremental[target_name] = {**cuts, "ci95": ci}

        fit_gp = ridge_from_arrays(f"ridge__global_plus_patch__{target_name}", weights)
        full_alt = predict_ridge(np.concatenate((representations["global_summary"], shuffled_patch), axis=1), fit_gp).reshape(-1)
        relevant_alt = predict_ridge(np.concatenate((representations["global_summary"], relevant_patch), axis=1), fit_gp).reshape(-1)
        fit_strict = ridge_from_arrays(f"ridge__global_plus_patch_plus_strict__{target_name}", weights)
        strict_original = predictions["global_plus_patch_plus_strict"][target_name]
        strict_alt = predict_ridge(np.concatenate((representations["global_plus_patch"], shuffled_strict), axis=1), fit_strict).reshape(-1)

        def degradation(original: np.ndarray, alternate: np.ndarray, indices: np.ndarray | None = None) -> float:
            base = rmse(original, target, indices)
            return (rmse(alternate, target, indices) - base) / max(base, 1.0e-12)

        records = {}
        for name, original, alternate in (
            ("full_patch", child, full_alt),
            ("relevant_patch_only", child, relevant_alt),
            ("strict_interaction_only", strict_original, strict_alt),
        ):
            use = np.flatnonzero(valid)
            point = degradation(original, alternate, use)
            boot = bootstrap_metric(draws, lambda draw, o=original, a=alternate, ok=valid: degradation(o, a, _masked_indices(draw, ok)))
            records[name] = {"original_rmse": rmse(original, target, use), "shuffled_rmse": rmse(alternate, target, use), "relative_degradation": point, "ci95": boot, "passes": bool(point > 0 and boot[0] > 0)}
        shuffles[target_name] = records
        tasks = sum(row["relative_rmse_improvement"] > 0 for row in cuts["task"])
        horizons = sum(row["relative_rmse_improvement"] > 0 for row in cuts["horizon"])
        pairs = sum(row["relative_rmse_improvement"] > 0 for row in cuts["pair"])
        passes = bool(cuts["overall"] >= gate["minimum_incremental_rmse_improvement"] and ci[0] > 0 and records["full_patch"]["passes"] and tasks >= gate["minimum_positive_tasks"] and horizons >= gate["minimum_positive_horizons"] and pairs >= gate["minimum_positive_pairs"])
        decisions[target_name] = {"passes_primary_gate": passes, "increment": cuts["overall"], "increment_ci95": ci, "full_patch_shuffle_pass": records["full_patch"]["passes"], "positive_tasks": tasks, "positive_horizons": horizons, "positive_pairs": pairs}

    family_decisions = {}
    for family, names in TARGET_FAMILIES.items():
        passing = [name for name in names if decisions[name]["passes_primary_gate"]]
        family_decisions[family] = {"targets": list(names), "passing_targets": passing, "passes": bool(passing)}
    strict_interaction_pass = any(
        decisions[name]["passes_primary_gate"] and shuffles[name]["strict_interaction_only"]["passes"]
        for name in TARGET_FAMILIES["interaction"] + TARGET_FAMILIES["relative"]
    )

    prior_result = json.loads((ROOT / protocol["comparison_to_3d"]["prior_diagnosis"]).read_text())
    mapping = protocol["comparison_to_3d"]["target_mapping"]
    comparisons = {}
    mismatch_supported = False
    for name, prior_name in mapping.items():
        prior_increment = float(prior_result["incremental_global_plus_patch"][prior_name]["overall"])
        current = float(decisions[name]["increment"])
        passes = bool(decisions[name]["passes_primary_gate"] and current >= prior_increment + protocol["comparison_to_3d"]["minimum_increment_advantage"])
        comparisons[name] = {"prior_3d_target": prior_name, "image_space_increment": current, "prior_3d_increment": prior_increment, "difference": current - prior_increment, "supports_mismatch": passes}
        mismatch_supported |= passes

    labels = []
    if family_decisions["robot"]["passes"]:
        labels.append("SPATIAL_EE_SIGNAL_CONFIRMED")
    if family_decisions["object"]["passes"]:
        labels.append("SPATIAL_OBJECT_IMAGE_SIGNAL_CONFIRMED")
    if family_decisions["relative"]["passes"]:
        labels.append("SPATIAL_RELATIVE_IMAGE_GEOMETRY_SIGNAL_CONFIRMED")
    if family_decisions["interaction"]["passes"] and strict_interaction_pass:
        labels.append("SPATIAL_INTERACTION_IMAGE_SIGNAL_CONFIRMED")
    if mismatch_supported:
        labels.append("IMAGE_SPACE_TARGET_MISMATCH_SUPPORTED")
    redesign = any(family_decisions[name]["passes"] for name in ("object", "relative", "interaction"))
    if redesign:
        labels.append("GLOBAL_POOLING_LOSES_IMAGE_SPACE_INTERACTION_SIGNAL")
        authorization = "GO_SPATIAL_TARGET_DESIGN"
    elif family_decisions["robot"]["passes"]:
        labels.append("SPATIAL_SIGNAL_REMAINS_NONUNIQUE")
        authorization = "BLOCK_VISUAL_TARGET_REDESIGN"
    else:
        labels.append("NATURAL_IMAGE_SPACE_RELEVANCE_UNRESOLVED")
        authorization = "BLOCK_VISUAL_TARGET_REDESIGN"

    near_contact = {}
    truth = dataset["diagnostics"]["near_contact_either"].astype(int)
    for representation in ("global_summary", "patch_summary", "global_plus_patch", "global_plus_patch_plus_strict"):
        probability = predict_logistic(representations[representation], logistic_from_arrays(f"near_contact__{representation}", weights))
        from src.probes.spatial_relevance import contact_metrics
        near_contact[representation] = contact_metrics(probability, truth)

    target_correlations = {}
    for first in TARGET_NAMES:
        target_correlations[first] = {}
        for second in TARGET_NAMES:
            use = dataset["valid"][first] & dataset["valid"][second]
            target_correlations[first][second] = safe_correlation(dataset["targets"][first][use], dataset["targets"][second][use])

    derived_path = ROOT / protocol["outputs"]["evaluation_derived"]
    save_derived(derived_path, dataset, representations, predictions)
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "date": protocol["date"],
        "status": "COMPLETE",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/natural_image_space_interaction_protocol.json"),
        "calibration_freeze_sha256": sha256(freeze_path),
        "data_boundary": {"calibration_init_states": [31, 32, 33, 34], "development_evaluation_init_states": [36, 37, 38, 39], "evaluation_is_fully_blind": False, "historical_reference_only": [30, 35], "init40_49_accessed": False, "branch_points": 40, "rows": 960},
        "camera_projection": {"inventory_status": "CAMERA_PROJECTION_VALIDATED", "calibration_validation": freeze["projection_validation"], "evaluation_validation": projection},
        "target_definitions": protocol["image_space_targets"],
        "target_valid_rows": {name: int(value.sum()) for name, value in dataset["valid"].items()},
        "target_distributions": {name: {"mean": float(np.mean(dataset["targets"][name][dataset["valid"][name]])), "median": float(np.median(dataset["targets"][name][dataset["valid"][name]])), "p75": float(np.quantile(dataset["targets"][name][dataset["valid"][name]], 0.75)), "p90": float(np.quantile(dataset["targets"][name][dataset["valid"][name]], 0.90)), "p95": float(np.quantile(dataset["targets"][name][dataset["valid"][name]], 0.95))} for name in TARGET_NAMES},
        "calibration_target_target_correlations": freeze["calibration_target_target_correlations"],
        "evaluation_target_target_correlations": target_correlations,
        "raw_correlations": raw_correlations,
        "delta_rho": delta_rho,
        "probe_metrics": probe_metrics,
        "incremental_global_plus_patch": incremental,
        "shuffle_controls": shuffles,
        "near_contact_secondary": near_contact,
        "target_decisions": decisions,
        "family_decisions": family_decisions,
        "strict_interaction_signal_pass": strict_interaction_pass,
        "comparison_to_frozen_3d_targets": comparisons,
        "candidate_headroom_reference": protocol["candidate_headroom_reference"],
        "decision": {"mechanism_labels": labels, "next_stage_authorization": authorization, "controller_authorization": False, "repeat_similar_representation_probe_if_unresolved": False},
        "artifacts": {"evaluation_derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)}},
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {"natural_rgb_modified": False, "masked_image_inference": False, "reward_done_success_or_task_outcome_read": False, "init40_49_access": False, "world_model_encoder_candidate_reranker_or_controller_change": False, "controller_success_run": False, "upstream_overwrite": False},
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    atomic_json(ROOT / protocol["outputs"]["diagnosis"], result)
    print(json.dumps(result["decision"], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("calibrate", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/natural_image_space_interaction_protocol.json")
    args = parser.parse_args()
    protocol, commit = load_protocol(args.protocol, args.phase)
    if args.phase == "calibrate":
        calibrate(protocol, commit)
    else:
        evaluate(protocol, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
