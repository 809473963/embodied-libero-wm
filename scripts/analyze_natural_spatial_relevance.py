#!/usr/bin/env python3
"""Calibrate and evaluate the Natural Spatial Target Relevance Audit."""

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

from src.probes.spatial_relevance import (  # noqa: E402
    PAIR_NAMES,
    SCALAR_REPRESENTATIONS,
    TARGET_FAMILIES,
    TARGET_NAMES,
    build_external_targets,
    build_natural_spatial_dataset,
    contact_metrics,
    cut_metrics,
    fit_logistic,
    fit_ridge,
    fit_randomized_pca,
    logistic_from_arrays,
    logistic_to_arrays,
    pca_from_arrays,
    pca_to_arrays,
    predict_logistic,
    predict_ridge,
    ridge_from_arrays,
    ridge_to_arrays,
    safe_correlation,
    safe_spearman,
    select_ridge_alpha,
    transform_pca,
)


PROTOCOL_VERSION = "natural-spatial-target-relevance-audit-v1"
REPRESENTATIONS = (
    "global_summary",
    "patch_summary",
    "global_plus_patch",
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


def normalization() -> tuple[np.ndarray, np.ndarray]:
    state = torch.load(
        ROOT / "results/world_model/d14_single_seed11/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    return (
        np.asarray(state["normalization"]["proprio"]["mean"], dtype=np.float64),
        np.asarray(state["normalization"]["proprio"]["std"], dtype=np.float64),
    )


def load_protocol(path: Path, phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected protocol version")
    if protocol["status"] != "PREREGISTERED_BEFORE_NATURAL_RELEVANCE_CALIBRATION_OR_EVALUATION":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise RuntimeError("protocol must be committed")
    for item in protocol["implementation"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"implementation changed after preregistration: {item['path']}")
    for item in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen upstream changed: {item['path']}")
    if phase == "calibrate":
        for relative in (
            protocol["outputs"]["calibration_freeze"],
            protocol["outputs"]["calibration_weights"],
            protocol["outputs"]["calibration_derived"],
            protocol["outputs"]["diagnosis"],
        ):
            if (ROOT / relative).exists():
                raise RuntimeError(f"calibration refuses pre-existing output: {relative}")
    else:
        freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("calibration freeze must be committed before evaluation")
        freeze = json.loads(freeze_path.read_text())
        if freeze["status"] != "FROZEN_BEFORE_INIT36_39_RELEVANCE_EVALUATION":
            raise ValueError("calibration freeze status changed")
        if freeze["protocol_sha256"] != sha256(path):
            raise ValueError("calibration freeze references another protocol")
        for item in freeze["artifacts"].values():
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"calibration artifact changed: {item['path']}")
    return protocol, commit


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def load_partition(protocol: dict[str, Any], partition: str) -> dict[str, Any]:
    record = protocol["partitions"][partition]
    visual = _load_npz(ROOT / record["visual_features"])
    masks = _load_npz(ROOT / record["masks"])
    base = _load_npz(ROOT / record["branch_sidecar"])
    state = _load_npz(ROOT / record["state_sidecar"])
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        if not np.array_equal(base[key], masks[key]) or not np.array_equal(base[key], state[key]):
            raise ValueError(f"partition identity mismatch: {key}")
    if set(np.unique(base["init_state_index"]).tolist()) != set(record["init_state_indices"]):
        raise ValueError("partition boundary changed")
    p_mean, p_std = normalization()
    state_dataset = build_external_targets(base, state, p_mean, p_std)
    dataset = build_natural_spatial_dataset(
        visual,
        masks,
        state_dataset,
        float(protocol["interaction_region"]["pixel_distance_threshold"]),
    )
    return {"dataset": dataset, "base": base, "state": state, "visual": visual, "masks": masks}


def build_representations(
    dataset: dict[str, Any],
    global_pca: Any,
    patch_pca: Any,
) -> dict[str, np.ndarray]:
    arrays = dataset["arrays"]
    global_scores = transform_pca(arrays["delta_global"], global_pca)
    patch_scores = transform_pca(arrays["delta_patch_flat"], patch_pca)
    global_summary = np.column_stack(
        [arrays["global_distance"], arrays["global_l2"], global_scores[:, :8]]
    )
    patch_summary = np.asarray(dataset["patch_summary"], dtype=np.float64)
    result = {
        "global_summary": global_summary,
        "patch_summary": patch_summary,
        "global_plus_patch": np.concatenate((global_summary, patch_summary), axis=1),
        "global_pca16": global_scores,
        "patch_pca16": patch_scores,
    }
    expected = {"global_summary": 10, "patch_summary": 10, "global_plus_patch": 20, "global_pca16": 16, "patch_pca16": 16}
    for name, value in result.items():
        if value.shape != (960, expected[name]) or not np.isfinite(value).all():
            raise ValueError(f"representation contract failed: {name} {value.shape}")
    return result


def save_derived(path: Path, dataset: dict[str, Any], representations: dict[str, np.ndarray]) -> None:
    arrays: dict[str, np.ndarray] = {}
    arrays.update({f"metadata__{name}": value for name, value in dataset["metadata"].items()})
    arrays.update({f"target__{name}": value.astype(np.float32) for name, value in dataset["targets"].items()})
    arrays["target__contact"] = dataset["contact"].astype(np.int8)
    arrays.update({f"representation__{name}": value.astype(np.float32) for name, value in representations.items()})
    for name in (
        "global_distance",
        "patch_mean",
        "patch_max",
        "patch_top5_mean",
        "patch_top10_mean",
        "robot_patch_mean",
        "relevant_patch_mean",
        "interaction_patch_mean",
        "background_patch_mean",
        "distance_map",
        "pair_fractional_coverage",
        "interaction_indicator",
        "top5_fractional_coverage",
        "top10_fractional_coverage",
        "top5_interaction_fraction",
        "top10_interaction_fraction",
    ):
        arrays[f"diagnostic__{name}"] = np.asarray(dataset["arrays"][name], dtype=np.float32)
    atomic_npz(path, arrays)


def calibrate(protocol: dict[str, Any], protocol_commit: str) -> None:
    loaded = load_partition(protocol, "calibration")
    dataset = loaded["dataset"]
    rank = int(protocol["equal_rank_pca"]["rank"])
    global_pca, _ = fit_randomized_pca(
        dataset["arrays"]["delta_global"],
        rank,
        int(protocol["equal_rank_pca"]["global_seed"]),
    )
    patch_pca, _ = fit_randomized_pca(
        dataset["arrays"]["delta_patch_flat"],
        rank,
        int(protocol["equal_rank_pca"]["patch_seed"]),
    )
    representations = build_representations(dataset, global_pca, patch_pca)
    weights: dict[str, np.ndarray] = {}
    weights.update(pca_to_arrays("global_pca", global_pca))
    weights.update(pca_to_arrays("patch_pca", patch_pca))
    models: dict[str, Any] = {}
    grid = [float(value) for value in protocol["probes"]["ridge_grid"]]
    init_state = dataset["metadata"]["init_state"]
    for representation, features in representations.items():
        models[representation] = {}
        for target_name in TARGET_NAMES:
            target = dataset["targets"][target_name]
            alpha, cv = select_ridge_alpha(features, target, init_state, grid)
            fit = fit_ridge(features, target, alpha)
            prefix = f"ridge__{representation}__{target_name}"
            weights.update(ridge_to_arrays(prefix, fit))
            models[representation][target_name] = {
                "dimensions": int(features.shape[1]),
                "selected_alpha": alpha,
                "cv": cv,
            }
        logistic = fit_logistic(
            features,
            dataset["contact"],
            float(protocol["contact_probe"]["alpha"]),
        )
        weights.update(logistic_to_arrays(f"contact__{representation}", logistic))
    reconstruction_alpha, reconstruction_cv = select_ridge_alpha(
        representations["patch_summary"],
        dataset["arrays"]["global_distance"],
        init_state,
        grid,
    )
    reconstruction_fit = fit_ridge(
        representations["patch_summary"],
        dataset["arrays"]["global_distance"],
        reconstruction_alpha,
    )
    weights.update(ridge_to_arrays("global_reconstruction", reconstruction_fit))
    thresholds = {
        name: float(np.quantile(dataset["targets"][name], 0.75)) for name in TARGET_NAMES
    }
    for name, value in thresholds.items():
        weights[f"target_p75__{name}"] = np.asarray(value, dtype=np.float64)

    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    derived_path = ROOT / protocol["outputs"]["calibration_derived"]
    atomic_npz(weights_path, weights)
    save_derived(derived_path, dataset, representations)
    freeze = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_INIT36_39_RELEVANCE_EVALUATION",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/natural_spatial_target_relevance_protocol.json"),
        "partition": {
            "init_state_indices": [31, 32, 33, 34],
            "branch_points": 40,
            "rows": 960,
            "evaluation_features_or_state_read": False,
        },
        "natural_image_only": True,
        "interaction_region": protocol["interaction_region"],
        "representation_dimensions": {name: int(value.shape[1]) for name, value in representations.items()},
        "equal_rank_pca": {
            "rank": rank,
            "global_explained_variance_proxy": None,
            "patch_explained_variance_proxy": None,
        },
        "models": models,
        "global_reconstruction": {
            "selected_alpha": reconstruction_alpha,
            "cv": reconstruction_cv,
        },
        "target_p75_calibration_only": thresholds,
        "artifacts": {
            "weights": {"path": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path)},
            "derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)},
        },
        "forbidden_activity_audit": {
            "evaluation_partition_read": False,
            "simulator_run": False,
            "masked_or_modified_image_used": False,
            "world_model_or_encoder_training": False,
            "controller_or_success_evaluation": False,
            "init40_49_access": False,
        },
    }
    atomic_json(ROOT / protocol["outputs"]["calibration_freeze"], freeze)
    print(json.dumps({"status": freeze["status"], "weights_sha256": freeze["artifacts"]["weights"]["sha256"]}, indent=2))


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
    return interval([float(function(indices)) for indices in draws])


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
                    raise RuntimeError("shuffle cell has fewer than two branches")
                shift = int(rng.integers(1, ordered.size))
                result[ordered] = source[np.roll(ordered, shift)]
    return result


def correlation_cuts(value: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": safe_correlation(value, target), "horizon": [], "task": [], "pair": []}
    for field, values in (("horizon", range(1, 9)), ("task", range(10)), ("pair", range(3))):
        for item in values:
            mask = metadata[field] == item
            label: Any = PAIR_NAMES[item] if field == "pair" else item
            result[field].append({field: label, **safe_correlation(value[mask], target[mask])})
    return result


def rmse(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is not None:
        prediction = prediction[indices]
        target = target[indices]
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(target)))))


def relative_improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    parent_rmse = rmse(parent, target, indices)
    child_rmse = rmse(child, target, indices)
    return (parent_rmse - child_rmse) / max(parent_rmse, 1.0e-12)


def cut_improvements(parent: np.ndarray, child: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": relative_improvement(parent, child, target), "horizon": [], "task": [], "pair": []}
    for field, values in (("horizon", range(1, 9)), ("task", range(10)), ("pair", range(3))):
        for item in values:
            mask = np.flatnonzero(metadata[field] == item)
            label: Any = PAIR_NAMES[item] if field == "pair" else item
            result[field].append({field: label, "relative_rmse_improvement": relative_improvement(parent, child, target, mask)})
    return result


def localization(dataset: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    arrays = dataset["arrays"]

    def summarize(indices: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(indices.size),
            "top5_fractional_coverage": np.mean(arrays["top5_fractional_coverage"][indices], axis=0).tolist(),
            "top10_fractional_coverage": np.mean(arrays["top10_fractional_coverage"][indices], axis=0).tolist(),
            "top5_interaction_fraction": float(np.mean(arrays["top5_interaction_fraction"][indices])),
            "top10_interaction_fraction": float(np.mean(arrays["top10_interaction_fraction"][indices])),
        }

    result = {"all_rows": summarize(np.arange(960)), "high_target": {}}
    for target_name in TARGET_FAMILIES["object"] + TARGET_FAMILIES["relative"]:
        selected = np.flatnonzero(dataset["targets"][target_name] >= thresholds[target_name])
        result["high_target"][target_name] = {"calibration_p75_threshold": thresholds[target_name], **summarize(selected)}
    return result


def evaluate(protocol: dict[str, Any], protocol_commit: str) -> None:
    freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    with np.load(weights_path, allow_pickle=False) as weights_file:
        weights = {name: np.asarray(weights_file[name]) for name in weights_file.files}
    loaded = load_partition(protocol, "development_evaluation")
    dataset = loaded["dataset"]
    representations = build_representations(
        dataset,
        pca_from_arrays("global_pca", weights),
        pca_from_arrays("patch_pca", weights),
    )
    metadata = dataset["metadata"]
    draws = bootstrap_indices(
        metadata,
        int(protocol["bootstrap"]["replicates"]),
        int(protocol["bootstrap"]["seed"]),
    )
    correlation_draws = draws[: int(protocol["bootstrap"]["correlation_replicates"])]

    raw_correlations: dict[str, Any] = {}
    delta_rho: dict[str, Any] = {}
    scalar_values = {name: dataset["arrays"][name] for name in SCALAR_REPRESENTATIONS}
    for target_name in TARGET_NAMES:
        target = dataset["targets"][target_name]
        raw_correlations[target_name] = {}
        delta_rho[target_name] = {}
        global_value = scalar_values["global_distance"]
        global_rho = safe_spearman(global_value, target)
        for representation, value in scalar_values.items():
            record = correlation_cuts(value, target, metadata)
            record["overall"]["spearman_ci95"] = bootstrap_metric(
                correlation_draws,
                lambda index, v=value, t=target: safe_spearman(v[index], t[index]),
            )
            raw_correlations[target_name][representation] = record
            if representation != "global_distance":
                point = safe_spearman(value, target) - global_rho
                ci = bootstrap_metric(
                    correlation_draws,
                    lambda index, v=value, t=target, g=global_value: safe_spearman(v[index], t[index])
                    - safe_spearman(g[index], t[index]),
                )
                delta_rho[target_name][representation] = {"delta_rho": point, "ci95": ci}

    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in REPRESENTATIONS}
    probe_metrics: dict[str, Any] = {name: {} for name in REPRESENTATIONS}
    for representation, features in representations.items():
        for target_name in TARGET_NAMES:
            fit = ridge_from_arrays(f"ridge__{representation}__{target_name}", weights)
            prediction = predict_ridge(features, fit).reshape(-1)
            predictions[representation][target_name] = prediction
            probe_metrics[representation][target_name] = cut_metrics(
                prediction, dataset["targets"][target_name], metadata
            )

    shuffled_patch = deranged_group(
        representations["patch_summary"], metadata, int(protocol["shuffle"]["seed"])
    )
    semantic_shuffles = {
        "relevant_patch": representations["patch_summary"].copy(),
        "interaction_patch": representations["patch_summary"].copy(),
    }
    semantic_shuffles["relevant_patch"][:, 5:6] = deranged_group(
        representations["patch_summary"][:, 5:6], metadata, int(protocol["shuffle"]["seed"]) + 1
    )
    semantic_shuffles["interaction_patch"][:, 6:7] = deranged_group(
        representations["patch_summary"][:, 6:7], metadata, int(protocol["shuffle"]["seed"]) + 2
    )

    incremental: dict[str, Any] = {}
    shuffles: dict[str, Any] = {}
    target_decisions: dict[str, Any] = {}
    gate = protocol["gates"]
    for target_name in TARGET_NAMES:
        target = dataset["targets"][target_name]
        parent = predictions["global_summary"][target_name]
        child = predictions["global_plus_patch"][target_name]
        cuts = cut_improvements(parent, child, target, metadata)
        ci = bootstrap_metric(draws, lambda index, p=parent, c=child, t=target: relative_improvement(p, c, t, index))
        incremental[target_name] = {**cuts, "ci95": ci}

        fit = ridge_from_arrays(f"ridge__global_plus_patch__{target_name}", weights)
        full_shuffled_features = np.concatenate((representations["global_summary"], shuffled_patch), axis=1)
        full_shuffled = predict_ridge(full_shuffled_features, fit).reshape(-1)
        semantic_predictions = {}
        for name, patch_value in semantic_shuffles.items():
            features = np.concatenate((representations["global_summary"], patch_value), axis=1)
            semantic_predictions[name] = predict_ridge(features, fit).reshape(-1)
        original_rmse = rmse(child, target)

        def degradation(alternate: np.ndarray, indices: np.ndarray | None = None) -> float:
            base = rmse(child, target, indices)
            changed = rmse(alternate, target, indices)
            return (changed - base) / max(base, 1.0e-12)

        full_degradation = degradation(full_shuffled)
        full_ci = bootstrap_metric(draws, lambda index, value=full_shuffled: degradation(value, index))
        semantic_records = {}
        for name, alternate in semantic_predictions.items():
            value = degradation(alternate)
            value_ci = bootstrap_metric(draws, lambda index, pred=alternate: degradation(pred, index))
            semantic_records[name] = {
                "original_rmse": original_rmse,
                "shuffled_rmse": rmse(alternate, target),
                "relative_degradation": value,
                "ci95": value_ci,
                "passes": bool(value > 0.0 and value_ci[0] > 0.0),
            }
        shuffles[target_name] = {
            "full_patch": {
                "original_rmse": original_rmse,
                "shuffled_rmse": rmse(full_shuffled, target),
                "relative_degradation": full_degradation,
                "ci95": full_ci,
                "passes": bool(full_degradation > 0.0 and full_ci[0] > 0.0),
            },
            **semantic_records,
        }
        task_positive = sum(row["relative_rmse_improvement"] > 0.0 for row in cuts["task"])
        horizon_positive = sum(row["relative_rmse_improvement"] > 0.0 for row in cuts["horizon"])
        pair_positive = sum(row["relative_rmse_improvement"] > 0.0 for row in cuts["pair"])
        passes = bool(
            cuts["overall"] >= float(gate["minimum_incremental_rmse_improvement"])
            and ci[0] > 0.0
            and shuffles[target_name]["full_patch"]["passes"]
            and task_positive >= int(gate["minimum_positive_tasks"])
            and horizon_positive >= int(gate["minimum_positive_horizons"])
            and pair_positive >= int(gate["minimum_positive_pairs"])
        )
        target_decisions[target_name] = {
            "passes_primary_gate": passes,
            "positive_tasks": task_positive,
            "positive_horizons": horizon_positive,
            "positive_pairs": pair_positive,
            "increment": cuts["overall"],
            "increment_ci95": ci,
            "patch_shuffle_pass": shuffles[target_name]["full_patch"]["passes"],
        }

    family_decisions = {
        family: {
            "targets": list(targets),
            "passing_targets": [name for name in targets if target_decisions[name]["passes_primary_gate"]],
        }
        for family, targets in TARGET_FAMILIES.items()
    }
    for record in family_decisions.values():
        record["passes"] = bool(record["passing_targets"])

    interaction_pass = bool(
        family_decisions["relative"]["passes"]
        and any(
            shuffles[name]["interaction_patch"]["passes"]
            for name in family_decisions["relative"]["passing_targets"]
        )
    )
    labels = []
    if family_decisions["robot"]["passes"]:
        labels.append("SPATIAL_REPRESENTATION_ROBOT_SENSITIVE")
    if family_decisions["object"]["passes"]:
        labels.append("SPATIAL_OBJECT_STATE_SIGNAL_CONFIRMED")
    if family_decisions["relative"]["passes"]:
        labels.append("SPATIAL_RELATIVE_GEOMETRY_SIGNAL_CONFIRMED")
    if interaction_pass:
        labels.append("SPATIAL_INTERACTION_SIGNAL_CONFIRMED")
    task_relevant = family_decisions["object"]["passes"] or family_decisions["relative"]["passes"]
    if task_relevant:
        labels.append("GLOBAL_POOLING_LOSES_TASK_RELEVANT_SIGNAL")
        authorization = "GO_SPATIAL_TARGET_DESIGN"
    elif family_decisions["robot"]["passes"]:
        labels.append("GLOBAL_POOLING_LOSES_MAINLY_ROBOT_APPEARANCE_SIGNAL")
        authorization = "BLOCK_VISUAL_TARGET_REDESIGN"
    else:
        labels.append("NATURAL_PATCH_TASK_RELEVANCE_UNRESOLVED")
        authorization = "BLOCK_VISUAL_TARGET_REDESIGN"

    contact_result = {}
    for representation, features in representations.items():
        fit = logistic_from_arrays(f"contact__{representation}", weights)
        contact_result[representation] = contact_metrics(
            predict_logistic(features, fit), dataset["contact"]
        )
    reconstruction_fit = ridge_from_arrays("global_reconstruction", weights)
    reconstruction_prediction = predict_ridge(
        representations["patch_summary"], reconstruction_fit
    ).reshape(-1)
    reconstruction = cut_metrics(
        reconstruction_prediction, dataset["arrays"]["global_distance"], metadata
    )
    thresholds = {name: float(np.asarray(weights[f"target_p75__{name}"])) for name in TARGET_NAMES}

    derived_path = ROOT / protocol["outputs"]["evaluation_derived"]
    derived_arrays: dict[str, np.ndarray] = {}
    derived_arrays.update({f"metadata__{name}": value for name, value in metadata.items()})
    derived_arrays.update({f"target__{name}": value.astype(np.float32) for name, value in dataset["targets"].items()})
    derived_arrays.update({f"representation__{name}": value.astype(np.float32) for name, value in representations.items()})
    for representation, by_target in predictions.items():
        for target_name, value in by_target.items():
            derived_arrays[f"prediction__{representation}__{target_name}"] = value.astype(np.float32)
    for name in (
        "global_distance", "patch_mean", "patch_max", "patch_top5_mean", "patch_top10_mean",
        "robot_patch_mean", "relevant_patch_mean", "interaction_patch_mean", "background_patch_mean",
        "distance_map", "pair_fractional_coverage", "interaction_indicator", "top5_fractional_coverage",
        "top10_fractional_coverage", "top5_interaction_fraction", "top10_interaction_fraction",
    ):
        derived_arrays[f"diagnostic__{name}"] = np.asarray(dataset["arrays"][name], dtype=np.float32)
    atomic_npz(derived_path, derived_arrays)

    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "date": protocol["date"],
        "status": "COMPLETE",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/natural_spatial_target_relevance_protocol.json"),
        "calibration_freeze_sha256": sha256(freeze_path),
        "data_boundary": {
            "calibration_init_states": [31, 32, 33, 34],
            "development_evaluation_init_states": [36, 37, 38, 39],
            "evaluation_is_fully_blind": False,
            "historical_reference_only": [30, 35],
            "init40_49_accessed": False,
            "natural_images_only": True,
            "rows": 960,
            "branch_points": 40,
        },
        "definitions": {
            "global": "frozen full-image L2-normalized CLIP ViT-B/32 visual_projection [512]",
            "patch": "frozen natural-image 7x7 post_layernorm visual_projection tokens [49,512]",
            "fractional_coverage_order": ["robot", "relevant_object", "other_object", "background_mixed"],
            "interaction_region": protocol["interaction_region"],
            "external_targets": {family: list(names) for family, names in TARGET_FAMILIES.items()},
        },
        "raw_correlations": raw_correlations,
        "delta_rho": delta_rho,
        "probe_metrics": probe_metrics,
        "incremental_global_plus_patch": incremental,
        "shuffle_controls": shuffles,
        "target_decisions": target_decisions,
        "family_decisions": family_decisions,
        "interaction_signal_pass": interaction_pass,
        "contact_secondary": contact_result,
        "global_pooling_reconstruction_reference": reconstruction,
        "information_localization": localization(dataset, thresholds),
        "decision": {
            "mechanism_labels": labels,
            "next_stage_authorization": authorization,
            "controller_authorization": False,
        },
        "artifacts": {
            "evaluation_derived": {
                "path": str(derived_path.relative_to(ROOT)),
                "sha256": sha256(derived_path),
            }
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "simulator_run": False,
            "masked_or_modified_image_used": False,
            "world_model_or_encoder_training": False,
            "candidate_reranker_or_controller_change": False,
            "reward_done_success_or_task_outcome_read": False,
            "init40_49_access": False,
            "seed23_47_run": False,
            "upstream_overwrite": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    result_path = ROOT / protocol["outputs"]["diagnosis"]
    atomic_json(result_path, result)
    print(json.dumps(result["decision"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/natural_spatial_target_relevance_protocol.json")
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, protocol_commit = load_protocol(args.protocol, args.phase)
    if args.phase == "calibrate":
        calibrate(protocol, protocol_commit)
    else:
        evaluate(protocol, protocol_commit)


if __name__ == "__main__":
    main()
