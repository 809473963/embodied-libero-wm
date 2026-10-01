#!/usr/bin/env python3
"""Calibrate and evaluate the preregistered continuous-footprint audit."""

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

from src.probes.continuous_footprint_decomposition import (
    CANDIDATES,
    DIMENSIONS,
    FootprintCandidate,
    footprint_candidates,
    pair_summary,
    representation_distances,
)
from src.probes.missing_state import (
    fit_ridge,
    predict_ridge,
    ridge_from_arrays,
    ridge_to_arrays,
    scalar_metrics,
    select_ridge_alpha,
)
from src.probes.spatial_relevance import PAIR_NAMES, safe_correlation, safe_spearman
from src.probes.spatial_target_design import footprint_from_object_masks, metadata_from_derived


PROTOCOL_VERSION = "continuous-footprint-information-decomposition-v1"
PROTOCOL_PATH = ROOT / "configs/continuous_footprint_information_decomposition_protocol.json"
TARGETS = (
    "object_mask_difference",
    "visible_area_difference",
    "object_mask_residual",
    "visible_area_residual",
)
RAW_TO_RESIDUAL = {
    "object_mask_difference": "object_mask_residual",
    "visible_area_difference": "visible_area_residual",
}


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
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True
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
    if protocol["protocol_version"] != PROTOCOL_VERSION or protocol["status"] != "PREREGISTERED_BEFORE_CALIBRATION_OR_EVALUATION_RESULTS":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(PROTOCOL_PATH)
    if not commit:
        raise RuntimeError("protocol must be committed")
    tag = protocol["frozen_upstream"]["tag"]
    if subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip() != "tag":
        raise RuntimeError("upstream tag is not annotated")
    if subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip() != protocol["frozen_upstream"]["commit"]:
        raise RuntimeError("upstream tag moved")
    for section in ("files",):
        for record in protocol["frozen_upstream"][section]:
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen upstream changed: {record['path']}")
    for record in protocol["implementation"].values():
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"implementation changed: {record['path']}")
    outputs = protocol["outputs"]
    if phase == "calibrate":
        for key in ("calibration_weights", "calibration_derived", "calibration_freeze", "evaluation_derived", "diagnosis"):
            if (ROOT / outputs[key]).exists():
                raise FileExistsError(outputs[key])
    else:
        freeze_path = ROOT / outputs["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("calibration freeze must be committed")
        freeze = json.loads(freeze_path.read_text())
        if freeze["status"] != "FROZEN_BEFORE_INIT36_39_CONTINUOUS_FOOTPRINT_EVALUATION":
            raise RuntimeError("calibration freeze status changed")
        if freeze["protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise RuntimeError("freeze references another protocol")
        for record in freeze["artifacts"].values():
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"calibration artifact changed: {record['path']}")
    return protocol, commit


def load_partition(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    record = protocol["partitions"][name]
    upstream = load_npz(ROOT / record["target_design_derived"])
    masks = load_npz(ROOT / record["per_object_masks"])
    metadata = metadata_from_derived(upstream)
    expected = set(record["init_state_indices"])
    if set(np.unique(metadata["init_state"]).tolist()) != expected or set(np.unique(masks["init_state_index"]).tolist()) != expected:
        raise ValueError("partition boundary changed")
    occupancy, area = footprint_from_object_masks(masks["per_object_masks"], 7)
    candidates = footprint_candidates(occupancy, area)
    targets = {name: np.asarray(upstream[f"target__{name}"], dtype=np.float64) for name in TARGETS}
    return {
        "metadata": metadata,
        "global": np.asarray(upstream["representation__global"], dtype=np.float64),
        "targets": targets,
        "occupancy": occupancy,
        "area": area,
        "candidates": candidates,
    }


def b2_candidate(occupancy: np.ndarray, area: np.ndarray) -> FootprintCandidate:
    binary = (np.asarray(occupancy) > 0.0).astype(np.float64)
    mass_scale = binary.mean(axis=(-2, -1))
    # B2 uses the same six-slot map summary as the frozen upstream audit.
    from src.probes.continuous_footprint_decomposition import _grid
    yy, xx = _grid()
    mass = binary.sum(axis=(-2, -1))
    centroid = np.stack(((binary * xx).sum((-2, -1)) / mass, (binary * yy).sum((-2, -1)) / mass), axis=-1)
    state = np.concatenate((binary.reshape(40, 3, 8, 49), np.asarray(area)[..., None]), axis=-1)
    return FootprintCandidate(state, binary.reshape(40, 3, 8, 49), centroid, np.asarray(area), True)


def oof_prediction(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    prediction = np.empty_like(np.asarray(y, dtype=np.float64).reshape(-1))
    for held_out in sorted(np.unique(groups).tolist()):
        train = groups != held_out
        fit = fit_ridge(x[train], y[train], alpha)
        prediction[~train] = predict_ridge(x[~train], fit).reshape(-1)
    return prediction


def rmse(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    pred, true = np.asarray(prediction), np.asarray(target)
    if indices is not None:
        pred, true = pred[indices], true[indices]
    return float(np.sqrt(np.mean(np.square(pred - true))))


def improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    parent_error = rmse(parent, target, indices)
    return (parent_error - rmse(child, target, indices)) / max(parent_error, 1.0e-12)


def fit_models(
    features: dict[str, np.ndarray], targets: dict[str, np.ndarray], groups: np.ndarray, grid: list[float]
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    weights: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, feature in features.items():
        records[name], predictions[name] = {}, {}
        for target_name, target in targets.items():
            alpha, cv = select_ridge_alpha(feature, target, groups, grid)
            fit = fit_ridge(feature, target, alpha)
            weights.update(ridge_to_arrays(f"ridge__{name}__{target_name}", fit))
            predictions[name][target_name] = oof_prediction(feature, target, groups, alpha)
            selected = next(row for row in cv if float(row["alpha"]) == alpha)
            records[name][target_name] = {"alpha": alpha, "cv_rmse": selected["mean_fold_rmse"], "dimensions": int(feature.shape[1]), "cv": cv}
    return weights, records, predictions


def calibrate(protocol: dict[str, Any], protocol_commit: str) -> None:
    partition = load_partition(protocol, "calibration")
    summaries = {name: pair_summary(candidate, partition["metadata"]) for name, candidate in partition["candidates"].items()}
    b2 = b2_candidate(partition["occupancy"], partition["area"])
    summaries["B2"] = pair_summary(b2, partition["metadata"])
    features = {"G": partition["global"]}
    features.update({name: np.concatenate((partition["global"], summary), axis=1) for name, summary in summaries.items()})
    weights, records, predictions = fit_models(
        features, partition["targets"], partition["metadata"]["init_state"], [float(v) for v in protocol["probes"]["ridge_grid"]]
    )
    baseline = predictions["G"]
    calibration_gains = {
        name: {target: improvement(baseline[target], predictions[name][target], partition["targets"][target]) for target in TARGETS}
        for name in summaries
    }
    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    derived_path = ROOT / protocol["outputs"]["calibration_derived"]
    atomic_npz(weights_path, weights)
    atomic_npz(
        derived_path,
        {
            **{f"metadata__{name}": value for name, value in partition["metadata"].items()},
            **{f"target__{name}": value.astype(np.float32) for name, value in partition["targets"].items()},
            **{f"summary__{name}": value.astype(np.float32) for name, value in summaries.items()},
            **{f"prediction__{name}__{target}": value.astype(np.float32) for name, by_target in predictions.items() for target, value in by_target.items()},
        },
    )
    f7 = partition["candidates"]["F7"].state
    freeze = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_INIT36_39_CONTINUOUS_FOOTPRINT_EVALUATION",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "partition": {"init_state_indices": [31, 32, 33, 34], "branch_points": 40, "pair_rows": 960, "evaluation_numeric_arrays_read": False},
        "missing_rule": {"zero_mass_count": 0, "rule": "zero occupancy mass is DATA_OR_IMPLEMENTATION_BLOCK; no centroid or moment imputation"},
        "f7_stability": {"finite": bool(np.isfinite(f7).all()), "maximum_absolute_value": float(np.max(np.abs(f7)))},
        "probe_models": records,
        "calibration_gains": calibration_gains,
        "candidate_order": list(CANDIDATES),
        "artifacts": {
            "weights": {"path": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path)},
            "derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)},
        },
        "forbidden_activity_audit": {
            "evaluation_numeric_arrays_read": False,
            "simulator_run": False,
            "natural_decoder_or_encoder_training": False,
            "world_model_reranker_or_controller_change": False,
            "reward_done_success_or_task_outcome_read": False,
            "init40_49_access": False,
        },
    }
    atomic_json(ROOT / protocol["outputs"]["calibration_freeze"], freeze)
    print(json.dumps({"status": freeze["status"], "f7_stability": freeze["f7_stability"], "calibration_gains": calibration_gains}, indent=2))


def bootstrap_indices(metadata: dict[str, np.ndarray], replicates: int, seed: int) -> list[np.ndarray]:
    branch, task = metadata["branch"], metadata["task"]
    rng = np.random.default_rng(seed)
    branches_by_task = {task_id: np.unique(branch[task == task_id]) for task_id in range(10)}
    rows = {int(value): np.flatnonzero(branch == value) for value in np.unique(branch)}
    draws = []
    for _ in range(replicates):
        selected = []
        for task_id in range(10):
            values = branches_by_task[task_id]
            selected.extend(rows[int(value)] for value in rng.choice(values, size=values.size, replace=True))
        draws.append(np.concatenate(selected))
    return draws


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_metric(draws: list[np.ndarray], function: Callable[[np.ndarray], float]) -> list[float]:
    return interval([float(function(draw)) for draw in draws])


def cut_improvements(parent: np.ndarray, child: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": improvement(parent, child, target), "task": [], "horizon": [], "pair": []}
    for field, values in (("task", range(10)), ("horizon", range(1, 9)), ("pair", range(3))):
        for value in values:
            index = np.flatnonzero(metadata[field] == value)
            label: Any = PAIR_NAMES[value] if field == "pair" else value
            result[field].append({field: label, "relative_rmse_improvement": improvement(parent, child, target, index)})
    return result


def derange(values: np.ndarray, metadata: dict[str, np.ndarray], seed: int) -> np.ndarray:
    source, output = np.asarray(values), np.asarray(values).copy()
    rng = np.random.default_rng(seed)
    for task in range(10):
        for horizon in range(1, 9):
            for pair in range(3):
                index = np.flatnonzero((metadata["task"] == task) & (metadata["horizon"] == horizon) & (metadata["pair"] == pair))
                ordered = index[np.argsort(metadata["branch"][index], kind="stable")]
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
    candidates = partition["candidates"]
    summaries = {name: pair_summary(candidate, partition["metadata"]) for name, candidate in candidates.items()}
    b2 = b2_candidate(partition["occupancy"], partition["area"])
    summaries["B2"] = pair_summary(b2, partition["metadata"])
    features = {"G": partition["global"]}
    features.update({name: np.concatenate((partition["global"], summary), axis=1) for name, summary in summaries.items()})
    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in features}
    metrics: dict[str, Any] = {name: {} for name in features}
    for name, feature in features.items():
        for target_name, target in partition["targets"].items():
            fit = ridge_from_arrays(f"ridge__{name}__{target_name}", weights)
            prediction = predict_ridge(feature, fit).reshape(-1)
            predictions[name][target_name] = prediction
            metrics[name][target_name] = scalar_metrics(prediction, target)

    draws = bootstrap_indices(partition["metadata"], int(protocol["bootstrap"]["replicates"]), int(protocol["bootstrap"]["seed"]))
    correlation_draws = draws[: int(protocol["bootstrap"]["correlation_replicates"])]
    increments: dict[str, Any] = {}
    shuffles: dict[str, Any] = {}
    baseline = predictions["G"]
    for candidate_index, name in enumerate((*CANDIDATES, "B2")):
        increments[name], shuffles[name] = {}, {}
        shuffled = derange(summaries[name], partition["metadata"], int(protocol["shuffle"]["seed"]) + candidate_index)
        for target_name, target in partition["targets"].items():
            cuts = cut_improvements(baseline[target_name], predictions[name][target_name], target, partition["metadata"])
            cuts["ci95"] = bootstrap_metric(draws, lambda index, n=name, t=target_name: improvement(baseline[t][index], predictions[n][t][index], partition["targets"][t][index]))
            increments[name][target_name] = cuts
            fit = ridge_from_arrays(f"ridge__{name}__{target_name}", weights)
            shuffled_prediction = predict_ridge(np.concatenate((partition["global"], shuffled), axis=1), fit).reshape(-1)
            degradation = (rmse(shuffled_prediction, target) - rmse(predictions[name][target_name], target)) / max(rmse(predictions[name][target_name], target), 1.0e-12)
            ci = bootstrap_metric(draws, lambda index, s=shuffled_prediction, n=name, t=target_name: (rmse(s, partition["targets"][t], index) - rmse(predictions[n][t], partition["targets"][t], index)) / max(rmse(predictions[n][t], partition["targets"][t], index), 1.0e-12))
            shuffles[name][target_name] = {"relative_degradation": degradation, "ci95": ci, "passes": bool(degradation > 0.0 and ci[0] > 0.0)}

    retentions = {
        name: {
            target: increments[name][target]["overall"] / max(increments["F0"][target]["overall"], 1.0e-12)
            for target in RAW_TO_RESIDUAL
        }
        for name in CANDIDATES
    }
    gates: dict[str, Any] = {}
    qualified: list[tuple[int, int, str, str]] = []
    for order, name in enumerate(CANDIDATES):
        target_gates = {}
        for target, residual in RAW_TO_RESIDUAL.items():
            raw = increments[name][target]
            residual_record = increments[name][residual]
            positive_tasks = sum(row["relative_rmse_improvement"] > 0.0 for row in raw["task"])
            positive_horizons = sum(row["relative_rmse_improvement"] > 0.0 for row in raw["horizon"])
            positive_pairs = sum(row["relative_rmse_improvement"] > 0.0 for row in raw["pair"])
            primary = bool(raw["overall"] >= protocol["gates"]["minimum_primary_increment"] and raw["ci95"][0] > 0.0 and shuffles[name][target]["passes"] and positive_tasks >= 7 and positive_horizons >= 4 and positive_pairs == 3)
            residual_pass = bool(residual_record["overall"] >= protocol["gates"]["minimum_residual_increment"] and residual_record["ci95"][0] > 0.0)
            retention_pass = bool(retentions[name][target] >= protocol["gates"]["minimum_retention_vs_F0"])
            complete = bool(name != "F0" and primary and residual_pass and retention_pass)
            target_gates[target] = {"primary": primary, "residual": residual_pass, "retention": retention_pass, "complete": complete, "positive_tasks": positive_tasks, "positive_horizons": positive_horizons, "positive_pairs": positive_pairs}
            if complete:
                qualified.append((DIMENSIONS[name], order, name, target))
        gates[name] = {"targets": target_gates, "any_complete": any(record["complete"] for record in target_gates.values())}

    def nested(parent: str, child: str) -> dict[str, Any]:
        result = {}
        for target_name, target in partition["targets"].items():
            value = cut_improvements(predictions[parent][target_name], predictions[child][target_name], target, partition["metadata"])
            value["ci95"] = bootstrap_metric(draws, lambda index, p=parent, c=child, t=target_name: improvement(predictions[p][t], predictions[c][t], partition["targets"][t], index))
            result[target_name] = value
        return result

    nested_analysis = {
        "F1_to_F3": nested("F1", "F3"),
        "F3_to_F4": nested("F3", "F4"),
        "F4_to_F5": nested("F4", "F5"),
        "area_to_soft_map_plus_area": nested("F1", "F0"),
        "area_to_binary_plus_area": nested("F1", "B2"),
        "soft_map_to_soft_map_plus_area": nested("F2", "F0"),
    }

    separation, correlations = {}, {}
    for name in CANDIDATES:
        candidate_distance, temporal_distance = representation_distances(candidates[name])
        separation[name] = {
            "candidate_mean": float(candidate_distance.mean()),
            "temporal_mean": float(temporal_distance.mean()),
            "candidate_temporal_ratio": float(candidate_distance.mean() / max(temporal_distance.mean(), 1.0e-12)),
            "pair_means": [float(candidate_distance[partition["metadata"]["pair"] == pair].mean()) for pair in range(3)],
        }
        correlations[name] = {target: correlation_record(candidate_distance, value, correlation_draws) for target, value in partition["targets"].items()}
        correlations[name]["ee_image_displacement"] = correlation_record(candidate_distance, np.asarray(load_npz(ROOT / protocol["partitions"]["development_evaluation"]["target_design_derived"])["target__ee_image_displacement"], dtype=np.float64), correlation_draws)

    labels: list[str] = []
    selected_target = None
    selected_dimension = None
    selected_gate_target = None
    if qualified:
        selected_dimension, _, selected_target, selected_gate_target = min(qualified)
        labels.append("MINIMAL_CONTINUOUS_FOOTPRINT_VALID")
        if selected_target == "F1":
            labels.insert(0, "VISIBLE_AREA_DOMINANT_TARGET")
        elif selected_target == "F3":
            labels.insert(0, "AREA_CENTROID_TARGET_VALID")
        elif selected_target in ("F4", "F5", "F6", "F7"):
            labels.insert(0, "CONTINUOUS_SHAPE_EXTENT_REQUIRED")
        authorization = "GO_MINIMAL_SPATIAL_EXTRACTION_AUDIT"
    else:
        labels.append("FULL_SOFT_FOOTPRINT_STRUCTURE_REQUIRED")
        authorization = "GO_SOFT_SPATIAL_EXTRACTION_AUDIT"
    if gates["F3"]["targets"]["object_mask_difference"]["complete"]:
        labels.append("SHAPE_DETAIL_NOT_REQUIRED")
    conditional_threshold = float(protocol["gates"]["minimum_conditional_increment"])
    binary_conditional = nested_analysis["area_to_binary_plus_area"]["object_mask_difference"]
    if binary_conditional["overall"] < conditional_threshold or binary_conditional["ci95"][0] <= 0.0:
        labels.append("BINARY_COMPONENT_REDUNDANT_GIVEN_AREA")
    soft_conditional = nested_analysis["area_to_soft_map_plus_area"]["object_mask_difference"]
    if soft_conditional["overall"] >= conditional_threshold and soft_conditional["ci95"][0] > 0.0:
        labels.append("SOFT_OCCUPANCY_ADDS_VALUE_BEYOND_AREA")
    area_conditional = nested_analysis["soft_map_to_soft_map_plus_area"]
    if any(area_conditional[target]["overall"] >= conditional_threshold and area_conditional[target]["ci95"][0] > 0.0 for target in RAW_TO_RESIDUAL):
        labels.append("VISIBLE_AREA_ADDS_VALUE_BEYOND_SOFT_MAP")
    if any(gates["F0"]["targets"][target]["residual"] for target in RAW_TO_RESIDUAL):
        labels.append("OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED")
    labels = list(dict.fromkeys(labels))

    derived_path = ROOT / protocol["outputs"]["evaluation_derived"]
    atomic_npz(
        derived_path,
        {
            **{f"metadata__{name}": value for name, value in partition["metadata"].items()},
            **{f"target__{name}": value.astype(np.float32) for name, value in partition["targets"].items()},
            **{f"summary__{name}": value.astype(np.float32) for name, value in summaries.items()},
            **{f"state__{name}": candidate.state.astype(np.float32) for name, candidate in candidates.items()},
            **{f"prediction__{name}__{target}": value.astype(np.float32) for name, by_target in predictions.items() for target, value in by_target.items()},
        },
    )
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "date": protocol["date"],
        "status": "COMPLETE",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "calibration_freeze_sha256": sha256(freeze_path),
        "data_boundary": {"calibration_init_states": [31, 32, 33, 34], "development_evaluation_init_states": [36, 37, 38, 39], "development_evaluation_is_blind_confirmation": False, "historical_reference_only": [30, 35], "init40_49_accessed": False, "branch_points": 40, "pair_rows": 960},
        "candidate_definitions": protocol["candidates"],
        "missing_rule": freeze["missing_rule"],
        "probe_metrics": metrics,
        "incremental": increments,
        "shuffle_controls": shuffles,
        "retention_vs_F0": retentions,
        "nested_analysis": nested_analysis,
        "candidate_separation": separation,
        "correlations": correlations,
        "gates": gates,
        "dimensional_efficiency": {name: {target: increments[name][target]["overall"] / DIMENSIONS[name] for target in RAW_TO_RESIDUAL} for name in CANDIDATES},
        "decision": {
            "mechanism_labels": labels,
            "selected_minimal_oracle_target": selected_target,
            "selected_dimension": selected_dimension,
            "selection_gate_target": selected_gate_target,
            "next_stage_authorization": authorization,
            "spatial_world_model_prototype_authorization": "BLOCK_SPATIAL_WM_PROTOTYPE",
            "alternative_spatial_encoder_audit_authorized": False,
            "controller_authorization": False,
        },
        "candidate_headroom_reference": protocol["candidate_headroom_reference"],
        "artifacts": {"evaluation_derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)}},
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {"simulator_run": False, "natural_decoder_or_encoder_training": False, "world_model_training": False, "candidate_or_reranker_change": False, "controller_or_success_run": False, "reward_done_success_or_task_outcome_read": False, "init40_49_access": False, "seed23_47_run": False, "upstream_overwrite": False},
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    atomic_json(ROOT / protocol["outputs"]["diagnosis"], result)
    print(json.dumps(result["decision"], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    protocol, commit = load_protocol(args.phase)
    if args.phase == "calibrate":
        calibrate(protocol, commit)
    else:
        evaluate(protocol, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
