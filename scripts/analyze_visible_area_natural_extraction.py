#!/usr/bin/env python3
"""Calibrate and evaluate natural-RGB extraction of the frozen F1 area target."""

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

import h5py
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
from src.probes.spatial_relevance import PAIR_NAMES, safe_correlation, safe_spearman
from src.probes.visible_area_extraction import (
    AreaHead,
    CONDITIONS,
    MODEL_FORMS,
    VISUAL_FORMS,
    candidate_ordering,
    fitted_from_arrays,
    fitted_to_arrays,
    image_features,
    pair_rows,
    parameter_count,
    predict_head,
    scalar_pair_summary,
    standardizer,
    train_head,
)


PROTOCOL_VERSION = "relevant-object-visible-area-natural-extraction-v1"
PROTOCOL_PATH = ROOT / "configs/relevant_object_visible_area_natural_extraction_protocol.json"
TARGETS = ("visible_area_difference", "visible_area_residual", "object_mask_difference", "object_mask_residual")


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


def load_protocol(phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION or protocol["status"] != "PREREGISTERED_BEFORE_CALIBRATION_OR_EVALUATION_RESULTS":
        raise ValueError("protocol not preregistered")
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
        for key in ("calibration_weights", "calibration_derived", "calibration_freeze", "evaluation_derived", "diagnosis"):
            if (ROOT / outputs[key]).exists():
                raise FileExistsError(outputs[key])
    else:
        freeze_path = ROOT / outputs["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("calibration freeze must be committed")
        freeze = json.loads(freeze_path.read_text())
        if freeze["status"] != "FROZEN_BEFORE_INIT36_39_VISIBLE_AREA_EVALUATION":
            raise RuntimeError("calibration freeze status changed")
        if freeze["protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise RuntimeError("freeze references another protocol")
        for record in freeze["artifacts"].values():
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"calibration artifact changed: {record['path']}")
    return protocol, commit


def load_partition(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    record = protocol["partitions"][name]
    visual = load_npz(ROOT / record["natural_features"])
    masks = load_npz(ROOT / record["per_object_masks"])
    downstream = load_npz(ROOT / record["continuous_footprint_derived"])
    global_features = np.asarray(visual["global_variant_features"], dtype=np.float32)[..., 0, :]
    patch_features = np.asarray(visual["patch_features"], dtype=np.float32)
    task_id = np.asarray(masks["task_id"], dtype=np.int64)
    init_state = np.asarray(masks["init_state_index"], dtype=np.int64)
    object_masks = np.asarray(masks["per_object_masks"], dtype=bool)
    area = object_masks.any(axis=3).mean(axis=(-2, -1)).astype(np.float64)
    with h5py.File(ROOT / protocol["frozen_upstream"]["language_feature_store"], "r") as store:
        language = store["task_metadata/language_features"][:].astype(np.float32)
    features = image_features(global_features, patch_features, language, task_id)
    expected = set(record["init_state_indices"])
    if set(np.unique(init_state).tolist()) != expected:
        raise ValueError("partition boundary changed")
    metadata = {key: np.asarray(downstream[f"metadata__{key}"]) for key in ("branch", "task", "init_state", "horizon", "pair")}
    if not np.array_equal(metadata["task"].reshape(40, 3, 8)[:, 0, 0], task_id):
        raise ValueError("task identities do not align")
    real_delta, pair_meta = pair_rows(area)
    if not np.array_equal(pair_meta["branch"], metadata["branch"]) or not np.array_equal(pair_meta["horizon"], metadata["horizon"]):
        raise ValueError("pair ordering changed")
    return {
        "area": area,
        "features": features,
        "task_id": task_id,
        "init_state": init_state,
        "image_task": np.repeat(task_id, 24),
        "image_init": np.repeat(init_state, 24),
        "metadata": metadata,
        "real_delta": real_delta,
        "global": np.asarray(downstream["representation__global"], dtype=np.float64),
        "targets": {target: np.asarray(downstream[f"target__{target}"], dtype=np.float64) for target in TARGETS},
    }


def fold_fit(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    form: str,
    maximum_epochs: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, int, float]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean, scale = standardizer(x[train])
    target_mean = float(np.mean(y[train]))
    target_scale = max(float(np.std(y[train])), 1.0e-8)
    train_x = torch.from_numpy(((x[train] - mean) / scale).astype(np.float32))
    valid_x = torch.from_numpy(((x[valid] - mean) / scale).astype(np.float32))
    train_y = torch.from_numpy(((y[train] - target_mean) / target_scale).astype(np.float32))
    valid_y = y[valid]
    model = AreaHead(x.shape[1], form)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_mae, best_epoch, best_prediction = float("inf"), 0, None
    for epoch in range(1, maximum_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(model(train_x) - train_y))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(valid_x).cpu().numpy().astype(np.float64)
            prediction = np.clip(prediction * target_scale + target_mean, 0.0, 1.0)
        model.train()
        mae = float(np.mean(np.abs(prediction - valid_y)))
        if mae < best_mae - 1.0e-12:
            best_mae, best_epoch, best_prediction = mae, epoch, prediction.copy()
    assert best_prediction is not None
    return best_prediction, best_epoch, best_mae


def round_half_up_median(values: list[int]) -> int:
    return int(math.floor(float(np.median(values)) + 0.5))


def oof_area_predictions(protocol: dict[str, Any], partition: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    y = partition["area"].reshape(-1)
    groups = partition["image_init"]
    settings = protocol["models"]
    predictions, records = {}, {}
    for base, x in partition["features"].items():
        for form in MODEL_FORMS:
            name = f"{base}__{form}"
            output = np.empty_like(y)
            epochs, folds = [], []
            for fold_index, held in enumerate(sorted(np.unique(groups).tolist())):
                valid = groups == held
                pred, epoch, mae = fold_fit(
                    x, y, ~valid, valid, form, int(settings["maximum_epochs"]),
                    int(settings["seed"]) + 101 * fold_index + len(records),
                    float(settings["learning_rate"]), float(settings["weight_decay"]),
                )
                output[valid] = pred
                epochs.append(epoch)
                folds.append({"held_out_init": int(held), "best_epoch": epoch, "mae": mae})
            predictions[name] = output
            records[name] = {
                "input_dimensions": int(x.shape[1]),
                "parameters": parameter_count(x.shape[1], form),
                "folds": folds,
                "frozen_epoch": round_half_up_median(epochs),
                "oof_mae": float(np.mean(np.abs(output - y))),
                "oof_rmse": float(np.sqrt(np.mean(np.square(output - y)))),
            }
    return predictions, records


def oof_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    output = np.empty_like(y)
    for held in sorted(np.unique(groups).tolist()):
        valid = groups == held
        fitted = fit_ridge(x[~valid], y[~valid], alpha)
        output[valid] = predict_ridge(x[valid], fitted).reshape(-1)
    return output


def rmse(prediction: np.ndarray, target: np.ndarray, index: np.ndarray | None = None) -> float:
    pred, true = np.asarray(prediction), np.asarray(target)
    if index is not None:
        pred, true = pred[index], true[index]
    return float(np.sqrt(np.mean(np.square(pred - true))))


def improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray, index: np.ndarray | None = None) -> float:
    base = rmse(parent, target, index)
    return (base - rmse(child, target, index)) / max(base, 1.0e-12)


def fit_downstream_calibration(
    protocol: dict[str, Any], partition: dict[str, Any], area_predictions: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    grid = [float(value) for value in protocol["downstream"]["ridge_grid"]]
    groups = partition["metadata"]["init_state"]
    weights, records, predictions = {}, {}, {}
    feature_sets = {"G": partition["global"]}
    for name, values in area_predictions.items():
        feature_sets[name] = np.concatenate((partition["global"], scalar_pair_summary(values.reshape(40, 3, 8))), axis=1)
    for name, feature in feature_sets.items():
        records[name], predictions[name] = {}, {}
        for target_name, target in partition["targets"].items():
            alpha, cv = select_ridge_alpha(feature, target, groups, grid)
            fitted = fit_ridge(feature, target, alpha)
            weights.update(ridge_to_arrays(f"ridge__{name}__{target_name}", fitted))
            predictions[name][target_name] = oof_ridge(feature, target, groups, alpha)
            records[name][target_name] = {"alpha": alpha, "dimensions": int(feature.shape[1]), "cv": cv}
    return weights, records, predictions


def calibrate(protocol: dict[str, Any], protocol_commit: str) -> None:
    partition = load_partition(protocol, "calibration")
    area_predictions, area_records = oof_area_predictions(protocol, partition)
    ridge_weights, ridge_records, downstream_predictions = fit_downstream_calibration(protocol, partition, area_predictions)
    target = partition["area"].reshape(-1)
    constant = float(np.mean(target))
    real_delta = partition["real_delta"]
    candidate_selection = {}
    for condition in CONDITIONS:
        rows = []
        for visual in VISUAL_FORMS:
            for form in MODEL_FORMS:
                name = f"{condition}__{visual}__{form}"
                pred_delta, _ = pair_rows(area_predictions[name].reshape(40, 3, 8))
                area_gain = improvement(
                    downstream_predictions["G"]["visible_area_difference"],
                    downstream_predictions[name]["visible_area_difference"],
                    partition["targets"]["visible_area_difference"],
                )
                row = {
                    "name": name,
                    "area_mae": float(np.mean(np.abs(area_predictions[name] - target))),
                    "pair_difference_rmse": rmse(pred_delta, real_delta),
                    "downstream_area_gain": area_gain,
                    "parameters": area_records[name]["parameters"],
                }
                rows.append(row)
            
        rows.sort(key=lambda item: (item["area_mae"], item["pair_difference_rmse"], -item["downstream_area_gain"], item["parameters"], item["name"]))
        candidate_selection[condition] = {"selected": rows[0]["name"], "ranking": rows}
    final_weights = dict(ridge_weights)
    for index, (name, prediction) in enumerate(area_predictions.items()):
        base, form = name.rsplit("__", 1)
        fitted = train_head(
            partition["features"][base], target, form, area_records[name]["frozen_epoch"],
            int(protocol["models"]["seed"]) + 1000 + index,
            float(protocol["models"]["learning_rate"]), float(protocol["models"]["weight_decay"]),
        )
        final_weights.update(fitted_to_arrays(f"area__{name}", fitted))
    positive = np.abs(real_delta[np.abs(real_delta) > 0.0])
    tie_threshold = float(np.quantile(positive, float(protocol["candidate_differential"]["tie_quantile"])))
    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    derived_path = ROOT / protocol["outputs"]["calibration_derived"]
    atomic_npz(weights_path, final_weights)
    atomic_npz(
        derived_path,
        {
            "target__area": target.astype(np.float32),
            "target__pair_delta": real_delta.astype(np.float32),
            **{f"prediction__{name}": value.astype(np.float32) for name, value in area_predictions.items()},
            **{f"downstream__{name}__{target_name}": value.astype(np.float32) for name, record in downstream_predictions.items() for target_name, value in record.items()},
        },
    )
    freeze = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_INIT36_39_VISIBLE_AREA_EVALUATION",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "partition": {"init_state_indices": [31, 32, 33, 34], "branch_points": 40, "image_states": 960, "pair_rows": 960, "evaluation_numeric_arrays_read": False},
        "constant_area": constant,
        "tie_threshold": tie_threshold,
        "area_models": area_records,
        "downstream_models": ridge_records,
        "condition_selection": candidate_selection,
        "artifacts": {
            "weights": {"path": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path)},
            "derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)},
        },
        "forbidden_activity_audit": {
            "evaluation_numeric_arrays_read": False,
            "simulator_run": False,
            "mask_or_state_used_as_extractor_input": False,
            "encoder_changed_or_selected": False,
            "world_model_reranker_or_controller_run": False,
            "reward_done_success_or_task_outcome_read": False,
            "init40_49_or_seed23_47_access": False,
        },
    }
    atomic_json(ROOT / protocol["outputs"]["calibration_freeze"], freeze)
    print(json.dumps({"status": freeze["status"], "condition_selection": candidate_selection, "tie_threshold": tie_threshold}, indent=2))


def bootstrap_draws(task_by_branch: np.ndarray, replicates: int, seed: int, rows_per_branch: int = 24) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    branches = np.arange(40)
    by_task = {task: branches[task_by_branch == task] for task in range(10)}
    row_lookup = {branch: np.arange(branch * rows_per_branch, (branch + 1) * rows_per_branch) for branch in branches}
    output = []
    for _ in range(replicates):
        chosen = []
        for task in range(10):
            values = by_task[task]
            chosen.extend(row_lookup[int(branch)] for branch in rng.choice(values, size=len(values), replace=True))
        output.append(np.concatenate(chosen))
    return output


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap(draws: list[np.ndarray], fn: Callable[[np.ndarray], float]) -> list[float]:
    return interval([float(fn(index)) for index in draws])


def derange_summary(summary: np.ndarray, metadata: dict[str, np.ndarray], seed: int) -> np.ndarray:
    source, output = np.asarray(summary), np.asarray(summary).copy()
    rng = np.random.default_rng(seed)
    for task in range(10):
        for horizon in range(1, 9):
            for pair in range(3):
                index = np.flatnonzero((metadata["task"] == task) & (metadata["horizon"] == horizon) & (metadata["pair"] == pair))
                shift = int(rng.integers(1, len(index)))
                output[index] = source[np.roll(index, shift)]
    return output


def cut_improvements(parent: np.ndarray, child: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {"task": [], "horizon": [], "pair": []}
    for field, values in (("task", range(10)), ("horizon", range(1, 9)), ("pair", range(3))):
        for value in values:
            index = np.flatnonzero(metadata[field] == value)
            result[field].append({
                field: PAIR_NAMES[value] if field == "pair" else value,
                "relative_rmse_improvement": improvement(parent, child, target, index),
            })
    return result


def evaluation_metrics(real: np.ndarray, predicted: np.ndarray, constant: float, tie_threshold: float) -> dict[str, Any]:
    truth, estimate = np.asarray(real).reshape(-1), np.asarray(predicted).reshape(-1)
    baseline = np.full_like(truth, constant)
    delta_real, _ = pair_rows(np.asarray(real))
    delta_pred, _ = pair_rows(np.asarray(predicted))
    keep = np.abs(delta_real) > tie_threshold
    slope, intercept = np.polyfit(truth, estimate, 1)
    temporal_real, temporal_pred = [], []
    for branch in range(40):
        for candidate in range(3):
            temporal_real.extend(np.abs(np.diff(real[branch, candidate])))
            temporal_pred.extend(np.abs(np.diff(predicted[branch, candidate])))
    candidate_real = np.mean(np.abs(delta_real))
    candidate_pred = np.mean(np.abs(delta_pred))
    real_ratio = candidate_real / max(float(np.mean(temporal_real)), 1.0e-12)
    predicted_ratio = candidate_pred / max(float(np.mean(temporal_pred)), 1.0e-12)
    return {
        "absolute": {
            **scalar_metrics(estimate, truth),
            "constant_mae": float(np.mean(np.abs(baseline - truth))),
            "mae_improvement_vs_constant": (float(np.mean(np.abs(baseline - truth))) - float(np.mean(np.abs(estimate - truth)))) / max(float(np.mean(np.abs(baseline - truth))), 1.0e-12),
            "prediction_mean": float(np.mean(estimate)), "real_mean": float(np.mean(truth)),
            "prediction_std": float(np.std(estimate)), "real_std": float(np.std(truth)),
            "std_ratio": float(np.std(estimate) / max(np.std(truth), 1.0e-12)),
            "calibration_slope": float(slope), "calibration_intercept": float(intercept),
        },
        "differential": {
            "rmse": rmse(delta_pred, delta_real),
            "mae": float(np.mean(np.abs(delta_pred - delta_real))),
            "pearson": safe_correlation(delta_pred, delta_real),
            "spearman": safe_spearman(delta_pred, delta_real),
            "direction_accuracy": float(np.mean(np.sign(delta_pred[keep]) == np.sign(delta_real[keep]))),
            "direction_rows": int(np.sum(keep)),
            "tie_threshold": tie_threshold,
            "prediction_std": float(np.std(delta_pred)), "real_std": float(np.std(delta_real)),
            "std_ratio": float(np.std(delta_pred) / max(np.std(delta_real), 1.0e-12)),
        },
        "ordering": candidate_ordering(real, predicted),
        "scale_control": {
            "real_candidate_temporal_ratio": real_ratio,
            "predicted_candidate_temporal_ratio": predicted_ratio,
            "ratio_of_ratios": predicted_ratio / max(real_ratio, 1.0e-12),
        },
    }


def evaluate(protocol: dict[str, Any]) -> None:
    partition = load_partition(protocol, "development_evaluation")
    freeze = json.loads((ROOT / protocol["outputs"]["calibration_freeze"]).read_text())
    weights = load_npz(ROOT / protocol["outputs"]["calibration_weights"])
    real = partition["area"]
    predictions = {}
    for base, x in partition["features"].items():
        for form in MODEL_FORMS:
            name = f"{base}__{form}"
            predictions[name] = predict_head(x, fitted_from_arrays(f"area__{name}", weights)).reshape(40, 3, 8)
    selected = {condition: freeze["condition_selection"][condition]["selected"] for condition in CONDITIONS}
    image_draws = bootstrap_draws(partition["task_id"], int(protocol["bootstrap"]["replicates"]), int(protocol["bootstrap"]["seed"]))
    pair_draws = bootstrap_draws(partition["task_id"], int(protocol["bootstrap"]["replicates"]), int(protocol["bootstrap"]["seed"]) + 1)
    model_metrics = {}
    truth_flat = real.reshape(-1)
    baseline_abs = np.abs(truth_flat - float(freeze["constant_area"]))
    for name, state in predictions.items():
        record = evaluation_metrics(real, state, float(freeze["constant_area"]), float(freeze["tie_threshold"]))
        pred_flat = state.reshape(-1)
        record["absolute"]["mae_improvement_ci"] = bootstrap(
            image_draws,
            lambda index, p=pred_flat: (float(np.mean(baseline_abs[index])) - float(np.mean(np.abs(p[index] - truth_flat[index])))) / max(float(np.mean(baseline_abs[index])), 1.0e-12),
        )
        model_metrics[name] = record
    downstream = {}
    metadata = partition["metadata"]
    baseline_predictions = {}
    for target_name, target in partition["targets"].items():
        baseline_predictions[target_name] = predict_ridge(partition["global"], ridge_from_arrays("ridge__G__" + target_name, weights)).reshape(-1)
    for condition, name in selected.items():
        summary = scalar_pair_summary(predictions[name])
        feature = np.concatenate((partition["global"], summary), axis=1)
        shuffled_feature = np.concatenate((partition["global"], derange_summary(summary, metadata, int(protocol["shuffle"]["seed"])),), axis=1)
        by_target = {}
        for target_name, target in partition["targets"].items():
            fitted = ridge_from_arrays(f"ridge__{name}__{target_name}", weights)
            child = predict_ridge(feature, fitted).reshape(-1)
            shuffled = predict_ridge(shuffled_feature, fitted).reshape(-1)
            parent = baseline_predictions[target_name]
            gain = improvement(parent, child, target)
            shuffle_degradation = (rmse(shuffled, target) - rmse(child, target)) / max(rmse(child, target), 1.0e-12)
            by_target[target_name] = {
                "global_rmse": rmse(parent, target), "global_plus_predicted_area_rmse": rmse(child, target),
                "relative_rmse_improvement": gain,
                "improvement_ci": bootstrap(pair_draws, lambda index, p=parent, c=child, t=target: improvement(p, c, t, index)),
                "shuffle_degradation": shuffle_degradation,
                "shuffle_ci": bootstrap(pair_draws, lambda index, s=shuffled, c=child, t=target: (rmse(s, t, index) - rmse(c, t, index)) / max(rmse(c, t, index), 1.0e-12)),
                "cuts": cut_improvements(parent, child, target, metadata),
                "prediction": child,
            }
        downstream[condition] = {"extractor": name, "targets": by_target}
    gates = {}
    thresholds = protocol["gates"]
    for condition, name in selected.items():
        metric = model_metrics[name]
        area = downstream[condition]["targets"]["visible_area_difference"]
        residual = downstream[condition]["targets"]["visible_area_residual"]
        cuts = area["cuts"]
        check = {
            "absolute_recovery": metric["absolute"]["mae_improvement_vs_constant"] >= thresholds["minimum_mae_improvement"] and metric["absolute"]["mae_improvement_ci"][0] > 0.0,
            "area_increment": area["relative_rmse_improvement"] >= thresholds["minimum_area_increment"] and area["improvement_ci"][0] > 0.0,
            "area_shuffle": area["shuffle_ci"][0] > 0.0,
            "area_residual": residual["relative_rmse_improvement"] >= thresholds["minimum_residual_increment"] and residual["improvement_ci"][0] > 0.0,
            "task_robustness": sum(row["relative_rmse_improvement"] > 0.0 for row in cuts["task"]) >= thresholds["minimum_positive_tasks"],
            "horizon_robustness": sum(row["relative_rmse_improvement"] > 0.0 for row in cuts["horizon"]) >= thresholds["minimum_positive_horizons"],
            "pair_robustness": sum(row["relative_rmse_improvement"] > 0.0 for row in cuts["pair"]) == thresholds["minimum_positive_pairs"],
            "image_dynamic_range": metric["absolute"]["std_ratio"] >= thresholds["minimum_dynamic_range_ratio"],
            "pair_dynamic_range": metric["differential"]["std_ratio"] >= thresholds["minimum_dynamic_range_ratio"],
        }
        check["complete"] = all(check.values())
        gates[condition] = check
    labels = []
    if any(not gates[c]["image_dynamic_range"] or not gates[c]["pair_dynamic_range"] for c in CONDITIONS):
        labels.append("AREA_DYNAMIC_RANGE_COLLAPSE")
    if gates["E0_visual"]["complete"]:
        labels.extend(("NATURAL_VISIBLE_AREA_PROXY_VALID", "VISUAL_ONLY_AREA_EXTRACTION_SUFFICIENT", "SPATIAL_TARGET_EXTRACTION_RESOLVED"))
        chosen_condition = "E0_visual"
        authorization = "GO_MINIMAL_SPATIAL_WM_PROTOTYPE"
    elif gates["E1_visual_language"]["complete"]:
        labels.extend(("NATURAL_VISIBLE_AREA_PROXY_VALID", "TASK_LANGUAGE_NEEDED_FOR_AREA_EXTRACTION", "SPATIAL_TARGET_EXTRACTION_RESOLVED"))
        chosen_condition = "E1_visual_language"
        authorization = "GO_MINIMAL_SPATIAL_WM_PROTOTYPE"
    else:
        chosen_condition = None
        authorization = "BLOCK_SPATIAL_WM_PROTOTYPE"
        absolute_pass = [gates[c]["absolute_recovery"] for c in CONDITIONS]
        if any(absolute_pass):
            labels.append("AREA_REGRESSION_ACCURATE_BUT_DECISION_SIGNAL_INSUFFICIENT")
        else:
            labels.append("NATURAL_VISIBLE_AREA_EXTRACTION_FAILED")
        labels.append("NATURAL_SPATIAL_EXTRACTION_BLOCK")
    compact_downstream = {}
    arrays = {
        "target__area": truth_flat.astype(np.float32),
        "target__pair_delta": partition["real_delta"].astype(np.float32),
    }
    for name, state in predictions.items():
        arrays[f"prediction__{name}"] = state.reshape(-1).astype(np.float32)
    for condition, record in downstream.items():
        compact_downstream[condition] = {"extractor": record["extractor"], "targets": {}}
        for target_name, target_record in record["targets"].items():
            arrays[f"downstream__{condition}__{target_name}"] = target_record.pop("prediction").astype(np.float32)
            compact_downstream[condition]["targets"][target_name] = target_record
    diagnosis = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
        "partition": {"init_state_indices": [36, 37, 38, 39], "branch_points": 40, "image_states": 960, "pair_rows": 960, "blind_confirmation": False},
        "selected_by_condition": selected,
        "selected_condition": chosen_condition,
        "selected_extractor": None if chosen_condition is None else selected[chosen_condition],
        "model_metrics": model_metrics,
        "downstream": compact_downstream,
        "gates": gates,
        "final_mechanism_labels": labels,
        "next_stage_authorization": authorization,
        "persistent_controller_authorization": False,
        "alternative_encoder_audit_authorization": False,
        "world_model_scope_if_go": "seed11 single-model offline action-conditioned future-visible-area prototype only; no reranker or controller",
        "forbidden_activity_audit": {
            "simulator_run": False, "natural_rgb_modified_or_masked": False,
            "mask_depth_state_proprio_pose_contact_action_candidate_id_used_as_extractor_input": False,
            "new_encoder_or_encoder_finetuning": False, "world_model_trained": False,
            "reranker_or_controller_run": False, "reward_done_success_or_task_outcome_read": False,
            "init40_49_or_seed23_47_access": False,
        },
    }
    derived_path = ROOT / protocol["outputs"]["evaluation_derived"]
    atomic_npz(derived_path, arrays)
    diagnosis["artifacts"] = {
        "protocol": {"path": str(PROTOCOL_PATH.relative_to(ROOT)), "sha256": sha256(PROTOCOL_PATH)},
        "calibration_freeze": {"path": protocol["outputs"]["calibration_freeze"], "sha256": sha256(ROOT / protocol["outputs"]["calibration_freeze"])},
        "calibration_weights": {"path": protocol["outputs"]["calibration_weights"], "sha256": sha256(ROOT / protocol["outputs"]["calibration_weights"])},
        "calibration_derived": {"path": protocol["outputs"]["calibration_derived"], "sha256": sha256(ROOT / protocol["outputs"]["calibration_derived"])},
        "evaluation_derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)},
    }
    if not finite_tree(diagnosis):
        raise FloatingPointError("nonfinite diagnosis")
    atomic_json(ROOT / protocol["outputs"]["diagnosis"], diagnosis)
    print(json.dumps({"labels": labels, "authorization": authorization, "selected": diagnosis["selected_extractor"], "gates": gates}, indent=2))


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    args = parser.parse_args()
    protocol, commit = load_protocol(args.phase)
    if args.phase == "calibrate":
        calibrate(protocol, commit)
    else:
        evaluate(protocol)


if __name__ == "__main__":
    main()
