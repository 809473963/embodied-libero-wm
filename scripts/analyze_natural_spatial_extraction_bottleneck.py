#!/usr/bin/env python3
"""Calibrate and evaluate the preregistered natural extraction bottleneck audit."""

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
from src.probes.natural_spatial_extraction import (
    CAPACITIES,
    DESIGNS,
    PatchDecoder,
    area_metrics,
    binary_metrics,
    decoder_from_arrays,
    decoder_to_arrays,
    fit_decoder,
    ordinal_metrics,
    predict_decoder,
)
from src.probes.spatial_relevance import PAIR_NAMES, safe_correlation, safe_spearman
from src.probes.spatial_target_design import (
    footprint_from_object_masks,
    metadata_from_derived,
    spatial_pair_summary,
)


PROTOCOL_VERSION = "natural-spatial-extraction-bottleneck-v1"
PROTOCOL_PATH = ROOT / "configs/natural_spatial_extraction_bottleneck_protocol.json"
RAW_TARGETS = ("object_mask_difference", "visible_area_difference", "ee_image_displacement")
RESIDUAL_TARGETS = ("object_mask_residual", "visible_area_residual")
ALL_TARGETS = RAW_TARGETS + RESIDUAL_TARGETS


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
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected extraction protocol")
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
        if freeze["status"] != "FROZEN_BEFORE_INIT36_39_NATURAL_EXTRACTION_EVALUATION":
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
    upstream = load_npz(ROOT / record["target_design_derived"])
    expected = set(record["init_state_indices"])
    metadata = metadata_from_derived(upstream)
    if set(np.unique(metadata["init_state"]).tolist()) != expected:
        raise ValueError("partition boundary changed")
    if set(np.unique(masks["init_state_index"]).tolist()) != expected:
        raise ValueError("mask partition boundary changed")
    patch = np.asarray(features["patch_features"], dtype=np.float64)
    if patch.shape != (40, 3, 8, 49, 512):
        raise ValueError("patch shape changed")
    if np.asarray(masks["per_object_masks"]).shape != (40, 3, 8, 2, 128, 128):
        raise ValueError("object-mask shape changed")
    global_summary = np.asarray(upstream["representation__global"], dtype=np.float64)
    return {
        "patch": patch,
        "masks": masks,
        "upstream": upstream,
        "metadata": metadata,
        "global": global_summary,
    }


def flatten_images(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    return source.reshape((40 * 3 * 8,) + source.shape[3:])


def image_init_states(partition: dict[str, Any]) -> np.ndarray:
    return np.repeat(np.asarray(partition["masks"]["init_state_index"]), 3 * 8)


def teacher_specifications(
    partition: dict[str, Any], ordinal_edge: float | None = None, area_edges: np.ndarray | None = None
) -> tuple[dict[str, dict[str, np.ndarray | str | int | None]], float, np.ndarray]:
    soft, area = footprint_from_object_masks(partition["masks"]["per_object_masks"], 7)
    binary = soft > 0.0
    if ordinal_edge is None:
        ordinal_edge = float(np.median(soft[soft > 0.0]))
    if area_edges is None:
        area_edges = np.quantile(area.reshape(-1), [1.0 / 3.0, 2.0 / 3.0]).astype(np.float64)
    ordinal = np.zeros_like(soft, dtype=np.int64)
    ordinal[soft > 0.0] = 1
    ordinal[soft > float(ordinal_edge)] = 2
    coarse_area = np.digitize(area, np.asarray(area_edges), right=False).astype(np.int64)
    designs: dict[str, dict[str, np.ndarray | str | int | None]] = {
        "B0-soft-area": {"state": soft, "area": area, "occupancy_target": soft, "area_target": area, "occupancy_classes": 0, "area_kind": "continuous"},
        "B1-binary": {"state": binary.astype(np.float64), "area": binary.mean(axis=(-2, -1)), "occupancy_target": binary.astype(np.float64), "area_target": None, "occupancy_classes": 2, "area_kind": "none"},
        "B2-binary-area": {"state": binary.astype(np.float64), "area": area, "occupancy_target": binary.astype(np.float64), "area_target": area, "occupancy_classes": 2, "area_kind": "continuous"},
        "B3-ordinal": {"state": ordinal.astype(np.float64) / 2.0, "area": (ordinal.astype(np.float64) / 2.0).mean(axis=(-2, -1)), "occupancy_target": ordinal, "area_target": None, "occupancy_classes": 3, "area_kind": "none"},
        "B4-binary-coarse-area": {"state": binary.astype(np.float64), "area": coarse_area.astype(np.float64) / 2.0, "occupancy_target": binary.astype(np.float64), "area_target": coarse_area, "occupancy_classes": 2, "area_kind": "coarse"},
    }
    return designs, float(ordinal_edge), np.asarray(area_edges, dtype=np.float64)


def construct_targets(partition: dict[str, Any], weights: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    upstream = partition["upstream"]
    targets = {name: np.asarray(upstream[f"target__{name}"], dtype=np.float64) for name in RAW_TARGETS}
    if weights is not None:
        ee = targets["ee_image_displacement"][:, None]
        for raw, residual in (("object_mask_difference", "object_mask_residual"), ("visible_area_difference", "visible_area_residual")):
            prediction = predict_ridge(ee, ridge_from_arrays(f"ee_control__{raw}", weights)).reshape(-1)
            targets[residual] = targets[raw] - prediction
    return targets


def oof_prediction(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    prediction = np.empty_like(np.asarray(y, dtype=np.float64).reshape(-1))
    for held_out in sorted(np.unique(groups).tolist()):
        train = groups != held_out
        fit = fit_ridge(np.asarray(x)[train], np.asarray(y)[train], alpha)
        prediction[~train] = predict_ridge(np.asarray(x)[~train], fit).reshape(-1)
    return prediction


def cv_probe(feature: np.ndarray, target: np.ndarray, groups: np.ndarray, grid: list[float]) -> tuple[float, np.ndarray, dict[str, float]]:
    alpha, cv = select_ridge_alpha(feature, target, groups, grid)
    prediction = oof_prediction(feature, target, groups, alpha)
    selected = next(row for row in cv if float(row["alpha"]) == float(alpha))
    return alpha, prediction, {"alpha": alpha, "cv_rmse": float(selected["mean_fold_rmse"]), "cv": cv}


def rmse(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if indices is not None:
        prediction, target = prediction[indices], target[indices]
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray, indices: np.ndarray | None = None) -> float:
    parent_error = rmse(parent, target, indices)
    return (parent_error - rmse(child, target, indices)) / max(parent_error, 1.0e-12)


def pair_summary(state: np.ndarray, area: np.ndarray, metadata: dict[str, np.ndarray]) -> np.ndarray:
    return spatial_pair_summary(np.asarray(state, dtype=np.float64), np.asarray(area, dtype=np.float64), metadata)[0]


def sanitize(value: str) -> str:
    return value.replace("-", "_").replace("+", "plus")


def decoder_prefix(design: str, capacity: str) -> str:
    return f"decoder__{sanitize(design)}__{sanitize(capacity)}"


def representation_name(kind: str, design: str, capacity: str | None = None) -> str:
    if capacity is None:
        return f"{kind}__{sanitize(design)}"
    return f"{kind}__{sanitize(design)}__{sanitize(capacity)}"


def choose_binary_threshold(probability: np.ndarray, target: np.ndarray, grid: list[float]) -> tuple[float, dict[str, float]]:
    records = {str(value): binary_metrics(probability, target, value) for value in grid}
    selected = max(grid, key=lambda value: (records[str(value)]["iou"], records[str(value)]["balanced_accuracy"], -value))
    return float(selected), records[str(selected)]


def decoded_state(
    design: str,
    occupancy_prediction: np.ndarray,
    area_prediction: np.ndarray | None,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if design in ("B1-binary", "B2-binary-area", "B4-binary-coarse-area"):
        state = (np.asarray(occupancy_prediction) >= float(threshold)).astype(np.float64)
    else:
        state = np.argmax(np.asarray(occupancy_prediction), axis=-1).astype(np.float64) / 2.0
    if design in ("B1-binary", "B3-ordinal"):
        area = state.mean(axis=(-2, -1))
    elif design == "B2-binary-area":
        area = np.asarray(area_prediction, dtype=np.float64)
    else:
        area = np.argmax(np.asarray(area_prediction), axis=-1).astype(np.float64) / 2.0
    return state, area


def recovery_record(
    design: str,
    occupancy_prediction: np.ndarray,
    area_prediction: np.ndarray | None,
    teacher: dict[str, Any],
    threshold: float,
    area_baseline: float,
) -> dict[str, Any]:
    if design == "B3-ordinal":
        occupancy = ordinal_metrics(occupancy_prediction, teacher["occupancy_target"])
    else:
        occupancy = binary_metrics(occupancy_prediction, teacher["occupancy_target"], threshold)
    result: dict[str, Any] = {"occupancy": occupancy}
    if design == "B2-binary-area":
        result["area"] = area_metrics(area_prediction, teacher["area_target"], area_baseline)
    elif design == "B4-binary-coarse-area":
        result["area"] = ordinal_metrics(area_prediction, teacher["area_target"])
    return result


def recovery_pass(design: str, record: dict[str, Any], gates: dict[str, Any]) -> bool:
    if design == "B3-ordinal":
        occupancy = record["occupancy"]
        return bool(
            occupancy["macro_f1"] >= gates["ordinal_minimum_macro_f1"]
            and occupancy["balanced_accuracy"] >= gates["ordinal_minimum_balanced_accuracy"]
            and occupancy["ordinal_mae"] <= gates["ordinal_maximum_mae"]
        )
    occupancy = record["occupancy"]
    passed = bool(
        occupancy["iou"] >= gates["binary_minimum_iou"]
        and occupancy["balanced_accuracy"] >= gates["binary_minimum_balanced_accuracy"]
        and occupancy["positive_recall"] >= gates["binary_minimum_positive_recall"]
    )
    if design == "B2-binary-area":
        passed = passed and record["area"]["mae_improvement_over_constant"] >= gates["continuous_area_minimum_mae_improvement"]
    if design == "B4-binary-coarse-area":
        area = record["area"]
        passed = passed and area["macro_f1"] >= gates["ordinal_minimum_macro_f1"] and area["balanced_accuracy"] >= gates["ordinal_minimum_balanced_accuracy"] and area["ordinal_mae"] <= gates["ordinal_maximum_mae"]
    return bool(passed)


def fit_probe_family(
    representations: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    groups: np.ndarray,
    grid: list[float],
    weights: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    records: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, feature in representations.items():
        records[name] = {}
        predictions[name] = {}
        for target_name, target in targets.items():
            alpha, oof, cv = cv_probe(feature, target, groups, grid)
            fit = fit_ridge(feature, target, alpha)
            weights.update(ridge_to_arrays(f"ridge__{name}__{target_name}", fit))
            records[name][target_name] = {**cv, "dimensions": int(feature.shape[1])}
            predictions[name][target_name] = oof
    return records, predictions


def calibrate(protocol: dict[str, Any], protocol_commit: str) -> None:
    partition = load_partition(protocol, "calibration")
    targets = construct_targets(partition)
    metadata = partition["metadata"]
    groups = metadata["init_state"]
    grid = [float(value) for value in protocol["probes"]["ridge_grid"]]
    weights: dict[str, np.ndarray] = {}
    ee = targets["ee_image_displacement"][:, None]
    ee_models = {}
    for raw, residual in (("object_mask_difference", "object_mask_residual"), ("visible_area_difference", "visible_area_residual")):
        alpha, cv = select_ridge_alpha(ee, targets[raw], groups, grid)
        fit = fit_ridge(ee, targets[raw], alpha)
        weights.update(ridge_to_arrays(f"ee_control__{raw}", fit))
        targets[residual] = targets[raw] - predict_ridge(ee, fit).reshape(-1)
        ee_models[raw] = {"alpha": alpha, "cv": cv}

    teachers, ordinal_edge, area_edges = teacher_specifications(partition)
    weights["teacher__ordinal_positive_median"] = np.asarray(ordinal_edge)
    weights["teacher__coarse_area_edges"] = np.asarray(area_edges)
    weights["teacher__continuous_area_baseline"] = np.asarray(float(np.mean(teachers["B0-soft-area"]["area"])))
    representations = {"global": partition["global"]}
    oracle_features = {}
    for design, teacher in teachers.items():
        summary = pair_summary(teacher["state"], teacher["area"], metadata)
        oracle_features[design] = summary
        representations[representation_name("oracle", design)] = np.concatenate((partition["global"], summary), axis=1)
    probe_records, oof_predictions = fit_probe_family(representations, targets, groups, grid, weights)

    oracle_screening = {}
    baseline = oof_predictions["global"]
    b0_gain = {}
    for target_name in ALL_TARGETS:
        b0_gain[target_name] = improvement(baseline[target_name], oof_predictions[representation_name("oracle", "B0-soft-area")][target_name], targets[target_name])
    for design in DESIGNS:
        name = representation_name("oracle", design)
        gains = {target: improvement(baseline[target], oof_predictions[name][target], targets[target]) for target in ALL_TARGETS}
        raw_pass = max(gains["object_mask_difference"], gains["visible_area_difference"]) >= float(protocol["calibration_screen"]["minimum_primary_increment"])
        residual_pass = max(gains["object_mask_residual"], gains["visible_area_residual"]) >= float(protocol["calibration_screen"]["minimum_residual_increment"])
        retention = {
            target: gains[target] / max(b0_gain[target], 1.0e-12)
            for target in ("object_mask_difference", "visible_area_difference")
        }
        retention_pass = max(retention.values()) >= float(protocol["calibration_screen"]["minimum_oracle_retention_vs_B0"])
        oracle_screening[design] = {"gains": gains, "retention_vs_B0": retention, "eligible_for_decoder": bool(raw_pass and residual_pass and retention_pass)}

    patch_flat = flatten_images(partition["patch"])
    image_groups = image_init_states(partition)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder_records: dict[str, Any] = {}
    natural_oof_arrays: dict[str, np.ndarray] = {}
    decoder_epochs: dict[str, int] = {}
    thresholds: dict[str, float] = {}
    final_decoder_fits = {}
    natural_representations: dict[str, np.ndarray] = {}
    natural_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for design in DESIGNS:
        decoder_records[design] = {}
        if not oracle_screening[design]["eligible_for_decoder"]:
            decoder_records[design]["status"] = "SKIPPED_BY_CALIBRATION_ORACLE_SCREEN"
            continue
        teacher = teachers[design]
        occ_flat = flatten_images(teacher["occupancy_target"])
        area_target = teacher["area_target"]
        area_flat = None if area_target is None else flatten_images(area_target)
        area_baseline = float(np.mean(flatten_images(teachers["B0-soft-area"]["area"])))
        for capacity_index, capacity in enumerate(CAPACITIES):
            oof_occ_shape = occ_flat.shape + (() if int(teacher["occupancy_classes"]) == 2 else (3,))
            oof_occ = np.empty(oof_occ_shape, dtype=np.float64)
            if teacher["area_kind"] == "none":
                oof_area = None
            elif teacher["area_kind"] == "continuous":
                oof_area = np.empty(area_flat.shape, dtype=np.float64)
            else:
                oof_area = np.empty(area_flat.shape + (3,), dtype=np.float64)
            folds = []
            for fold, held_out in enumerate(sorted(np.unique(image_groups).tolist())):
                train = image_groups != held_out
                valid = ~train
                fit, record = fit_decoder(
                    patch_flat[train], occ_flat[train], None if area_flat is None else area_flat[train],
                    capacity, int(teacher["occupancy_classes"]), str(teacher["area_kind"]),
                    int(protocol["decoders"]["maximum_epochs"]), float(protocol["decoders"]["learning_rate"]),
                    float(protocol["decoders"]["weight_decay"]), int(protocol["decoders"]["seed"]) + 100 * capacity_index + fold,
                    device, patch_flat[valid], occ_flat[valid], None if area_flat is None else area_flat[valid],
                )
                pred_occ, pred_area = predict_decoder(patch_flat[valid], fit, device)
                oof_occ[valid] = pred_occ
                if oof_area is not None:
                    oof_area[valid] = pred_area
                folds.append({"held_out_init": int(held_out), **record})
            if int(teacher["occupancy_classes"]) == 2:
                threshold, _ = choose_binary_threshold(oof_occ, occ_flat, [float(value) for value in protocol["decoders"]["threshold_grid"]])
            else:
                threshold = 0.5
            recovery = recovery_record(design, oof_occ, oof_area, {"occupancy_target": occ_flat, "area_target": area_flat}, threshold, area_baseline)
            recovery["passes"] = recovery_pass(design, recovery, protocol["gates"]["natural_recovery"])
            epochs = int(math.floor(float(np.median([row["best_epoch"] for row in folds])) + 0.5))
            prefix = decoder_prefix(design, capacity)
            decoder_epochs[prefix] = epochs
            thresholds[prefix] = threshold
            weights[f"{prefix}__threshold"] = np.asarray(threshold)
            final_fit, final_record = fit_decoder(
                patch_flat, occ_flat, area_flat, capacity, int(teacher["occupancy_classes"]), str(teacher["area_kind"]),
                epochs, float(protocol["decoders"]["learning_rate"]), float(protocol["decoders"]["weight_decay"]),
                int(protocol["decoders"]["seed"]) + 1000 + 100 * capacity_index, device,
            )
            weights.update(decoder_to_arrays(prefix, final_fit))
            final_decoder_fits[prefix] = final_fit
            state_flat, state_area_flat = decoded_state(design, oof_occ, oof_area, threshold)
            state = state_flat.reshape(40, 3, 8, 7, 7)
            state_area = state_area_flat.reshape(40, 3, 8)
            summary = pair_summary(state, state_area, metadata)
            natural_name = representation_name("natural", design, capacity)
            natural_representations[natural_name] = np.concatenate((partition["global"], summary), axis=1)
            natural_states[natural_name] = (state, state_area)
            natural_oof_arrays[f"oof__{natural_name}__occupancy_probability"] = oof_occ.astype(np.float32)
            if oof_area is not None:
                natural_oof_arrays[f"oof__{natural_name}__area_prediction"] = oof_area.astype(np.float32)
            decoder_records[design][capacity] = {
                "folds": folds,
                "selected_final_epochs": epochs,
                "threshold": threshold,
                "recovery": recovery,
                "final_fit": final_record,
                "parameter_count": int(folds[0]["parameter_count"]),
            }

    natural_probe_records, natural_oof_predictions = fit_probe_family(natural_representations, targets, groups, grid, weights)
    probe_records.update(natural_probe_records)
    oof_predictions.update(natural_oof_predictions)
    calibration_bridge = {}
    for name in natural_representations:
        calibration_bridge[name] = {
            target: improvement(baseline[target], oof_predictions[name][target], targets[target]) for target in ALL_TARGETS
        }

    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    derived_path = ROOT / protocol["outputs"]["calibration_derived"]
    atomic_npz(weights_path, weights)
    derived_arrays = {
        **{f"metadata__{name}": value for name, value in metadata.items()},
        **{f"target__{name}": value.astype(np.float32) for name, value in targets.items()},
        **{f"representation__{name}": value.astype(np.float32) for name, value in {**representations, **natural_representations}.items()},
        **natural_oof_arrays,
    }
    for name, values in natural_states.items():
        derived_arrays[f"state__{name}"] = values[0].astype(np.float32)
        derived_arrays[f"area__{name}"] = values[1].astype(np.float32)
    atomic_npz(derived_path, derived_arrays)
    freeze = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_INIT36_39_NATURAL_EXTRACTION_EVALUATION",
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "partition": {"init_state_indices": [31, 32, 33, 34], "branch_points": 40, "pair_rows": 960, "evaluation_numeric_arrays_read": False},
        "teacher_freeze": {"binary_rule": "cell occupancy > 0", "ordinal_positive_median": ordinal_edge, "coarse_area_tertile_edges": area_edges.tolist()},
        "oracle_screening": oracle_screening,
        "decoder_records": decoder_records,
        "probe_models": probe_records,
        "calibration_bridge": calibration_bridge,
        "ee_control_models": ee_models,
        "device": device,
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
    print(json.dumps({"status": freeze["status"], "teacher_freeze": freeze["teacher_freeze"], "trained_decoder_designs": [name for name in DESIGNS if oracle_screening[name]["eligible_for_decoder"]]}, indent=2))


def bootstrap_indices(metadata: dict[str, np.ndarray], replicates: int, seed: int) -> list[np.ndarray]:
    branch = metadata["branch"]
    task = metadata["task"]
    rng = np.random.default_rng(seed)
    branches_by_task = {task_id: np.unique(branch[task == task_id]) for task_id in range(10)}
    rows = {int(value): np.flatnonzero(branch == value) for value in np.unique(branch)}
    draws = []
    for _ in range(int(replicates)):
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


def cut_improvements(parent: np.ndarray, child: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": improvement(parent, child, target), "task": [], "horizon": [], "pair": []}
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


def candidate_temporal_distance(state: np.ndarray, area: np.ndarray, include_area: bool) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    def distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
        a = state[left].reshape(-1)
        b = state[right].reshape(-1)
        values = np.abs(a - b)
        if include_area:
            values = np.concatenate((values, [abs(float(area[left]) - float(area[right]))]))
        return float(np.mean(values))
    candidate = []
    for branch in range(40):
        for left, right in ((0, 1), (0, 2), (1, 2)):
            for horizon in range(8):
                candidate.append(distance((branch, left, horizon), (branch, right, horizon)))
    temporal = []
    for branch in range(40):
        for candidate_index in range(3):
            for horizon in range(7):
                temporal.append(distance((branch, candidate_index, horizon), (branch, candidate_index, horizon + 1)))
    return np.asarray(candidate), np.asarray(temporal)


def correlation_record(value: np.ndarray, target: np.ndarray, draws: list[np.ndarray]) -> dict[str, Any]:
    record = safe_correlation(value, target)
    record["spearman_ci95"] = bootstrap_metric(draws, lambda index: safe_spearman(value[index], target[index]))
    return record


def evaluate(protocol: dict[str, Any], protocol_commit: str) -> None:
    freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    weights = load_npz(ROOT / protocol["outputs"]["calibration_weights"])
    partition = load_partition(protocol, "development_evaluation")
    metadata = partition["metadata"]
    targets = construct_targets(partition, weights)
    teachers, _, _ = teacher_specifications(
        partition,
        float(np.asarray(weights["teacher__ordinal_positive_median"])),
        np.asarray(weights["teacher__coarse_area_edges"], dtype=np.float64),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_flat = flatten_images(partition["patch"])
    representations = {"global": partition["global"]}
    spatial_summaries: dict[str, np.ndarray] = {}
    states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    recovery: dict[str, Any] = {}
    for design, teacher in teachers.items():
        summary = pair_summary(teacher["state"], teacher["area"], metadata)
        name = representation_name("oracle", design)
        spatial_summaries[name] = summary
        representations[name] = np.concatenate((partition["global"], summary), axis=1)
        states[name] = (np.asarray(teacher["state"]), np.asarray(teacher["area"]))
    for design in DESIGNS:
        if not freeze["oracle_screening"][design]["eligible_for_decoder"]:
            continue
        teacher = teachers[design]
        area_baseline = float(np.asarray(weights["teacher__continuous_area_baseline"]))
        for capacity in CAPACITIES:
            prefix = decoder_prefix(design, capacity)
            fit = decoder_from_arrays(prefix, weights)
            pred_occ, pred_area = predict_decoder(patch_flat, fit, device)
            threshold = float(np.asarray(weights[f"{prefix}__threshold"]))
            teacher_flat = {
                "occupancy_target": flatten_images(teacher["occupancy_target"]),
                "area_target": None if teacher["area_target"] is None else flatten_images(teacher["area_target"]),
            }
            record = recovery_record(design, pred_occ, pred_area, teacher_flat, threshold, area_baseline)
            record["passes"] = recovery_pass(design, record, protocol["gates"]["natural_recovery"])
            recovery_name = representation_name("natural", design, capacity)
            recovery[recovery_name] = record
            state_flat, state_area_flat = decoded_state(design, pred_occ, pred_area, threshold)
            state = state_flat.reshape(40, 3, 8, 7, 7)
            state_area = state_area_flat.reshape(40, 3, 8)
            summary = pair_summary(state, state_area, metadata)
            spatial_summaries[recovery_name] = summary
            representations[recovery_name] = np.concatenate((partition["global"], summary), axis=1)
            states[recovery_name] = (state, state_area)

    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in representations}
    metrics: dict[str, Any] = {name: {} for name in representations}
    for name, feature in representations.items():
        for target_name, target in targets.items():
            fit = ridge_from_arrays(f"ridge__{name}__{target_name}", weights)
            prediction = predict_ridge(feature, fit).reshape(-1)
            predictions[name][target_name] = prediction
            metrics[name][target_name] = scalar_metrics(prediction, target)

    draws = bootstrap_indices(metadata, int(protocol["bootstrap"]["replicates"]), int(protocol["bootstrap"]["seed"]))
    correlation_draws = draws[: int(protocol["bootstrap"]["correlation_replicates"])]
    baseline = predictions["global"]
    incremental: dict[str, Any] = {}
    shuffles: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for name in representations:
        if name == "global":
            continue
        incremental[name] = {}
        shuffles[name] = {}
        spatial = spatial_summaries[name]
        shuffled = derange(spatial, metadata, int(protocol["shuffle"]["seed"]) + len(incremental))
        for target_name, target in targets.items():
            parent = baseline[target_name]
            child = predictions[name][target_name]
            cuts = cut_improvements(parent, child, target, metadata)
            cuts["ci95"] = bootstrap_metric(draws, lambda index, p=parent, c=child, t=target: improvement(p, c, t, index))
            incremental[name][target_name] = cuts
            feature = np.concatenate((partition["global"], shuffled), axis=1)
            alternate = predict_ridge(feature, ridge_from_arrays(f"ridge__{name}__{target_name}", weights)).reshape(-1)
            original_error = rmse(child, target)
            degradation = (rmse(alternate, target) - original_error) / max(original_error, 1.0e-12)
            ci = bootstrap_metric(draws, lambda index, o=child, a=alternate, t=target: (rmse(a, t, index) - rmse(o, t, index)) / max(rmse(o, t, index), 1.0e-12))
            shuffles[name][target_name] = {"relative_degradation": degradation, "ci95": ci, "passes": bool(degradation > 0.0 and ci[0] > 0.0)}

    separation: dict[str, Any] = {}
    candidate_distance_arrays: dict[str, np.ndarray] = {}
    temporal_distance_arrays: dict[str, np.ndarray] = {}
    for name, (state, area) in states.items():
        design = "B0-soft-area" if "B0_soft_area" in name else next(item for item in ("B1-binary", "B2-binary-area", "B3-ordinal", "B4-binary-coarse-area") if sanitize(item) in name)
        candidate, temporal = candidate_temporal_distance(state, area, design in ("B0-soft-area", "B2-binary-area", "B4-binary-coarse-area"))
        candidate_distance_arrays[name] = candidate
        temporal_distance_arrays[name] = temporal
        separation[name] = {
            "candidate_mean": float(np.mean(candidate)),
            "ordinary_temporal_mean": float(np.mean(temporal)),
            "candidate_to_temporal_ratio": float(np.mean(candidate) / max(float(np.mean(temporal)), 1.0e-12)),
            "per_pair": [
                {"pair": PAIR_NAMES[pair], "candidate_mean": float(np.mean(candidate[metadata["pair"] == pair]))}
                for pair in range(3)
            ],
            "correlations": {target: correlation_record(candidate, targets[target], correlation_draws) for target in ALL_TARGETS},
        }

    for name in incremental:
        primary = []
        residual = []
        for target_name in ("object_mask_difference", "visible_area_difference"):
            record = incremental[name][target_name]
            positive_tasks = sum(row["relative_rmse_improvement"] > 0.0 for row in record["task"])
            positive_horizons = sum(row["relative_rmse_improvement"] > 0.0 for row in record["horizon"])
            positive_pairs = sum(row["relative_rmse_improvement"] > 0.0 for row in record["pair"])
            if record["overall"] >= float(protocol["gates"]["minimum_primary_increment"]) and record["ci95"][0] > 0.0 and shuffles[name][target_name]["passes"] and positive_tasks >= 7 and positive_horizons >= 4 and positive_pairs == 3:
                primary.append(target_name)
        for target_name in RESIDUAL_TARGETS:
            record = incremental[name][target_name]
            if record["overall"] >= float(protocol["gates"]["minimum_residual_increment"]) and record["ci95"][0] > 0.0:
                residual.append(target_name)
        oracle = name.startswith("oracle__")
        natural_recovery = True if oracle else recovery[name]["passes"]
        if oracle and "B0_soft_area" not in name:
            retention_targets = [
                target_name for target_name in primary
                if incremental[name][target_name]["overall"]
                / max(incremental[representation_name("oracle", "B0-soft-area")][target_name]["overall"], 1.0e-12)
                >= float(protocol["gates"]["minimum_oracle_retention_vs_B0"])
            ]
        else:
            retention_targets = list(primary)
        gates[name] = {
            "primary_passing_targets": primary,
            "residual_passing_targets": residual,
            "oracle_retention_passing_targets": retention_targets,
            "recovery_pass": natural_recovery,
            "passes": bool(primary and residual and retention_targets and natural_recovery),
        }

    oracle_b0 = representation_name("oracle", "B0-soft-area")
    oracle_b1 = representation_name("oracle", "B1-binary")
    information_loss = {}
    for target_name in ("object_mask_difference", "visible_area_difference", "object_mask_residual", "visible_area_residual"):
        soft_gain = incremental[oracle_b0][target_name]["overall"]
        binary_gain = incremental[oracle_b1][target_name]["overall"]
        natural_rows = {}
        for capacity in CAPACITIES:
            name = representation_name("natural", "B1-binary", capacity)
            if name in incremental:
                natural_gain = incremental[name][target_name]["overall"]
                natural_rows[capacity] = {"gain": natural_gain, "retention_vs_oracle_binary": natural_gain / max(binary_gain, 1.0e-12), "loss_oracle_binary_to_natural": 1.0 - natural_gain / max(binary_gain, 1.0e-12)}
        information_loss[target_name] = {
            "oracle_soft_gain": soft_gain,
            "oracle_binary_gain": binary_gain,
            "binary_retention_vs_soft": binary_gain / max(soft_gain, 1.0e-12),
            "loss_soft_to_binary": 1.0 - binary_gain / max(soft_gain, 1.0e-12),
            "natural_binary": natural_rows,
        }

    valid_natural = [name for name in gates if name.startswith("natural__") and gates[name]["passes"]]
    labels = []
    if gates.get(oracle_b1, {}).get("passes"):
        labels.append("BINARY_FOOTPRINT_TARGET_VALID")
    ordinal_oracle = representation_name("oracle", "B3-ordinal")
    if gates.get(ordinal_oracle, {}).get("passes"):
        labels.append("COARSE_FOOTPRINT_TARGET_VALID")
    if any(name.startswith("natural__B1_binary") and gates[name]["passes"] for name in valid_natural):
        labels.extend(["SOFT_TARGET_OVERPRECISION_SUPPORTED", "NATURAL_BINARY_SPATIAL_PROXY_VALID"])
    b1_valid = any(name.startswith("natural__B1_binary") and gates[name]["passes"] for name in valid_natural)
    area_valid = any(("B2_binary_area" in name or "B4_binary_coarse_area" in name) and gates[name]["passes"] for name in valid_natural)
    if b1_valid and not area_valid:
        labels.append("VISIBLE_AREA_NOT_REQUIRED")
    elif area_valid and not b1_valid:
        labels.append("VISIBLE_SCALE_SIGNAL_NEEDED")
    capacity_rank = {capacity: index for index, capacity in enumerate(CAPACITIES)}
    passing_capacities = [next(capacity for capacity in CAPACITIES if sanitize(capacity) in name) for name in valid_natural]
    if passing_capacities and min(capacity_rank[value] for value in passing_capacities) >= 2:
        labels.append("DECODER_CAPACITY_LIMIT_SUPPORTED")
    if passing_capacities and all(value == "C3-local-context" for value in passing_capacities):
        labels.append("LOCAL_SPATIAL_CONTEXT_NEEDED")
    any_recovery_without_bridge = any(recovery[name]["passes"] and not gates[name]["passes"] for name in recovery)
    if any_recovery_without_bridge:
        labels.append("LOCALIZATION_NOT_SUFFICIENT")
    calibration_bridge = freeze["calibration_bridge"]
    if any(
        freeze["decoder_records"][design][capacity]["recovery"]["passes"]
        and max(calibration_bridge[representation_name("natural", design, capacity)]["object_mask_difference"], calibration_bridge[representation_name("natural", design, capacity)]["visible_area_difference"]) >= 0.1
        and not gates[representation_name("natural", design, capacity)]["passes"]
        for design in DESIGNS if freeze["oracle_screening"][design]["eligible_for_decoder"] for capacity in CAPACITIES
    ):
        labels.append("EXTRACTION_GENERALIZATION_LIMIT")

    selected = None
    for design in DESIGNS:
        for capacity in CAPACITIES:
            name = representation_name("natural", design, capacity)
            if name in valid_natural:
                selected = name
                break
        if selected is not None:
            break
    binary_oracle_pass = gates.get(oracle_b1, {}).get("passes", False)
    if selected is not None:
        labels.append("SPATIAL_TARGET_EXTRACTION_RESOLVED")
        authorization = "GO_SPATIAL_WM_PROTOTYPE"
        alternative_encoder_authorization = False
        block_type = None
    else:
        labels.append("NATURAL_SPATIAL_EXTRACTION_BLOCK")
        if not binary_oracle_pass:
            labels.append("SOFT_FOOTPRINT_INFORMATION_REQUIRED")
            block_type = "target simplification failure"
            alternative_encoder_authorization = False
        else:
            labels.append("FROZEN_CLIP_SPATIAL_INFORMATION_LIMIT")
            block_type = "frozen CLIP information limit"
            alternative_encoder_authorization = True
        authorization = "BLOCK_SPATIAL_WM_PROTOTYPE"
    if any(gates[name]["residual_passing_targets"] for name in gates):
        labels.append("OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED")
    labels = list(dict.fromkeys(labels))

    calibration_evaluation_gap = {}
    for name, record in recovery.items():
        design = next(item for item in DESIGNS if sanitize(item) in name)
        capacity = next(item for item in CAPACITIES if sanitize(item) in name)
        calibration_record = freeze["decoder_records"][design][capacity]["recovery"]
        calibration_evaluation_gap[name] = {"calibration": calibration_record, "evaluation": record}

    derived_path = ROOT / protocol["outputs"]["evaluation_derived"]
    arrays = {
        **{f"metadata__{name}": value for name, value in metadata.items()},
        **{f"target__{name}": value.astype(np.float32) for name, value in targets.items()},
        **{f"representation__{name}": value.astype(np.float32) for name, value in representations.items()},
        **{f"prediction__{name}__{target}": prediction.astype(np.float32) for name, target_predictions in predictions.items() for target, prediction in target_predictions.items()},
        **{f"candidate_distance__{name}": value.astype(np.float32) for name, value in candidate_distance_arrays.items()},
        **{f"temporal_distance__{name}": value.astype(np.float32) for name, value in temporal_distance_arrays.items()},
    }
    for name, (state, area) in states.items():
        arrays[f"state__{name}"] = state.astype(np.float32)
        arrays[f"area__{name}"] = area.astype(np.float32)
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
        "teacher_definitions": protocol["teachers"],
        "teacher_freeze": freeze["teacher_freeze"],
        "decoder_definitions": protocol["decoders"],
        "oracle_screening": freeze["oracle_screening"],
        "decoder_records": freeze["decoder_records"],
        "probe_metrics": metrics,
        "incremental": incremental,
        "shuffle_controls": shuffles,
        "candidate_separation": separation,
        "natural_recovery": recovery,
        "calibration_evaluation_gap": calibration_evaluation_gap,
        "information_loss_decomposition": information_loss,
        "gates": gates,
        "decision": {
            "mechanism_labels": labels,
            "selected_target": selected,
            "next_stage_authorization": authorization,
            "alternative_spatial_encoder_audit_authorized": alternative_encoder_authorization,
            "block_type": block_type,
            "controller_authorization": False,
            "prototype_scope_if_go": "seed11 single-model offline spatial World Model mechanism prototype only",
        },
        "candidate_headroom_reference": protocol["candidate_headroom_reference"],
        "artifacts": {"evaluation_derived": {"path": str(derived_path.relative_to(ROOT)), "sha256": sha256(derived_path)}},
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {"simulator_run": False, "world_model_or_encoder_training": False, "candidate_or_reranker_change": False, "controller_or_success_run": False, "reward_done_success_or_task_outcome_read": False, "init40_49_access": False, "seed23_47_run": False, "upstream_overwrite": False},
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
