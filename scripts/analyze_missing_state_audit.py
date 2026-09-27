#!/usr/bin/env python3
"""Calibrate and evaluate the preregistered missing-state ridge probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

from src.probes.missing_state import (
    FEATURE_GROUPS,
    FEATURE_SET_GROUPS,
    PAIR_NAMES,
    build_pair_dataset,
    fit_ridge,
    predict_ridge,
    ridge_from_arrays,
    ridge_to_arrays,
    scalar_metrics,
    select_ridge_alpha,
)


PROTOCOL_VERSION = "local-visual-counterfactual-missing-state-audit-v1"


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


def load_protocol(path: Path, phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_STATE_INVENTORY_OR_REPLAY":
        raise ValueError("protocol is not preregistered")
    preregistered_commit = last_commit(path)
    if not preregistered_commit:
        raise ValueError("protocol must be committed")
    implementation = protocol["implementation"]["analyzer"]
    if sha256(ROOT / implementation["path"]) != implementation["sha256"]:
        raise ValueError("analyzer changed after preregistration")
    inventory = ROOT / protocol["outputs"]["state_inventory"]
    if not inventory.exists() or not last_commit(inventory):
        raise RuntimeError("state inventory must be committed")
    calibration = protocol["partitions"]["calibration"]
    for key in ("source_sidecar", "state_sidecar", "state_manifest"):
        if not (ROOT / calibration[key]).exists():
            raise FileNotFoundError(calibration[key])
    if phase == "calibrate":
        for forbidden in (
            protocol["partitions"]["evaluation"]["state_sidecar"],
            protocol["partitions"]["evaluation"]["state_manifest"],
            protocol["outputs"]["calibration_freeze"],
            protocol["outputs"]["diagnosis"],
        ):
            if (ROOT / forbidden).exists():
                raise RuntimeError(f"calibration phase refuses pre-existing artifact: {forbidden}")
    else:
        freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("calibration freeze must be committed before evaluation")
        evaluation = protocol["partitions"]["evaluation"]
        for key in ("source_sidecar", "state_sidecar", "state_manifest"):
            if not (ROOT / evaluation[key]).exists():
                raise FileNotFoundError(evaluation[key])
    return protocol, preregistered_commit


def load_partition(protocol: dict[str, Any], partition: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    record = protocol["partitions"][partition]
    base_file = np.load(ROOT / record["source_sidecar"], allow_pickle=False)
    state_file = np.load(ROOT / record["state_sidecar"], allow_pickle=False)
    base = {key: np.asarray(base_file[key]) for key in base_file.files}
    state = {key: np.asarray(state_file[key]) for key in state_file.files}
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        if not np.array_equal(base[key], state[key]):
            raise ValueError(f"partition identity mismatch: {key}")
    if set(np.unique(base["init_state_index"]).tolist()) != set(record["init_state_indices"]):
        raise ValueError("partition init-state boundary changed")
    return base, state


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


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    first = np.asarray(x, dtype=np.float64).reshape(-1)
    second = np.asarray(y, dtype=np.float64).reshape(-1)
    if first.size < 3 or np.std(first) <= 1.0e-20 or np.std(second) <= 1.0e-20:
        return {"n": int(first.size), "pearson": None, "pearson_p": None, "spearman": None, "spearman_p": None}
    pearson = stats.pearsonr(first, second)
    spearman = stats.spearmanr(first, second)
    return {
        "n": int(first.size),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_indices(metadata: dict[str, np.ndarray], replicates: int, seed: int) -> list[np.ndarray]:
    branch = metadata["branch"]
    task = metadata["task"]
    rng = np.random.default_rng(seed)
    branches_by_task = {
        task_id: np.unique(branch[task == task_id]) for task_id in sorted(np.unique(task).tolist())
    }
    row_indices = {int(value): np.flatnonzero(branch == value) for value in np.unique(branch)}
    draws = []
    for _ in range(replicates):
        rows = []
        for task_id, candidates in branches_by_task.items():
            del task_id
            sampled = rng.choice(candidates, size=candidates.size, replace=True)
            rows.extend(row_indices[int(value)] for value in sampled)
        draws.append(np.concatenate(rows))
    return draws


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def metric_bootstrap(
    draws: list[np.ndarray],
    function: Callable[[np.ndarray], float],
) -> list[float]:
    return interval([function(indices) for indices in draws])


def calibrate(protocol: dict[str, Any], preregistered_commit: str) -> None:
    base, state = load_partition(protocol, "calibration")
    p_mean, p_std = normalization()
    dataset = build_pair_dataset(base, state, p_mean, p_std)
    grid = [float(value) for value in protocol["probe"]["ridge_grid"]]
    weights: dict[str, np.ndarray] = {}
    model_records: dict[str, Any] = {}
    for set_name, features in dataset["sets"].items():
        scalar_alpha, scalar_cv = select_ridge_alpha(
            features, dataset["target_visual"], dataset["metadata"]["init_state"], grid
        )
        scalar_fit = fit_ridge(features, dataset["target_visual"], scalar_alpha)
        weights.update(ridge_to_arrays(f"{set_name}_scalar", scalar_fit))
        direction_alpha, direction_cv = select_ridge_alpha(
            features, dataset["target_delta_z"], dataset["metadata"]["init_state"], grid
        )
        direction_fit = fit_ridge(features, dataset["target_delta_z"], direction_alpha)
        weights.update(ridge_to_arrays(f"{set_name}_direction", direction_fit))
        model_records[set_name] = {
            "dimensions": int(features.shape[1]),
            "scalar_selected_alpha": scalar_alpha,
            "scalar_cv": scalar_cv,
            "direction_selected_alpha": direction_alpha,
            "direction_cv": direction_cv,
        }
    weights["constant_visual"] = np.asarray(dataset["target_visual"].mean(), dtype=np.float64)
    thresholds = np.quantile(dataset["scalars"]["object_translation_max"], [1 / 3, 2 / 3])
    weights["object_motion_thresholds"] = thresholds.astype(np.float64)
    output = ROOT / protocol["outputs"]["calibration_weights"]
    if output.exists():
        raise FileExistsError(output)
    atomic_npz(output, weights)
    freeze = {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_HELD_OUT_STATE_REPLAY_OR_EVALUATION",
        "protocol_sha256": sha256(ROOT / "configs/missing_state_audit_protocol.json"),
        "preregistered_commit": preregistered_commit,
        "partition": {
            "init_state_indices": sorted(np.unique(dataset["metadata"]["init_state"]).astype(int).tolist()),
            "branch_points": int(np.unique(dataset["metadata"]["branch"]).size),
            "rows": int(dataset["target_visual"].size),
            "evaluation_state_sidecar_read": False,
        },
        "feature_sets": model_records,
        "feature_group_dimensions": {name: int(value.shape[1]) for name, value in dataset["groups"].items()},
        "constant_visual_baseline": float(weights["constant_visual"]),
        "object_motion_bin_thresholds_calibration_only": thresholds.tolist(),
        "weights": {"path": str(output.relative_to(ROOT)), "sha256": sha256(output)},
        "sources": {
            "base_sidecar_sha256": sha256(ROOT / protocol["partitions"]["calibration"]["source_sidecar"]),
            "state_sidecar_sha256": sha256(ROOT / protocol["partitions"]["calibration"]["state_sidecar"]),
            "state_inventory_sha256": sha256(ROOT / protocol["outputs"]["state_inventory"]),
        },
        "forbidden_activity_audit": {
            "evaluation_state_read": False,
            "init_40_49_access": False,
            "success_reward_done_read": False,
            "model_or_controller_training": False,
            "evaluation_hyperparameter_selection": False,
        },
    }
    freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
    atomic_json(freeze_path, freeze)
    print(json.dumps({"status": "CALIBRATION_FROZEN", "weights_sha256": freeze["weights"]["sha256"]}))


def cut_correlations(scalar: np.ndarray, target: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {"overall": correlation(scalar, target), "horizon": [], "task": [], "pair": []}
    for horizon in range(1, 9):
        mask = metadata["horizon"] == horizon
        result["horizon"].append({"horizon": horizon, **correlation(scalar[mask], target[mask])})
    for task in range(10):
        mask = metadata["task"] == task
        result["task"].append({"task": task, **correlation(scalar[mask], target[mask])})
    for pair, pair_name in enumerate(PAIR_NAMES):
        mask = metadata["pair"] == pair
        result["pair"].append({"pair": pair_name, **correlation(scalar[mask], target[mask])})
    return result


def feature_slices(dataset: dict[str, Any], set_name: str) -> dict[str, slice]:
    offset = 0
    result = {}
    for group in FEATURE_SET_GROUPS[set_name]:
        width = dataset["groups"][group].shape[1]
        result[group] = slice(offset, offset + width)
        offset += width
    return result


def shuffled_group(
    values: np.ndarray,
    metadata: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    result = np.asarray(values).copy()
    rng = np.random.default_rng(seed)
    for task in range(10):
        for horizon in range(1, 9):
            for pair in range(3):
                indices = np.flatnonzero(
                    (metadata["task"] == task)
                    & (metadata["horizon"] == horizon)
                    & (metadata["pair"] == pair)
                )
                if indices.size < 2:
                    raise ValueError("shuffle stratum has fewer than two branches")
                order = np.arange(indices.size)
                for _ in range(100):
                    rng.shuffle(order)
                    if np.all(order != np.arange(indices.size)):
                        break
                else:
                    raise RuntimeError("could not construct derangement")
                result[indices] = values[indices[order]]
    return result


def direction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    pred_norm = np.linalg.norm(pred, axis=1)
    true_norm = np.linalg.norm(true, axis=1)
    direction = np.sum(pred * true, axis=1) / np.maximum(pred_norm * true_norm, 1.0e-12)
    rmse = float(np.sqrt(np.mean(np.square(pred - true))))
    zero = float(np.sqrt(np.mean(np.square(true))))
    return {
        "rmse": rmse,
        "zero_rmse": zero,
        "improvement_over_zero": (zero - rmse) / zero,
        "direction_mean": float(direction.mean()),
        "norm_ratio_of_means": float(pred_norm.mean() / true_norm.mean()),
    }


def evaluate(protocol: dict[str, Any], preregistered_commit: str) -> None:
    freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    if freeze["protocol_sha256"] != sha256(ROOT / "configs/missing_state_audit_protocol.json"):
        raise ValueError("calibration freeze references another protocol")
    weights_path = ROOT / freeze["weights"]["path"]
    if sha256(weights_path) != freeze["weights"]["sha256"]:
        raise ValueError("calibration weights changed")
    base, state = load_partition(protocol, "evaluation")
    p_mean, p_std = normalization()
    dataset = build_pair_dataset(base, state, p_mean, p_std)
    weights = np.load(weights_path, allow_pickle=False)
    target = dataset["target_visual"]
    metadata = dataset["metadata"]
    draws = bootstrap_indices(
        metadata,
        int(protocol["bootstrap"]["replicates"]),
        int(protocol["bootstrap"]["seed"]),
    )

    constant_prediction = np.full_like(target, float(weights["constant_visual"]))
    model_predictions: dict[str, np.ndarray] = {}
    scalar_results: dict[str, Any] = {
        "constant": scalar_metrics(constant_prediction, target),
        "sets": {},
        "increments": {},
    }
    for set_name, features in dataset["sets"].items():
        fit = ridge_from_arrays(f"{set_name}_scalar", weights)
        prediction = predict_ridge(features, fit).reshape(-1)
        model_predictions[set_name] = prediction
        scalar_results["sets"][set_name] = {
            **scalar_metrics(prediction, target),
            "alpha": fit.alpha,
            "dimensions": int(features.shape[1]),
        }
    comparison_names = (("S0", "constant"), ("S1", "S0"), ("S2", "S1"), ("S3", "S2"), ("S4", "S3"))
    for current, previous in comparison_names:
        current_prediction = model_predictions[current]
        previous_prediction = constant_prediction if previous == "constant" else model_predictions[previous]
        current_rmse = scalar_metrics(current_prediction, target)["rmse"]
        previous_rmse = scalar_metrics(previous_prediction, target)["rmse"]
        improvement = (previous_rmse - current_rmse) / previous_rmse
        ci = metric_bootstrap(
            draws,
            lambda indices, cp=current_prediction, pp=previous_prediction: (
                np.sqrt(np.mean(np.square(pp[indices] - target[indices])))
                - np.sqrt(np.mean(np.square(cp[indices] - target[indices])))
            )
            / np.sqrt(np.mean(np.square(pp[indices] - target[indices]))),
        )
        scalar_results["increments"][f"{current}_vs_{previous}"] = {
            "relative_rmse_improvement": float(improvement),
            "ci95": ci,
        }

    group_parent = {"rotation": ("S1", "S0"), "object": ("S2", "S1"), "contact": ("S3", "S2"), "relative": ("S4", "S3")}
    shuffle_results = {}
    for index, (group, (set_name, _)) in enumerate(group_parent.items()):
        shuffled = shuffled_group(dataset["groups"][group], metadata, int(protocol["shuffle"]["seed"]) + index)
        features = dataset["sets"][set_name].copy()
        features[:, feature_slices(dataset, set_name)[group]] = shuffled
        prediction = predict_ridge(features, ridge_from_arrays(f"{set_name}_scalar", weights)).reshape(-1)
        original = model_predictions[set_name]
        original_rmse = scalar_metrics(original, target)["rmse"]
        shuffled_rmse = scalar_metrics(prediction, target)["rmse"]
        degradation = (shuffled_rmse - original_rmse) / original_rmse
        ci = metric_bootstrap(
            draws,
            lambda indices, sp=prediction, op=original: (
                np.sqrt(np.mean(np.square(sp[indices] - target[indices])))
                - np.sqrt(np.mean(np.square(op[indices] - target[indices])))
            )
            / np.sqrt(np.mean(np.square(op[indices] - target[indices]))),
        )
        shuffle_results[group] = {
            "original_rmse": original_rmse,
            "shuffled_rmse": shuffled_rmse,
            "relative_degradation": float(degradation),
            "ci95": ci,
        }

    association_keys = (
        "ee_translation",
        "ee_rotation",
        "action_rotation",
        "object_translation_max",
        "object_translation_mean",
        "object_rotation_max",
        "object_rotation_mean",
        "contact_different",
        "relative_translation_max",
        "relative_translation_mean",
        "relative_rotation_max",
        "relative_rotation_mean",
    )
    associations = {}
    for key in association_keys:
        scalar = dataset["scalars"][key]
        record = cut_correlations(scalar, target, metadata)
        record["overall"]["spearman_ci95"] = metric_bootstrap(
            draws,
            lambda indices, value=scalar: float(stats.spearmanr(value[indices], target[indices]).statistic),
        )
        associations[key] = record
    action_to_rotation = cut_correlations(
        dataset["scalars"]["action_rotation"], dataset["scalars"]["ee_rotation"], metadata
    )
    action_to_rotation["overall"]["spearman_ci95"] = metric_bootstrap(
        draws,
        lambda indices: float(
            stats.spearmanr(
                dataset["scalars"]["action_rotation"][indices],
                dataset["scalars"]["ee_rotation"][indices],
            ).statistic
        ),
    )

    different = dataset["scalars"]["contact_different"] > 0.5
    contact_conditioned = {
        "same": summary(target[~different]),
        "different": summary(target[different]),
        "mean_difference": float(target[different].mean() - target[~different].mean()),
        "mean_difference_ci95": metric_bootstrap(
            draws,
            lambda indices: float(
                target[indices][different[indices]].mean() - target[indices][~different[indices]].mean()
            )
            if np.any(different[indices]) and np.any(~different[indices])
            else 0.0,
        ),
    }
    thresholds = np.asarray(weights["object_motion_thresholds"], dtype=np.float64)
    object_motion = dataset["scalars"]["object_translation_max"]
    bin_index = np.digitize(object_motion, thresholds, right=False)
    object_conditioned = {
        "thresholds_from_calibration": thresholds.tolist(),
        "low": summary(target[bin_index == 0]),
        "medium": summary(target[bin_index == 1]),
        "high": summary(target[bin_index == 2]),
    }

    minimum = float(protocol["gates"]["minimum_incremental_rmse_improvement"])
    corr_min = float(protocol["gates"]["minimum_spearman"])

    def increment_pass(name: str) -> bool:
        value = scalar_results["increments"][name]
        return value["relative_rmse_improvement"] >= minimum and value["ci95"][0] > 0

    def shuffle_pass(group: str) -> bool:
        value = shuffle_results[group]
        return value["relative_degradation"] > 0 and value["ci95"][0] > 0

    def association_pass(key: str) -> bool:
        value = associations[key]["overall"]
        return (
            value["spearman"] is not None
            and value["spearman"] >= corr_min
            and value["spearman_ci95"][0] > 0
        )

    def positive_tasks(key: str) -> int:
        return sum(
            row["spearman"] is not None and row["spearman"] > 0
            for row in associations[key]["task"]
        )

    rotation_chain = {
        "action_to_real_rotation": bool(
            action_to_rotation["overall"]["spearman"] is not None
            and action_to_rotation["overall"]["spearman"] >= corr_min
            and action_to_rotation["overall"]["spearman_ci95"][0] > 0
        ),
        "real_rotation_to_visual": association_pass("ee_rotation"),
        "S1_over_S0": increment_pass("S1_vs_S0"),
    }
    rotation_supported = all(rotation_chain.values())
    object_supported = bool(
        increment_pass("S2_vs_S1")
        and shuffle_pass("object")
        and (association_pass("object_translation_max") or association_pass("object_rotation_max"))
        and max(positive_tasks("object_translation_max"), positive_tasks("object_rotation_max"))
        >= int(protocol["gates"]["minimum_positive_tasks"])
    )
    contact_supported = bool(
        increment_pass("S3_vs_S2")
        and shuffle_pass("contact")
        and contact_conditioned["mean_difference"] > 0
        and contact_conditioned["mean_difference_ci95"][0] > 0
    )
    relative_supported = bool(
        increment_pass("S4_vs_S3")
        and shuffle_pass("relative")
        and (association_pass("relative_translation_max") or association_pass("relative_rotation_max"))
    )
    drivers = []
    if rotation_supported:
        drivers.append("EE_ORIENTATION_MISSING_STATE_SUPPORTED")
    if object_supported:
        drivers.append("OBJECT_DYNAMICS_VISUAL_DRIVER")
    if contact_supported:
        drivers.append("CONTACT_BRANCHING_VISUAL_DRIVER")
    if relative_supported:
        drivers.append("RELATIVE_GEOMETRY_VISUAL_DRIVER")
    baseline_insufficient = not increment_pass("S0_vs_constant")
    labels = list(drivers)
    if baseline_insufficient:
        labels.append("ROBOT_KINEMATICS_ONLY_INSUFFICIENT")
    if len(drivers) >= 2:
        labels.append("MULTI_FACTOR_VISUAL_COUNTERFACTUAL")
    if not drivers:
        labels.append("NO_TESTED_STATE_EXPLAINS_VISUAL_COUNTERFACTUAL")

    if len(drivers) >= 2:
        authorization = "GO_STRUCTURED_STATE_DYNAMICS_DESIGN"
    elif drivers == ["EE_ORIENTATION_MISSING_STATE_SUPPORTED"]:
        authorization = "GO_FULL_EE_POSE_DYNAMICS_DESIGN"
    elif drivers == ["OBJECT_DYNAMICS_VISUAL_DRIVER"]:
        authorization = "GO_OBJECT_AWARE_DYNAMICS_DIAGNOSIS"
    elif drivers == ["CONTACT_BRANCHING_VISUAL_DRIVER"]:
        authorization = "GO_INTERACTION_MODE_DYNAMICS_DIAGNOSIS"
    elif drivers == ["RELATIVE_GEOMETRY_VISUAL_DRIVER"]:
        authorization = "GO_OBJECT_RELATIVE_STATE_DIAGNOSIS"
    else:
        authorization = "BLOCK_STRUCTURED_PHYSICAL_WM_ROUTE"

    secondary = {}
    extension_sets = {
        "rotation": "S1",
        "object": "S2",
        "contact": "S3",
        "relative": "S4",
    }
    for group, set_name in extension_sets.items():
        if {
            "rotation": rotation_supported,
            "object": object_supported,
            "contact": contact_supported,
            "relative": relative_supported,
        }[group]:
            prediction = predict_ridge(
                dataset["sets"][set_name], ridge_from_arrays(f"{set_name}_direction", weights)
            )
            secondary[group] = direction_metrics(prediction, dataset["target_delta_z"])

    result = {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "final_mechanism_labels": labels,
        "next_stage_authorization": authorization,
        "controller_authorized": False,
        "protocol": {
            "path": "configs/missing_state_audit_protocol.json",
            "sha256": sha256(ROOT / "configs/missing_state_audit_protocol.json"),
            "preregistered_commit": preregistered_commit,
        },
        "calibration_freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha256(freeze_path)},
        "evaluation_scope": {
            "init_state_indices": sorted(np.unique(metadata["init_state"]).astype(int).tolist()),
            "branch_points": int(np.unique(metadata["branch"]).size),
            "rows": int(target.size),
            "tasks": sorted(np.unique(metadata["task"]).astype(int).tolist()),
            "pairs": list(PAIR_NAMES),
            "horizons": list(range(1, 9)),
        },
        "feature_dimensions": {
            "groups": {name: int(value.shape[1]) for name, value in dataset["groups"].items()},
            "sets": {name: int(value.shape[1]) for name, value in dataset["sets"].items()},
        },
        "visual_target": summary(target),
        "raw_associations": associations,
        "rotation_chain": {**rotation_chain, "action_to_rotation_association": action_to_rotation},
        "contact_conditioned_visual": contact_conditioned,
        "object_motion_conditioned_visual": object_conditioned,
        "scalar_probes": scalar_results,
        "group_shuffle_controls": shuffle_results,
        "secondary_direction_probes": secondary,
        "gate_results": {
            "rotation_supported": rotation_supported,
            "object_supported": object_supported,
            "contact_supported": contact_supported,
            "relative_supported": relative_supported,
            "robot_kinematics_only_insufficient": baseline_insufficient,
            "minimum_incremental_rmse_improvement": minimum,
            "minimum_spearman": corr_min,
            "minimum_positive_tasks": int(protocol["gates"]["minimum_positive_tasks"]),
        },
        "sources": {
            "state_inventory_sha256": sha256(ROOT / protocol["outputs"]["state_inventory"]),
            "evaluation_source_sidecar_sha256": sha256(ROOT / protocol["partitions"]["evaluation"]["source_sidecar"]),
            "evaluation_state_sidecar_sha256": sha256(ROOT / protocol["partitions"]["evaluation"]["state_sidecar"]),
            "calibration_weights_sha256": sha256(weights_path),
        },
        "nan_inf_audit": "pass",
        "oracle_boundary": "Future EE/object/contact/relative state is a mechanism-diagnosis oracle, not a deployable World-Model or controller input.",
        "forbidden_activity_audit": {
            "init_40_49_access": False,
            "reward_done_success_read": False,
            "controller_or_reranker_run": False,
            "world_model_training_or_change": False,
            "candidate_generator_or_scale_change": False,
            "evaluation_fit_or_threshold_selection": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    output = ROOT / protocol["outputs"]["diagnosis"]
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    derived = ROOT / protocol["outputs"]["evaluation_metrics"]
    atomic_npz(
        derived,
        {
            "target_visual": target,
            "target_delta_z": dataset["target_delta_z"],
            **{f"group_{key}": value for key, value in dataset["groups"].items()},
            **{f"prediction_{key}": value for key, value in model_predictions.items()},
            **{f"meta_{key}": value for key, value in metadata.items()},
        },
    )
    result["artifacts"] = {
        "diagnosis_sha256_before_artifact_field": sha256(output),
        "evaluation_metrics": str(derived.relative_to(ROOT)),
        "evaluation_metrics_sha256": sha256(derived),
    }
    atomic_json(output, result)
    print(json.dumps({"status": "COMPLETE", "labels": labels, "authorization": authorization}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/missing_state_audit_protocol.json")
    args = parser.parse_args()
    protocol, commit = load_protocol(args.protocol, args.phase)
    if args.phase == "calibrate":
        calibrate(protocol, commit)
    else:
        evaluate(protocol, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
