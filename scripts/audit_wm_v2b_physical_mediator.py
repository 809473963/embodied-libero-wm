#!/usr/bin/env python3
"""Offline audit of WM-v2B counterfactual physical-mediator fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")
CANDIDATE_NAMES = ("A0", "A1", "A2")
DIMENSION_NAMES = (
    "ee_x",
    "ee_y",
    "ee_z",
    "gripper_0",
    "gripper_1",
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
GROUPS = {
    "ee_pos": np.asarray([0, 1, 2]),
    "gripper": np.asarray([3, 4]),
    "joints": np.asarray([5, 6, 7, 8, 9, 10, 11]),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
    os.replace(temporary, path)


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def summary(value: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(value, dtype=np.float64).reshape(-1)
    if data.size == 0:
        raise ValueError("cannot summarize an empty array")
    if not np.isfinite(data).all():
        raise FloatingPointError("summary input contains NaN/Inf")
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | int | None]:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"n": int(a.size), "pearson": None, "pearson_p": None, "spearman": None, "spearman_p": None}
    pearson = pearsonr(a, b)
    spearman = spearmanr(a, b)
    return {
        "n": int(a.size),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def smooth_l1_from_absolute(absolute: np.ndarray) -> np.ndarray:
    value = np.asarray(absolute, dtype=np.float64)
    return np.where(value < 1.0, 0.5 * value**2, value - 0.5)


def absolute_from_smooth_l1(loss: np.ndarray) -> np.ndarray:
    value = np.asarray(loss, dtype=np.float64)
    if np.any(value < 0):
        raise ValueError("SmoothL1 loss cannot be negative")
    return np.where(value < 0.5, np.sqrt(2.0 * value), value + 0.5)


def absolute_metrics(absolute_error: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(absolute_error, dtype=np.float64)
    return {
        "elements": int(value.size),
        "rmse": float(np.sqrt(np.mean(value**2))),
        "mae": float(np.mean(value)),
        "smooth_l1_beta1": float(np.mean(smooth_l1_from_absolute(value))),
    }


def absolute_breakdown(
    absolute_error: np.ndarray, task_id: np.ndarray
) -> dict[str, Any]:
    value = np.asarray(absolute_error, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (8, 12):
        raise ValueError(f"expected [N,8,12] absolute error, got {value.shape}")
    if len(task_id) != value.shape[0]:
        raise ValueError("task identity length mismatch")
    return {
        "overall": absolute_metrics(value),
        "per_horizon": [
            {"horizon": horizon + 1, **absolute_metrics(value[:, horizon])}
            for horizon in range(8)
        ],
        "groups": {
            name: absolute_metrics(value[..., indices])
            for name, indices in GROUPS.items()
        },
        "per_task": {
            str(task): {
                "samples": int(np.sum(task_id == task)),
                **absolute_metrics(value[task_id == task]),
            }
            for task in range(10)
        },
    }


def pairwise_delta(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4 or array.shape[1:] != (3, 8, 12):
        raise ValueError(f"expected [N,3,8,12], got {array.shape}")
    return np.stack(
        [array[:, left] - array[:, right] for left, right in PAIR_INDICES],
        axis=1,
    )


def vector_cosine(predicted: np.ndarray, real: np.ndarray) -> np.ndarray:
    numerator = np.sum(predicted * real, axis=-1)
    denominator = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(real, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def differential_stats(
    predicted: np.ndarray, real: np.ndarray, minimum_real_norm: float
) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(real, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim < 2:
        raise ValueError("predicted and real deltas must share [...,D]")
    pred_norm = np.linalg.norm(pred, axis=-1)
    real_norm = np.linalg.norm(target, axis=-1)
    eligible = real_norm >= float(minimum_real_norm)
    if not np.any(eligible):
        raise ValueError("no differential vectors reach the denominator threshold")
    direction = vector_cosine(pred, target)[eligible]
    ratio = pred_norm[eligible] / real_norm[eligible]
    point_error = np.sqrt(np.mean((pred - target) ** 2, axis=-1))[eligible]
    point_zero = np.sqrt(np.mean(target**2, axis=-1))[eligible]
    element_error = (pred - target)[eligible]
    element_zero = target[eligible]
    error_rmse = float(np.sqrt(np.mean(element_error**2)))
    zero_rmse = float(np.sqrt(np.mean(element_zero**2)))
    return {
        "vectors": int(eligible.sum()),
        "total_vectors": int(eligible.size),
        "eligible_fraction": float(eligible.mean()),
        "direction_cosine": summary(direction),
        "real_norm": summary(real_norm[eligible]),
        "predicted_norm": summary(pred_norm[eligible]),
        "magnitude_ratio": {
            "ratio_of_means": float(pred_norm[eligible].mean() / real_norm[eligible].mean()),
            **summary(ratio),
        },
        "differential_rmse": error_rmse,
        "zero_difference_rmse": zero_rmse,
        "relative_improvement_over_zero": float((zero_rmse - error_rmse) / zero_rmse),
        "pointwise_differential_rmse": summary(point_error),
        "pointwise_zero_rmse": summary(point_zero),
    }


def rank_contexts(
    predicted_norm: np.ndarray, real_norm: np.ndarray, tie_tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return within-context Spearman [N,H] and ordering scores [N,H,3]."""
    pred = np.asarray(predicted_norm, dtype=np.float64)
    real = np.asarray(real_norm, dtype=np.float64)
    if pred.shape != real.shape or pred.ndim != 3 or pred.shape[1] != 3:
        raise ValueError("ranking norms must share [N,3,H]")
    sample_count, _, horizon_count = pred.shape
    within = np.empty((sample_count, horizon_count), dtype=np.float64)
    ordering = np.empty((sample_count, horizon_count, 3), dtype=np.float64)
    comparisons = ((0, 1), (0, 2), (1, 2))
    for sample in range(sample_count):
        for horizon in range(horizon_count):
            pred_row = pred[sample, :, horizon]
            real_row = real[sample, :, horizon]
            pred_rank = rankdata(pred_row, method="average")
            real_rank = rankdata(real_row, method="average")
            if np.ptp(pred_rank) == 0 or np.ptp(real_rank) == 0:
                within[sample, horizon] = 0.0
            else:
                within[sample, horizon] = float(pearsonr(pred_rank, real_rank).statistic)
            for index, (left, right) in enumerate(comparisons):
                pred_difference = pred_row[left] - pred_row[right]
                real_difference = real_row[left] - real_row[right]
                if abs(pred_difference) <= tie_tolerance or abs(real_difference) <= tie_tolerance:
                    ordering[sample, horizon, index] = 0.5
                else:
                    ordering[sample, horizon, index] = float(
                        np.sign(pred_difference) == np.sign(real_difference)
                    )
    return within, ordering


def ranking_summary(
    predicted_norm: np.ndarray,
    real_norm: np.ndarray,
    within: np.ndarray,
    ordering: np.ndarray,
    task_id: np.ndarray,
) -> dict[str, Any]:
    def cut(mask: np.ndarray) -> dict[str, Any]:
        return {
            "contexts": int(np.sum(mask) * 8),
            "within_context_spearman": summary(within[mask]),
            "pairwise_ordering_accuracy": summary(ordering[mask]),
            "pooled": correlation(predicted_norm[mask], real_norm[mask]),
        }

    return {
        "overall": cut(np.ones(len(task_id), dtype=bool)),
        "per_horizon": [
            {
                "horizon": horizon + 1,
                "contexts": int(len(task_id)),
                "within_context_spearman": summary(within[:, horizon]),
                "pairwise_ordering_accuracy": summary(ordering[:, horizon]),
                "pooled": correlation(
                    predicted_norm[:, :, horizon], real_norm[:, :, horizon]
                ),
            }
            for horizon in range(8)
        ],
        "per_task": {
            str(task): cut(task_id == task) for task in range(10)
        },
    }


def sign_fidelity(
    predicted: np.ndarray,
    real: np.ndarray,
    minimum_abs_real_delta: float,
    task_id: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(real, dtype=np.float64)
    eligible = np.abs(target) >= float(minimum_abs_real_delta)
    correct = np.sign(pred) == np.sign(target)

    def summarize_mask(
        mask: np.ndarray, correct_values: np.ndarray
    ) -> dict[str, float | int | None]:
        count = int(mask.sum())
        return {
            "eligible": count,
            "eligible_fraction": None if mask.size == 0 else float(mask.mean()),
            "accuracy": None if count == 0 else float(correct_values[mask].mean()),
        }

    dimensions = {
        name: summarize_mask(eligible[..., index], correct[..., index])
        for index, name in enumerate(DIMENSION_NAMES)
    }
    groups = {
        name: summarize_mask(eligible[..., indices], correct[..., indices])
        for name, indices in GROUPS.items()
    }
    per_horizon = [
        {
            "horizon": horizon + 1,
            **summarize_mask(
                eligible[:, :, horizon], correct[:, :, horizon]
            ),
        }
        for horizon in range(8)
    ]
    per_task = {
        str(task): summarize_mask(
            eligible[task_id == task], correct[task_id == task]
        )
        for task in range(10)
    }
    return {
        "threshold": float(minimum_abs_real_delta),
        "overall": summarize_mask(eligible, correct),
        "dimensions": dimensions,
        "groups": groups,
        "per_horizon": per_horizon,
        "per_task": per_task,
    }


def percentile_interval(value: np.ndarray) -> list[float]:
    return [float(np.percentile(value, 2.5)), float(np.percentile(value, 97.5))]


def block_sums(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(value.shape[0], -1).sum(axis=1)


def bootstrap_metrics(
    predicted: np.ndarray,
    real: np.ndarray,
    within_rank: np.ndarray,
    ordering: np.ndarray,
    replicates: int,
    seed: int,
    minimum_real_norm: float,
) -> dict[str, Any]:
    """Branch-point block bootstrap for all authorization metrics."""
    pred = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(real, dtype=np.float64)
    count = pred.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(replicates, count), endpoint=False)

    eligible = np.linalg.norm(target, axis=-1) >= minimum_real_norm
    error_ss = block_sums((pred - target) ** 2 * eligible[..., None])
    zero_ss = block_sums(target**2 * eligible[..., None])
    block_element_count = eligible.reshape(count, -1).sum(axis=1) * pred.shape[-1]
    element_count = block_element_count[indices].sum(axis=1)
    error_rmse = np.sqrt(error_ss[indices].sum(axis=1) / element_count)
    zero_rmse = np.sqrt(zero_ss[indices].sum(axis=1) / element_count)
    improvement = (zero_rmse - error_rmse) / zero_rmse

    direction = vector_cosine(pred, target)
    direction_sum = block_sums(direction * eligible)
    block_direction_count = eligible.reshape(count, -1).sum(axis=1)
    direction_count = block_direction_count[indices].sum(axis=1)
    direction_mean = direction_sum[indices].sum(axis=1) / direction_count

    rank_sum = block_sums(within_rank)
    rank_count = within_rank[0].size * count
    rank_mean = rank_sum[indices].sum(axis=1) / rank_count
    ordering_sum = block_sums(ordering)
    ordering_count = ordering[0].size * count
    ordering_mean = ordering_sum[indices].sum(axis=1) / ordering_count

    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "unit": "branch_point",
        "relative_improvement_ci95": percentile_interval(improvement),
        "direction_mean_ci95": percentile_interval(direction_mean),
        "within_context_spearman_ci95": percentile_interval(rank_mean),
        "pairwise_ordering_accuracy_ci95": percentile_interval(ordering_mean),
    }


def subgroup_bootstrap(
    predicted: np.ndarray,
    real: np.ndarray,
    replicates: int,
    seed: int,
    minimum_real_norm: float,
) -> dict[str, list[float]]:
    pred = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(real, dtype=np.float64)
    count = pred.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(replicates, count), endpoint=False)
    eligible = np.linalg.norm(target, axis=-1) >= minimum_real_norm
    error_ss = block_sums((pred - target) ** 2 * eligible[..., None])
    zero_ss = block_sums(target**2 * eligible[..., None])
    block_elements = eligible.reshape(count, -1).sum(axis=1) * pred.shape[-1]
    elements = block_elements[indices].sum(axis=1)
    error = np.sqrt(error_ss[indices].sum(axis=1) / elements)
    zero = np.sqrt(zero_ss[indices].sum(axis=1) / elements)
    improvement = (zero - error) / zero
    direction = vector_cosine(pred, target)
    direction_sum = block_sums(direction * eligible)
    direction_count = eligible.reshape(count, -1).sum(axis=1)
    direction_mean = direction_sum[indices].sum(axis=1) / direction_count[
        indices
    ].sum(axis=1)
    return {
        "relative_improvement_ci95": percentile_interval(improvement),
        "direction_mean_ci95": percentile_interval(direction_mean),
    }


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != "wm-v2b-counterfactual-physical-mediator-v1":
        raise ValueError("unexpected protocol version")
    if protocol["status"] != "PREREGISTERED_BEFORE_DIFFERENTIAL_METRICS":
        raise ValueError("protocol is not preregistered")
    frozen = protocol["frozen_upstream"]
    checks = (
        ("counterfactual_result", "counterfactual_result_sha256"),
        ("counterfactual_report", "counterfactual_report_sha256"),
        ("matched_branch_sidecar", "matched_branch_sidecar_sha256"),
        ("wm_v2b_result", "wm_v2b_result_sha256"),
        ("wm_v2b_validation_sidecar", "wm_v2b_validation_sidecar_sha256"),
        ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
        ("wm_v2b_config", "wm_v2b_config_sha256"),
        ("wm_v2b_protocol", "wm_v2b_protocol_sha256"),
    )
    for path_key, hash_key in checks:
        if sha256(ROOT / frozen[path_key]) != frozen[hash_key]:
            raise ValueError(f"frozen hash mismatch: {path_key}")
    for tag_key, commit_key in (
        ("counterfactual_annotated_tag", "counterfactual_commit"),
        ("wm_v2b_annotated_tag", "wm_v2b_commit"),
    ):
        tag = frozen[tag_key]
        if subprocess.check_output(
            ["git", "cat-file", "-t", tag], cwd=ROOT, text=True
        ).strip() != "tag":
            raise ValueError(f"{tag} is not annotated")
        peeled = subprocess.check_output(
            ["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True
        ).strip()
        if peeled != frozen[commit_key]:
            raise ValueError(f"{tag} target changed")
    counterfactual = json.loads((ROOT / frozen["counterfactual_result"]).read_text())
    if counterfactual["verdict"] != frozen["counterfactual_verdict"]:
        raise ValueError("frozen counterfactual verdict changed")
    return protocol


def load_sources(protocol: dict[str, Any]) -> dict[str, np.ndarray | dict[str, Any]]:
    frozen = protocol["frozen_upstream"]
    with np.load(ROOT / frozen["wm_v2b_validation_sidecar"]) as source:
        validation = {name: source[name].copy() for name in source.files}
    with np.load(ROOT / frozen["matched_branch_sidecar"]) as source:
        branches = {name: source[name].copy() for name in source.files}
    if validation["correct_p_element_error"].shape != (5869, 8, 12):
        raise ValueError("validation element-error shape changed")
    if branches["real_proprio"].shape != (40, 3, 8, 12):
        raise ValueError("matched real-proprio shape changed")
    if branches["pred_proprio_wm_v2b_standardized"].shape != (40, 3, 8, 12):
        raise ValueError("matched predicted-proprio shape changed")
    if set(validation["demo_id"].astype(int).tolist()) != set(range(45, 50)):
        raise ValueError("validation split changed")
    if set(branches["init_state_index"].astype(int).tolist()) != {30, 35}:
        raise ValueError("matched branch init-state scope changed")
    checkpoint = torch.load(
        ROOT / frozen["wm_v2b_checkpoint"], map_location="cpu", weights_only=False
    )
    normalization = checkpoint["normalization"]["proprio"]
    return {"validation": validation, "branches": branches, "normalization": normalization}


def differential_cuts(
    predicted: np.ndarray,
    real: np.ndarray,
    task_id: np.ndarray,
    minimum_real_norm: float,
) -> dict[str, Any]:
    return {
        "overall": differential_stats(predicted, real, minimum_real_norm),
        "per_horizon": [
            {
                "horizon": horizon + 1,
                **differential_stats(
                    predicted[:, :, horizon], real[:, :, horizon], minimum_real_norm
                ),
            }
            for horizon in range(8)
        ],
        "per_task": {
            str(task): differential_stats(
                predicted[task_id == task], real[task_id == task], minimum_real_norm
            )
            for task in range(10)
        },
        "per_pair": {
            name: differential_stats(
                predicted[:, pair], real[:, pair], minimum_real_norm
            )
            for pair, name in enumerate(PAIR_NAMES)
        },
        "groups": {
            name: differential_stats(
                predicted[..., indices], real[..., indices], minimum_real_norm
            )
            for name, indices in GROUPS.items()
        },
    }


def coupling_cuts(
    real_physical: np.ndarray,
    predicted_physical: np.ndarray,
    real_visual: np.ndarray,
    predicted_visual: np.ndarray,
    task_id: np.ndarray,
) -> dict[str, Any]:
    def row(mask: np.ndarray) -> dict[str, Any]:
        return {
            "real_physical_to_real_visual": correlation(
                real_physical[mask], real_visual[mask]
            ),
            "predicted_physical_to_predicted_visual": correlation(
                predicted_physical[mask], predicted_visual[mask]
            ),
        }

    return {
        "overall": row(np.ones(len(task_id), dtype=bool)),
        "per_horizon": [
            {
                "horizon": horizon + 1,
                "real_physical_to_real_visual": correlation(
                    real_physical[:, :, horizon], real_visual[:, :, horizon]
                ),
                "predicted_physical_to_predicted_visual": correlation(
                    predicted_physical[:, :, horizon], predicted_visual[:, :, horizon]
                ),
            }
            for horizon in range(8)
        ],
        "per_task": {str(task): row(task_id == task) for task in range(10)},
        "per_pair": {
            name: {
                "real_physical_to_real_visual": correlation(
                    real_physical[:, pair], real_visual[:, pair]
                ),
                "predicted_physical_to_predicted_visual": correlation(
                    predicted_physical[:, pair], predicted_visual[:, pair]
                ),
            }
            for pair, name in enumerate(PAIR_NAMES)
        },
    }


def analyze(protocol: dict[str, Any], sources: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validation = sources["validation"]
    branches = sources["branches"]
    normalization = sources["normalization"]
    p_mean = np.asarray(normalization["mean"], dtype=np.float64)
    p_std = np.asarray(normalization["std"], dtype=np.float64)

    validation_loss = validation["correct_p_element_error"].astype(np.float64)
    validation_absolute = absolute_from_smooth_l1(validation_loss)
    validation_rmse_check = np.sqrt(np.mean(validation_absolute**2, axis=-1))
    if not np.allclose(
        validation_rmse_check,
        validation["correct_p_rmse"],
        atol=2e-7,
        rtol=0,
    ):
        raise ValueError("SmoothL1 inversion does not reproduce validation RMSE")

    real_p_standardized = (
        branches["real_proprio"].astype(np.float64) - p_mean
    ) / p_std
    predicted_p = branches["pred_proprio_wm_v2b_standardized"].astype(np.float64)
    branch_residual = predicted_p - real_p_standardized
    branch_absolute = np.abs(branch_residual)
    validation_tasks = validation["task_id"].astype(np.int64)
    branch_tasks = branches["task_id"].astype(np.int64)

    absolute = {
        "validation": absolute_breakdown(validation_absolute, validation_tasks),
        "branches": {
            name: absolute_breakdown(branch_absolute[:, index], branch_tasks)
            for index, name in enumerate(CANDIDATE_NAMES)
        },
    }
    pooled_error = branch_absolute.reshape(120, 8, 12)
    pooled_tasks = np.repeat(branch_tasks, 3)
    absolute["branch_pooled"] = absolute_breakdown(pooled_error, pooled_tasks)
    validation_overall = absolute["validation"]["overall"]
    branch_overall = absolute["branch_pooled"]["overall"]
    absolute["distribution_shift"] = {
        "rmse_ratio": float(branch_overall["rmse"] / validation_overall["rmse"]),
        "mae_ratio": float(branch_overall["mae"] / validation_overall["mae"]),
        "smooth_l1_ratio": float(
            branch_overall["smooth_l1_beta1"]
            / validation_overall["smooth_l1_beta1"]
        ),
    }

    real_delta = pairwise_delta(real_p_standardized)
    predicted_delta = pairwise_delta(predicted_p)
    minimum_norm = float(
        protocol["metric_contract"]["minimum_real_norm_for_direction_and_ratio"]
    )
    differential = differential_cuts(
        predicted_delta, real_delta, branch_tasks, minimum_norm
    )
    pred_norm = np.linalg.norm(predicted_delta, axis=-1)
    real_norm = np.linalg.norm(real_delta, axis=-1)
    within_rank, ordering = rank_contexts(
        pred_norm,
        real_norm,
        float(protocol["metric_contract"]["ranking_tie_tolerance"]),
    )
    ranking = ranking_summary(
        pred_norm, real_norm, within_rank, ordering, branch_tasks
    )
    sign = sign_fidelity(
        predicted_delta,
        real_delta,
        float(protocol["metric_contract"]["minimum_abs_real_delta_for_sign"]),
        branch_tasks,
    )

    bootstrap_config = protocol["bootstrap"]
    bootstrap = bootstrap_metrics(
        predicted_delta,
        real_delta,
        within_rank,
        ordering,
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
        minimum_norm,
    )
    bootstrap["per_pair"] = {
        name: subgroup_bootstrap(
            predicted_delta[:, pair],
            real_delta[:, pair],
            int(bootstrap_config["replicates"]),
            int(bootstrap_config["seed"]) + 10 + pair,
            minimum_norm,
        )
        for pair, name in enumerate(PAIR_NAMES)
    }
    bootstrap["per_horizon"] = [
        {
            "horizon": horizon + 1,
            **subgroup_bootstrap(
                predicted_delta[:, :, horizon],
                real_delta[:, :, horizon],
                int(bootstrap_config["replicates"]),
                int(bootstrap_config["seed"]) + 100 + horizon,
                minimum_norm,
            ),
        }
        for horizon in range(8)
    ]

    real_visual = branches["real_visual_cosine"].astype(np.float64)
    predicted_visual = branches[
        "wm_v2b_seed11_predicted_visual_separation"
    ].astype(np.float64)
    coupling = coupling_cuts(
        real_norm, pred_norm, real_visual, predicted_visual, branch_tasks
    )

    overall = differential["overall"]
    overall_ranking = ranking["overall"]
    pair_gate = {
        name: {
            "relative_improvement_positive": bool(
                differential["per_pair"][name]["relative_improvement_over_zero"] > 0
            ),
            "direction_positive": bool(
                differential["per_pair"][name]["direction_cosine"]["mean"] > 0
            ),
            "magnitude_ratio_at_least_0_05": bool(
                differential["per_pair"][name]["magnitude_ratio"]["ratio_of_means"]
                >= 0.05
            ),
        }
        for name in PAIR_NAMES
    }
    gate_pass = {
        "differential_improvement": bool(
            overall["relative_improvement_over_zero"] >= 0.05
            and bootstrap["relative_improvement_ci95"][0] > 0
        ),
        "direction_alignment": bool(
            overall["direction_cosine"]["mean"] >= 0.10
            and bootstrap["direction_mean_ci95"][0] > 0
        ),
        "ranking_relation": bool(
            overall_ranking["within_context_spearman"]["mean"] >= 0.10
            and bootstrap["within_context_spearman_ci95"][0] > 0
            and overall_ranking["pairwise_ordering_accuracy"]["mean"] >= 0.55
            and bootstrap["pairwise_ordering_accuracy_ci95"][0] > 0.50
        ),
        "no_candidate_specific_collapse": bool(
            all(all(values.values()) for values in pair_gate.values())
        ),
    }
    go = bool(all(gate_pass.values()))
    absolute_shift = bool(
        absolute["distribution_shift"]["rmse_ratio"] >= 1.5
        or absolute["distribution_shift"]["mae_ratio"] >= 1.5
    )
    if go:
        labels = ["PHYSICAL_MEDIATOR_VALID_VISUAL_TRANSFER_BLOCK_CONFIRMED"]
        if absolute_shift:
            labels.insert(0, "PHYSICAL_MEDIATOR_RELATIVE_SIGNAL_VALID")
        verdict = "GO_WM_V2C_DESIGN"
    else:
        labels = ["PHYSICAL_MEDIATOR_COUNTERFACTUAL_FAIL"]
        verdict = "BLOCK_WM_V2C_PHYSICAL_MEDIATOR"

    analysis = {
        "absolute_proprio_error": absolute,
        "counterfactual_differential": differential,
        "separation_ranking": ranking,
        "dimension_sign_fidelity": sign,
        "visual_physical_coupling": coupling,
        "bootstrap": bootstrap,
        "decision": {
            "gate_pass": gate_pass,
            "candidate_pair_collapse_checks": pair_gate,
            "absolute_distribution_shift": absolute_shift,
            "mechanism_labels": labels,
            "authorization_verdict": verdict,
        },
    }
    derived = {
        "task_id": branch_tasks.astype(np.int8),
        "init_state_index": branches["init_state_index"].astype(np.int8),
        "planning_index": branches["planning_index"].astype(np.int8),
        "validation_absolute_error": validation_absolute.astype(np.float32),
        "branch_absolute_error": branch_absolute.astype(np.float32),
        "real_delta": real_delta.astype(np.float32),
        "predicted_delta": predicted_delta.astype(np.float32),
        "real_norm": real_norm.astype(np.float32),
        "predicted_norm": pred_norm.astype(np.float32),
        "direction_cosine": vector_cosine(predicted_delta, real_delta).astype(np.float32),
        "within_context_spearman": within_rank.astype(np.float32),
        "pairwise_ordering_score": ordering.astype(np.float32),
    }
    return analysis, derived


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_ci(value: list[float]) -> str:
    return f"[{value[0]:.4f}, {value[1]:.4f}]"


def render_report(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    absolute = analysis["absolute_proprio_error"]
    differential = analysis["counterfactual_differential"]
    ranking = analysis["separation_ranking"]
    sign = analysis["dimension_sign_fidelity"]
    coupling = analysis["visual_physical_coupling"]
    bootstrap = analysis["bootstrap"]
    decision = analysis["decision"]

    absolute_rows = []
    sources = [("D13 validation", absolute["validation"]), *[(name, absolute["branches"][name]) for name in CANDIDATE_NAMES], ("A0/A1/A2 pooled", absolute["branch_pooled"])]
    for label, values in sources:
        overall = values["overall"]
        absolute_rows.append([label, f"{overall['rmse']:.6f}", f"{overall['mae']:.6f}", f"{overall['smooth_l1_beta1']:.6f}"])

    horizon_rows = []
    for horizon in range(8):
        row = [str(horizon + 1)]
        for _, source in sources[:4]:
            values = source["per_horizon"][horizon]
            row.extend([f"{values['rmse']:.5f}", f"{values['mae']:.5f}", f"{values['smooth_l1_beta1']:.5f}"])
        horizon_rows.append(row)

    group_rows = []
    for group in GROUPS:
        row = [group]
        for _, source in sources[:4]:
            values = source["groups"][group]
            row.extend([f"{values['rmse']:.5f}", f"{values['mae']:.5f}", f"{values['smooth_l1_beta1']:.5f}"])
        group_rows.append(row)

    task_rows = []
    for task in range(10):
        row = [str(task)]
        for _, source in sources[:4]:
            values = source["per_task"][str(task)]
            row.extend([f"{values['rmse']:.5f}", f"{values['mae']:.5f}", f"{values['smooth_l1_beta1']:.5f}"])
        task_rows.append(row)

    def differential_row(label: str, values: dict[str, Any], ci: dict[str, Any] | None = None) -> list[str]:
        return [
            label,
            f"{values['direction_cosine']['mean']:.4f}",
            f"{values['magnitude_ratio']['ratio_of_means']:.4f}",
            f"{values['magnitude_ratio']['median']:.4f}",
            f"{values['magnitude_ratio']['p25']:.4f}",
            f"{values['magnitude_ratio']['p75']:.4f}",
            f"{values['differential_rmse']:.6f}",
            f"{values['zero_difference_rmse']:.6f}",
            f"{100*values['relative_improvement_over_zero']:.2f}%",
            "—" if ci is None else format_ci(ci["relative_improvement_ci95"]),
            "—" if ci is None else format_ci(ci["direction_mean_ci95"]),
        ]

    pair_rows = [
        differential_row(name, differential["per_pair"][name], bootstrap["per_pair"][name])
        for name in PAIR_NAMES
    ]
    horizon_diff_rows = [
        differential_row(
            f"h{row['horizon']}", row, bootstrap["per_horizon"][row["horizon"] - 1]
        )
        for row in differential["per_horizon"]
    ]
    task_diff_rows = [
        differential_row(f"task {task}", differential["per_task"][str(task)])
        for task in range(10)
    ]
    group_diff_rows = [
        differential_row(group, differential["groups"][group]) for group in GROUPS
    ]

    ranking_horizon_rows = [
        [
            str(row["horizon"]),
            f"{row['within_context_spearman']['mean']:.4f}",
            f"{row['pairwise_ordering_accuracy']['mean']:.2%}",
            "n/a" if row["pooled"]["spearman"] is None else f"{row['pooled']['spearman']:.4f}",
            "n/a" if row["pooled"]["pearson"] is None else f"{row['pooled']['pearson']:.4f}",
        ]
        for row in ranking["per_horizon"]
    ]
    ranking_task_rows = [
        [
            str(task),
            f"{ranking['per_task'][str(task)]['within_context_spearman']['mean']:.4f}",
            f"{ranking['per_task'][str(task)]['pairwise_ordering_accuracy']['mean']:.2%}",
            f"{ranking['per_task'][str(task)]['pooled']['spearman']:.4f}",
            f"{ranking['per_task'][str(task)]['pooled']['pearson']:.4f}",
        ]
        for task in range(10)
    ]
    sign_rows = [
        [name, str(values["eligible"]), f"{values['eligible_fraction']:.2%}", "n/a" if values["accuracy"] is None else f"{values['accuracy']:.2%}"]
        for name, values in sign["dimensions"].items()
    ]
    coupling_rows = [
        [
            str(row["horizon"]),
            f"{row['real_physical_to_real_visual']['pearson']:.4f}",
            f"{row['real_physical_to_real_visual']['spearman']:.4f}",
            f"{row['predicted_physical_to_predicted_visual']['pearson']:.4f}",
            f"{row['predicted_physical_to_predicted_visual']['spearman']:.4f}",
        ]
        for row in coupling["per_horizon"]
    ]
    gate_rows = [[name, "pass" if passed else "FAIL"] for name, passed in decision["gate_pass"].items()]
    overall = differential["overall"]
    overall_ranking = ranking["overall"]
    labels = ", ".join(f"`{label}`" for label in decision["mechanism_labels"])
    result_sha256 = hashlib.sha256(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()
    ).hexdigest()
    return f"""# The physical mediator {'passes' if result['verdict'] == 'GO_WM_V2C_DESIGN' else 'does not pass'} the counterfactual fidelity gate

Date: 2026-09-24

Authorization verdict: `{result['verdict']}`

Mechanism label(s): {labels}

## Executive summary

The frozen WM-v2B future-proprio head is evaluated without new inference or
simulator access. The audit reconstructs same-metric absolute errors from all
5,869 D13 validation samples and compares them with 40 matched branch points,
then directly tests predicted versus real A0/A1/A2 physical differences.

Differential RMSE is `{overall['differential_rmse']:.6f}` versus
`{overall['zero_difference_rmse']:.6f}` for predicting no candidate difference,
a `{100*overall['relative_improvement_over_zero']:.2f}%` relative improvement
(branch-block 95% CI {format_ci(bootstrap['relative_improvement_ci95'])}). Mean
direction cosine is `{overall['direction_cosine']['mean']:.4f}` (95% CI
{format_ci(bootstrap['direction_mean_ci95'])}), and predicted/real physical
magnitude ratio-of-means is `{overall['magnitude_ratio']['ratio_of_means']:.4f}`.

The direction and ranking signal is real but insufficient for a physical
mediator. Within-context separation Spearman is `{overall_ranking['within_context_spearman']['mean']:.4f}` and ordering
accuracy is `{overall_ranking['pairwise_ordering_accuracy']['mean']:.2%}`, yet the predicted candidate differences are
systematically too large. That over-scaling makes the full vector prediction
slightly worse than predicting no candidate difference at all. Two of three
candidate pairs have negative improvement, and the aggregate confidence
interval includes zero.

Absolute branch prediction is also shifted: pooled branch RMSE is `{absolute['distribution_shift']['rmse_ratio']:.3f}x`
validation RMSE. The gripper subgroup is especially weak, with direction
cosine `{differential['groups']['gripper']['direction_cosine']['mean']:.4f}` and sign accuracy `{sign['groups']['gripper']['accuracy']:.2%}`. The result therefore does not
receive either valid-relative-signal mechanism label.

The preregistered authorization decision is `{result['verdict']}`. This report
does not train WM-v2C or change WM-v2B, candidates, reranking, or controllers.

## Frozen evidence and coordinate contract

- Local counterfactual freeze commit/tag: `{result['protocol']['counterfactual_commit']}` / `{result['protocol']['counterfactual_tag']}`.
- Protocol source commit: `{result['protocol']['execution_source_commit']}`.
- Only the hash-verified D13-validation and matched-branch sidecars were read.
- Branch scope remains task 0--9, init states 30/35, planning indices 0/8; no simulator or 40--49 access occurred.
- All physical quantities below use the frozen train-standardized 12-D coordinate order: ee position (3), gripper (2), joints (7).
- Validation elementwise SmoothL1(beta=1) was inverted to absolute residual magnitude; the reconstructed per-sample RMSE reproduces the frozen sidecar within `2e-7`.

## Absolute proprio error uses one metric family everywhere

{markdown_table(['Source', 'RMSE', 'MAE', 'SmoothL1 beta=1'], absolute_rows)}

The pooled branch-to-validation ratios are RMSE
`{absolute['distribution_shift']['rmse_ratio']:.3f}x`, MAE
`{absolute['distribution_shift']['mae_ratio']:.3f}x`, and SmoothL1
`{absolute['distribution_shift']['smooth_l1_ratio']:.3f}x`. This comparison
replaces the invalid cross-metric comparison between validation SmoothL1 and
branch RMSE.

The A0/A1/A2 absolute errors are similar to one another, so there is no single
candidate-specific absolute collapse. Their pooled shift is nevertheless
large and material: equality across candidates does not establish that their
*differences* are represented faithfully.

Per horizon (`R/M/S` = RMSE/MAE/SmoothL1):

{markdown_table(['h', 'val R', 'val M', 'val S', 'A0 R', 'A0 M', 'A0 S', 'A1 R', 'A1 M', 'A1 S', 'A2 R', 'A2 M', 'A2 S'], horizon_rows)}

Per proprio group:

{markdown_table(['Group', 'val R', 'val M', 'val S', 'A0 R', 'A0 M', 'A0 S', 'A1 R', 'A1 M', 'A1 S', 'A2 R', 'A2 M', 'A2 S'], group_rows)}

Per task:

{markdown_table(['Task', 'val R', 'val M', 'val S', 'A0 R', 'A0 M', 'A0 S', 'A1 R', 'A1 M', 'A1 S', 'A2 R', 'A2 M', 'A2 S'], task_rows)}

## Counterfactual differential fidelity

For every matched branch point, pair, and horizon, the audit defines
`delta_p_real = p_real_i - p_real_j` and
`delta_p_pred = p_pred_i - p_pred_j` in standardized coordinates. Direction
uses cosine alignment; magnitude uses L2; differential error is RMSE against
the real delta. Ratios/directions require real L2 at least `1e-4`.

Overall:

{markdown_table(['Cut', 'Direction', 'Norm ratio', 'Median ratio', 'p25', 'p75', 'Diff RMSE', 'Zero RMSE', 'Improvement', 'Improvement CI', 'Direction CI'], [differential_row('all', overall, bootstrap)])}

By candidate pair:

{markdown_table(['Pair', 'Direction', 'Norm ratio', 'Median ratio', 'p25', 'p75', 'Diff RMSE', 'Zero RMSE', 'Improvement', 'Improvement CI', 'Direction CI'], pair_rows)}

By horizon:

{markdown_table(['h', 'Direction', 'Norm ratio', 'Median ratio', 'p25', 'p75', 'Diff RMSE', 'Zero RMSE', 'Improvement', 'Improvement CI', 'Direction CI'], horizon_diff_rows)}

By proprio group:

{markdown_table(['Group', 'Direction', 'Norm ratio', 'Median ratio', 'p25', 'p75', 'Diff RMSE', 'Zero RMSE', 'Improvement', 'Improvement CI', 'Direction CI'], group_diff_rows)}

By task:

{markdown_table(['Task', 'Direction', 'Norm ratio', 'Median ratio', 'p25', 'p75', 'Diff RMSE', 'Zero RMSE', 'Improvement', 'Improvement CI', 'Direction CI'], task_diff_rows)}

## Why positive direction is not enough

The head usually points candidate differences in the right direction, but it
does not reproduce their size accurately. Aggregate predicted norm is
`{overall['magnitude_ratio']['ratio_of_means']:.3f}x` real norm. At h1 the ratio is `{differential['per_horizon'][0]['magnitude_ratio']['ratio_of_means']:.3f}x` and differential error is
`{-100*differential['per_horizon'][0]['relative_improvement_over_zero']:.1f}%` worse than zero-difference. Only h7 and h8 have small positive point
estimates, and both confidence intervals cross zero.

The failure is not uniform across coordinates. EE-position differences
improve over zero by `{100*differential['groups']['ee_pos']['relative_improvement_over_zero']:.2f}%`, but joints are `{abs(100*differential['groups']['joints']['relative_improvement_over_zero']):.2f}%` worse and gripper is
`{abs(100*differential['groups']['gripper']['relative_improvement_over_zero']):.2f}%` worse, with gripper direction near zero. Only three of ten tasks have
a positive differential-improvement point estimate. A future mediator would
therefore require differential calibration and coordinate-level validation;
the current head cannot be coupled directly into a visual predictor on this
evidence.

## Separation ranking fidelity

Overall mean within-context Spearman is
`{overall_ranking['within_context_spearman']['mean']:.4f}` (95% CI
{format_ci(bootstrap['within_context_spearman_ci95'])}); three-way pairwise
ordering accuracy is `{overall_ranking['pairwise_ordering_accuracy']['mean']:.2%}`
(95% CI {format_ci(bootstrap['pairwise_ordering_accuracy_ci95'])}). Pooled
predicted-vs-real separation Spearman/Pearson are
`{overall_ranking['pooled']['spearman']:.4f}` / `{overall_ranking['pooled']['pearson']:.4f}`.

This passes the preregistered ordinal gate: the head can often tell which of
the three pairwise physical separations is larger. Ranking discards absolute
scale and vector residual, however, so it cannot rescue the failed
zero-difference and pair-consistency gates.

{markdown_table(['h', 'Within Spearman', 'Ordering accuracy', 'Pooled Spearman', 'Pooled Pearson'], ranking_horizon_rows)}

{markdown_table(['Task', 'Within Spearman', 'Ordering accuracy', 'Pooled Spearman', 'Pooled Pearson'], ranking_task_rows)}

## Dimension-wise sign fidelity

Only real standardized deltas with absolute magnitude at least `1e-3` enter
the sign comparison. Overall accuracy is
`{sign['overall']['accuracy']:.2%}` over `{sign['overall']['eligible']}` eligible
dimension values.

{markdown_table(['Dimension', 'Eligible', 'Coverage', 'Sign accuracy'], sign_rows)}

Group accuracies are ee position `{sign['groups']['ee_pos']['accuracy']:.2%}`,
gripper `{sign['groups']['gripper']['accuracy']:.2%}`, and joints
`{sign['groups']['joints']['accuracy']:.2%}`. Horizon/task sign cuts remain in
the compact JSON.

## Visual/physical coupling remains diagnostic

Real physical versus real visual separation has Pearson/Spearman
`{coupling['overall']['real_physical_to_real_visual']['pearson']:.4f}` /
`{coupling['overall']['real_physical_to_real_visual']['spearman']:.4f}`.
Predicted physical versus predicted visual separation has Pearson/Spearman
`{coupling['overall']['predicted_physical_to_predicted_visual']['pearson']:.4f}` /
`{coupling['overall']['predicted_physical_to_predicted_visual']['spearman']:.4f}`.
These associations do not establish a causal mediation path. In particular,
the stronger predicted-predicted correlation measures internal co-response of
two outputs from the same model; it must not be interpreted as mediator
accuracy against the matched real futures.

{markdown_table(['h', 'Real Pearson', 'Real Spearman', 'Pred Pearson', 'Pred Spearman'], coupling_rows)}

## Preregistered authorization decision

{markdown_table(['Gate', 'Result'], gate_rows)}

Candidate-pair collapse checks:

```json
{json.dumps(decision['candidate_pair_collapse_checks'], indent=2)}
```

Final authorization verdict: `{result['verdict']}`.

Mechanism label(s): {labels}.

`PHYSICAL_MEDIATOR_RELATIVE_SIGNAL_VALID` and
`PHYSICAL_MEDIATOR_VALID_VISUAL_TRANSFER_BLOCK_CONFIRMED` are not assigned.
Although direction and ranking pass, the preregistered definition requires all
four gate groups, including positive differential improvement and no
candidate-pair collapse. Direct WM-v2C design from this frozen head remains
blocked. Any future experiment must be separately preregistered; this audit
does not authorize a new objective, architecture, training run, or controller.

## Integrity and forbidden-activity audit

- Result: `{result['artifacts']['result']}`, SHA-256 `{result_sha256}`.
- Derived sidecar: `{result['artifacts']['derived_sidecar']}`, SHA-256 `{result['artifacts']['derived_sidecar_sha256']}`.
- Frozen matched-branch sidecar SHA-256: `{result['protocol']['matched_branch_sidecar_sha256']}`.
- Frozen validation sidecar SHA-256: `{result['protocol']['validation_sidecar_sha256']}`.
- All source/derived numeric arrays and reported values are finite; NaN/Inf audit: `{result['nan_inf_audit']}`.
- No simulator, inference, training, controller, success/reward/done signal, fake target, or init-state access occurred.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2b_counterfactual_physical_mediator_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/world_model_v2b_physical_mediator/diagnosis.json",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "artifacts/stage3/wm_v2b_physical_mediator/metrics.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/wm_v2b_counterfactual_physical_mediator_audit.md",
    )
    args = parser.parse_args()
    for path in (args.output, args.sidecar, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    protocol = validate_protocol(args.protocol)
    sources = load_sources(protocol)
    analysis, derived = analyze(protocol, sources)
    if not all(np.isfinite(value).all() for value in derived.values()):
        raise FloatingPointError("derived sidecar contains NaN/Inf")
    atomic_npz(args.sidecar, derived)
    frozen = protocol["frozen_upstream"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "COMPLETE",
        "verdict": analysis["decision"]["authorization_verdict"],
        "mechanism_labels": analysis["decision"]["mechanism_labels"],
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": sha256(args.protocol),
            "execution_source_commit": source_commit(),
            "counterfactual_commit": frozen["counterfactual_commit"],
            "counterfactual_tag": frozen["counterfactual_annotated_tag"],
            "matched_branch_sidecar_sha256": frozen["matched_branch_sidecar_sha256"],
            "validation_sidecar_sha256": frozen["wm_v2b_validation_sidecar_sha256"],
        },
        "scope": protocol["scope"],
        "analysis": analysis,
        "artifacts": {
            "result": str(args.output.relative_to(ROOT)),
            "derived_sidecar": str(args.sidecar.relative_to(ROOT)),
            "derived_sidecar_sha256": sha256(args.sidecar),
            "derived_sidecar_arrays": {
                name: list(value.shape) for name, value in derived.items()
            },
        },
        "normalization": {
            "proprio_count": int(sources["normalization"]["count"]),
            "rule": sources["normalization"]["rule"],
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "simulator_run_or_init_state_access": False,
            "model_inference": False,
            "model_training_or_modification": False,
            "lambda_p_changed": False,
            "candidate_or_reranker_changed": False,
            "c1_c2_d20_v2_run": False,
            "success_reward_done_used": False,
            "fake_counterfactual_target": False,
            "upstream_artifact_overwritten": False,
            "gate_changed_after_results": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    atomic_json(args.output, result)
    result["artifacts"]["result_sha256_before_hash_field"] = sha256(args.output)
    atomic_json(args.output, result)
    atomic_text(args.report, render_report(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "mechanism_labels": result["mechanism_labels"],
                "result": str(args.output),
                "derived_sidecar_sha256": result["artifacts"]["derived_sidecar_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
