"""Helpers for the frozen expanded one-shot candidate-interface audit.

The only objective used here is expert-action MSE.  It is an offline
diagnostic oracle, never a deployment score, reward, or success measure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.probes.planning_architecture import (
    ACTION_DIM,
    GRIPPER,
    HORIZON,
    OfflineContext,
    antithetic_noise,
    bootstrap_gains,
    summarize_family,
)


def continuous_candidates(
    default: np.ndarray,
    unit: np.ndarray,
    count: int,
    std: np.ndarray,
    scale: float,
    bank_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return A0 plus a fixed number of continuous non-gripper proposals."""

    alternatives = count - 1
    if alternatives < 1:
        raise ValueError("count must include A0 and at least one alternative")
    if bank_style == "antithetic":
        noise = antithetic_noise(unit, alternatives)
    elif bank_style == "independent":
        if alternatives > unit.shape[1]:
            raise ValueError("unit bank too small")
        noise = unit[:, :alternatives]
    else:
        raise ValueError(bank_style)
    multiplier = np.asarray(std, dtype=np.float32).copy()
    multiplier[6] = 0.0
    proposed = default[:, None] + noise * multiplier[None, None, None] * np.float32(scale)
    preclip = np.concatenate((default[:, None], proposed), axis=1).astype(np.float32)
    candidates = np.clip(preclip, -1.0, 1.0).astype(np.float32)
    category = np.concatenate((np.zeros(1, dtype=np.int8), np.ones(alternatives, dtype=np.int8)))
    forms = ["a0"] + [f"continuous_{bank_style}_{index:02d}" for index in range(alternatives)]
    return preclip, candidates, category, forms


def _binarize_gripper(default: np.ndarray) -> np.ndarray:
    return np.where(default[:, :, 6] >= 0.0, 1.0, -1.0).astype(np.float32)


def gripper_form_bank(default: np.ndarray) -> dict[str, np.ndarray]:
    """Construct the finite binary-like gripper family.

    Switches occur after steps 1,2,4,6,7.  Constant endpoints cover the
    otherwise duplicate switch-at-boundary cases.  C0 sign ties map to +1.
    """

    samples = len(default)
    bank: dict[str, np.ndarray] = {
        "c0_binarized": _binarize_gripper(default),
        "all_minus": -np.ones((samples, HORIZON), dtype=np.float32),
        "all_plus": np.ones((samples, HORIZON), dtype=np.float32),
    }
    for boundary in (1, 2, 4, 6, 7):
        negative_to_positive = np.ones((samples, HORIZON), dtype=np.float32)
        negative_to_positive[:, :boundary] = -1.0
        positive_to_negative = -negative_to_positive
        bank[f"minus_to_plus_after_{boundary}"] = negative_to_positive
        bank[f"plus_to_minus_after_{boundary}"] = positive_to_negative
    return bank


ALLOCATION_ORDER = (
    "all_minus",
    "all_plus",
    "minus_to_plus_after_4",
    "plus_to_minus_after_4",
    "minus_to_plus_after_2",
    "plus_to_minus_after_2",
    "minus_to_plus_after_6",
    "plus_to_minus_after_6",
    "minus_to_plus_after_1",
    "plus_to_minus_after_1",
    "minus_to_plus_after_7",
    "plus_to_minus_after_7",
)


def structured_gripper_candidates(
    default: np.ndarray,
    form_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    bank = gripper_form_bank(default)
    candidates = np.repeat(default[:, None], 1 + len(form_names), axis=1).astype(np.float32)
    for index, name in enumerate(form_names, 1):
        candidates[:, index, :, 6] = bank[name]
    category = np.concatenate((np.zeros(1, dtype=np.int8), np.full(len(form_names), 2, dtype=np.int8)))
    return candidates.copy(), candidates, category, ["a0", *form_names]


def combined_candidates(
    default: np.ndarray,
    unit: np.ndarray,
    total_count: int,
    gripper_count: int,
    std: np.ndarray,
    scale: float,
    bank_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if gripper_count not in (0, 4, 8, 12):
        raise ValueError("frozen allocation must use 0/4/8/12 gripper proposals")
    continuous_count = total_count - 1 - gripper_count
    if continuous_count < 1:
        raise ValueError("allocation leaves no continuous proposal")
    pre, candidates, category, forms = continuous_candidates(
        default, unit, 1 + continuous_count, std, scale, bank_style
    )
    if gripper_count == 0:
        return pre, candidates, category, forms
    bank = gripper_form_bank(default)
    selected_forms = ALLOCATION_ORDER[:gripper_count]
    gripper = np.repeat(default[:, None], gripper_count, axis=1).astype(np.float32)
    for index, name in enumerate(selected_forms):
        gripper[:, index, :, 6] = bank[name]
    return (
        np.concatenate((pre, gripper), axis=1),
        np.concatenate((candidates, gripper), axis=1),
        np.concatenate((category, np.full(gripper_count, 2, dtype=np.int8))),
        forms + list(selected_forms),
    )


def _relative_gain(default: np.ndarray, oracle: np.ndarray) -> float:
    return float((np.mean(default) - np.mean(oracle)) / np.mean(default))


def summarize_interface(
    name: str,
    preclip: np.ndarray,
    candidates: np.ndarray,
    category: np.ndarray,
    forms: list[str],
    context: OfflineContext,
    demo_values: tuple[int, ...],
    draws: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary, arrays = summarize_family(name, preclip, candidates, context, demo_values, draws)
    if category.shape != (candidates.shape[1],) or len(forms) != candidates.shape[1]:
        raise ValueError("candidate metadata shape mismatch")
    winner = arrays["winner"].astype(np.int64)
    winner_category = category[winner]
    errors = np.square(candidates.astype(np.float64) - context.expert[:, None].astype(np.float64)).mean(axis=(2, 3))
    default_error = errors[:, 0]
    full_selected = candidates[np.arange(len(candidates)), winner]

    category_names = {0: "a0", 1: "continuous", 2: "structured_gripper"}
    summary["candidate_categories"] = {
        category_names[value]: int(np.sum(category == value)) for value in category_names
    }
    summary["winner_category_fraction"] = {
        category_names[value]: float(np.mean(winner_category == value)) for value in category_names
    }
    summary["winner_form_fraction"] = {
        form: float(np.mean(winner == index)) for index, form in enumerate(forms)
    }
    summary["continuous_candidate_count"] = int(np.sum(category == 1))
    summary["structured_gripper_candidate_count"] = int(np.sum(category == 2))

    gripper_error = np.square(candidates[:, :, :, 6].astype(np.float64) - context.expert[:, None, :, 6].astype(np.float64)).mean(axis=2)
    gripper_winner = np.argmin(gripper_error, axis=1)
    gripper_selected = candidates[np.arange(len(candidates)), gripper_winner, :, 6]
    summary["gripper_diagnostics"] = {
        "oracle_relative_mse_gain": _relative_gain(gripper_error[:, 0], gripper_error[np.arange(len(candidates)), gripper_winner]),
        "oracle_sign_accuracy": float(np.mean(np.where(gripper_selected >= 0, 1.0, -1.0) == context.expert[:, :, 6])),
        "full_winner_sign_accuracy": float(np.mean(np.where(full_selected[:, :, 6] >= 0, 1.0, -1.0) == context.expert[:, :, 6])),
        "structured_full_winner_fraction": float(np.mean(winner_category == 2)),
        "switch_form_full_winner_fraction": float(np.mean(["_to_" in forms[index] for index in winner])),
        "binary_expert_fraction": float(np.mean(np.isin(context.expert[:, :, 6], (-1.0, 1.0)))),
    }

    flat = candidates.reshape(len(candidates), candidates.shape[1], -1).astype(np.float64)
    if candidates.shape[1] > 1:
        distance = np.sqrt(np.mean(np.square(flat[:, :, None] - flat[:, None, :]), axis=3))
        diagonal = np.arange(candidates.shape[1])
        distance[:, diagonal, diagonal] = np.inf
        nearest = distance.min(axis=2)
        duplicate_pairs = np.triu(distance <= 1.0e-8, 1)
        total_pairs = candidates.shape[1] * (candidates.shape[1] - 1) / 2
        duplicate_fraction = float(duplicate_pairs.sum() / (len(candidates) * total_pairs))
        nearest_mean = float(nearest.mean())
        near_duplicate_fraction = float(np.mean(nearest <= 1.0e-6))
    else:
        duplicate_fraction = nearest_mean = near_duplicate_fraction = 0.0
    summary["coverage"] = {
        "mean_nearest_neighbor_action_rms": nearest_mean,
        "candidate_near_duplicate_fraction": near_duplicate_fraction,
        "exact_duplicate_pair_fraction": duplicate_fraction,
    }
    outside = np.abs(preclip) > 1.0
    summary["candidate_clip_fraction"] = float(np.mean(np.any(outside, axis=(2, 3))))
    summary["gain_per_candidate"] = float(summary["relative_mse_gain"] / candidates.shape[1])
    summary["per_task_prefix_h1_gain"] = {
        str(task): _relative_gain(
            np.square(candidates[context.task_id == task, 0, 0].astype(np.float64) - context.expert[context.task_id == task, 0].astype(np.float64)).mean(axis=1),
            np.square(candidates[context.task_id == task, :, 0].astype(np.float64) - context.expert[context.task_id == task, None, 0].astype(np.float64)).mean(axis=2).min(axis=1),
        )
        for task in range(10)
    }
    arrays["winner_category"] = winner_category.astype(np.int8)
    arrays["category"] = category.astype(np.int8)
    arrays["gripper_winner"] = gripper_winner.astype(np.int16)
    return summary, arrays


def paired_delta(
    child: dict[str, Any],
    parent: dict[str, Any],
    child_bootstrap: np.ndarray,
    parent_bootstrap: np.ndarray,
) -> dict[str, Any]:
    values = child_bootstrap - parent_bootstrap
    return {
        "point": float(child["relative_mse_gain"] - parent["relative_mse_gain"]),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }
