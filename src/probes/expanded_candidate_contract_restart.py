"""Exact N64 candidate-contract correction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.probes.expanded_candidate_interface import summarize_interface
from src.probes.planning_architecture import HORIZON, OfflineContext


SWITCH_ORDER = (
    "minus_to_plus_after_4", "plus_to_minus_after_4",
    "minus_to_plus_after_3", "plus_to_minus_after_3",
    "minus_to_plus_after_5", "plus_to_minus_after_5",
    "minus_to_plus_after_2", "plus_to_minus_after_2",
    "minus_to_plus_after_6", "plus_to_minus_after_6",
    "minus_to_plus_after_1", "plus_to_minus_after_1",
    "minus_to_plus_after_7", "plus_to_minus_after_7",
)
MANDATORY_GRIPPER = ("c0_binarized", "all_negative", "all_positive")


def gripper_universe(default: np.ndarray) -> dict[str, np.ndarray]:
    samples = len(default)
    bank: dict[str, np.ndarray] = {
        "c0_binarized": np.where(default[:, :, 6] >= 0.0, 1.0, -1.0).astype(np.float32),
        "all_negative": -np.ones((samples, HORIZON), dtype=np.float32),
        "all_positive": np.ones((samples, HORIZON), dtype=np.float32),
    }
    for boundary in range(1, HORIZON):
        minus_plus = np.ones((samples, HORIZON), dtype=np.float32)
        minus_plus[:, :boundary] = -1.0
        bank[f"minus_to_plus_after_{boundary}"] = minus_plus
        bank[f"plus_to_minus_after_{boundary}"] = -minus_plus
    return bank


def exact_antithetic_continuous(
    default: np.ndarray,
    unit: np.ndarray,
    alternatives: int,
    std: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if alternatives % 2:
        raise ValueError("continuous alternatives must be even")
    pairs = alternatives // 2
    if pairs > unit.shape[1]:
        raise ValueError("unit bank too small")
    deviations = []
    forms = []
    multiplier = np.asarray(std, dtype=np.float32).copy()
    multiplier[6] = 0.0
    for index in range(pairs):
        epsilon = unit[:, index] * multiplier[None, None] * np.float32(scale)
        deviations.extend((epsilon, -epsilon))
        forms.extend((f"plus_epsilon_{index:02d}", f"minus_epsilon_{index:02d}"))
    proposed = default[:, None] + np.stack(deviations, axis=1)
    return proposed.astype(np.float32), np.clip(proposed, -1.0, 1.0).astype(np.float32), forms


def build_contract_candidates(
    default: np.ndarray,
    unit: np.ndarray,
    continuous_count: int,
    gripper_forms: tuple[str, ...],
    std: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    continuous_pre, continuous, continuous_forms = exact_antithetic_continuous(
        default, unit, continuous_count, std, scale
    )
    bank = gripper_universe(default)
    special = np.repeat(default[:, None], len(gripper_forms), axis=1).astype(np.float32)
    for index, name in enumerate(gripper_forms):
        special[:, index, :, 6] = bank[name]
    preclip = np.concatenate((default[:, None], continuous_pre, special), axis=1)
    candidates = np.concatenate((default[:, None], continuous, special), axis=1)
    category = np.concatenate((
        np.zeros(1, dtype=np.int8),
        np.ones(continuous_count, dtype=np.int8),
        np.full(len(gripper_forms), 2, dtype=np.int8),
    ))
    return preclip, candidates, category, ["a0", *continuous_forms, *gripper_forms]


def _gain(default: np.ndarray, oracle: np.ndarray) -> float:
    return float((default.mean() - oracle.mean()) / default.mean())


def quick_oracle_metrics(candidates: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    squared = np.square(candidates.astype(np.float64) - expert[:, None].astype(np.float64))
    full = squared.mean(axis=(2, 3))
    prefix = {}
    for width in (1, 2, 4, 8):
        value = squared[:, :, :width].mean(axis=(2, 3))
        prefix[str(width)] = _gain(value[:, 0], value.min(axis=1))
    return {
        "h8": _gain(full[:, 0], full.min(axis=1)),
        "h1": prefix["1"], "h2": prefix["2"], "h4": prefix["4"],
        "mean_h1_h2": 0.5 * (prefix["1"] + prefix["2"]),
    }


def select_gripper_subset(
    default: np.ndarray,
    unit: np.ndarray,
    continuous_count: int,
    gripper_count: int,
    std: np.ndarray,
    scale: float,
    expert: np.ndarray,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    selected = list(MANDATORY_GRIPPER)
    trace: list[dict[str, Any]] = []
    remaining = list(SWITCH_ORDER)
    while len(selected) < gripper_count:
        trials = []
        for order, name in enumerate(remaining):
            forms = tuple((*selected, name))
            generated = build_contract_candidates(default, unit, continuous_count, forms, std, scale)
            metric = quick_oracle_metrics(generated[1], expert)
            trials.append((metric["mean_h1_h2"], metric["h8"], -order, name, metric))
        winner = max(trials, key=lambda value: value[:3])
        selected.append(winner[3])
        remaining.remove(winner[3])
        trace.append({"step": len(selected), "selected": winner[3], "metrics": winner[4]})
    return tuple(selected), trace


def validate_exact_contract(
    preclip: np.ndarray,
    candidates: np.ndarray,
    category: np.ndarray,
    forms: list[str],
    default: np.ndarray,
) -> dict[str, Any]:
    continuous_indices = np.flatnonzero(category == 1)
    deviations = preclip[:, continuous_indices] - default[:, None]
    pair_checks = []
    pair_errors = []
    for offset in range(0, len(continuous_indices), 2):
        error = float(np.max(np.abs(deviations[:, offset] + deviations[:, offset + 1])))
        pair_errors.append(error)
        pair_checks.append(error <= 2.0e-7)
    special_forms = forms[1 + len(continuous_indices):]
    return {
        "exact_total_64": candidates.shape[1] == 64,
        "a0_once_by_contract": forms.count("a0") == 1 and np.array_equal(candidates[:, 0], default),
        "continuous_count_even": len(continuous_indices) % 2 == 0,
        "continuous_pair_count": len(continuous_indices) // 2,
        "all_antithetic_pairs_exact": all(pair_checks),
        "max_float32_pair_antisymmetry_error": max(pair_errors, default=0.0),
        "pair_antisymmetry_tolerance": 2.0e-7,
        "unmatched_continuous_direction_count": 0 if all(pair_checks) else int(sum(not value for value in pair_checks)),
        "g3_included": "c0_binarized" in special_forms,
        "mandatory_constants_included": all(name in special_forms for name in ("all_negative", "all_positive")),
        "unique_special_form_contract": len(special_forms) == len(set(special_forms)),
        "finite": bool(np.isfinite(preclip).all() and np.isfinite(candidates).all()),
        "bounds_valid": bool(np.all(np.abs(candidates) <= 1.0)),
        "pass": bool(
            candidates.shape[1] == 64
            and forms.count("a0") == 1
            and len(continuous_indices) % 2 == 0
            and all(pair_checks)
            and "c0_binarized" in special_forms
            and all(name in special_forms for name in ("all_negative", "all_positive"))
            and len(special_forms) == len(set(special_forms))
            and np.isfinite(preclip).all()
            and np.isfinite(candidates).all()
            and np.all(np.abs(candidates) <= 1.0)
        ),
    }


def summarize_contract(
    name: str,
    generated: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
    context: OfflineContext,
    demos: tuple[int, ...],
    draws: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary, arrays = summarize_interface(name, *generated, context, demos, draws)
    preclip, candidates, category, forms = generated
    summary["exact_contract"] = validate_exact_contract(
        preclip, candidates, category, forms, context.actor_default
    )
    winner = arrays["winner"].astype(np.int64)
    winner_forms = np.asarray(forms, dtype=object)[winner]
    summary["structured_subtype_winner_fraction"] = {
        "g3": float(np.mean(winner_forms == "c0_binarized")),
        "constant": float(np.mean(np.isin(winner_forms, ("all_negative", "all_positive")))),
        "single_switch": float(np.mean(["_to_" in value for value in winner_forms])),
    }
    return summary, arrays
