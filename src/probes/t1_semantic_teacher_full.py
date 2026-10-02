"""Statistics for the independently frozen T1 semantic-teacher audit."""

from __future__ import annotations

from typing import Any

import numpy as np


MARGINS = ("M0", "M1", "M2")
LOCAL_MAX_GAP = 8
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261002


def rank_auc(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("invalid AUC inputs")
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("distribution requires finite nonempty values")
    return {
        "n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)), "p90": float(np.quantile(values, 0.90)),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def overlap_coefficient(positive: np.ndarray, negative: np.ndarray, bins: int = 100) -> float:
    positive, negative = np.asarray(positive), np.asarray(negative)
    low, high = float(min(positive.min(), negative.min())), float(max(positive.max(), negative.max()))
    if high == low:
        return 1.0
    hp, edges = np.histogram(positive, bins=bins, range=(low, high), density=False)
    hn, _ = np.histogram(negative, bins=edges, density=False)
    return float(np.minimum(hp / hp.sum(), hn / hn.sum()).sum())


def margin_audit(margins: np.ndarray, goal: np.ndarray, task_id: np.ndarray, episode_id: np.ndarray, timestep: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, family in enumerate(MARGINS):
        values = margins[:, :, index]
        positive, negative = values[goal], values[~goal]
        task_auc = {}
        for task in range(10):
            mask = np.repeat(task_id == task, 2).reshape(values.shape)
            task_auc[str(task)] = rank_auc(values[mask], goal[mask]) if np.any(goal[mask]) and np.any(~goal[mask]) else None
        jumps = []
        for episode in np.unique(episode_id):
            ids = np.flatnonzero(episode_id == episode)
            order = ids[np.argsort(timestep[ids])]
            jumps.extend(np.abs(np.diff(values[order], axis=0)).reshape(-1).tolist())
        jump = np.asarray(jumps, dtype=np.float64)
        auc = rank_auc(values, goal)
        pos_stats, neg_stats = distribution(positive), distribution(negative)
        task_pass = sum(value is not None and value > 0.5 for value in task_auc.values())
        row = {
            "family": family, "finite": bool(np.isfinite(values).all()), "goal_auc": auc,
            "goal_positive": pos_stats, "goal_negative": neg_stats,
            "median_separation": pos_stats["median"] - neg_stats["median"],
            "overlap_coefficient": overlap_coefficient(positive, negative),
            "per_task_auc": task_auc, "tasks_auc_above_half": task_pass,
            "adjacent_absolute_jump": {**distribution(jump), "p99": float(np.quantile(jump, 0.99))},
        }
        row["admissible"] = bool(row["finite"] and auc >= 0.80 and row["median_separation"] > 0 and task_pass >= 7 and row["adjacent_absolute_jump"]["p99"] <= 5.0)
        rows.append(row)
    return rows


def choose_margin(rows: list[dict[str, Any]]) -> str:
    official_rank = {"M2": 0, "M1": 1, "M0": 2}
    eligible = [row for row in rows if row["admissible"]]
    if not eligible:
        return "none"
    return sorted(eligible, key=lambda row: (official_rank[row["family"]], -row["goal_auc"], -row["tasks_auc_above_half"], row["adjacent_absolute_jump"]["p99"], MARGINS.index(row["family"])))[0]["family"]


def block_bootstrap_ci(task: np.ndarray, demo: np.ndarray, numerator: np.ndarray, denominator: np.ndarray, seed_offset: int = 0) -> list[float]:
    task, demo = np.asarray(task), np.asarray(demo)
    numerator, denominator = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    if not (len(task) == len(demo) == len(numerator) == len(denominator)):
        raise ValueError("bootstrap arrays differ in length")
    num = np.zeros((10, 4), dtype=np.float64); den = np.zeros((10, 4), dtype=np.float64)
    for t in range(10):
        demos = np.sort(np.unique(demo[task == t]))
        if len(demos) != 4:
            raise ValueError(f"task {t} does not have four demo blocks")
        for j, d in enumerate(demos):
            mask = (task == t) & (demo == d)
            num[t, j], den[t, j] = numerator[mask].sum(), denominator[mask].sum()
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, 4, size=(BOOTSTRAP_DRAWS, 10, 4))
    tindex = np.arange(10)[None, :, None]
    draws = num[tindex, indices].sum(axis=(1, 2)) / den[tindex, indices].sum(axis=(1, 2))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return rank_auc(np.asarray(scores), np.asarray(labels) > 0)
