"""Clean semantic goal-progress teacher utilities.

This module contains no HDF5 discovery. Demonstration arrays must be supplied
by :class:`AllowedDemoGuard`; task grounding is performed against a restored
initial physical scene and persists the selected MuJoCo object identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.probes.semantic_goal_contract import parse_sexpression


BOWLS = ("akita_black_bowl_1", "akita_black_bowl_2")
PLATE = "plate_1"
OFFICIAL_XY_LIMIT_METERS = 0.03
GAP_BINS = ((1, 2), (3, 4), (5, 8), (9, 16))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _section(root: list[Any], name: str) -> list[Any]:
    for row in root:
        if isinstance(row, list) and row and str(row[0]).lower() == name.lower():
            return row
    raise ValueError(f"missing BDDL section: {name}")


def task_grounding_contract(path: Path, task_id: int) -> dict[str, Any]:
    root = parse_sexpression(path.read_text())
    language = _section(root, ":language")
    initial = _section(root, ":init")
    goal = _section(root, ":goal")
    source_atoms = []
    for row in initial[1:]:
        if isinstance(row, list) and len(row) in (2, 3) and str(row[1]).lower() == BOWLS[0]:
            source_atoms.append([str(value) for value in row])
    if len(source_atoms) != 1:
        raise ValueError(f"task{task_id} does not have one bowl1 source atom")
    goal_expression = goal[1]
    if (
        not isinstance(goal_expression, list)
        or len(goal_expression) != 2
        or str(goal_expression[0]).lower() != "and"
    ):
        raise ValueError(f"task{task_id} has unsupported goal wrapper")
    goal_atom = [str(value) for value in goal_expression[1]]
    if [value.lower() for value in goal_atom] != ["on", BOWLS[0], PLATE]:
        raise ValueError(f"task{task_id} goal is not On(selected bowl, plate)")
    return {
        "task_id": int(task_id),
        "instruction": " ".join(str(value) for value in language[1:]),
        "source_atom": source_atoms[0],
        "source_predicate": source_atoms[0][0],
        "source_target": source_atoms[0][2] if len(source_atoms[0]) == 3 else None,
        "candidate_referents": list(BOWLS),
        "plate_target": PLATE,
        "exact_final_goal": goal_atom,
        "bddl_path": path.as_posix(),
        "bddl_sha256": sha256(path),
    }


def ground_instruction(inner: Any, contract: dict[str, Any]) -> dict[str, Any]:
    atom = list(contract["source_atom"])
    matches: list[str] = []
    values: dict[str, bool] = {}
    for bowl in BOWLS:
        query = [atom[0], bowl, *atom[2:]]
        value = bool(inner._eval_predicate(query))
        values[bowl] = value
        if value:
            matches.append(bowl)
    if len(matches) == 1:
        status, referent = "UNIQUE", matches[0]
    elif len(matches) == 0:
        status, referent = "NOT_FOUND", None
    else:
        status, referent = "AMBIGUOUS", None
    return {
        "instruction_task_id": int(contract["task_id"]),
        "status": status,
        "referent": referent,
        "predicate_values": values,
    }


def _object_position(inner: Any, name: str) -> np.ndarray:
    return np.asarray(inner.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)


def official_goal_and_margin(inner: Any, bowl: str) -> tuple[bool, float, dict[str, float]]:
    """Evaluate exact On and a continuous margin derived from its geometry.

    The exact Boolean remains the installed predicate. The continuous margin
    is the minimum normalized clearance of the two explicit numerical On
    inequalities and the negative absolute object-surface gap. The latter is
    computed from MuJoCo model offsets and prevents a bowl high above a plate
    from receiving a falsely favorable margin. The official 0.03 m horizontal
    threshold supplies the single task-invariant physical scale.
    """

    bowl_pos = _object_position(inner, bowl)
    plate_pos = _object_position(inner, PLATE)
    bowl_model = inner.get_object(bowl)
    plate_model = inner.get_object(PLATE)
    bowl_bottom = float(bowl_pos[2] + np.asarray(bowl_model.bottom_offset)[2])
    plate_top = float(plate_pos[2] + np.asarray(plate_model.top_offset)[2])
    xy_distance = float(np.linalg.norm(bowl_pos[:2] - plate_pos[:2]))
    xy_clearance = OFFICIAL_XY_LIMIT_METERS - xy_distance
    z_order_clearance = float(bowl_pos[2] - plate_pos[2])
    surface_gap = bowl_bottom - plate_top
    scale = OFFICIAL_XY_LIMIT_METERS
    margin = min(xy_clearance / scale, z_order_clearance / scale, -abs(surface_gap) / scale)
    goal = bool(inner._eval_predicate(["On", bowl, PLATE]))
    values = {
        "xy_distance_m": xy_distance,
        "xy_clearance_m": xy_clearance,
        "z_order_clearance_m": z_order_clearance,
        "surface_gap_m": surface_gap,
        "normalized_margin": float(margin),
    }
    if not np.isfinite(np.asarray(list(values.values()), dtype=np.float64)).all():
        raise FloatingPointError("goal geometry is not finite")
    return goal, float(margin), values


def controlled_and_reach(inner: Any, bowl: str) -> tuple[bool, float]:
    model = inner.get_object(bowl)
    controlled = bool(
        inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=model)
    )
    ee = np.asarray(
        inner.sim.data.get_site_xpos(inner.robots[0].gripper.important_sites["grip_site"]),
        dtype=np.float64,
    )
    reach = -float(np.linalg.norm(ee - _object_position(inner, bowl))) / OFFICIAL_XY_LIMIT_METERS
    if not math.isfinite(reach):
        raise FloatingPointError("reach readiness is not finite")
    return controlled, reach


def proprio_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    result = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float64).reshape(3),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64).reshape(2),
            np.asarray(observation["robot0_joint_pos"], dtype=np.float64).reshape(7),
        )
    )
    if result.shape != (12,) or not np.isfinite(result).all():
        raise FloatingPointError("invalid 12-D proprioception")
    return result


def calibrated_epsilon(changes: np.ndarray, floor: float = 1.0e-4) -> float:
    positive = np.asarray(changes, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > floor)]
    if not len(positive):
        return float(floor)
    return float(max(floor, np.quantile(positive, 0.10)))


def compare_teacher(
    goal_a: bool,
    margin_a: float,
    controlled_a: bool,
    reach_a: float,
    goal_b: bool,
    margin_b: float,
    controlled_b: bool,
    reach_b: float,
    family: str,
    epsilon_goal: float,
    epsilon_reach: float,
) -> int:
    """Return -1 / 0 / +1 for b relative to a under a lexicographic teacher."""

    if bool(goal_a) != bool(goal_b):
        return 1 if goal_b else -1
    if family == "T2" and not goal_a:
        if bool(controlled_a) != bool(controlled_b):
            return 1 if controlled_b else -1
        if controlled_a:
            delta, epsilon = margin_b - margin_a, epsilon_goal
        else:
            delta, epsilon = reach_b - reach_a, epsilon_reach
    else:
        delta, epsilon = margin_b - margin_a, epsilon_goal
    if delta > epsilon:
        return 1
    if delta < -epsilon:
        return -1
    return 0


def pair_rows(episode_indices: np.ndarray, timesteps: np.ndarray) -> Iterable[tuple[int, int, int]]:
    episodes = np.unique(episode_indices)
    for episode in episodes:
        index = np.flatnonzero(episode_indices == episode)
        by_time = {int(timesteps[row]): int(row) for row in index}
        for row in index:
            start = int(timesteps[row])
            for gap in range(1, 17):
                other = by_time.get(start + gap)
                if other is not None:
                    yield int(row), int(other), gap


def summarize_labels(labels: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    n = int(len(labels))
    return {
        "n": n,
        "non_tie_fraction": float(np.mean(labels != 0)) if n else 0.0,
        "progress_fraction": float(np.mean(labels > 0)) if n else 0.0,
        "regression_fraction": float(np.mean(labels < 0)) if n else 0.0,
        "tie_fraction": float(np.mean(labels == 0)) if n else 0.0,
        "time_agreement_among_non_ties": float(np.mean(labels[labels != 0] > 0))
        if np.any(labels != 0)
        else 0.0,
    }


def fit_linear_pair_probe(features: np.ndarray, labels: np.ndarray, ridge: float) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    keep = y != 0
    x, y = x[keep], y[keep]
    if len(y) == 0:
        raise ValueError("no non-tie rows for proprio probe")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-8] = 1.0
    z = (x - mean) / std
    design = np.column_stack((z, np.ones(len(z))))
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    prediction = np.where(design @ weights >= 0.0, 1, -1)
    return {
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weights": weights.tolist(),
        "ridge": float(ridge),
        "train_accuracy": float(np.mean(prediction == y)),
        "train_rows": int(len(y)),
    }


def apply_linear_pair_probe(features: np.ndarray, labels: np.ndarray, model: dict[str, Any]) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    keep = y != 0
    x, y = x[keep], y[keep]
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    design = np.column_stack(((x - mean) / std, np.ones(len(x))))
    prediction = np.where(design @ weights >= 0.0, 1, -1)
    return {"rows": int(len(y)), "accuracy": float(np.mean(prediction == y))}

