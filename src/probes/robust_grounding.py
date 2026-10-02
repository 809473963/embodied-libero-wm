"""Explicit source-geometry compatibility for same-scene bowl grounding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


BOWLS = ("akita_black_bowl_1", "akita_black_bowl_2")
FAMILIES = ("G1", "G2", "G3")
PERTURBATIONS_MM = (1, 2, 5)


def perturbation_vectors() -> np.ndarray:
    """Return the preregistered ±axis perturbations in meters."""

    rows = []
    for millimeters in PERTURBATIONS_MM:
        for axis in range(3):
            for sign in (-1.0, 1.0):
                value = np.zeros(3, dtype=np.float64)
                value[axis] = sign * millimeters / 1000.0
                rows.append(value)
    return np.asarray(rows, dtype=np.float64)


def _position(inner: Any, name: str) -> np.ndarray:
    return np.asarray(inner.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)


def _body_rotation(inner: Any, name: str) -> np.ndarray:
    body_id = inner.obj_body_id[name]
    return np.asarray(inner.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)


def _world_offset(inner: Any, name: str, attribute: str) -> np.ndarray:
    model = inner.get_object(name)
    return _position(inner, name) + _body_rotation(inner, name) @ np.asarray(
        getattr(model, attribute), dtype=np.float64
    )


def exact_source_match(inner: Any, contract: dict[str, Any], bowl: str) -> bool:
    atom = list(contract["source_atom"])
    return bool(inner._eval_predicate([str(atom[0]).lower(), bowl, *atom[2:]]))


def _outside_distance(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(lower - value, value - upper), 0.0)


def _site_features(
    inner: Any, contract: dict[str, Any], bowl: str, target_shift: np.ndarray
) -> dict[str, float]:
    target = str(contract["source_target"])
    site_state = inner.object_states_dict[target]
    site = inner.object_sites_dict[target]
    site_pos = np.asarray(inner.sim.data.get_site_xpos(target), dtype=np.float64) + target_shift
    rotation = np.asarray(inner.sim.data.get_site_xmat(target), dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (_position(inner, bowl) - site_pos)
    size = np.maximum(np.asarray(site.size, dtype=np.float64), 0.005)
    predicate = str(contract["source_predicate"]).lower()
    if predicate == "in":
        lower = -size.copy()
        lower[2] -= 0.01
        upper = size.copy()
        outside = _outside_distance(local, lower, upper)
        raw = float(np.linalg.norm(outside))
        normalized = float(np.linalg.norm(local / size))
        kind = "containment_region"
    else:
        lower = np.asarray([-size[0], -size[1], size[2] - 0.005], dtype=np.float64)
        upper = np.asarray([size[0], size[1], size[2] + 0.10], dtype=np.float64)
        outside = _outside_distance(local, lower, upper)
        raw = float(np.linalg.norm(outside))
        normalized = float(
            np.linalg.norm(
                np.asarray(
                    [local[0] / size[0], local[1] / size[1], outside[2] / 0.10],
                    dtype=np.float64,
                )
            )
        )
        kind = "source_site"
    return {
        "raw_distance": raw,
        "normalized_distance": normalized,
        "horizontal_distance": float(np.linalg.norm(local[:2])),
        "vertical_mismatch": float(outside[2]),
        "kind": kind,
    }


def _support_features(
    inner: Any, contract: dict[str, Any], bowl: str, target_shift: np.ndarray
) -> dict[str, float]:
    target = str(contract["source_target"])
    bowl_pos = _position(inner, bowl)
    support_pos = _position(inner, target) + target_shift
    bowl_model = inner.get_object(bowl)
    support_model = inner.get_object(target)
    horizontal = float(np.linalg.norm(bowl_pos[:2] - support_pos[:2]))
    horizontal_scale = max(
        float(bowl_model.horizontal_radius) + float(support_model.horizontal_radius), 0.01
    )
    horizontal_surface_gap = max(horizontal - horizontal_scale, 0.0)
    bowl_bottom = _world_offset(inner, bowl, "bottom_offset")
    support_top = _world_offset(inner, target, "top_offset") + target_shift
    vertical_gap = abs(float(bowl_bottom[2] - support_top[2]))
    vertical_scale = max(
        abs(float(bowl_model.top_offset[2] - bowl_model.bottom_offset[2])),
        abs(float(support_model.top_offset[2] - support_model.bottom_offset[2])),
        0.01,
    )
    raw = float(math.hypot(horizontal_surface_gap, vertical_gap))
    normalized = float(
        math.hypot(horizontal / horizontal_scale, vertical_gap / vertical_scale)
    )
    return {
        "raw_distance": raw,
        "normalized_distance": normalized,
        "horizontal_distance": horizontal,
        "vertical_mismatch": vertical_gap,
        "kind": "object_support",
    }


def compatibility(
    inner: Any,
    contract: dict[str, Any],
    bowl: str,
    family: str,
    target_shift: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return a lower-is-better explicit compatibility score."""

    if family not in FAMILIES:
        raise ValueError(f"unsupported family: {family}")
    shift = np.zeros(3, dtype=np.float64) if target_shift is None else np.asarray(target_shift, dtype=np.float64)
    target = str(contract["source_target"])
    state = inner.object_states_dict[target]
    if state.object_state_type == "site":
        detail = _site_features(inner, contract, bowl, shift)
    else:
        if str(contract["source_predicate"]).lower() != "on":
            raise ValueError("only On(object, object-support) is admitted")
        detail = _support_features(inner, contract, bowl, shift)
    exact = exact_source_match(inner, contract, bowl)
    if family == "G1":
        score = float(detail["raw_distance"])
    elif family == "G2":
        score = float(detail["normalized_distance"])
    else:
        bounded = float(detail["normalized_distance"] / (1.0 + detail["normalized_distance"]))
        score = bounded if exact else 1.0 + bounded
    detail = dict(detail)
    detail.update({"exact": exact, "family": family, "score": score})
    if not math.isfinite(score):
        raise FloatingPointError("non-finite grounding compatibility")
    return score, detail


def resolve_scores(scores: np.ndarray, threshold: float, max_score: float) -> tuple[str, int, float]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (2,) or not np.isfinite(scores).all():
        return "NOT_FOUND", -1, float("nan")
    order = np.argsort(scores, kind="stable")
    best, second = int(order[0]), int(order[1])
    gap = float(scores[second] - scores[best])
    if float(scores[best]) > float(max_score):
        return "NOT_FOUND", -1, gap
    if gap <= float(threshold):
        return "AMBIGUOUS", -1, gap
    return "UNIQUE", best, gap


def exact_resolve(values: np.ndarray) -> tuple[str, int]:
    values = np.asarray(values, dtype=np.bool_)
    indices = np.flatnonzero(values)
    if len(indices) == 1:
        return "UNIQUE", int(indices[0])
    if len(indices) == 0:
        return "NOT_FOUND", -1
    return "AMBIGUOUS", -1


def threshold_grid(family: str) -> tuple[float, ...]:
    if family == "G1":
        return (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
    if family == "G2":
        return (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
    if family == "G3":
        return (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75)
    raise ValueError(family)


def maximum_score(family: str) -> float:
    return {"G1": 0.50, "G2": 20.0, "G3": 2.0}[family]
