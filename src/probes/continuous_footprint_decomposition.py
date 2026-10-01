"""Deterministic continuous-footprint decompositions for the oracle audit.

All states are computed from already-frozen 7x7 relevant-object occupancy
maps.  They are diagnostic per-image oracle targets, never deployment inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.probes.spatial_target_design import gather_pair_values


CANDIDATES = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
DIMENSIONS = {"F0": 50, "F1": 1, "F2": 49, "F3": 3, "F4": 5, "F5": 6, "F6": 7, "F7": 8}


@dataclass(frozen=True)
class FootprintCandidate:
    state: np.ndarray
    spatial: np.ndarray
    centroid: np.ndarray | None
    scale: np.ndarray
    has_exact_area: bool


def _grid() -> tuple[np.ndarray, np.ndarray]:
    coordinate = (np.arange(7, dtype=np.float64) + 0.5) / 7.0
    return np.meshgrid(coordinate, coordinate, indexing="ij")


def footprint_candidates(occupancy: np.ndarray, area: np.ndarray) -> dict[str, FootprintCandidate]:
    """Construct the frozen F0--F7 per-image states.

    Zero occupancy mass is an explicit data block.  Centroids and moments are
    never imputed with zero, the image centre, or a previous value.
    """

    soft = np.asarray(occupancy, dtype=np.float64)
    visible_area = np.asarray(area, dtype=np.float64)
    if soft.shape != (40, 3, 8, 7, 7) or visible_area.shape != (40, 3, 8):
        raise ValueError(f"unexpected footprint shapes: {soft.shape}, {visible_area.shape}")
    yy, xx = _grid()
    mass = soft.sum(axis=(-2, -1))
    if np.any(mass <= 0.0):
        raise ValueError("ZERO_OCCUPANCY_MASS_DATA_BLOCK")
    cx = (soft * xx).sum(axis=(-2, -1)) / mass
    cy = (soft * yy).sum(axis=(-2, -1)) / mass
    dx = xx - cx[..., None, None]
    dy = yy - cy[..., None, None]
    var_x = (soft * dx * dx).sum(axis=(-2, -1)) / mass
    var_y = (soft * dy * dy).sum(axis=(-2, -1)) / mass
    sigma_x = np.sqrt(np.maximum(var_x, 0.0))
    sigma_y = np.sqrt(np.maximum(var_y, 0.0))
    covariance = (soft * dx * dy).sum(axis=(-2, -1)) / mass
    skew_x = (soft * dx**3).sum(axis=(-2, -1)) / mass / np.maximum(sigma_x, 1.0e-12) ** 3
    skew_y = (soft * dy**3).sum(axis=(-2, -1)) / mass / np.maximum(sigma_y, 1.0e-12) ** 3

    support = soft > 0.0
    x_values = np.broadcast_to(xx, soft.shape)
    y_values = np.broadcast_to(yy, soft.shape)
    xmin = np.min(np.where(support, x_values, np.inf), axis=(-2, -1))
    xmax = np.max(np.where(support, x_values, -np.inf), axis=(-2, -1))
    ymin = np.min(np.where(support, y_values, np.inf), axis=(-2, -1))
    ymax = np.max(np.where(support, y_values, -np.inf), axis=(-2, -1))

    centroid = np.stack((cx, cy), axis=-1)
    spread = np.stack((sigma_x, sigma_y), axis=-1)
    second = np.concatenate((centroid, spread, covariance[..., None]), axis=-1)
    bbox = np.stack((cx, cy, xmin, xmax, ymin, ymax), axis=-1)
    compact = np.concatenate((second, skew_x[..., None], skew_y[..., None]), axis=-1)
    soft_flat = soft.reshape(40, 3, 8, 49)
    map_scale = soft.mean(axis=(-2, -1))

    values = {
        "F0": FootprintCandidate(np.concatenate((soft_flat, visible_area[..., None]), axis=-1), soft_flat, centroid, visible_area, True),
        "F1": FootprintCandidate(visible_area[..., None], visible_area[..., None], None, visible_area, True),
        "F2": FootprintCandidate(soft_flat, soft_flat, centroid, map_scale, False),
        "F3": FootprintCandidate(np.concatenate((visible_area[..., None], centroid), axis=-1), centroid, centroid, visible_area, True),
        "F4": FootprintCandidate(np.concatenate((visible_area[..., None], centroid, spread), axis=-1), np.concatenate((centroid, spread), axis=-1), centroid, visible_area, True),
        "F5": FootprintCandidate(np.concatenate((visible_area[..., None], second), axis=-1), second, centroid, visible_area, True),
        "F6": FootprintCandidate(np.concatenate((visible_area[..., None], bbox), axis=-1), bbox, centroid, visible_area, True),
        "F7": FootprintCandidate(np.concatenate((visible_area[..., None], compact), axis=-1), compact, centroid, visible_area, True),
    }
    for name, candidate in values.items():
        if candidate.state.shape[-1] != DIMENSIONS[name] or not np.isfinite(candidate.state).all():
            raise FloatingPointError(f"invalid {name} state")
    return values


def pair_summary(candidate: FootprintCandidate, metadata: dict[str, np.ndarray]) -> np.ndarray:
    """Return the frozen six-slot low-capacity summary for one candidate."""

    first, second = gather_pair_values(candidate.spatial, metadata)
    difference = np.abs(first - second).reshape(first.shape[0], -1)
    ordered = np.sort(difference, axis=1)
    top_count = max(1, difference.shape[1] // 4)
    if candidate.centroid is None:
        centroid_delta = np.zeros(first.shape[0], dtype=np.float64)
    else:
        centroid_first, centroid_second = gather_pair_values(candidate.centroid, metadata)
        centroid_delta = np.linalg.norm(centroid_first - centroid_second, axis=1)
    scale_first, scale_second = gather_pair_values(candidate.scale, metadata)
    summary = np.column_stack(
        (
            np.sqrt(np.mean(np.square(difference), axis=1)),
            np.mean(difference, axis=1),
            np.max(difference, axis=1),
            np.mean(ordered[:, -top_count:], axis=1),
            centroid_delta,
            np.abs(scale_first - scale_second),
        )
    )
    if summary.shape != (960, 6) or not np.isfinite(summary).all():
        raise FloatingPointError("invalid pair summary")
    return summary


def representation_distances(candidate: FootprintCandidate) -> tuple[np.ndarray, np.ndarray]:
    """Candidate-pair and adjacent-horizon mean-absolute state distances."""

    state = np.asarray(candidate.state, dtype=np.float64)
    candidate_distance = []
    for branch in range(40):
        for left, right in ((0, 1), (0, 2), (1, 2)):
            for horizon in range(8):
                candidate_distance.append(float(np.mean(np.abs(state[branch, left, horizon] - state[branch, right, horizon]))))
    temporal_distance = []
    for branch in range(40):
        for candidate_index in range(3):
            for horizon in range(7):
                temporal_distance.append(float(np.mean(np.abs(state[branch, candidate_index, horizon + 1] - state[branch, candidate_index, horizon]))))
    return np.asarray(candidate_distance), np.asarray(temporal_distance)
