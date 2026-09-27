"""Utilities for the local visual counterfactual missing-state audit.

The functions in this module are deliberately model-light.  They construct
matched candidate-pair features from simulator truth and fit closed-form ridge
probes.  Simulator truth is a diagnostic oracle, never a deployment input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")
FEATURE_GROUPS = ("ej", "rotation", "object", "contact", "relative")
FEATURE_SET_GROUPS = {
    "S0": ("ej",),
    "S1": ("ej", "rotation"),
    "S2": ("ej", "rotation", "object"),
    "S3": ("ej", "rotation", "object", "contact"),
    "S4": ("ej", "rotation", "object", "contact", "relative"),
}


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1.0e-12)


def rotation_pair_features(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous relative rotvec and its geodesic angle.

    Inputs are rotation matrices, so quaternion q/-q double cover has already
    been removed.  The relative matrix is first @ second.T.
    """

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    relative = first_array @ np.swapaxes(second_array, -1, -2)
    rotvec = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(
        relative.shape[:-2] + (3,)
    )
    angle = np.linalg.norm(rotvec, axis=-1)
    return rotvec, angle


def quaternion_geodesic(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Quaternion geodesic with q/-q invariance."""

    first_q = normalize_quaternion(first)
    second_q = normalize_quaternion(second)
    dot = np.abs(np.sum(first_q * second_q, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


@dataclass(frozen=True)
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    intercept: np.ndarray
    alpha: float


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeFit:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if y_array.ndim == 1:
        y_array = y_array[:, None]
    mean = x_array.mean(axis=0)
    scale = x_array.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (x_array - mean) / scale
    intercept = y_array.mean(axis=0)
    centered_y = y_array - intercept
    n = max(int(x_array.shape[0]), 1)
    gram = standardized.T @ standardized / n
    rhs = standardized.T @ centered_y / n
    weight = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    return RidgeFit(mean, scale, weight, intercept, float(alpha))


def predict_ridge(x: np.ndarray, fit: RidgeFit) -> np.ndarray:
    prediction = ((np.asarray(x, dtype=np.float64) - fit.mean) / fit.scale) @ fit.weight
    prediction = prediction + fit.intercept
    return prediction


def ridge_to_arrays(prefix: str, fit: RidgeFit) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__mean": fit.mean,
        f"{prefix}__scale": fit.scale,
        f"{prefix}__weight": fit.weight,
        f"{prefix}__intercept": fit.intercept,
        f"{prefix}__alpha": np.asarray(fit.alpha, dtype=np.float64),
    }


def ridge_from_arrays(prefix: str, arrays: Any) -> RidgeFit:
    return RidgeFit(
        np.asarray(arrays[f"{prefix}__mean"], dtype=np.float64),
        np.asarray(arrays[f"{prefix}__scale"], dtype=np.float64),
        np.asarray(arrays[f"{prefix}__weight"], dtype=np.float64),
        np.asarray(arrays[f"{prefix}__intercept"], dtype=np.float64),
        float(np.asarray(arrays[f"{prefix}__alpha"])),
    )


def scalar_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    true = np.asarray(target, dtype=np.float64).reshape(-1)
    residual = pred - true
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum(np.square(true - true.mean())))
    r2 = None if denominator <= 1.0e-20 else float(1.0 - np.sum(np.square(residual)) / denominator)
    if pred.size < 3 or np.std(pred) <= 1.0e-20 or np.std(true) <= 1.0e-20:
        pearson = None
        spearman = None
    else:
        from scipy import stats

        pearson = float(stats.pearsonr(pred, true).statistic)
        spearman = float(stats.spearmanr(pred, true).statistic)
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    init_state: np.ndarray,
    grid: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    init = np.asarray(init_state)
    for alpha in grid:
        folds = []
        for held_out in sorted(np.unique(init).tolist()):
            train = init != held_out
            valid = ~train
            fit = fit_ridge(x[train], y[train], alpha)
            pred = predict_ridge(x[valid], fit)
            true = np.asarray(y)[valid]
            rmse = float(np.sqrt(np.mean(np.square(pred.reshape(true.shape) - true))))
            folds.append({"held_out_init": int(held_out), "rmse": rmse})
        rows.append(
            {
                "alpha": float(alpha),
                "mean_fold_rmse": float(np.mean([row["rmse"] for row in folds])),
                "folds": folds,
            }
        )
    selected = min(rows, key=lambda row: (row["mean_fold_rmse"], -row["alpha"]))
    return float(selected["alpha"]), rows


def _contact_features(trace_first: np.ndarray, trace_second: np.ndarray, horizon: int) -> np.ndarray:
    """Current split, onset, and duration differences for four contact types."""

    first = np.asarray(trace_first[: horizon + 1], dtype=bool)
    second = np.asarray(trace_second[: horizon + 1], dtype=bool)
    current = np.logical_xor(first[-1], second[-1]).astype(np.float64)
    any_different = np.asarray([float(np.any(current))])

    def onset(trace: np.ndarray) -> np.ndarray:
        result = np.full(trace.shape[1], horizon + 1, dtype=np.float64)
        for dimension in range(trace.shape[1]):
            indices = np.flatnonzero(trace[:, dimension])
            if indices.size:
                result[dimension] = float(indices[0])
        return result

    onset_difference = np.abs(onset(first) - onset(second)) / 8.0
    duration_difference = np.abs(first.sum(axis=0) - second.sum(axis=0)) / float(horizon + 1)
    return np.concatenate((current, any_different, onset_difference, duration_difference))


def _aggregate_pose_features(
    first_position: np.ndarray,
    second_position: np.ndarray,
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    position_delta = np.asarray(first_position) - np.asarray(second_position)
    rotation_delta, rotation_angle = rotation_pair_features(first_rotation, second_rotation)
    translation = np.linalg.norm(position_delta, axis=-1)
    feature = np.concatenate(
        (
            position_delta.mean(axis=0),
            rotation_delta.mean(axis=0),
            np.asarray([translation.max(), translation.mean(), rotation_angle.max(), rotation_angle.mean()]),
        )
    )
    scalar = {
        "translation_max": float(translation.max()),
        "translation_mean": float(translation.mean()),
        "rotation_max": float(rotation_angle.max()),
        "rotation_mean": float(rotation_angle.mean()),
    }
    return feature, scalar


def build_pair_dataset(
    base: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    proprio_mean: np.ndarray,
    proprio_std: np.ndarray,
) -> dict[str, Any]:
    """Construct the fixed S0--S4 feature hierarchy at branch/pair/horizon grain."""

    real_p = np.asarray(base["real_proprio"], dtype=np.float64)
    real_z = np.asarray(base["real_z"], dtype=np.float64)
    candidates = np.asarray(base["candidates"], dtype=np.float64)
    p_standardized = (real_p - np.asarray(proprio_mean).reshape(1, 1, 1, 12)) / np.asarray(
        proprio_std
    ).reshape(1, 1, 1, 12)
    ee_rot = np.asarray(state["ee_rotmat"], dtype=np.float64)
    object_pos = np.asarray(state["object_pos"], dtype=np.float64)
    object_rot = np.asarray(state["object_rotmat"], dtype=np.float64)
    object_mask = np.asarray(state["object_mask"], dtype=bool)
    contact = np.asarray(state["contact_any"], dtype=bool)

    groups = {name: [] for name in FEATURE_GROUPS}
    target_visual = []
    target_delta_z = []
    metadata = {name: [] for name in ("branch", "task", "init_state", "horizon", "pair")}
    scalars = {
        name: []
        for name in (
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
    }
    for branch in range(real_p.shape[0]):
        valid = object_mask[branch]
        if not np.any(valid):
            raise ValueError(f"branch {branch} has no relevant object")
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            for horizon in range(8):
                ej = np.concatenate(
                    (
                        p_standardized[branch, first, horizon, 0:3]
                        - p_standardized[branch, second, horizon, 0:3],
                        p_standardized[branch, first, horizon, 5:12]
                        - p_standardized[branch, second, horizon, 5:12],
                    )
                )
                rotvec, angle = rotation_pair_features(
                    ee_rot[branch, first, horizon], ee_rot[branch, second, horizon]
                )
                rotation = np.concatenate((rotvec, np.asarray(angle).reshape(1)))
                obj_feature, obj_scalar = _aggregate_pose_features(
                    object_pos[branch, first, horizon, valid],
                    object_pos[branch, second, horizon, valid],
                    object_rot[branch, first, horizon, valid],
                    object_rot[branch, second, horizon, valid],
                )
                contact_feature = _contact_features(
                    contact[branch, first], contact[branch, second], horizon
                )
                ee_first = real_p[branch, first, horizon, 0:3]
                ee_second = real_p[branch, second, horizon, 0:3]
                relative_position_first = object_pos[branch, first, horizon, valid] - ee_first
                relative_position_second = object_pos[branch, second, horizon, valid] - ee_second
                relative_rotation_first = object_rot[branch, first, horizon, valid] @ np.swapaxes(
                    ee_rot[branch, first, horizon], -1, -2
                )
                relative_rotation_second = object_rot[branch, second, horizon, valid] @ np.swapaxes(
                    ee_rot[branch, second, horizon], -1, -2
                )
                relative_feature, relative_scalar = _aggregate_pose_features(
                    relative_position_first,
                    relative_position_second,
                    relative_rotation_first,
                    relative_rotation_second,
                )
                z_first = real_z[branch, first, horizon]
                z_second = real_z[branch, second, horizon]
                visual = max(
                    1.0
                    - float(
                        np.dot(z_first, z_second)
                        / max(np.linalg.norm(z_first) * np.linalg.norm(z_second), 1.0e-12)
                    ),
                    0.0,
                )
                action_rotation = float(
                    np.linalg.norm(
                        candidates[branch, first, : horizon + 1, 3:6]
                        - candidates[branch, second, : horizon + 1, 3:6]
                    )
                )

                groups["ej"].append(ej)
                groups["rotation"].append(rotation)
                groups["object"].append(obj_feature)
                groups["contact"].append(contact_feature)
                groups["relative"].append(relative_feature)
                target_visual.append(visual)
                target_delta_z.append(z_first - z_second)
                metadata["branch"].append(branch)
                metadata["task"].append(int(base["task_id"][branch]))
                metadata["init_state"].append(int(base["init_state_index"][branch]))
                metadata["horizon"].append(horizon + 1)
                metadata["pair"].append(pair_index)
                scalars["ee_translation"].append(float(np.linalg.norm(ej[:3])))
                scalars["ee_rotation"].append(float(angle))
                scalars["action_rotation"].append(action_rotation)
                for key, value in obj_scalar.items():
                    scalars[f"object_{key}"].append(value)
                scalars["contact_different"].append(float(np.any(contact_feature[:4])))
                for key, value in relative_scalar.items():
                    scalars[f"relative_{key}"].append(value)

    group_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in groups.items()}
    sets = {
        name: np.concatenate([group_arrays[group] for group in feature_groups], axis=1)
        for name, feature_groups in FEATURE_SET_GROUPS.items()
    }
    return {
        "groups": group_arrays,
        "sets": sets,
        "target_visual": np.asarray(target_visual, dtype=np.float64),
        "target_delta_z": np.asarray(target_delta_z, dtype=np.float64),
        "metadata": {key: np.asarray(value, dtype=np.int64) for key, value in metadata.items()},
        "scalars": {key: np.asarray(value, dtype=np.float64) for key, value in scalars.items()},
    }
