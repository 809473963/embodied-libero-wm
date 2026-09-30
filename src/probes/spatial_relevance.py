"""Natural-image spatial representation relevance utilities.

The module deliberately contains only deterministic feature construction and
low-capacity probe helpers.  Future simulator state is a diagnostic oracle and
is never exposed as a deployable model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")


TARGET_NAMES = (
    "ee_translation",
    "ee_rotation",
    "object_translation",
    "object_rotation",
    "relative_translation",
    "relative_rotation",
)
TARGET_FAMILIES = {
    "robot": ("ee_translation", "ee_rotation"),
    "object": ("object_translation", "object_rotation"),
    "relative": ("relative_translation", "relative_rotation"),
}
SCALAR_REPRESENTATIONS = (
    "global_distance",
    "patch_mean",
    "patch_max",
    "patch_top5_mean",
    "patch_top10_mean",
    "robot_patch_mean",
    "relevant_patch_mean",
    "interaction_patch_mean",
    "background_patch_mean",
)


def _rankdata(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(value, kind="stable")
    sorted_value = value[order]
    ranks = np.empty(value.size, dtype=np.float64)
    start = 0
    while start < value.size:
        stop = start + 1
        while stop < value.size and sorted_value[stop] == sorted_value[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if x.size < 3 or denominator <= 1.0e-15:
        return None
    return float(np.dot(x, y) / denominator)


def safe_correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size < 3 or np.std(x) <= 1.0e-15 or np.std(y) <= 1.0e-15:
        return {"n": int(x.size), "pearson": None, "spearman": None}
    pearson = _pearson(x, y)
    spearman = _pearson(_rankdata(x), _rankdata(y))
    return {
        "n": int(x.size),
        "pearson": pearson if pearson is not None and np.isfinite(pearson) else None,
        "spearman": spearman if spearman is not None and np.isfinite(spearman) else None,
    }


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = safe_correlation(first, second)["spearman"]
    return 0.0 if value is None else float(value)


def _normalized_cosine_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1.0e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1.0e-12)
    return np.maximum(1.0 - np.sum(a * b, axis=-1), 0.0)


def interaction_patch_indicator(
    masks: np.ndarray,
    coverage: np.ndarray,
    pixel_distance_threshold: float,
) -> np.ndarray:
    """Return [49] frozen geometry-defined interaction indicators.

    A patch is active when it overlaps both robot and relevant-object pixels,
    or when its center is within the fixed threshold of both masks.  Masks are
    natural-image semantic masks; RGB is never replaced or modified.
    """

    region = np.asarray(masks, dtype=bool)
    fraction = np.asarray(coverage, dtype=np.float64)
    if region.shape != (4, 128, 128) or fraction.shape != (4, 49):
        raise ValueError(f"unexpected interaction inputs: {region.shape}, {fraction.shape}")
    robot = region[0]
    relevant = region[1]
    if not np.any(robot) or not np.any(relevant):
        return np.zeros(49, dtype=np.float64)
    centers = np.asarray([(index + 0.5) * 128.0 / 7.0 for index in range(7)])
    centers = np.clip(np.rint(centers).astype(np.int64), 0, 127)
    center_y, center_x = np.meshgrid(centers, centers, indexing="ij")
    center_points = np.column_stack((center_y.reshape(-1), center_x.reshape(-1)))

    def nearest_distance(mask: np.ndarray) -> np.ndarray:
        points = np.argwhere(mask)
        squared = np.sum(
            np.square(center_points[:, None, :].astype(np.float64) - points[None, :, :]),
            axis=-1,
        )
        return np.sqrt(np.min(squared, axis=1)).reshape(7, 7)

    distance_robot = nearest_distance(robot)
    distance_relevant = nearest_distance(relevant)
    near = np.zeros((7, 7), dtype=bool)
    near[:] = (distance_robot <= pixel_distance_threshold) & (
        distance_relevant <= pixel_distance_threshold
    )
    overlap = (fraction[0] > 0.0) & (fraction[1] > 0.0)
    return np.logical_or(overlap, near.reshape(49)).astype(np.float64)


def _weighted_region_mean(distance: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(distance * weight) / max(float(np.sum(weight)), 1.0e-12))


def _spatial_moments(distance: np.ndarray) -> tuple[float, float, float]:
    grid = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    weight = np.asarray(distance, dtype=np.float64).reshape(7, 7)
    total = float(weight.sum())
    if total <= 1.0e-12:
        return 0.0, 0.0, 0.0
    cx = float(np.sum(weight * xx) / total)
    cy = float(np.sum(weight * yy) / total)
    spread = float(np.sqrt(np.sum(weight * ((xx - cx) ** 2 + (yy - cy) ** 2)) / total))
    return cx, cy, spread


def build_natural_spatial_dataset(
    visual: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    external: dict[str, Any],
    pixel_distance_threshold: float,
) -> dict[str, Any]:
    """Construct natural global/patch representations and external targets."""

    global_features = np.asarray(visual["global_variant_features"], dtype=np.float32)[..., 0, :]
    patch_features = np.asarray(visual["patch_features"], dtype=np.float32)
    patch_coverage = np.asarray(visual["patch_coverage"], dtype=np.float64)
    semantic_masks = np.asarray(masks["masks"], dtype=bool)
    expected = global_features.shape[:3]
    if expected != (40, 3, 8):
        raise ValueError(f"unexpected natural global feature shape: {global_features.shape}")
    if patch_features.shape != expected + (49, 512):
        raise ValueError(f"unexpected patch shape: {patch_features.shape}")
    if patch_coverage.shape != expected + (4, 49):
        raise ValueError(f"unexpected coverage shape: {patch_coverage.shape}")
    if semantic_masks.shape != expected + (4, 128, 128):
        raise ValueError(f"unexpected mask shape: {semantic_masks.shape}")

    visual_metadata = {
        name: np.asarray(visual[f"metadata__{name}"], dtype=np.int64)
        for name in ("branch", "task", "init_state", "horizon", "pair")
    }
    for name, values in visual_metadata.items():
        if not np.array_equal(values, np.asarray(external["metadata"][name])):
            raise ValueError(f"visual/state row identity mismatch: {name}")

    rows: dict[str, list[Any]] = {
        "delta_global": [],
        "delta_patch_flat": [],
        "global_distance": [],
        "global_l2": [],
        "patch_mean": [],
        "patch_max": [],
        "patch_top5_mean": [],
        "patch_top10_mean": [],
        "robot_patch_mean": [],
        "relevant_patch_mean": [],
        "interaction_patch_mean": [],
        "background_patch_mean": [],
        "centroid_x": [],
        "centroid_y": [],
        "spatial_spread": [],
        "distance_map": [],
        "pair_fractional_coverage": [],
        "interaction_indicator": [],
        "top5_fractional_coverage": [],
        "top10_fractional_coverage": [],
        "top5_interaction_fraction": [],
        "top10_interaction_fraction": [],
    }
    for branch in range(40):
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            del pair_index
            for horizon in range(8):
                g_first = global_features[branch, first, horizon]
                g_second = global_features[branch, second, horizon]
                p_first = patch_features[branch, first, horizon]
                p_second = patch_features[branch, second, horizon]
                delta_global = g_first.astype(np.float64) - g_second.astype(np.float64)
                delta_patch = p_first.astype(np.float64) - p_second.astype(np.float64)
                distance = _normalized_cosine_distance(p_first, p_second)
                coverage = (
                    patch_coverage[branch, first, horizon]
                    + patch_coverage[branch, second, horizon]
                ) / 2.0
                interaction = np.maximum(
                    interaction_patch_indicator(
                        semantic_masks[branch, first, horizon],
                        patch_coverage[branch, first, horizon],
                        pixel_distance_threshold,
                    ),
                    interaction_patch_indicator(
                        semantic_masks[branch, second, horizon],
                        patch_coverage[branch, second, horizon],
                        pixel_distance_threshold,
                    ),
                )
                order = np.argsort(distance, kind="stable")
                top5 = order[-5:]
                top10 = order[-10:]
                cx, cy, spread = _spatial_moments(distance)
                values = {
                    "delta_global": delta_global.astype(np.float32),
                    "delta_patch_flat": delta_patch.reshape(-1).astype(np.float32),
                    "global_distance": float(_normalized_cosine_distance(g_first, g_second)),
                    "global_l2": float(np.linalg.norm(delta_global)),
                    "patch_mean": float(distance.mean()),
                    "patch_max": float(distance.max()),
                    "patch_top5_mean": float(distance[top5].mean()),
                    "patch_top10_mean": float(distance[top10].mean()),
                    "robot_patch_mean": _weighted_region_mean(distance, coverage[0]),
                    "relevant_patch_mean": _weighted_region_mean(distance, coverage[1]),
                    "interaction_patch_mean": _weighted_region_mean(distance, interaction),
                    "background_patch_mean": _weighted_region_mean(distance, coverage[3]),
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "spatial_spread": spread,
                    "distance_map": distance.astype(np.float32),
                    "pair_fractional_coverage": coverage.astype(np.float32),
                    "interaction_indicator": interaction.astype(np.float32),
                    "top5_fractional_coverage": coverage[:, top5].mean(axis=1).astype(np.float32),
                    "top10_fractional_coverage": coverage[:, top10].mean(axis=1).astype(np.float32),
                    "top5_interaction_fraction": float(interaction[top5].mean()),
                    "top10_interaction_fraction": float(interaction[top10].mean()),
                }
                for name, value in values.items():
                    rows[name].append(value)

    arrays = {name: np.asarray(values) for name, values in rows.items()}
    patch_summary = np.column_stack(
        [
            arrays["patch_mean"],
            arrays["patch_max"],
            arrays["patch_top5_mean"],
            arrays["patch_top10_mean"],
            arrays["robot_patch_mean"],
            arrays["relevant_patch_mean"],
            arrays["interaction_patch_mean"],
            arrays["centroid_x"],
            arrays["centroid_y"],
            arrays["spatial_spread"],
        ]
    )
    targets = {
        name: np.asarray(external["targets"][name], dtype=np.float64) for name in TARGET_NAMES
    }
    contact = np.asarray(external["contact"], dtype=np.int64)
    if not all(value.shape == (960,) for value in targets.values()) or contact.shape != (960,):
        raise ValueError("external target shape changed")
    if not all(np.isfinite(value).all() for value in (*arrays.values(), *targets.values())):
        raise FloatingPointError("natural spatial dataset contains NaN/Inf")
    return {
        "metadata": visual_metadata,
        "arrays": arrays,
        "patch_summary": patch_summary.astype(np.float64),
        "targets": targets,
        "contact": contact,
    }


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    components: np.ndarray


def fit_randomized_pca(
    values: np.ndarray,
    components: int,
    seed: int,
    oversample: int = 8,
    power_iterations: int = 2,
) -> tuple[PCAFit, np.ndarray]:
    """Fit deterministic low-rank PCA without materializing a D-by-D covariance."""

    x = np.asarray(values, dtype=np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    width = min(components + oversample, centered.shape[0], centered.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(centered.shape[1], width))
    projected = centered @ omega
    for _ in range(power_iterations):
        projected = centered @ (centered.T @ projected)
    q, _ = np.linalg.qr(projected, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
    basis = vt[:components]
    scores = centered @ basis.T
    return PCAFit(mean.astype(np.float64), basis.astype(np.float64)), scores.astype(np.float64)


def transform_pca(values: np.ndarray, fit: PCAFit) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - fit.mean) @ fit.components.T


def pca_to_arrays(prefix: str, fit: PCAFit) -> dict[str, np.ndarray]:
    return {f"{prefix}__mean": fit.mean, f"{prefix}__components": fit.components}


def pca_from_arrays(prefix: str, arrays: Any) -> PCAFit:
    return PCAFit(
        np.asarray(arrays[f"{prefix}__mean"], dtype=np.float64),
        np.asarray(arrays[f"{prefix}__components"], dtype=np.float64),
    )


@dataclass(frozen=True)
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    intercept: np.ndarray
    alpha: float


def fit_ridge(values: np.ndarray, target: np.ndarray, alpha: float) -> RidgeFit:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (x - mean) / scale
    intercept = y.mean(axis=0)
    centered = y - intercept
    n = max(x.shape[0], 1)
    gram = standardized.T @ standardized / n
    rhs = standardized.T @ centered / n
    weight = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    return RidgeFit(mean, scale, weight, intercept, float(alpha))


def predict_ridge(values: np.ndarray, fit: RidgeFit) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) - fit.mean) / fit.scale) @ fit.weight + fit.intercept


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


def select_ridge_alpha(
    values: np.ndarray,
    target: np.ndarray,
    init_state: np.ndarray,
    grid: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    init = np.asarray(init_state)
    for alpha in grid:
        folds = []
        for held_out in sorted(np.unique(init).tolist()):
            train = init != held_out
            valid = ~train
            fit = fit_ridge(values[train], target[train], alpha)
            prediction = predict_ridge(values[valid], fit).reshape(np.asarray(target[valid]).shape)
            rmse = float(np.sqrt(np.mean(np.square(prediction - target[valid]))))
            folds.append({"held_out_init": int(held_out), "rmse": rmse})
        rows.append(
            {
                "alpha": float(alpha),
                "mean_fold_rmse": float(np.mean([fold["rmse"] for fold in folds])),
                "folds": folds,
            }
        )
    selected = min(rows, key=lambda row: (row["mean_fold_rmse"], -row["alpha"]))
    return float(selected["alpha"]), rows


def scalar_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    true = np.asarray(target, dtype=np.float64).reshape(-1)
    residual = pred - true
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum(np.square(true - true.mean())))
    r2 = None if denominator <= 1.0e-20 else float(1.0 - np.sum(np.square(residual)) / denominator)
    correlation = safe_correlation(pred, true)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": correlation["pearson"],
        "spearman": correlation["spearman"],
    }


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.asarray(first, dtype=np.float64) @ np.swapaxes(
        np.asarray(second, dtype=np.float64), -1, -2
    )
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def build_external_targets(
    base: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    proprio_mean: np.ndarray,
    proprio_std: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct the frozen missing-state scalar targets without SciPy."""

    real_p = np.asarray(base["real_proprio"], dtype=np.float64)
    standardized = (real_p - np.asarray(proprio_mean).reshape(1, 1, 1, 12)) / np.asarray(
        proprio_std
    ).reshape(1, 1, 1, 12)
    ee_rot = np.asarray(state["ee_rotmat"], dtype=np.float64)
    object_pos = np.asarray(state["object_pos"], dtype=np.float64)
    object_rot = np.asarray(state["object_rotmat"], dtype=np.float64)
    object_mask = np.asarray(state["object_mask"], dtype=bool)
    contact = np.asarray(state["contact_any"], dtype=bool)
    targets = {name: [] for name in TARGET_NAMES}
    contact_target = []
    metadata = {name: [] for name in ("branch", "task", "init_state", "horizon", "pair")}
    for branch in range(40):
        valid = object_mask[branch]
        if not np.any(valid):
            raise ValueError(f"branch {branch} has no relevant object")
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            for horizon in range(8):
                ee_delta = standardized[branch, first, horizon, 0:3] - standardized[
                    branch, second, horizon, 0:3
                ]
                targets["ee_translation"].append(float(np.linalg.norm(ee_delta)))
                targets["ee_rotation"].append(
                    float(_rotation_angle(ee_rot[branch, first, horizon], ee_rot[branch, second, horizon]))
                )
                object_delta = object_pos[branch, first, horizon, valid] - object_pos[
                    branch, second, horizon, valid
                ]
                object_angle = _rotation_angle(
                    object_rot[branch, first, horizon, valid],
                    object_rot[branch, second, horizon, valid],
                )
                targets["object_translation"].append(
                    float(np.linalg.norm(object_delta, axis=-1).max())
                )
                targets["object_rotation"].append(float(object_angle.max()))
                relative_first_pos = object_pos[branch, first, horizon, valid] - real_p[
                    branch, first, horizon, 0:3
                ]
                relative_second_pos = object_pos[branch, second, horizon, valid] - real_p[
                    branch, second, horizon, 0:3
                ]
                relative_first_rot = object_rot[branch, first, horizon, valid] @ np.swapaxes(
                    ee_rot[branch, first, horizon], -1, -2
                )
                relative_second_rot = object_rot[branch, second, horizon, valid] @ np.swapaxes(
                    ee_rot[branch, second, horizon], -1, -2
                )
                targets["relative_translation"].append(
                    float(np.linalg.norm(relative_first_pos - relative_second_pos, axis=-1).max())
                )
                targets["relative_rotation"].append(
                    float(_rotation_angle(relative_first_rot, relative_second_rot).max())
                )
                contact_target.append(
                    int(
                        np.any(
                            np.logical_xor(
                                contact[branch, first, horizon], contact[branch, second, horizon]
                            )
                        )
                    )
                )
                metadata["branch"].append(branch)
                metadata["task"].append(int(base["task_id"][branch]))
                metadata["init_state"].append(int(base["init_state_index"][branch]))
                metadata["horizon"].append(horizon + 1)
                metadata["pair"].append(pair_index)
    return {
        "metadata": {name: np.asarray(value, dtype=np.int64) for name, value in metadata.items()},
        "targets": {name: np.asarray(value, dtype=np.float64) for name, value in targets.items()},
        "contact": np.asarray(contact_target, dtype=np.int64),
    }


@dataclass(frozen=True)
class LogisticFit:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    intercept: float
    alpha: float


def fit_logistic(values: np.ndarray, target: np.ndarray, alpha: float = 1.0) -> LogisticFit:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError("contact logistic target must contain both classes")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (x - mean) / scale
    positive = max(float(np.sum(y == 1.0)), 1.0)
    negative = max(float(np.sum(y == 0.0)), 1.0)
    sample_weight = np.where(y == 1.0, y.size / (2.0 * positive), y.size / (2.0 * negative))

    design = np.column_stack((standardized, np.ones(standardized.shape[0])))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha) / y.size
    penalty[-1, -1] = 0.0
    for _ in range(100):
        logits = np.clip(design @ parameters, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = sample_weight * (probability - y) / y.size
        gradient = design.T @ residual + penalty @ parameters
        curvature = sample_weight * probability * (1.0 - probability) / y.size
        hessian = design.T @ (design * curvature[:, None]) + penalty + 1.0e-8 * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        parameters -= step
        if np.linalg.norm(step) <= 1.0e-9:
            break
    if not np.isfinite(parameters).all():
        raise RuntimeError("logistic fit produced non-finite parameters")
    return LogisticFit(mean, scale, parameters[:-1], float(parameters[-1]), float(alpha))


def predict_logistic(values: np.ndarray, fit: LogisticFit) -> np.ndarray:
    standardized = (np.asarray(values, dtype=np.float64) - fit.mean) / fit.scale
    logits = np.clip(standardized @ fit.weight + fit.intercept, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def logistic_to_arrays(prefix: str, fit: LogisticFit) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__mean": fit.mean,
        f"{prefix}__scale": fit.scale,
        f"{prefix}__weight": fit.weight,
        f"{prefix}__intercept": np.asarray(fit.intercept, dtype=np.float64),
        f"{prefix}__alpha": np.asarray(fit.alpha, dtype=np.float64),
    }


def logistic_from_arrays(prefix: str, arrays: Any) -> LogisticFit:
    return LogisticFit(
        np.asarray(arrays[f"{prefix}__mean"], dtype=np.float64),
        np.asarray(arrays[f"{prefix}__scale"], dtype=np.float64),
        np.asarray(arrays[f"{prefix}__weight"], dtype=np.float64),
        float(np.asarray(arrays[f"{prefix}__intercept"])),
        float(np.asarray(arrays[f"{prefix}__alpha"])),
    )


def contact_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    true = np.asarray(target, dtype=np.int64).reshape(-1)
    positive = true == 1
    negative = true == 0
    if not np.any(positive) or not np.any(negative):
        return {"n": int(true.size), "positives": int(positive.sum()), "auroc": None, "balanced_accuracy": None}
    ranks = _rankdata(score)
    auc = (float(ranks[positive].sum()) - positive.sum() * (positive.sum() + 1) / 2.0) / (
        positive.sum() * negative.sum()
    )
    predicted = score >= 0.5
    tpr = float(np.mean(predicted[positive]))
    tnr = float(np.mean(~predicted[negative]))
    return {
        "n": int(true.size),
        "positives": int(positive.sum()),
        "auroc": float(auc),
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }


def cut_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": scalar_metrics(prediction, target), "horizon": [], "task": [], "pair": []}
    for field, values in (("horizon", range(1, 9)), ("task", range(10)), ("pair", range(3))):
        for value in values:
            mask = metadata[field] == value
            label: Any = PAIR_NAMES[value] if field == "pair" else value
            result[field].append({field: label, **scalar_metrics(prediction[mask], target[mask])})
    return result
