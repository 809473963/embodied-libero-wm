"""Feature construction for the frozen visual-target suitability audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.probes.missing_state import PAIR_INDICES, PAIR_NAMES


MASK_NAMES = ("robot", "relevant", "other", "background")
MASK_INDEX = {name: index for index, name in enumerate(MASK_NAMES)}
VARIANT_NAMES = (
    "full",
    "robot_only",
    "relevant_object_only",
    "robot_removed",
    "relevant_object_removed",
    "background_only",
)
FEATURE_GROUPS = ("full_pixel", "robot_pixel", "relevant_pixel", "visibility", "patch")
FEATURE_SET_GROUPS = {
    "V0": ("full_pixel",),
    "V1": ("full_pixel", "robot_pixel"),
    "V2": ("full_pixel", "robot_pixel", "relevant_pixel"),
    "V3": ("full_pixel", "robot_pixel", "relevant_pixel", "visibility"),
    "V4": ("full_pixel", "robot_pixel", "relevant_pixel", "visibility", "patch"),
}
FEATURE_NAMES = {
    "full_pixel": ("pixel_mae", "pixel_rmse", "unequal_fraction", "pixel_p90"),
    "robot_pixel": (
        "robot_mean",
        "robot_p90",
        "robot_changed_fraction",
        "robot_energy_contribution",
        "robot_union_area_fraction",
        "robot_area_difference",
    ),
    "relevant_pixel": (
        "relevant_mean",
        "relevant_p90",
        "relevant_changed_fraction",
        "relevant_energy_contribution",
        "relevant_union_area_fraction",
        "relevant_area_difference",
    ),
    "visibility": (
        "robot_visible_area_difference",
        "relevant_visible_area_difference",
        "object_occlusion_fraction_difference",
        "object_occlusion_fraction_mean",
    ),
    "patch": (
        "patch_mean",
        "patch_max",
        "patch_top10_mean",
        "patch_robot_mean",
        "patch_relevant_mean",
        "patch_other_mean",
        "patch_background_mean",
        "top10_robot_fraction",
        "top10_relevant_fraction",
        "top10_other_fraction",
        "top10_background_fraction",
    ),
}


def masked_variants(image: np.ndarray, masks: np.ndarray, replacement_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Create the six frozen diagnostic variants for one RGB frame."""

    rgb = np.asarray(image, dtype=np.uint8)
    region = np.asarray(masks, dtype=bool)
    if rgb.shape != (128, 128, 3) or region.shape != (4, 128, 128):
        raise ValueError(f"unexpected image/mask shape: {rgb.shape}, {region.shape}")
    if not np.all(region.sum(axis=0) == 1):
        raise ValueError("semantic masks must be an exact partition")
    replacement = np.asarray(replacement_rgb, dtype=np.uint8).reshape(1, 1, 3)

    def keep(mask: np.ndarray) -> np.ndarray:
        return np.where(mask[..., None], rgb, replacement).astype(np.uint8)

    def remove(mask: np.ndarray) -> np.ndarray:
        return np.where(mask[..., None], replacement, rgb).astype(np.uint8)

    robot = region[MASK_INDEX["robot"]]
    relevant = region[MASK_INDEX["relevant"]]
    non_robot_non_relevant = ~(robot | relevant)
    return {
        "full": rgb.copy(),
        "robot_only": keep(robot),
        "relevant_object_only": keep(relevant),
        "robot_removed": remove(robot),
        "relevant_object_removed": remove(relevant),
        "background_only": keep(non_robot_non_relevant),
    }


def patch_mask_coverage(mask: np.ndarray) -> np.ndarray:
    """Align one 128x128 mask to CLIP's 7x7 patch grid.

    The pinned CLIP processor maps the square 128x128 input to 224x224 without
    a crop.  Nearest-neighbor mask resizing followed by exact 32x32 pooling is
    the preregistered spatial alignment; it is diagnostic rather than a claim
    of transformer attention attribution.
    """

    value = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    resized = F.interpolate(value, size=(224, 224), mode="nearest")
    pooled = F.avg_pool2d(resized, kernel_size=32, stride=32)
    return pooled[0, 0].reshape(49).numpy().astype(np.float64)


def all_patch_coverages(masks: np.ndarray) -> np.ndarray:
    array = np.asarray(masks, dtype=bool)
    if array.ndim != 6 or array.shape[3:] != (4, 128, 128):
        raise ValueError(f"expected [B,3,8,4,128,128], got {array.shape}")
    result = np.empty(array.shape[:3] + (4, 49), dtype=np.float32)
    for branch in range(array.shape[0]):
        for candidate in range(3):
            for horizon in range(8):
                for region in range(4):
                    result[branch, candidate, horizon, region] = patch_mask_coverage(
                        array[branch, candidate, horizon, region]
                    )
    return result


def _safe_region_summary(pixel_difference: np.ndarray, region: np.ndarray) -> tuple[float, float, float, float]:
    gray = np.asarray(pixel_difference, dtype=np.float64).mean(axis=-1)
    selected = gray[np.asarray(region, dtype=bool)]
    if selected.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(selected.mean()),
        float(np.quantile(selected, 0.90)),
        float(np.mean(selected > 0.0)),
        float(selected.size / gray.size),
    )


def _exclusive_pair_regions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Build a disjoint pair mask with frozen robot>relevant>other priority."""

    union = np.asarray(first, dtype=bool) | np.asarray(second, dtype=bool)
    result = np.zeros_like(union)
    available = np.ones(union.shape[1:], dtype=bool)
    for index in range(4):
        selected = union[index] & available
        result[index] = selected
        available &= ~selected
    # Candidate-specific partitions guarantee full coverage; keep an explicit
    # guard rather than silently assigning uncovered pixels.
    if np.any(available) or not np.all(result.sum(axis=0) == 1):
        raise ValueError("exclusive pair masks do not partition the image")
    return result


def _patch_pair_features(
    first_tokens: np.ndarray,
    second_tokens: np.ndarray,
    first_coverage: np.ndarray,
    second_coverage: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, dict[str, float]]:
    first = np.asarray(first_tokens, dtype=np.float64)
    second = np.asarray(second_tokens, dtype=np.float64)
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1.0e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1.0e-12)
    distance = np.maximum(1.0 - np.sum(first * second, axis=-1), 0.0)
    coverage = (np.asarray(first_coverage, dtype=np.float64) + np.asarray(second_coverage, dtype=np.float64)) / 2.0
    regional = []
    for region in range(4):
        weight = coverage[region]
        regional.append(float(np.sum(distance * weight) / max(np.sum(weight), 1.0e-12)))
    order = np.argsort(distance, kind="stable")[-top_k:]
    category = np.argmax(coverage, axis=0)
    top_fraction = [float(np.mean(category[order] == region)) for region in range(4)]
    features = np.asarray(
        [
            float(distance.mean()),
            float(distance.max()),
            float(distance[order].mean()),
            *regional,
            *top_fraction,
        ],
        dtype=np.float64,
    )
    scalars = {
        "patch_mean": float(distance.mean()),
        "patch_max": float(distance.max()),
        "patch_top10_mean": float(distance[order].mean()),
        "patch_robot_mean": regional[0],
        "patch_relevant_mean": regional[1],
        "patch_other_mean": regional[2],
        "patch_background_mean": regional[3],
        "top10_robot_fraction": top_fraction[0],
        "top10_relevant_fraction": top_fraction[1],
        "top10_other_fraction": top_fraction[2],
        "top10_background_fraction": top_fraction[3],
    }
    return features, scalars


def build_visual_pair_dataset(
    base: dict[str, np.ndarray],
    mask_data: dict[str, np.ndarray],
    patch_tokens: np.ndarray,
    patch_coverage: np.ndarray,
    top_k: int = 10,
) -> dict[str, Any]:
    """Construct frozen V0--V4 features at branch/pair/horizon grain."""

    rgb = np.asarray(base["real_rgb"], dtype=np.uint8)
    real_z = np.asarray(base["real_z"], dtype=np.float64)
    actions = np.asarray(base["candidates"], dtype=np.float64)
    masks = np.asarray(mask_data["masks"], dtype=bool)
    no_robot_relevant = np.asarray(mask_data["no_robot_relevant_mask"], dtype=bool)
    tokens = np.asarray(patch_tokens, dtype=np.float64)
    coverage = np.asarray(patch_coverage, dtype=np.float64)
    expected = (rgb.shape[0], 3, 8)
    if masks.shape != expected + (4, 128, 128):
        raise ValueError(f"mask shape mismatch: {masks.shape}")
    if no_robot_relevant.shape != expected + (128, 128):
        raise ValueError(f"no-robot relevant mask shape mismatch: {no_robot_relevant.shape}")
    if tokens.shape != expected + (49, 512):
        raise ValueError(f"patch token shape mismatch: {tokens.shape}")
    if coverage.shape != expected + (4, 49):
        raise ValueError(f"patch coverage shape mismatch: {coverage.shape}")

    groups = {name: [] for name in FEATURE_GROUPS}
    target = []
    delta_z = []
    metadata = {name: [] for name in ("branch", "task", "init_state", "horizon", "pair")}
    scalar_names = (
        "pixel_mae",
        "pixel_rmse",
        "unequal_pixel_fraction",
        "pixel_p90",
        "robot_pixel_mean",
        "robot_pixel_p90",
        "robot_changed_fraction",
        "robot_energy_contribution",
        "robot_union_area_fraction",
        "robot_area_difference",
        "relevant_pixel_mean",
        "relevant_pixel_p90",
        "relevant_changed_fraction",
        "relevant_energy_contribution",
        "relevant_union_area_fraction",
        "relevant_area_difference",
        "other_pixel_mean",
        "other_energy_contribution",
        "background_pixel_mean",
        "background_energy_contribution",
        "robot_visible_area_difference",
        "relevant_visible_area_difference",
        "object_occlusion_fraction_difference",
        "object_occlusion_fraction_mean",
        "action_translation",
        "action_rotation",
        *FEATURE_NAMES["patch"],
    )
    scalars = {name: [] for name in scalar_names}
    for branch in range(rgb.shape[0]):
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            for horizon in range(8):
                image_first = rgb[branch, first, horizon].astype(np.float64) / 255.0
                image_second = rgb[branch, second, horizon].astype(np.float64) / 255.0
                difference = np.abs(image_first - image_second)
                gray = difference.mean(axis=-1)
                full_pixel = np.asarray(
                    [
                        float(difference.mean()),
                        float(np.sqrt(np.mean(np.square(difference)))),
                        float(np.mean(np.any(difference > 0.0, axis=-1))),
                        float(np.quantile(gray, 0.90)),
                    ],
                    dtype=np.float64,
                )
                union = masks[branch, first, horizon] | masks[branch, second, horizon]
                exclusive = _exclusive_pair_regions(
                    masks[branch, first, horizon], masks[branch, second, horizon]
                )
                total_energy = float(np.sum(difference))
                region_values: dict[str, tuple[float, float, float, float]] = {}
                contributions: dict[str, float] = {}
                area_difference: dict[str, float] = {}
                for region_index, region_name in enumerate(MASK_NAMES):
                    region_values[region_name] = _safe_region_summary(difference, union[region_index])
                    contributions[region_name] = float(
                        np.sum(difference[exclusive[region_index]]) / max(total_energy, 1.0e-12)
                    )
                    area_difference[region_name] = float(
                        abs(
                            masks[branch, first, horizon, region_index].sum()
                            - masks[branch, second, horizon, region_index].sum()
                        )
                        / (128 * 128)
                    )
                robot_summary = region_values["robot"]
                relevant_summary = region_values["relevant"]
                robot_pixel = np.asarray(
                    [*robot_summary[:3], contributions["robot"], robot_summary[3], area_difference["robot"]],
                    dtype=np.float64,
                )
                relevant_pixel = np.asarray(
                    [
                        *relevant_summary[:3],
                        contributions["relevant"],
                        relevant_summary[3],
                        area_difference["relevant"],
                    ],
                    dtype=np.float64,
                )
                object_visible = [
                    float(masks[branch, candidate, horizon, MASK_INDEX["relevant"]].sum())
                    for candidate in (first, second)
                ]
                object_no_robot = [
                    float(no_robot_relevant[branch, candidate, horizon].sum())
                    for candidate in (first, second)
                ]
                occlusion = [
                    max(potential - visible, 0.0) / max(potential, 1.0)
                    for visible, potential in zip(object_visible, object_no_robot)
                ]
                visibility = np.asarray(
                    [
                        area_difference["robot"],
                        area_difference["relevant"],
                        abs(occlusion[0] - occlusion[1]),
                        float(np.mean(occlusion)),
                    ],
                    dtype=np.float64,
                )
                patch, patch_scalar = _patch_pair_features(
                    tokens[branch, first, horizon],
                    tokens[branch, second, horizon],
                    coverage[branch, first, horizon],
                    coverage[branch, second, horizon],
                    top_k,
                )
                z_first = real_z[branch, first, horizon]
                z_second = real_z[branch, second, horizon]
                clip_distance = max(
                    1.0
                    - float(
                        np.dot(z_first, z_second)
                        / max(np.linalg.norm(z_first) * np.linalg.norm(z_second), 1.0e-12)
                    ),
                    0.0,
                )
                action_translation = float(
                    np.linalg.norm(actions[branch, first, : horizon + 1, 0:3] - actions[branch, second, : horizon + 1, 0:3])
                )
                action_rotation = float(
                    np.linalg.norm(actions[branch, first, : horizon + 1, 3:6] - actions[branch, second, : horizon + 1, 3:6])
                )

                groups["full_pixel"].append(full_pixel)
                groups["robot_pixel"].append(robot_pixel)
                groups["relevant_pixel"].append(relevant_pixel)
                groups["visibility"].append(visibility)
                groups["patch"].append(patch)
                target.append(clip_distance)
                delta_z.append(z_first - z_second)
                metadata["branch"].append(branch)
                metadata["task"].append(int(base["task_id"][branch]))
                metadata["init_state"].append(int(base["init_state_index"][branch]))
                metadata["horizon"].append(horizon + 1)
                metadata["pair"].append(pair_index)
                scalar_row = {
                    "pixel_mae": full_pixel[0],
                    "pixel_rmse": full_pixel[1],
                    "unequal_pixel_fraction": full_pixel[2],
                    "pixel_p90": full_pixel[3],
                    "robot_pixel_mean": robot_summary[0],
                    "robot_pixel_p90": robot_summary[1],
                    "robot_changed_fraction": robot_summary[2],
                    "robot_energy_contribution": contributions["robot"],
                    "robot_union_area_fraction": robot_summary[3],
                    "robot_area_difference": area_difference["robot"],
                    "relevant_pixel_mean": relevant_summary[0],
                    "relevant_pixel_p90": relevant_summary[1],
                    "relevant_changed_fraction": relevant_summary[2],
                    "relevant_energy_contribution": contributions["relevant"],
                    "relevant_union_area_fraction": relevant_summary[3],
                    "relevant_area_difference": area_difference["relevant"],
                    "other_pixel_mean": region_values["other"][0],
                    "other_energy_contribution": contributions["other"],
                    "background_pixel_mean": region_values["background"][0],
                    "background_energy_contribution": contributions["background"],
                    "robot_visible_area_difference": visibility[0],
                    "relevant_visible_area_difference": visibility[1],
                    "object_occlusion_fraction_difference": visibility[2],
                    "object_occlusion_fraction_mean": visibility[3],
                    "action_translation": action_translation,
                    "action_rotation": action_rotation,
                    **patch_scalar,
                }
                for name in scalars:
                    scalars[name].append(float(scalar_row[name]))

    group_arrays = {name: np.asarray(value, dtype=np.float64) for name, value in groups.items()}
    sets = {
        name: np.concatenate([group_arrays[group] for group in group_names], axis=1)
        for name, group_names in FEATURE_SET_GROUPS.items()
    }
    result = {
        "groups": group_arrays,
        "sets": sets,
        "target_clip": np.asarray(target, dtype=np.float64),
        "target_delta_z": np.asarray(delta_z, dtype=np.float64),
        "metadata": {name: np.asarray(value, dtype=np.int64) for name, value in metadata.items()},
        "scalars": {name: np.asarray(value, dtype=np.float64) for name, value in scalars.items()},
    }
    numeric = [*group_arrays.values(), *sets.values(), result["target_clip"], result["target_delta_z"], *result["scalars"].values()]
    if not all(np.isfinite(value).all() for value in numeric):
        raise FloatingPointError("visual pair dataset contains NaN/Inf")
    return result

