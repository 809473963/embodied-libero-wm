"""Natural image-space targets for the interaction-relevance audit.

The functions in this module are deterministic and operate only on frozen
natural-image features, semantic masks, camera matrices, and matched simulator
state.  They do not read rewards, success, or controller outcomes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.probes.spatial_relevance import PAIR_INDICES, PAIR_NAMES


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
PATCH_GRID = 7
OBJECT_SLOTS = 2

TARGET_NAMES = (
    "ee_image_displacement",
    "object_centroid_displacement",
    "object_mask_difference",
    "visible_area_difference",
    "relative_image_displacement",
    "ee_object_distance_change",
    "strict_proximity_change",
    "robot_object_adjacency_change",
)

TARGET_FAMILIES = {
    "robot": ("ee_image_displacement",),
    "object": (
        "object_centroid_displacement",
        "object_mask_difference",
        "visible_area_difference",
    ),
    "relative": (
        "relative_image_displacement",
        "ee_object_distance_change",
    ),
    "interaction": (
        "strict_proximity_change",
        "robot_object_adjacency_change",
    ),
}

SCALAR_REPRESENTATIONS = (
    "global_distance",
    "patch_mean",
    "patch_top5_mean",
    "patch_top10_mean",
    "robot_patch_mean",
    "relevant_patch_mean",
    "strict_interaction_patch_mean",
    "background_patch_mean",
)


def project_world_points(
    points: np.ndarray,
    world_to_pixel: np.ndarray,
    convention: str,
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world xyz to unclipped natural-image (u, v) coordinates."""

    xyz = np.asarray(points, dtype=np.float64)
    transform = np.asarray(world_to_pixel, dtype=np.float64)
    if xyz.shape[-1] != 3 or transform.shape != (4, 4):
        raise ValueError("unexpected camera projection shape")
    homogeneous = np.concatenate((xyz, np.ones(xyz.shape[:-1] + (1,))), axis=-1)
    projected = homogeneous @ transform.T
    depth = projected[..., 2]
    safe_depth = np.where(np.abs(depth) > 1.0e-12, depth, np.nan)
    u = projected[..., 0] / safe_depth
    v = projected[..., 1] / safe_depth
    if convention == "vertical_flip_to_direct_render":
        v = (height - 1.0) - v
    elif convention != "direct_render":
        raise ValueError(f"unknown projection convention: {convention}")
    return np.stack((u, v), axis=-1), depth


def point_to_mask_distance(point_uv: np.ndarray, mask: np.ndarray) -> float | None:
    region = np.asarray(mask, dtype=bool)
    if region.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or not np.any(region):
        return None
    yx = np.argwhere(region).astype(np.float64)
    point = np.asarray(point_uv, dtype=np.float64)
    delta = yx - np.asarray([point[1], point[0]], dtype=np.float64)
    return float(np.sqrt(np.min(np.sum(np.square(delta), axis=1))))


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    region = np.asarray(mask, dtype=bool)
    if region.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"unexpected object mask shape: {region.shape}")
    visible_pixels = int(region.sum())
    if visible_pixels == 0:
        return {
            "visible": False,
            "area_fraction": 0.0,
            "centroid": None,
            "bbox": None,
        }
    yx = np.argwhere(region)
    ymin, xmin = yx.min(axis=0)
    ymax, xmax = yx.max(axis=0)
    centroid = np.asarray(
        [float(yx[:, 1].mean()) / IMAGE_WIDTH, float(yx[:, 0].mean()) / IMAGE_HEIGHT],
        dtype=np.float64,
    )
    bbox = np.asarray(
        [
            float(xmin) / IMAGE_WIDTH,
            float(ymin) / IMAGE_HEIGHT,
            float(xmax - xmin + 1) / IMAGE_WIDTH,
            float(ymax - ymin + 1) / IMAGE_HEIGHT,
        ],
        dtype=np.float64,
    )
    return {
        "visible": True,
        "area_fraction": visible_pixels / float(IMAGE_HEIGHT * IMAGE_WIDTH),
        "centroid": centroid,
        "bbox": bbox,
    }


def mask_pair_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = np.logical_or(a, b)
    intersection = np.logical_and(a, b)
    union_count = int(union.sum())
    iou = 1.0 if union_count == 0 else float(intersection.sum() / union_count)
    return {
        "iou_distance": 1.0 - iou,
        "symmetric_difference_fraction": float(np.logical_xor(a, b).mean()),
    }


def _binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-radius binary dilation implemented with an integral image."""

    source = np.asarray(mask, dtype=np.uint8)
    padded = np.pad(source, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    diameter = 2 * radius + 1
    window = (
        integral[diameter:, diameter:]
        - integral[:-diameter, diameter:]
        - integral[diameter:, :-diameter]
        + integral[:-diameter, :-diameter]
    )
    return window > 0


def _boundary(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if not np.any(source):
        return source.copy()
    interior = source.copy()
    interior[1:, :] &= source[:-1, :]
    interior[:-1, :] &= source[1:, :]
    interior[:, 1:] &= source[:, :-1]
    interior[:, :-1] &= source[:, 1:]
    return source & ~interior


def boundary_adjacency(object_mask: np.ndarray, robot_mask: np.ndarray, radius: int) -> float:
    edge = _boundary(object_mask)
    count = int(edge.sum())
    if count == 0:
        return 0.0
    near_robot = _binary_dilation(robot_mask, radius)
    return float(np.logical_and(edge, near_robot).sum() / count)


def strict_patch_weights(
    relevant_fraction: np.ndarray,
    ee_uv: np.ndarray,
    sigma_pixels: float,
) -> np.ndarray:
    coverage = np.asarray(relevant_fraction, dtype=np.float64).reshape(49)
    coordinate = (np.arange(PATCH_GRID, dtype=np.float64) + 0.5) * IMAGE_WIDTH / PATCH_GRID
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    point = np.asarray(ee_uv, dtype=np.float64)
    squared = np.square(xx - point[0]) + np.square(yy - point[1])
    return coverage * np.exp(-0.5 * squared.reshape(-1) / float(sigma_pixels) ** 2)


def build_image_space_dataset(
    spatial: dict[str, np.ndarray],
    frozen_masks: dict[str, np.ndarray],
    object_masks: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    camera_inventory: dict[str, Any],
    strict_sigma_pixels: float,
    adjacency_radius_pixels: int,
    near_contact_pixels: float,
) -> dict[str, Any]:
    """Construct aligned image-space targets and strict local patch summary."""

    metadata = {
        key: np.asarray(spatial[f"metadata__{key}"], dtype=np.int64)
        for key in ("branch", "task", "init_state", "horizon", "pair")
    }
    distance_map = np.asarray(spatial["diagnostic__distance_map"], dtype=np.float64)
    pair_coverage = np.asarray(
        spatial["diagnostic__pair_fractional_coverage"], dtype=np.float64
    )
    semantic = np.asarray(frozen_masks["masks"], dtype=bool)
    per_object = np.asarray(object_masks["per_object_masks"], dtype=bool)
    ee_world = np.asarray(state["ee_pos"], dtype=np.float64)
    valid_objects = np.asarray(state["object_mask"], dtype=bool)
    expected = (40, 3, 8)
    if semantic.shape != expected + (4, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("frozen semantic mask shape changed")
    if per_object.shape != expected + (OBJECT_SLOTS, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("per-object mask shape changed")
    if ee_world.shape != expected + (3,) or valid_objects.shape != (40, OBJECT_SLOTS):
        raise ValueError("state shape changed")
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        if not np.array_equal(np.asarray(object_masks[key]), np.asarray(state[key])):
            raise ValueError(f"image/state identity mismatch: {key}")

    convention = camera_inventory["selected_projection_convention"]
    task_matrices = {
        int(row["task_id"]): np.asarray(row["world_to_pixel"], dtype=np.float64)
        for row in camera_inventory["tasks"]
    }
    projected_ee = np.empty(expected + (2,), dtype=np.float64)
    projection_depth = np.empty(expected, dtype=np.float64)
    for branch in range(40):
        task_id = int(state["task_id"][branch])
        pixels, depth = project_world_points(
            ee_world[branch], task_matrices[task_id], convention
        )
        projected_ee[branch] = pixels
        projection_depth[branch] = depth

    targets = {name: [] for name in TARGET_NAMES}
    valid = {name: [] for name in TARGET_NAMES}
    diagnostics: dict[str, list[Any]] = {
        "objectwise_centroid_displacement": [],
        "objectwise_iou_distance": [],
        "objectwise_symmetric_difference": [],
        "objectwise_visible_area_difference": [],
        "objectwise_relative_displacement": [],
        "objectwise_distance_change": [],
        "objectwise_proximity_change": [],
        "objectwise_adjacency_change": [],
        "objectwise_visible_both": [],
        "objectwise_centroid_first": [],
        "objectwise_centroid_second": [],
        "objectwise_bbox_first": [],
        "objectwise_bbox_second": [],
        "strict_interaction_patch_mean": [],
        "near_contact_either": [],
    }
    diagonal = float(np.hypot(IMAGE_WIDTH, IMAGE_HEIGHT))

    def aggregate(values: list[float], name: str) -> None:
        finite = [float(value) for value in values if np.isfinite(value)]
        targets[name].append(max(finite) if finite else np.nan)
        valid[name].append(bool(finite))

    row_index = 0
    for branch in range(40):
        objects = valid_objects[branch]
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            del pair_index
            for horizon in range(8):
                ee_first_px = projected_ee[branch, first, horizon]
                ee_second_px = projected_ee[branch, second, horizon]
                ee_first = ee_first_px / np.asarray([IMAGE_WIDTH, IMAGE_HEIGHT])
                ee_second = ee_second_px / np.asarray([IMAGE_WIDTH, IMAGE_HEIGHT])
                ee_distance = float(np.linalg.norm(ee_first - ee_second))
                targets["ee_image_displacement"].append(ee_distance)
                valid["ee_image_displacement"].append(True)

                centroid_values: list[float] = []
                mask_values: list[float] = []
                area_values: list[float] = []
                relative_values: list[float] = []
                distance_values: list[float] = []
                proximity_values: list[float] = []
                adjacency_values: list[float] = []
                objectwise = {key: [] for key in diagnostics if key.startswith("objectwise_")}
                near = False
                for object_index in range(OBJECT_SLOTS):
                    if not objects[object_index]:
                        for key in objectwise:
                            if key == "objectwise_visible_both":
                                objectwise[key].append(False)
                            elif "centroid" in key:
                                objectwise[key].append([np.nan, np.nan])
                            elif "bbox" in key:
                                objectwise[key].append([np.nan] * 4)
                            else:
                                objectwise[key].append(np.nan)
                        continue
                    mask_first = per_object[branch, first, horizon, object_index]
                    mask_second = per_object[branch, second, horizon, object_index]
                    geometry_first = mask_geometry(mask_first)
                    geometry_second = mask_geometry(mask_second)
                    pair_metric = mask_pair_metrics(mask_first, mask_second)
                    mask_values.append(pair_metric["symmetric_difference_fraction"])
                    area_change = abs(
                        float(geometry_first["area_fraction"])
                        - float(geometry_second["area_fraction"])
                    )
                    area_values.append(area_change)
                    adjacency_first = boundary_adjacency(
                        mask_first,
                        semantic[branch, first, horizon, 0],
                        adjacency_radius_pixels,
                    )
                    adjacency_second = boundary_adjacency(
                        mask_second,
                        semantic[branch, second, horizon, 0],
                        adjacency_radius_pixels,
                    )
                    adjacency_change = abs(adjacency_first - adjacency_second)
                    adjacency_values.append(adjacency_change)
                    visible_both = bool(
                        geometry_first["visible"] and geometry_second["visible"]
                    )
                    objectwise["objectwise_iou_distance"].append(pair_metric["iou_distance"])
                    objectwise["objectwise_symmetric_difference"].append(
                        pair_metric["symmetric_difference_fraction"]
                    )
                    objectwise["objectwise_visible_area_difference"].append(area_change)
                    objectwise["objectwise_adjacency_change"].append(adjacency_change)
                    objectwise["objectwise_visible_both"].append(visible_both)
                    objectwise["objectwise_centroid_first"].append(
                        geometry_first["centroid"] if geometry_first["centroid"] is not None else [np.nan, np.nan]
                    )
                    objectwise["objectwise_centroid_second"].append(
                        geometry_second["centroid"] if geometry_second["centroid"] is not None else [np.nan, np.nan]
                    )
                    objectwise["objectwise_bbox_first"].append(
                        geometry_first["bbox"] if geometry_first["bbox"] is not None else [np.nan] * 4
                    )
                    objectwise["objectwise_bbox_second"].append(
                        geometry_second["bbox"] if geometry_second["bbox"] is not None else [np.nan] * 4
                    )
                    if visible_both:
                        centroid_first = np.asarray(geometry_first["centroid"], dtype=np.float64)
                        centroid_second = np.asarray(geometry_second["centroid"], dtype=np.float64)
                        centroid = float(np.linalg.norm(centroid_first - centroid_second))
                        relative = float(
                            np.linalg.norm(
                                (centroid_first - ee_first) - (centroid_second - ee_second)
                            )
                        )
                        distance_change = abs(
                            float(np.linalg.norm(centroid_first - ee_first))
                            - float(np.linalg.norm(centroid_second - ee_second))
                        )
                        proximity_first = point_to_mask_distance(ee_first_px, mask_first)
                        proximity_second = point_to_mask_distance(ee_second_px, mask_second)
                        if proximity_first is None or proximity_second is None:
                            proximity_change = np.nan
                        else:
                            proximity_change = abs(proximity_first - proximity_second) / diagonal
                            near |= min(proximity_first, proximity_second) <= near_contact_pixels
                        centroid_values.append(centroid)
                        relative_values.append(relative)
                        distance_values.append(distance_change)
                        if np.isfinite(proximity_change):
                            proximity_values.append(float(proximity_change))
                        objectwise["objectwise_centroid_displacement"].append(centroid)
                        objectwise["objectwise_relative_displacement"].append(relative)
                        objectwise["objectwise_distance_change"].append(distance_change)
                        objectwise["objectwise_proximity_change"].append(proximity_change)
                    else:
                        objectwise["objectwise_centroid_displacement"].append(np.nan)
                        objectwise["objectwise_relative_displacement"].append(np.nan)
                        objectwise["objectwise_distance_change"].append(np.nan)
                        objectwise["objectwise_proximity_change"].append(np.nan)

                aggregate(centroid_values, "object_centroid_displacement")
                aggregate(mask_values, "object_mask_difference")
                aggregate(area_values, "visible_area_difference")
                aggregate(relative_values, "relative_image_displacement")
                aggregate(distance_values, "ee_object_distance_change")
                aggregate(proximity_values, "strict_proximity_change")
                aggregate(adjacency_values, "robot_object_adjacency_change")
                for key, values in objectwise.items():
                    diagnostics[key].append(values)
                diagnostics["near_contact_either"].append(near)

                weights_first = strict_patch_weights(
                    np.asarray(frozen_masks["patch_coverage"])[branch, first, horizon, 1],
                    ee_first_px,
                    strict_sigma_pixels,
                )
                weights_second = strict_patch_weights(
                    np.asarray(frozen_masks["patch_coverage"])[branch, second, horizon, 1],
                    ee_second_px,
                    strict_sigma_pixels,
                )
                weights = 0.5 * (weights_first + weights_second)
                distance = distance_map[row_index]
                diagnostics["strict_interaction_patch_mean"].append(
                    float(np.sum(distance * weights) / max(float(weights.sum()), 1.0e-12))
                )
                row_index += 1

    target_arrays = {name: np.asarray(values, dtype=np.float64) for name, values in targets.items()}
    valid_arrays = {name: np.asarray(values, dtype=bool) for name, values in valid.items()}
    diagnostic_arrays = {name: np.asarray(values) for name, values in diagnostics.items()}
    if row_index != 960 or distance_map.shape != (960, 49) or pair_coverage.shape != (960, 4, 49):
        raise ValueError("row construction changed")
    if not np.isfinite(projected_ee).all() or not np.isfinite(projection_depth).all():
        raise FloatingPointError("camera projection contains NaN/Inf")
    for name in TARGET_NAMES:
        if not np.isfinite(target_arrays[name][valid_arrays[name]]).all():
            raise FloatingPointError(f"valid target is non-finite: {name}")
    return {
        "metadata": metadata,
        "targets": target_arrays,
        "valid": valid_arrays,
        "diagnostics": diagnostic_arrays,
        "projected_ee": projected_ee,
        "projection_depth": projection_depth,
    }


def grouped_horizon_masks(metadata: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    horizon = np.asarray(metadata["horizon"], dtype=np.int64)
    return {
        "h1_h2": np.isin(horizon, [1, 2]),
        "h3_h5": np.isin(horizon, [3, 4, 5]),
        "h6_h8": np.isin(horizon, [6, 7, 8]),
    }

