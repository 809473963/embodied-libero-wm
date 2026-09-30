"""Deterministic camera projection helpers for the contract-resolution audit."""

from __future__ import annotations

from typing import Any

import numpy as np


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


def official_float_projection(points: np.ndarray, world_to_pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply robosuite's 4x4 transform without its rounding/clipping/swap."""

    xyz = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(world_to_pixel, dtype=np.float64)
    if xyz.shape[-1] != 3 or matrix.shape != (4, 4):
        raise ValueError("unexpected projection input")
    homogeneous = np.concatenate((xyz, np.ones(xyz.shape[:-1] + (1,))), axis=-1)
    projected = homogeneous @ matrix.T
    depth = projected[..., 2]
    if np.any(np.abs(depth) <= 1.0e-12):
        raise FloatingPointError("near-zero camera depth")
    return projected[..., :2] / depth[..., None], depth


def independent_projection(
    points: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation_world: np.ndarray,
    fovy_degrees: float,
    image_height: int = IMAGE_HEIGHT,
    image_width: int = IMAGE_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project from MuJoCo camera pose using an explicit independent formula.

    MuJoCo's camera local rotation maps local axes to world. Robosuite converts
    the MuJoCo/OpenGL camera to a +z-forward pinhole frame with
    diag(1, -1, -1). The returned raw pixel coordinates have a bottom-left
    vertical convention relative to the direct upright RGB used by this repo.
    """

    xyz = np.asarray(points, dtype=np.float64)
    center = np.asarray(camera_position, dtype=np.float64)
    rotation = np.asarray(camera_rotation_world, dtype=np.float64).reshape(3, 3)
    camera_mujoco = (xyz - center) @ rotation
    camera_cv = camera_mujoco * np.asarray([1.0, -1.0, -1.0])
    depth = camera_cv[..., 2]
    if np.any(np.abs(depth) <= 1.0e-12):
        raise FloatingPointError("near-zero independent camera depth")
    focal = 0.5 * image_height / np.tan(float(fovy_degrees) * np.pi / 360.0)
    raw_u = focal * camera_cv[..., 0] / depth + image_width / 2.0
    raw_v = focal * camera_cv[..., 1] / depth + image_height / 2.0
    raw = np.stack((raw_u, raw_v), axis=-1)
    upright = np.stack((raw_u, image_height - 1.0 - raw_v), axis=-1)
    return raw, upright, depth


def to_upright(raw_uv: np.ndarray, convention: str) -> np.ndarray:
    raw = np.asarray(raw_uv, dtype=np.float64)
    if convention == "direct_render":
        return raw.copy()
    if convention == "vertical_flip_to_direct_render":
        result = raw.copy()
        result[..., 1] = IMAGE_HEIGHT - 1.0 - result[..., 1]
        return result
    raise ValueError(f"unknown convention: {convention}")


def point_to_mask_distance(point_uv: np.ndarray, mask: np.ndarray) -> float | None:
    region = np.asarray(mask, dtype=bool)
    if region.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or not np.any(region):
        return None
    yx = np.argwhere(region).astype(np.float64)
    point = np.asarray(point_uv, dtype=np.float64)
    difference = yx - np.asarray([point[1], point[0]], dtype=np.float64)
    return float(np.sqrt(np.min(np.sum(np.square(difference), axis=1))))


def summarize(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary requires finite non-empty values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def absolute_alignment_pass(summary: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return bool(
        summary["median"] <= thresholds["maximum_median_error_pixels"]
        and summary["p95"] <= thresholds["maximum_p95_error_pixels"]
        and summary["max"] <= thresholds["maximum_error_pixels"]
    )

