"""Frozen-camera adapter for the natural image-space audit restart.

The restart deliberately reuses the already-preregistered image-space target
definitions while sourcing every world-to-pixel projection from the validated
camera contract.  It contains no projection-selection logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.probes.image_space_relevance import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    build_image_space_dataset,
    point_to_mask_distance,
    project_world_points,
)


EXPECTED_CONTRACT_SHA256 = (
    "5eb187c5b3b7eab7a8f2503f9ee8e9dc4bb12e2e65cfb5d017b1913c2d614e0e"
)
EXPECTED_CONVENTION = "vertical_flip_to_direct_render"


def validate_contract(contract: dict[str, Any]) -> None:
    if contract["status"] != "CAMERA_PROJECTION_CONTRACT_VALIDATED":
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: camera contract verdict")
    if contract["authorization"] != "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART":
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: camera contract authorization")
    if contract["selected_convention"] != EXPECTED_CONVENTION:
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: projection convention changed")
    if contract["image_shape"] != [IMAGE_HEIGHT, IMAGE_WIDTH, 3]:
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: camera image shape changed")
    matrix = np.asarray(contract["world_to_raw_pixel"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: invalid frozen projection matrix")


def project_with_contract(
    points: np.ndarray, contract: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Project with the frozen floating, unclipped upright-image contract."""

    validate_contract(contract)
    return project_world_points(
        points,
        np.asarray(contract["world_to_raw_pixel"], dtype=np.float64),
        contract["selected_convention"],
        height=int(contract["image_shape"][0]),
        width=int(contract["image_shape"][1]),
    )


def camera_adapter(contract: dict[str, Any], task_ids: list[int]) -> dict[str, Any]:
    """Adapt the frozen contract to the unchanged target-construction API."""

    validate_contract(contract)
    matrix = contract["world_to_raw_pixel"]
    return {
        "status": "CAMERA_PROJECTION_CONTRACT_VALIDATED",
        "selected_projection_convention": contract["selected_convention"],
        "tasks": [
            {"task_id": int(task_id), "world_to_pixel": matrix}
            for task_id in task_ids
        ],
    }


def build_dataset_from_contract(
    spatial: dict[str, np.ndarray],
    frozen_masks: dict[str, np.ndarray],
    object_masks: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    contract: dict[str, Any],
    strict_sigma_pixels: float,
    adjacency_radius_pixels: int,
    near_contact_pixels: float,
) -> dict[str, Any]:
    task_ids = sorted({int(value) for value in np.asarray(state["task_id"]).tolist()})
    return build_image_space_dataset(
        spatial,
        frozen_masks,
        object_masks,
        state,
        camera_adapter(contract, task_ids),
        strict_sigma_pixels,
        adjacency_radius_pixels,
        near_contact_pixels,
    )


def implementation_sanity_distance(
    world_point: np.ndarray,
    exact_site_mask: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    projected, depth = project_with_contract(world_point, contract)
    distance = point_to_mask_distance(projected, exact_site_mask)
    if distance is None:
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: sanity site mask is empty")
    in_frame = bool(
        0.0 <= projected[0] < IMAGE_WIDTH and 0.0 <= projected[1] < IMAGE_HEIGHT
    )
    return {
        "projected_uv": np.asarray(projected, dtype=np.float64).tolist(),
        "depth": float(depth),
        "distance_pixels": float(distance),
        "finite": bool(np.isfinite(projected).all() and np.isfinite(depth)),
        "positive_depth": bool(depth > 0.0),
        "in_frame": in_frame,
    }
