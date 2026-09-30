import json
from pathlib import Path

import numpy as np

from src.probes.image_space_relevance import (
    _binary_dilation,
    boundary_adjacency,
    mask_geometry,
    mask_pair_metrics,
    point_to_mask_distance,
    project_world_points,
    strict_patch_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def test_projection_contract_direct_and_flipped() -> None:
    matrix = np.eye(4)
    point = np.asarray([12.0, 24.0, 2.0])
    direct, depth = project_world_points(point, matrix, "direct_render")
    flipped, _ = project_world_points(point, matrix, "vertical_flip_to_direct_render")
    assert depth == 2.0
    assert np.allclose(direct, [6.0, 12.0])
    assert np.allclose(flipped, [6.0, 115.0])


def test_mask_targets_do_not_impute_invisible_centroid() -> None:
    empty = np.zeros((128, 128), dtype=bool)
    visible = empty.copy()
    visible[10:20, 30:50] = True
    assert mask_geometry(empty)["centroid"] is None
    geometry = mask_geometry(visible)
    assert geometry["visible"] is True
    assert np.allclose(geometry["centroid"], [39.5 / 128.0, 14.5 / 128.0])
    assert mask_pair_metrics(empty, empty)["iou_distance"] == 0.0
    assert mask_pair_metrics(empty, visible)["symmetric_difference_fraction"] > 0.0


def test_proximity_adjacency_and_strict_patch_are_finite() -> None:
    object_mask = np.zeros((128, 128), dtype=bool)
    robot_mask = np.zeros((128, 128), dtype=bool)
    object_mask[60:70, 60:70] = True
    robot_mask[64, 55] = True
    assert _binary_dilation(robot_mask, 4).shape == robot_mask.shape
    assert boundary_adjacency(object_mask, robot_mask, 5) > 0.0
    assert point_to_mask_distance(np.asarray([55.0, 64.0]), object_mask) == 5.0
    coverage = np.zeros(49)
    coverage[24] = 1.0
    weights = strict_patch_weights(coverage, np.asarray([64.0, 64.0]), 16.0)
    assert weights.shape == (49,)
    assert np.isfinite(weights).all() and weights.sum() > 0.0


def test_protocol_boundaries_and_authorization_rule_when_present() -> None:
    path = ROOT / "configs/natural_image_space_interaction_protocol.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text())
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["development_evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert protocol["partitions"]["historical_reference_only"] == [30, 35]
    assert list(range(40, 50)) == protocol["partitions"]["forbidden_init_state_indices"][-10:]
    assert protocol["gates"]["minimum_incremental_rmse_improvement"] == 0.10
    assert protocol["gates"]["authorization_families"] == ["object", "relative", "interaction"]
    assert protocol["decision"]["controller_authorization"] is False


def test_collector_contains_no_outcome_access() -> None:
    source = (ROOT / "scripts/collect_image_space_targets.py").read_text()
    for forbidden in ("get_reward", "check_success", "_check_success"):
        assert forbidden not in source
    assert "trusted_init_states(suite, task_id)[31]" in source

