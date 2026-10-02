from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.probes.spatial_visual_representation import (
    FAMILIES, SpatialRepresentationProbe, fixed_patch_coordinates, parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_r_protocol_is_calibration_only_and_hashes_match() -> None:
    protocol = json.loads((ROOT / "configs/goal_conditioned_spatial_visual_representation_protocol.json").read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_STAGE_R_FEATURE_EXTRACTION_OR_TRAINING"
    assert protocol["partitions"]["selection_train_demo_ids"] == [31, 32, 33]
    assert protocol["partitions"]["selection_validation_demo_ids"] == [34]
    assert protocol["partitions"]["stage_r_not_opened_demo_ids"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_including_metadata"] == list(range(40, 50))
    for row in protocol["frozen_files"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]


def test_patch_coordinates_are_fixed_row_major_cell_centers() -> None:
    coordinate = fixed_patch_coordinates()
    assert coordinate.shape == (49, 2)
    np.testing.assert_allclose(coordinate[0], [-6 / 7, -6 / 7])
    np.testing.assert_allclose(coordinate[6], [6 / 7, -6 / 7])
    np.testing.assert_allclose(coordinate[-1], [6 / 7, 6 / 7])
    assert len(np.unique(coordinate, axis=0)) == 49


def test_registered_family_shapes_parameters_and_antisymmetry() -> None:
    expected = {"R0": 131201, "R1": 78753, "R2": 95137, "R3": 124801}
    coordinate = torch.from_numpy(fixed_patch_coordinates())
    language = torch.randn(3, 512)
    for family in FAMILIES:
        torch.manual_seed(3)
        model = SpatialRepresentationProbe(family).eval()
        assert parameter_count(model) == expected[family]
        visual = torch.randn(3, 512) if family == "R0" else torch.randn(3, 49, 512)
        with torch.inference_mode():
            forward = model(visual, visual.roll(1, 0), language, coordinate)
            reverse = model(visual.roll(1, 0), visual, language, coordinate)
        torch.testing.assert_close(forward, -reverse)


def test_r1_patch_selection_is_language_independent() -> None:
    torch.manual_seed(4)
    model = SpatialRepresentationProbe("R1").eval()
    patch = torch.randn(2, 49, 512)
    coordinate = torch.from_numpy(fixed_patch_coordinates())
    with torch.inference_mode():
        _, first = model.represent(patch, torch.randn(2, 512), coordinate, True)
        _, second = model.represent(patch, torch.randn(2, 512), coordinate, True)
    assert first is not None and second is not None
    torch.testing.assert_close(first, second)


def test_stage_r_decision_thresholds_are_exact() -> None:
    protocol = json.loads((ROOT / "configs/goal_conditioned_spatial_visual_representation_protocol.json").read_text())
    gate = protocol["material_improvement_gate_relative_to_R0"]
    assert gate["teacher_change_agreement_increment_min_points"] == 4.0
    assert gate["and_at_least_one"]["cross_demo_match_mean_abs_improvement_min"] == 0.02
    assert gate["and_at_least_one"]["cross_demo_difference_accuracy_increment_min_points"] == 5.0
    assert gate["ordinary_accuracy_max_regression_points"] == 2.0
    assert protocol["decision_mapping"]["fail_authorization"] == "BLOCK_SEMANTIC_VALUE_REDISTILLATION"
