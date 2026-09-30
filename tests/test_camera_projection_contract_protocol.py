import json
from pathlib import Path

import numpy as np

from src.probes.camera_projection_contract import (
    absolute_alignment_pass,
    independent_projection,
    official_float_projection,
    to_upright,
)


ROOT = Path(__file__).resolve().parents[1]


def test_independent_formula_matches_constructed_transform() -> None:
    position = np.asarray([0.3, -0.2, 1.1])
    rotation = np.eye(3)
    point = np.asarray([0.5, -0.4, -2.9])
    raw, upright, depth = independent_projection(point, position, rotation, 90.0)
    focal = 64.0
    camera = (point - position) * np.asarray([1.0, -1.0, -1.0])
    expected = np.asarray([focal * camera[0] / camera[2] + 64, focal * camera[1] / camera[2] + 64])
    assert np.allclose(raw, expected)
    assert np.allclose(upright, [expected[0], 127 - expected[1]])
    assert depth > 0


def test_official_float_transform_and_upright_conversion() -> None:
    matrix = np.eye(4)
    point = np.asarray([12.0, 24.0, 2.0])
    raw, depth = official_float_projection(point, matrix)
    assert depth == 2.0
    assert np.allclose(raw, [6.0, 12.0])
    assert np.allclose(to_upright(raw, "direct_render"), [6.0, 12.0])
    assert np.allclose(to_upright(raw, "vertical_flip_to_direct_render"), [6.0, 115.0])


def test_rasterization_gate_is_absolute_not_old_margin() -> None:
    thresholds = {
        "maximum_median_error_pixels": 1.0,
        "maximum_p95_error_pixels": 1.5,
        "maximum_error_pixels": 2.0,
    }
    assert absolute_alignment_pass({"median": 0.8, "p95": 1.4, "max": 1.9}, thresholds)
    assert not absolute_alignment_pass({"median": 1.1, "p95": 1.4, "max": 1.9}, thresholds)


def test_protocol_scope_and_old_block_preservation_when_present() -> None:
    path = ROOT / "configs/camera_projection_contract_protocol.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text())
    assert protocol["scope"]["init_state_indices"] == [31, 32, 33, 34]
    assert set(protocol["scope"]["forbidden_init_state_indices"]) == set(range(50)) - {31, 32, 33, 34}
    assert protocol["historical_block"]["old_margin_threshold_pixels"] == 4.0
    assert protocol["historical_block"]["old_verdict_changed"] is False
    assert protocol["gates"]["old_robot_mask_margin_used"] is False
    assert protocol["decision"]["success_authorization"] == "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART"
    assert protocol["decision"]["controller_authorization"] is False


def test_audit_source_contains_no_candidate_or_outcome_access() -> None:
    source = (ROOT / "scripts/audit_camera_projection_contract.py").read_text()
    for forbidden in ("env.step(", "check_success", "_check_success", "get_reward"):
        assert forbidden not in source
