import hashlib
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
PROTOCOL = json.loads((ROOT / "configs/camera_projection_contract_protocol.json").read_text())
RESULT = json.loads((ROOT / "results/camera_projection_contract/diagnosis.json").read_text())
CONTRACT = json.loads((ROOT / "artifacts/camera_projection_contract/frozen_contract.json").read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_formula_recomputes_every_primary_projection() -> None:
    camera = RESULT["camera"]
    transform = np.asarray(camera["world_to_raw_pixel"], dtype=np.float64)
    position = np.asarray(camera["camera_position_world"], dtype=np.float64)
    rotation = np.asarray(camera["camera_rotation_world"], dtype=np.float64)
    fovy = float(camera["fovy_degrees"])

    for row in RESULT["primary_rows"]:
        point = np.asarray(row["world_position"], dtype=np.float64)
        official_raw, official_depth = official_float_projection(point, transform)
        independent_raw, independent_upright, independent_depth = independent_projection(
            point, position, rotation, fovy
        )
        assert np.allclose(official_raw, row["raw_utility_float_uv"], atol=1e-12, rtol=0)
        assert np.allclose(independent_raw, official_raw, atol=1e-12, rtol=0)
        assert np.allclose(
            independent_upright,
            to_upright(official_raw, RESULT["selected_convention"]),
            atol=1e-12,
            rtol=0,
        )
        assert np.isclose(official_depth, row["camera_depth"], atol=1e-12, rtol=0)
        assert np.isclose(independent_depth, official_depth, atol=1e-12, rtol=0)


def test_direct_rgb_orientation_is_exact_for_all_40_states() -> None:
    orientation = RESULT["rgb_orientation"]
    assert len(orientation["rows"]) == 40
    assert orientation["all_bitwise_equal"] is True
    assert orientation["maximum_absolute_difference"] == 0
    for row in orientation["rows"]:
        assert row["shape"] == [128, 128, 3]
        assert row["dtype"] == "uint8"
        assert row["bitwise_equal"] is True
        assert row["max_abs_difference"] == 0


def test_primary_landmark_world_points_are_finite_positive_and_in_frame() -> None:
    assert len(RESULT["primary_rows"]) == 40
    identities = set()
    for row in RESULT["primary_rows"]:
        identities.add((row["task_id"], row["init_state_index"], row["landmark_name"]))
        assert row["landmark_name"] == "gripper0_grip_site"
        assert row["landmark_id"] == 13
        assert 1 <= row["site_visible_pixels"] <= 64
        assert np.isfinite(np.asarray(row["world_position"], dtype=np.float64)).all()
        assert row["finite"] is True
        assert row["camera_depth"] > 0
        assert row["independent_depth"] > 0
        assert row["selected_in_frame"] is True
        u, v = row["pixels"][RESULT["selected_convention"]]
        assert 0 <= u < 128
        assert 0 <= v < 128
    assert identities == {
        (task, init_state, "gripper0_grip_site")
        for task in range(10)
        for init_state in (31, 32, 33, 34)
    }


def test_selected_convention_passes_and_alternative_fails_absolute_gate() -> None:
    thresholds = PROTOCOL["gates"]["absolute_landmark_alignment"]
    selected = RESULT["selected_convention"]
    alternative = ({"direct_render", "vertical_flip_to_direct_render"} - {selected}).pop()
    metrics = RESULT["primary_metrics"]

    assert absolute_alignment_pass(metrics["pooled"][selected], thresholds)
    assert not absolute_alignment_pass(metrics["pooled"][alternative], thresholds)
    for grouped in (metrics["per_task"], metrics["per_init"], metrics["per_landmark"]):
        assert all(absolute_alignment_pass(value[selected], thresholds) for value in grouped.values())


def test_source_independent_agreement_and_camera_stability_pass() -> None:
    comparison = RESULT["source_vs_independent"]
    assert comparison["maximum_raw_pixel_difference"] <= PROTOCOL["gates"][
        "maximum_source_vs_independent_pixel_difference"
    ]
    assert comparison["maximum_depth_difference"] <= PROTOCOL["gates"][
        "maximum_source_vs_independent_depth_difference"
    ]
    assert comparison["maximum_camera_matrix_drift"] <= PROTOCOL["gates"][
        "maximum_camera_matrix_drift"
    ]
    assert comparison["pass"] is True


def test_upstream_camera_block_files_are_unchanged() -> None:
    upstream = PROTOCOL["frozen_upstream"]
    assert upstream["primary_commit"] == "d4dd36113d4381b821238dcd29101546f64860f1"
    assert set(upstream["preserved_verdicts"]) == {
        "DATA_OR_IMPLEMENTATION_BLOCK",
        "CAMERA_PROJECTION_BLOCK",
        "BLOCK_VISUAL_TARGET_REDESIGN",
    }
    for artifact in upstream["files"]:
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert RESULT["old_block_preserved"]["old_result_changed"] is False
    assert RESULT["old_block_preserved"]["old_robot_mask_margin_threshold_pixels"] == 4.0


def test_mechanical_verdict_and_authorization_are_exactly_scoped() -> None:
    assert all(RESULT["gates"].values())
    assert RESULT["nan_inf_depth_in_frame_audit"] == "pass"
    assert RESULT["decision"] == {
        "final_verdict": "CAMERA_PROJECTION_CONTRACT_VALIDATED",
        "authorization": "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART",
        "spatial_target_design_authorized": False,
        "world_model_reranker_controller_authorized": False,
    }
    assert CONTRACT["status"] == "CAMERA_PROJECTION_CONTRACT_VALIDATED"
    assert CONTRACT["authorization"] == "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART"
    assert CONTRACT["selected_convention"] == "vertical_flip_to_direct_render"
    assert CONTRACT["downstream_constraint"] == (
        "all future 2D image-space target construction must read this frozen contract; "
        "it does not authorize target redesign, model, reranker, or controller work"
    )
    assert not any(RESULT["forbidden_activity_audit"].values())
