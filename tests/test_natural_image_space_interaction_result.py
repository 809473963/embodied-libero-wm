import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_camera_gate_and_blocked_decision_are_consistent() -> None:
    camera = json.loads((ROOT / "results/image_space_target_audit/camera_inventory.json").read_text())
    diagnosis = json.loads((ROOT / "results/image_space_target_audit/diagnosis.json").read_text())
    assert camera["status"] == "CAMERA_PROJECTION_BLOCK"
    assert camera["gate"]["pass"] is False
    assert camera["selected_median_margin_pixels"] < camera["gate"]["minimum_median_margin_over_alternative_pixels"]
    assert diagnosis["blocking_condition"] == "CAMERA_PROJECTION_BLOCK"
    assert diagnosis["decision"]["final_mechanism_verdict"] == "DATA_OR_IMPLEMENTATION_BLOCK"
    assert diagnosis["decision"]["next_stage_authorization"] == "BLOCK_VISUAL_TARGET_REDESIGN"
    assert diagnosis["decision"]["controller_authorization"] is False


def test_block_stopped_before_target_or_evaluation_outputs() -> None:
    diagnosis = json.loads((ROOT / "results/image_space_target_audit/diagnosis.json").read_text())
    boundary = diagnosis["execution_boundary"]
    assert boundary["calibration_target_construction_run"] is False
    assert boundary["development_evaluation_init36_39_opened"] is False
    assert boundary["init40_49_accessed"] is False
    assert not (ROOT / "artifacts/stage3/image_space_target_audit/calibration_derived.npz").exists()
    assert not (ROOT / "artifacts/stage3/image_space_target_audit/evaluation_derived.npz").exists()


def test_reported_hashes_and_forbidden_activity() -> None:
    diagnosis = json.loads((ROOT / "results/image_space_target_audit/diagnosis.json").read_text())
    camera_path = ROOT / diagnosis["artifacts"]["camera_inventory"]["path"]
    assert digest(camera_path) == diagnosis["artifacts"]["camera_inventory"]["sha256"]
    assert diagnosis["nan_inf_audit"] == "pass"
    assert not any(diagnosis["forbidden_activity_audit"].values())


def test_report_preserves_upstream_and_answers_four_questions() -> None:
    report = (ROOT / "docs/natural_image_space_interaction_target_audit.md").read_text()
    for value in (
        "GLOBAL_CLIP_SPATIAL_COMPRESSION",
        "NATURAL_PATCH_TASK_RELEVANCE_UNRESOLVED",
        "BLOCK_VISUAL_TARGET_REDESIGN",
        "CAMERA_PROJECTION_BLOCK",
        "DATA_OR_IMPLEMENTATION_BLOCK",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "1.648%",
    ):
        assert value in report

