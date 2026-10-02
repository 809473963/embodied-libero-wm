import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_access_union_is_calibration_only_and_forbidden_zero():
    audit = json.loads((ROOT / "results/semantic_teacher_clean/clean_restart_access_audit.json").read_text())
    assert audit["union_opened_demo_ids"] == [31, 32, 33, 34]
    assert audit["evaluation_demo_ids_opened"] == []
    assert audit["historical_demo_ids_opened"] == []
    assert audit["forbidden_demo_ids_opened"] == []
    assert audit["metadata_operations_outside_allowlist"] == 0
    assert audit["forbidden_access_count"] == 0


def test_calibration_sidecar_identity_and_finiteness():
    sidecar_path = "results/semantic_teacher_clean/calibration_sidecar.npz"
    result = json.loads((ROOT / "results/semantic_teacher_calibration.json").read_text())
    assert digest(sidecar_path) == result["sidecar_sha256"]
    with np.load(ROOT / sidecar_path) as sidecar:
        assert set(np.unique(sidecar["demo_id"]).tolist()) == {31, 32, 33, 34}
        assert set(np.unique(sidecar["task_id"]).tolist()) == set(range(10))
        assert sidecar["goal"].shape[1] == 2
        assert sidecar["proprio"].shape[1] == 12
        for key in sidecar.files:
            assert np.isfinite(sidecar[key]).all()


def test_grounding_gate_mechanically_blocks_teacher():
    result = json.loads((ROOT / "results/semantic_teacher_calibration.json").read_text())
    definition = json.loads((ROOT / "configs/teacher_definition.json").read_text())
    assert result["grounding"]["correct_instruction_unique_rate"] == 0.575
    assert result["grounding"]["ambiguous_rate"] == 0.0
    assert result["gates"]["same_scene_grounding"] is False
    assert result["complete_gate"] is False
    assert definition["selected_semantic_teacher"] == "none"
    assert definition["evaluation_authorized"] is False


def test_subset_diagnostics_do_not_override_grounding_stop():
    result = json.loads((ROOT / "results/semantic_teacher_calibration.json").read_text())
    assert result["selected_family"] == "T1"
    assert result["local_signal"]["non_tie_fraction"] >= 0.30
    assert result["usable_task_count"] == 7
    assert result["language"]["semantic_state_change_rate"] == 1.0
    assert result["language"]["tasks_with_material_change"] == 7
    assert result["proprio_probe"]["accuracy"] < 0.90
    assert result["complete_gate"] is False


def test_evaluation_partition_was_not_opened():
    evaluation = json.loads((ROOT / "results/semantic_teacher_eval.json").read_text())
    assert evaluation["opened"] is False
    assert evaluation["numeric_rows_read"] == 0
    assert evaluation["metadata_operations"] == 0
    assert evaluation["selected_semantic_teacher"] == "none"
    assert evaluation["mechanical_labels"] == ["SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED"]
    assert "BLOCK_VALUE_DISTILLATION" in evaluation["authorizations"]


def test_prior_blocked_artifacts_remain_exact():
    assert digest("docs/semantic_goal_progress_teacher_audit.md") == "0147cb42277dcaa8de446a841a79118016bbea42274bd0995ebf75801f00a3de"
    assert digest("results/semantic_goal_progress_teacher/boundary_incident.json") == "a114c1c1d26f25b04c5ef33fb11291a2d8f03eedeb2a2789e85e5ede5f4b1b16"
    assert digest("results/semantic_goal_progress_teacher/corrected_diagnosis.json") == "9ebf0fdd5bf9f2458587f933d2e55322ab5f4cbbf97e0d6f6d88fa18ed9599fa"


def test_report_has_required_decision_and_answers():
    report = (ROOT / "docs/semantic_goal_progress_teacher_clean_restart.md").read_text()
    for question in range(1, 15):
        assert f"**Q{question} " in report
    assert "SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED" in report
    assert "BLOCK_VALUE_DISTILLATION" in report
    assert "GO_VALUE_DISTILLATION" in report
    assert "No." in report

