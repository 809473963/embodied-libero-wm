"""Independent result checks for the blocked Stage-B1 semantic audit."""

from __future__ import annotations

import json
from pathlib import Path

from src.probes.semantic_goal_contract import sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/semantic_goal_progress_teacher/corrected_diagnosis.json"


def load_result() -> dict:
    result = json.loads(RESULT.read_text())
    for field in ("source_protocol", "source_inventory", "source_diagnosis", "decision_amendment", "boundary_incident"):
        record = result[field]
        assert sha256(ROOT / record["path"]) == record["sha256"]
    return result


def test_source_facts_are_narrow_and_language_claim_is_withdrawn() -> None:
    result = load_result()
    facts = result["admitted_source_findings"]
    assert facts["task_count"] == 10
    assert facts["common_local_bddl_goal"] == ["on", "akita_black_bowl_1", "plate_1"]
    assert facts["distinct_relevant_initial_relations"] == 10
    assert facts["second_same_type_bowl_present_in_every_task"] is True
    assert "SEMANTIC_TEACHER_LANGUAGE_INSUFFICIENT" in result["withdrawn_findings"]
    assert result["unresolved_corrected_route"]["status"] == "NOT_PREREGISTERED_OR_RUN"


def test_boundary_incident_is_explicit_and_no_values_were_loaded() -> None:
    result = load_result()
    incident = json.loads((ROOT / result["boundary_incident"]["path"]).read_text())
    assert incident["incident"] == "FORBIDDEN_PARTITION_METADATA_ENUMERATION"
    assert incident["task_ids_affected"] == [0]
    assert incident["forbidden_demo_ids_whose_metadata_was_enumerated"] == list(range(40, 50))
    assert incident["numeric_array_values_read_from_forbidden_groups"] == 0
    assert incident["models_trained_after_incident"] == 0


def test_no_teacher_distillation_candidate_or_controller_work_occurred() -> None:
    scope = load_result()["scope"]
    assert scope["calibration_teacher_rows_constructed"] == 0
    assert scope["evaluation_teacher_rows_constructed"] == 0
    assert scope["simulator_runs"] == 0
    assert scope["teacher_models_trained"] == 0
    assert scope["value_models_trained"] == 0
    assert scope["candidate_futures_executed"] == 0


def test_mechanical_decision_blocks_all_downstream_stages() -> None:
    decision = load_result()["decision"]
    assert decision["pass"] is False
    assert decision["highest_completed_stage"] == "B1"
    assert decision["final_mechanism_labels"] == [
        "SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED",
        "DATA_OR_IMPLEMENTATION_BLOCK",
    ]
    assert decision["semantic_teacher_selected"] is False
    assert decision["deployable_value_evaluator_selected"] is False
    assert decision["real_candidate_value_bridge_validated"] is False
    assert decision["value_distillation_authorization"] == "BLOCK_VALUE_DISTILLATION"
    assert decision["world_model_interface_authorization"] == "BLOCK_WORLD_MODEL_INTERFACE_AUDIT"
    assert decision["spatial_world_model_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert decision["controller_authorization"] is False


def test_required_reports_exist_and_state_no_selection() -> None:
    report = (ROOT / "docs/semantic_goal_progress_teacher_audit.md").read_text()
    summary = (ROOT / "docs/semantic_candidate_value_research_summary.md").read_text()
    assert "Selected semantic teacher: `none`" in report
    assert "Highest completed stage: `B1`" in summary
    assert "Semantic teacher selected: `no`" in summary
    assert "Controller authorized: `NO`" in summary
    for index in range(1, 16):
        assert f"## Q{index} " in summary

