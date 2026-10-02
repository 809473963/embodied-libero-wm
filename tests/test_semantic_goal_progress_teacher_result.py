"""Independent result checks for the blocked Stage-B1 semantic audit."""

from __future__ import annotations

import json
from pathlib import Path

from src.probes.semantic_goal_contract import sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/semantic_goal_progress_teacher/diagnosis.json"


def load_result() -> tuple[dict, dict]:
    result = json.loads(RESULT.read_text())
    inventory_path = ROOT / result["inventory"]["path"]
    assert sha256(inventory_path) == result["inventory"]["sha256"]
    return result, json.loads(inventory_path.read_text())


def test_all_tasks_share_one_goal_but_have_distinct_initial_contracts() -> None:
    result, inventory = load_result()
    summary = result["semantic_inventory_summary"]
    assert summary["task_count"] == 10
    assert summary["distinct_goal_signature_count"] == 1
    assert summary["distinct_relevant_initial_signature_count"] == 10
    assert inventory["distinct_goal_signatures"] == [["and", ["on", "akita_black_bowl_1", "plate_1"]]]
    assert all(row["goal_atoms"] == [["on", "akita_black_bowl_1", "plate_1"]] for row in inventory["contracts"])


def test_no_admissible_goal_changing_language_counterfactual_exists() -> None:
    result, inventory = load_result()
    summary = result["semantic_inventory_summary"]
    assert summary["directed_real_language_swap_count"] == 90
    assert summary["directed_goal_changing_counterfactual_count"] == 0
    assert summary["task_count_with_goal_changing_counterfactual"] == 0
    assert inventory["goal_changing_counterfactuals"] == []
    assert set(inventory["counterfactual_coverage_by_task"].values()) == {0}


def test_data_boundary_was_preserved() -> None:
    result, _ = load_result()
    scope = result["scope"]
    assert scope["task_definition_files_read"] == 10
    assert scope["demo_rows_read"] == 0
    assert scope["calibration_demo_rows_read"] == 0
    assert scope["evaluation_demo_rows_read"] == 0
    assert scope["forbidden_demo_rows_read"] == 0
    assert scope["simulator_runs"] == 0
    assert scope["teacher_models_trained"] == 0
    assert scope["value_models_trained"] == 0


def test_mechanical_decision_blocks_b2_b3_and_world_model() -> None:
    result, _ = load_result()
    decision = result["decision"]
    assert decision["identifiability_gate_pass"] is False
    assert decision["gates"] == {
        "all_tasks_parsed": True,
        "multiple_goal_signatures": False,
        "counterfactual_task_coverage": False,
    }
    assert decision["final_mechanism_labels"] == [
        "SEMANTIC_TEACHER_LANGUAGE_INSUFFICIENT",
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

