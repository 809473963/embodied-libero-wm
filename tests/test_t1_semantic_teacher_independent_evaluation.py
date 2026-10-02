import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    return json.loads((ROOT / path).read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_inputs_and_calibration_definition_remain_unchanged():
    protocol = _load("configs/t1_semantic_teacher_protocol.json")
    for entry in protocol["frozen_files"]:
        assert _sha256(ROOT / entry["path"]) == entry["sha256"]

    definition = _load("configs/t1_teacher_frozen_definition.json")
    assert definition["grounding_hash"] == protocol["expected_grounding_hash"]
    assert definition["selected_margin"] == "M2"
    assert definition["selected_teacher"] == "T1"
    assert definition["conditional_t2_activated"] is False
    assert definition["bootstrap"] == {
        "draws": 10000,
        "seed": 20261002,
        "unit": "task-stratified demo block",
    }


def test_evaluation_recomputes_from_the_frozen_sidecar():
    result = _load("results/t1_teacher_evaluation.json")
    with np.load(ROOT / "results/t1_semantic_teacher/evaluation_sidecar.npz") as sidecar:
        local = sidecar["pair_gap"] <= 8
        labels = sidecar["pair_label"][local]
        t0_labels = sidecar["t0_pair_label"][local]
        assert np.isclose(np.mean(labels != 0), result["t1_local"]["non_tie_fraction"])
        assert np.isclose(np.mean(labels > 0), result["t1_local"]["progress_fraction"])
        assert np.isclose(np.mean(labels < 0), result["t1_local"]["regression_fraction"])
        assert np.isclose(np.mean(labels == 0), result["t1_local"]["tie_fraction"])
        assert np.isclose(np.mean(t0_labels != 0), result["t0_local"]["non_tie_fraction"])
        for key in sidecar.files:
            array = sidecar[key]
            if np.issubdtype(array.dtype, np.number):
                assert np.isfinite(array).all(), key


def test_margin_density_and_task_gates_pass_held_out():
    result = _load("results/t1_teacher_evaluation.json")
    selected = next(row for row in result["margin_audit"] if row["family"] == "M2")
    assert selected["admissible"]
    assert selected["goal_auc"] >= 0.80
    assert selected["goal_positive"]["median"] > selected["goal_negative"]["median"]
    assert selected["tasks_auc_above_half"] == 10
    assert result["t1_local"]["non_tie_fraction"] >= 0.30
    assert result["t1_local"]["non_tie_ci95"][0] > 0.30
    assert result["usable_task_count"] == 10
    assert all(row["non_tie_fraction"] >= 0.30 for row in result["per_task"].values())
    assert result["teacher_time_agreement"] < 1.0


def test_language_counterfactual_gate_and_taxonomy_pass():
    result = _load("results/t1_language_counterfactual.json")["evaluation"]
    assert result["material_pair_preference_change_rate"] >= 0.20
    assert result["material_change_ci95"][0] > 0.20
    assert result["tasks_material_at_least_20pct"] >= 7
    assert sum(row["count"] for row in result["taxonomy"].values()) == result[
        "eligible_local_pair_counterfactuals"
    ]
    assert result["referent_switch_stratification"]["same"]["change_rate"] == 0.0
    assert result["referent_switch_stratification"]["different"]["change_rate"] == 1.0
    assert result["preference_reversal_rate"] > 0.0
    assert result["reversal_ci95"][0] > 0.0


def test_robot_controls_and_shortcut_gate_pass_held_out():
    result = _load("results/t1_robot_shortcut_controls.json")
    evaluation = result["evaluation"]
    assert evaluation["semantic_match_robot_difference"]["pass"]
    assert evaluation["semantic_match_robot_difference"]["task_count"] == 10
    assert evaluation["robot_match_semantic_difference"]["pass"]
    assert evaluation["robot_match_semantic_difference"]["task_count"] == 10
    probe = evaluation["proprio_probe"]
    assert probe["accuracy"] < 0.90
    assert probe["accuracy_ci95"][1] < 0.90
    assert 0.5 < probe["auroc"] < 1.0
    assert len(probe["per_task"]) == 10
    assert result["evaluation_reduction_points"] < -10.0


def test_all_mechanical_gates_pass_without_forbidden_access():
    result = _load("results/t1_teacher_evaluation.json")
    assert all(result["gates"].values())
    assert result["complete_gate"] is True

    ledger = _load("results/t1_semantic_teacher/evaluation_access_ledger.json")
    assert ledger["opened_demo_ids"] == [36, 37, 38, 39]
    assert ledger["metadata_operations"] == []
    assert ledger["forbidden_attempts"] == []
    assert ledger["forbidden_access_count"] == 0
    assert all(path.endswith("/states") for path in ledger["numeric_datasets_accessed"])
    assert not any(f"demo_{demo}" in path for demo in range(40, 50) for path in ledger["numeric_datasets_accessed"])


def test_final_decision_mapping_and_artifact_hashes_are_exact():
    decision = _load("results/t1_semantic_teacher_final_decision.json")
    assert decision["complete_gate"] is True
    assert decision["authorization"] == "GO_VALUE_DISTILLATION"
    assert "SEMANTIC_GOAL_PROGRESS_TEACHER_VALID" in decision["final_labels"]
    assert decision["value_distillation_executed"] is False
    assert decision["controller_authorization"] is False
    for path, expected in decision["artifact_sha256"].items():
        assert _sha256(ROOT / path) == expected


def test_report_contains_the_required_structure_and_answers():
    report = (ROOT / "docs/t1_semantic_goal_progress_teacher_full_evaluation.md").read_text()
    for section in range(1, 31):
        assert f"## {section}." in report
    for question in range(1, 21):
        assert f"**Q{question} " in report
    assert "GO_VALUE_DISTILLATION" in report
    assert "Value distillation executed: `false`" in report
