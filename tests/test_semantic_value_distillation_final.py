import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = ROOT / "results/semantic_value_final_decision.json"
REPORT_PATH = ROOT / "docs/goal_conditioned_semantic_value_distillation_audit.md"


def load_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_decision_artifact_hashes_are_exact() -> None:
    decision = load_json(DECISION_PATH)
    for relative, expected in decision["artifact_sha256"].items():
        assert digest(ROOT / relative) == expected


def test_final_decision_matches_frozen_evaluation_and_controls() -> None:
    decision = load_json(DECISION_PATH)
    evaluation = load_json(ROOT / "results/semantic_value_evaluation.json")
    language = load_json(ROOT / "results/semantic_language_counterfactual_eval.json")
    cross_demo = load_json(ROOT / "results/semantic_cross_demo_controls.json")

    assert decision["frozen_calibration_representative"] == evaluation["frozen_representative"]
    assert decision["selected_semantic_value_student"] == evaluation["selected_semantic_value_student"]
    assert decision["complete_gate"] is evaluation["complete_gate"] is False
    assert decision["authorization"] == evaluation["authorization"] == "BLOCK_REAL_CANDIDATE_VALUE_BRIDGE"
    assert decision["final_labels"] == evaluation["final_labels"]

    primary = decision["held_out_primary"]
    s2 = evaluation["metrics"]["S2"]
    assert primary["ordinary_local_non_tie_accuracy"] == s2["ordinary"]["local_non_tie"]["accuracy"]
    assert primary["teacher_change_agreement"] == language["primary"]["teacher_change_agreement"]
    assert primary["semantic_match_cross_demo_mean_abs_from_half"] == (
        cross_demo["families"]["S2"]["semantic_match"]["mean_abs_from_half"]
    )


def test_final_block_is_mechanically_required() -> None:
    decision = load_json(DECISION_PATH)
    primary = decision["held_out_primary"]
    assert primary["ordinary_local_non_tie_accuracy"] < 0.70
    assert primary["teacher_change_agreement"] < 0.70
    assert primary["semantic_match_cross_demo_mean_abs_from_half"] > 0.15
    assert decision["selected_semantic_value_student"] == "none"
    assert decision["real_candidate_bridge_executed"] is False
    assert not any(label.startswith("GO_") for label in decision["final_labels"])


def test_final_report_has_complete_required_structure_and_decision() -> None:
    report = REPORT_PATH.read_text()
    evaluation = load_json(ROOT / "results/semantic_value_evaluation.json")
    numbered_sections = re.findall(r"^## (\d+)\.", report, flags=re.MULTILINE)
    questions = re.findall(r"^\*\*Q(\d+) ", report, flags=re.MULTILINE)
    assert numbered_sections == [str(index) for index in range(1, 37)]
    assert questions == [str(index) for index in range(1, 23)]
    assert "Selected semantic value student: `none`" in report
    assert "BLOCK_REAL_CANDIDATE_VALUE_BRIDGE" in report
    assert "Next-stage authorization: `GO_REAL_CANDIDATE_VALUE_BRIDGE`" not in report
    assert digest(DECISION_PATH) in report
    for gap, metrics in evaluation["metrics"]["S2"]["ordinary"]["per_gap"].items():
        if metrics["rows"]:
            report_gap = gap.replace("-", "--")
            expected = f"| {report_gap} | {metrics['rows']:,} | {metrics['accuracy']:.2%} |"
            assert expected in report
