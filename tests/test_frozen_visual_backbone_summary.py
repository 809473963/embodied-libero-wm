from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_summary_reports_highest_stage_and_every_required_answer() -> None:
    summary = (ROOT / "docs/frozen_visual_backbone_research_summary.md").read_text()
    assert "Highest completed stage: E" in summary
    assert "CLIP internal layer valid: no" in summary
    assert "Alternate encoder audit executed: yes" in summary
    assert "Selected visual encoder: none" in summary
    assert "Selected visual layer: none" in summary
    assert "Selected semantic value student: none" in summary
    assert "Real candidate-value bridge authorized: no" in summary
    assert "Real candidate-value bridge executed: NO" in summary
    assert "World Model authorization: BLOCKED" in summary
    assert "Controller authorization: false" in summary
    for question in range(1, 24):
        assert f"**Q{question} —" in summary


def test_stage_e_report_and_decision_agree_and_stage_v_is_absent() -> None:
    report = (ROOT / "docs/alternate_frozen_visual_encoder_audit.md").read_text()
    decision = json.loads((ROOT / "results/alternate_encoder_final_decision.json").read_text())
    for label in decision["mechanism_labels"]:
        assert label in report
    for authorization in decision["next_stage_authorizations"]:
        assert authorization in report
    assert decision["selected_visual_encoder"] is None
    assert decision["selected_visual_layer"] is None
    assert decision["stop_chain"] is True
    assert "Stage V executed: `false`" in report
    assert not (ROOT / "configs/redistillation_protocol.json").exists()
    assert not (ROOT / "results/redistillation_evaluation.json").exists()
