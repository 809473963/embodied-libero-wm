import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_t1_protocol_is_conditional_and_bounded():
    p = json.loads((ROOT / "configs/t1_teacher_full_protocol.json").read_text())
    assert p["status"] == "PREREGISTERED_AFTER_STAGE_G_PASS"
    assert p["partitions"]["forbidden_including_metadata"] == list(range(40, 50))
    assert p["gates"]["local_non_tie_min"] == 0.3
    assert p["gates"]["severe_proprio_accuracy"] == 0.9
    assert p["decision"]["maximum_action"].endswith("do not execute distillation")


def test_margin_selection_is_not_temporal():
    p = json.loads((ROOT / "configs/t1_teacher_full_protocol.json").read_text())
    assert set(p["margin_candidates"]) == {"M0", "M1", "M2"}
    assert "no time" in p["margin_selection"]
    assert "pair-density" in p["margin_selection"]
