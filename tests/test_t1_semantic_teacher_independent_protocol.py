import json
from pathlib import Path

import numpy as np

from src.probes.t1_semantic_teacher_full import block_bootstrap_ci, choose_margin, rank_auc


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_freezes_g2_data_boundary_and_stop():
    p = json.loads((ROOT / "configs/t1_semantic_teacher_protocol.json").read_text())
    assert p["expected_grounding_hash"] == "337ebe063792402369e779bcec4800cb0ce542ea10fd8992eb8f66796dd56881"
    assert p["partitions"]["calibration"] == [31, 32, 33, 34]
    assert p["partitions"]["frozen_development_evaluation"] == [36, 37, 38, 39]
    assert p["partitions"]["forbidden_including_metadata"] == list(range(40, 50))
    assert p["bootstrap"]["draws"] == 10000
    assert p["decision"]["maximum_action"].endswith("do not execute value distillation")


def test_margin_contract_is_limited_and_semantic():
    m = json.loads((ROOT / "configs/official_goal_margin.json").read_text())
    assert set(m["families"]) == {"M0", "M1", "M2"}
    assert "body_xmat" in m["families"]["M1"]["orientation"]
    assert "installed numeric" in m["families"]["M2"]["role"]
    assert "temporal order accuracy" in m["selection"]["forbidden_selection_inputs"]


def test_auc_and_margin_selection_are_mechanical():
    assert rank_auc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1])) == 1.0
    rows = [
        {"family": "M0", "admissible": True, "goal_auc": .99, "tasks_auc_above_half": 10, "adjacent_absolute_jump": {"p99": .1}},
        {"family": "M2", "admissible": True, "goal_auc": .81, "tasks_auc_above_half": 7, "adjacent_absolute_jump": {"p99": 1.0}},
    ]
    assert choose_margin(rows) == "M2"


def test_task_stratified_bootstrap_is_deterministic_and_finite():
    task = np.repeat(np.arange(10), 4)
    demo = np.tile(np.arange(4), 10)
    numerator = np.arange(40) % 2
    denominator = np.ones(40)
    one = block_bootstrap_ci(task, demo, numerator, denominator)
    two = block_bootstrap_ci(task, demo, numerator, denominator)
    assert one == two
    assert np.isfinite(one).all() and one[0] <= one[1]
