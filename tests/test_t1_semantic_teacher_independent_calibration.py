import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_margin_and_teacher_are_selected_without_t2():
    result = json.loads((ROOT / "results/t1_teacher_calibration.json").read_text())
    definition = json.loads((ROOT / "configs/t1_teacher_frozen_definition.json").read_text())
    assert result["selected_margin"] == definition["selected_margin"] == "M2"
    assert result["selected_teacher"] == definition["selected_teacher"] == "T1"
    assert definition["conditional_t2_activated"] is False
    assert definition["bootstrap"]["draws"] == 10000
    assert definition["grounding_hash"] == "337ebe063792402369e779bcec4800cb0ce542ea10fd8992eb8f66796dd56881"


def test_calibration_margin_density_language_and_controls_pass():
    r = json.loads((ROOT / "results/t1_teacher_calibration.json").read_text())
    selected = next(row for row in r["margin_audit"] if row["family"] == "M2")
    assert selected["admissible"] and selected["goal_auc"] >= 0.80
    assert selected["goal_positive"]["median"] > selected["goal_negative"]["median"]
    assert selected["tasks_auc_above_half"] == 10
    assert r["t1_local"]["non_tie_fraction"] >= 0.30
    assert r["usable_task_count"] == 10
    assert r["language"]["material_pair_preference_change_rate"] >= 0.20
    assert r["language"]["tasks_material_at_least_20pct"] >= 7
    assert r["controls"]["semantic_match_robot_difference"]["pass"]
    assert r["controls"]["robot_match_semantic_difference"]["pass"]
    assert r["controls"]["proprio_probe"]["accuracy"] < 0.90
    assert all(r["gates"].values()) and r["complete_gate"]


def test_flip_taxonomy_and_sidecar_recompute():
    r = json.loads((ROOT / "results/t1_teacher_calibration.json").read_text())
    language = r["language"]
    assert sum(row["count"] for row in language["taxonomy"].values()) == language["eligible_local_pair_counterfactuals"]
    assert language["referent_switch_stratification"]["same"]["change_rate"] == 0.0
    assert language["referent_switch_stratification"]["different"]["change_rate"] == 1.0
    sidecar = np.load(ROOT / "results/t1_semantic_teacher/calibration_sidecar.npz")
    local = sidecar["pair_gap"] <= 8
    labels = sidecar["pair_label"][local]
    assert np.isclose(np.mean(labels != 0), r["t1_local"]["non_tie_fraction"])
    assert np.isfinite(sidecar["margins"]).all()


def test_calibration_access_is_exact_and_clean():
    ledger = json.loads((ROOT / "results/t1_semantic_teacher/calibration_access_ledger.json").read_text())
    assert ledger["opened_demo_ids"] == [31, 32, 33, 34]
    assert ledger["metadata_operations"] == []
    assert ledger["forbidden_access_count"] == 0
    assert all(path.endswith("/states") for path in ledger["numeric_datasets_accessed"])
