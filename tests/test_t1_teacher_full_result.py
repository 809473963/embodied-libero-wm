import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_FREEZE = "semantic-goal-progress-teacher-valid"


def load_frozen_json(path: str):
    content = subprocess.check_output(
        ["git", "show", f"{HISTORICAL_FREEZE}:{path}"], cwd=ROOT, text=True
    )
    return json.loads(content)


def test_calibration_mechanically_selects_m2_and_t1():
    # These paths are intentionally reused by the later independent T1
    # evaluation.  Validate this historical study at its annotated freeze.
    r = load_frozen_json("results/t1_teacher_calibration.json")
    d = load_frozen_json("configs/t1_teacher_frozen_definition.json")
    assert r["selected_margin_family"] == d["selected_margin_family"] == "M2"
    assert r["selected_family"] == d["selected_family"] == "T1"
    assert r["stage_t_complete_gate"] is True
    assert r["grounding"]["correct_instruction_unique_rate"] == 1.0
    assert r["usable_task_count"] == 10


def test_calibration_signal_controls_and_access_pass():
    r = load_frozen_json("results/t1_teacher_calibration.json")
    assert r["local_signal"]["non_tie_fraction"] >= 0.30
    assert r["language"]["tasks_with_material_change"] >= 7
    assert r["controls"]["semantic_match_robot_difference"]["pass"]
    assert r["controls"]["robot_match_semantic_difference"]["pass"]
    assert r["proprio_probe"]["accuracy"] < 0.90
    assert r["access_ledger"]["opened_demo_ids"] == [31, 32, 33, 34]
    assert r["access_ledger"]["metadata_operations"] == []
    assert r["access_ledger"]["forbidden_access_count"] == 0


def test_calibration_sidecar_is_finite_and_uses_all_ten_tasks():
    s = np.load(ROOT / "results/t1_teacher/calibration_sidecar.npz")
    assert set(np.unique(s["task_id"])) == set(range(10))
    assert np.isfinite(s["margin"]).all()
    assert np.isfinite(s["proprio"]).all()
    assert np.all(s["ground_status"][np.arange(40), np.repeat(np.arange(10), 4)] == 1)


def test_frozen_evaluation_passes_every_teacher_gate():
    r = json.loads((ROOT / "results/t1_teacher_eval.json").read_text())
    assert r["stage_t_complete_gate"] is True
    assert r["selected_family"] == "T1"
    assert r["selected_margin_family"] == "M2"
    assert r["local_signal"]["non_tie_fraction"] >= 0.30
    assert r["usable_task_count"] == 10
    assert r["language"]["semantic_state_change_rate"] >= 0.20
    assert r["language"]["tasks_with_material_change"] == 10
    assert r["proprio_probe"]["accuracy"] < 0.90
    assert all(r["gates"].values())


def test_frozen_evaluation_access_and_persistent_identity():
    r = json.loads((ROOT / "results/t1_teacher_eval.json").read_text())
    assert r["access_ledger"]["opened_demo_ids"] == [36, 37, 38, 39]
    assert r["access_ledger"]["metadata_operations"] == []
    assert r["access_ledger"]["forbidden_access_count"] == 0
    s = np.load(ROOT / "results/t1_teacher/evaluation_sidecar.npz")
    assert np.all(s["ground_status"][np.arange(40), np.repeat(np.arange(10), 4)] == 1)
    assert np.isfinite(s["margin"]).all()
