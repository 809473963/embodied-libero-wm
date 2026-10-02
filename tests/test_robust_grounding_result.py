import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_calibration_selects_identity_free_g2_and_passes():
    result = json.loads((ROOT / "results/robust_grounding_calibration.json").read_text())
    resolver = json.loads((ROOT / "configs/robust_grounding_frozen_resolver.json").read_text())
    assert result["complete_gate"] is True
    assert result["selected"]["family"] == resolver["family"] == "G2"
    assert result["selected"]["unique_precision"] >= 0.95
    assert result["selected"]["coverage"] >= 0.90
    assert result["selected"]["task_coverage_ge_75_count"] >= 9
    assert result["access_ledger"]["forbidden_access_count"] == 0


def test_calibration_sidecar_is_finite_and_identity_is_not_a_score_input():
    sidecar = np.load(ROOT / "results/robust_grounding/calibration_sidecar.npz")
    assert sidecar["scores"].shape == (40, 10, 3, 2)
    assert sidecar["perturbed_scores"].shape == (40, 10, 3, 18, 2)
    assert np.isfinite(sidecar["scores"]).all()
    result = json.loads((ROOT / "results/robust_grounding_calibration.json").read_text())
    assert result["selected"]["wrong_unique"] == 0
    assert result["perturbation"]["overall_decision_stability"] == 1.0


def test_calibration_ledger_opens_only_allowed_states():
    ledger = json.loads((ROOT / "results/robust_grounding/calibration_access_ledger.json").read_text())
    assert ledger["opened_demo_ids"] == [31, 32, 33, 34]
    assert ledger["metadata_operations"] == []
    assert ledger["forbidden_access_count"] == 0
    assert all(path.endswith("/states") for path in ledger["numeric_datasets_accessed"])
