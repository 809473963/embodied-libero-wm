import json
from pathlib import Path

import numpy as np

from src.probes.robust_grounding import maximum_score, perturbation_vectors, resolve_scores, threshold_grid


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_freezes_boundaries_and_stop_rules():
    protocol = json.loads((ROOT / "configs/robust_grounding_protocol.json").read_text())
    assert protocol["partitions"]["calibration"] == [31, 32, 33, 34]
    assert protocol["partitions"]["frozen_development_evaluation"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_including_metadata"] == list(range(40, 50))
    assert protocol["gates"]["precision_min"] == 0.95
    assert protocol["gates"]["coverage_min"] == 0.90
    assert "value distillation" in protocol["forbidden_activity"]


def test_confidence_resolution_is_strict_and_finite():
    assert resolve_scores(np.array([0.1, 0.3]), 0.1, 1.0)[:2] == ("UNIQUE", 0)
    assert resolve_scores(np.array([0.1, 0.2]), 0.1, 1.0)[0] == "AMBIGUOUS"
    assert resolve_scores(np.array([1.1, 1.2]), 0.0, 1.0)[0] == "NOT_FOUND"
    assert resolve_scores(np.array([np.nan, 0.2]), 0.0, 1.0)[0] == "NOT_FOUND"


def test_family_thresholds_and_perturbations_are_frozen():
    assert threshold_grid("G1")[0] == 0.0
    assert threshold_grid("G3")[-1] == 0.75
    assert maximum_score("G2") == 20.0
    shifts = perturbation_vectors()
    assert shifts.shape == (18, 3)
    assert set(np.round(np.max(np.abs(shifts), axis=1) * 1000).astype(int)) == {1, 2, 5}


def test_geometry_forbids_identity_shortcuts():
    geometry = json.loads((ROOT / "configs/robust_grounding_geometry.json").read_text())
    assert "known construction bowl" in geometry["identity_inputs_forbidden"]
    assert "horizontal" in geometry["object_support"]
    assert "vertical" in geometry["object_support"]
