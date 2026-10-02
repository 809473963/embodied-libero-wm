import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_null(value):
    if isinstance(value, dict):
        return all(finite_or_null(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_or_null(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def test_calibration_freeze_hashes_and_selection() -> None:
    result = json.loads((ROOT / "results/semantic_value_calibration.json").read_text())
    freeze = json.loads((ROOT / "results/semantic_value_calibration_freeze.json").read_text())
    assert result["selection"]["selected_family"] == "S2"
    assert freeze["selected_family"] == "S2"
    for relative, expected in freeze["artifact_sha256"].items():
        assert digest(ROOT / relative) == expected


def test_selection_rule_was_applied_mechanically() -> None:
    result = json.loads((ROOT / "results/semantic_value_calibration.json").read_text())
    trace = result["selection"]["trace"]
    s15, s2 = trace["metrics"]["S1.5"], trace["metrics"]["S2"]
    assert s15["language"] >= s2["language"] - trace["tolerance"]
    assert s15["ordinary"] < s2["ordinary"] - trace["tolerance"]
    assert trace["reason"] == "frozen lexicographic fallback"


def test_calibration_access_and_finiteness() -> None:
    result = json.loads((ROOT / "results/semantic_value_calibration.json").read_text())
    ledger = result["access_ledger"]
    assert result["all_finite"] is True
    assert finite_or_null(result)
    assert ledger["opened_demo_ids"] == [31, 32, 33, 34]
    assert ledger["metadata_operations"] == []
    assert ledger["forbidden_access_count"] == 0
    assert all("/vision" in path or "language_features" in path
               for path in ledger["numeric_datasets_accessed"])
    assert not any(f"demo_{demo}" in path for demo in range(40, 50)
                   for path in ledger["numeric_datasets_accessed"])


def test_frozen_models_respect_capacity_and_protocol_hash() -> None:
    result = json.loads((ROOT / "results/semantic_value_calibration.json").read_text())
    assert all(value <= 200_000 for value in result["parameter_counts"].values())
    weights = torch.load(
        ROOT / "results/semantic_value_distillation/calibration_weights.pt",
        map_location="cpu", weights_only=False,
    )
    assert weights["selected_family"] == "S2"
    assert weights["protocol_sha256"] == result["protocol_sha256"]


def test_optional_unavailable_gap_is_explicit_null() -> None:
    result = json.loads((ROOT / "results/semantic_value_calibration.json").read_text())
    for family in ("S0", "S1", "S1.5", "S2", "IMAGE_ONLY", "LANGUAGE_ONLY"):
        row = result["selection"]["metrics"][family]["ordinary"]["per_gap"][">16"]
        assert row == {"rows": 0, "accuracy": None, "auroc": None,
                       "log_loss": None, "brier": None}
