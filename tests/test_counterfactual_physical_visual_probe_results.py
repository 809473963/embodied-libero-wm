#!/usr/bin/env python3
"""Independent result checks for the held-out physical/visual probe."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/counterfactual_physical_visual_probe/diagnosis.json"
REPORT_PATH = ROOT / "docs/counterfactual_physical_visual_sufficiency_probe.md"
RESULT_SHA256 = "b2ea73ae18c96dd2c3fc3c990750705b51195b2f4982bbc747a32eed9696407d"
REPORT_SHA256 = "6e816968d63c6e0041f6bb3f491f118c40692ddc7b9826715df1188d2f36e8f5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def direction(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum(prediction.astype(np.float64) * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-12,
    )
    return float(values.mean())


def test_result_and_report_hashes_status_and_forbidden_audit() -> None:
    assert sha256(RESULT_PATH) == RESULT_SHA256
    assert sha256(REPORT_PATH) == REPORT_SHA256
    result = json.loads(RESULT_PATH.read_text())
    assert result["status"] == "COMPLETE_ONE_SHOT_HELD_OUT_EVALUATION"
    assert result["decision"]["mechanism_labels"] == [
        "CONTEXT_DOMINATES_PHYSICAL_SIGNAL_NOT_ADDITIVE"
    ]
    assert result["decision"]["authorization"] == "BLOCK_CONTEXT_CONDITIONED_WM_DESIGN"
    assert result["nan_inf_audit"] == "pass"
    assert finite_tree(result)
    assert not any(result["forbidden_activity_audit"].values())
    assert result["evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert result["evaluation"]["simulator_rerun"] is False


def test_sidecar_recomputes_primary_metrics_and_preserves_boundaries() -> None:
    result = json.loads(RESULT_PATH.read_text())
    sidecar_path = ROOT / result["artifacts"]["sidecar"]
    assert sha256(sidecar_path) == result["artifacts"]["sidecar_sha256"]
    with np.load(sidecar_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    assert arrays["context"].shape == (40, 256)
    assert arrays["real_physical"].shape == (40, 3, 8, 10)
    assert arrays["predicted_physical"].shape == (40, 3, 8, 10)
    assert arrays["real_visual"].shape == (40, 3, 8, 512)
    assert set(arrays["init_state_index"].tolist()) == {36, 37, 38, 39}
    assert set(arrays["task_id"].tolist()) == set(range(10))
    assert all(np.isfinite(value).all() for value in arrays.values())

    target = arrays["real_visual"]
    for name in (
        "p0",
        "p1",
        "p2",
        "p3_real",
        "p3_predicted",
        "p3_physical_shuffle",
        "p3_context_shuffle",
    ):
        prediction = arrays[f"prediction__{name}"]
        assert prediction.shape == target.shape
        expected = result["metrics"][name]["overall"]
        measured_rmse = float(
            np.sqrt(np.mean((prediction.astype(np.float64) - target) ** 2))
        )
        assert math.isclose(measured_rmse, expected["rmse"], abs_tol=1.0e-12)
        assert math.isclose(direction(prediction, target), expected["direction"], abs_tol=1.0e-9)
    assert np.count_nonzero(arrays["prediction__p0"]) == 0

    for mapping_name in ("physical_shuffle_mapping", "context_shuffle_mapping"):
        mapping = arrays[mapping_name]
        assert np.all(mapping != np.arange(40))
        assert np.array_equal(arrays["task_id"][mapping], arrays["task_id"])


def test_gates_and_report_cover_required_interpretation() -> None:
    result = json.loads(RESULT_PATH.read_text())
    decision = result["decision"]
    assert decision["p1_pass"] is False
    assert decision["p3_real_pass"] is False
    assert decision["p3_predicted_pass"] is False
    assert decision["context_dominates"] is True
    assert decision["p3_real_checks"]["better_than_zero"] is False
    assert decision["p3_real_checks"]["physical_shuffle_worse"] is False
    assert decision["p3_real_checks"]["context_shuffle_worse"] is False
    assert decision["p3_real_checks"]["positive_direction"] is True

    report = REPORT_PATH.read_text()
    for phrase in (
        "does **not** support the hypothesis",
        "context-only P2 is also worse than",
        "scientifically supported",
        "Exact branch split and isolation",
        "Physical and context shuffle controls",
        "Real versus predicted physical mediator",
        "Per-horizon evidence",
        "Per-task evidence",
        "Per-pair evidence",
        "Antisymmetry, finite values, and mechanical decision",
        "Forbidden-activity audit",
    ):
        assert phrase in report
