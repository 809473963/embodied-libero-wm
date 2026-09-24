#!/usr/bin/env python3
"""Independently recompute the WM-v2B.1 held-out decision evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/wm_v2b_1_differential_calibration_protocol.json"
FREEZE_PATH = ROOT / "configs/wm_v2b_1_calibration_freeze.json"
RESULT_PATH = ROOT / "results/world_model_v2b_1/diagnosis.json"
REPORT_PATH = ROOT / "docs/wm_v2b_1_differential_physical_calibration_audit.md"
RESULT_SHA256 = "5690d3b30642143d605c7adddcb6f945c230ac2134a861a854bbae5b531f2c23"
REPORT_SHA256 = "a26622a15a7ce46035963afd54c2ba304a549fa3547101044e7e97024f62e50b"
EVALUATION_SIDECAR_SHA256 = (
    "76ff1ee009b686c5d71cd7968c37aeec02afcce9485bd3f1bd8ea232faca60fe"
)
ANALYSIS_SIDECAR_SHA256 = (
    "bc74c881324970bdde82a44de92a7f48162de17e9faa320d61061736a87dea6c"
)
PAIRS = ((0, 1), (0, 2), (1, 2))
GROUPS = {
    "ee_pos": np.asarray([0, 1, 2]),
    "gripper": np.asarray([3, 4]),
    "joints": np.asarray([5, 6, 7, 8, 9, 10, 11]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairwise(value: np.ndarray) -> np.ndarray:
    return np.stack(
        [value[:, left] - value[:, right] for left, right in PAIRS], axis=1
    )


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))


def eligible_vector_rmse(error: np.ndarray, real: np.ndarray) -> float:
    """Match the preregistered vector denominator floor for cut metrics."""
    eligible = np.linalg.norm(real, axis=-1) >= 1e-4
    return rmse(error[eligible])


def direction_mean(predicted: np.ndarray, real: np.ndarray) -> float:
    pred = predicted.reshape(-1, predicted.shape[-1]).astype(np.float64)
    target = real.reshape(-1, real.shape[-1]).astype(np.float64)
    target_norm = np.linalg.norm(target, axis=-1)
    pred_norm = np.linalg.norm(pred, axis=-1)
    eligible = (target_norm >= 1e-4) & (pred_norm > 0)
    cosine = np.sum(pred[eligible] * target[eligible], axis=-1) / (
        pred_norm[eligible] * target_norm[eligible]
    )
    return float(np.mean(cosine))


def sign_accuracy(predicted: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    eligible = np.abs(real) >= 1e-3
    accuracy = np.mean(np.sign(predicted[eligible]) == np.sign(real[eligible]))
    return float(accuracy), float(np.mean(eligible))


def load_deltas() -> tuple[np.ndarray, np.ndarray]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    checkpoint = torch.load(
        ROOT / protocol["frozen_upstream"]["wm_v2b_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    mean = np.asarray(checkpoint["normalization"]["proprio"]["mean"], dtype=np.float64)
    std = np.asarray(checkpoint["normalization"]["proprio"]["std"], dtype=np.float64)
    sidecar_path = ROOT / protocol["artifacts"]["evaluation_sidecar"]
    assert sha256(sidecar_path) == EVALUATION_SIDECAR_SHA256
    with np.load(sidecar_path) as sidecar:
        real = (sidecar["real_proprio"].astype(np.float64) - mean) / std
        predicted = sidecar["pred_proprio_wm_v2b_standardized"].astype(np.float64)
    return pairwise(real), pairwise(predicted)


def test_result_hashes_provenance_and_forbidden_activity() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    assert sha256(RESULT_PATH) == RESULT_SHA256
    assert sha256(REPORT_PATH) == REPORT_SHA256
    assert result["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert result["calibration_freeze"]["status"] == "FROZEN_BEFORE_EVALUATION"
    assert result["calibration_freeze"]["evaluation_artifacts_present_at_fit_time"] is False
    assert result["artifacts"]["evaluation_sidecar_sha256"] == EVALUATION_SIDECAR_SHA256
    assert result["artifacts"]["derived_sidecar_sha256"] == ANALYSIS_SIDECAR_SHA256
    assert sha256(ROOT / result["artifacts"]["derived_sidecar"]) == ANALYSIS_SIDECAR_SHA256
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())

    without_self_hash = copy.deepcopy(result)
    expected = without_self_hash["artifacts"].pop("result_sha256_before_hash_field")
    payload = (json.dumps(without_self_hash, indent=2, ensure_ascii=False) + "\n").encode()
    assert hashlib.sha256(payload).hexdigest() == expected


def test_primary_metrics_recompute_from_held_out_sidecar() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    real_delta, raw_delta = load_deltas()
    alpha = np.asarray(
        freeze["coefficients"]["per_horizon_scalar"], dtype=np.float64
    )[None, None, :, None]
    calibrated = raw_delta * alpha

    calculated_zero = rmse(real_delta)
    calculated_raw = rmse(raw_delta - real_delta)
    calculated_calibrated = rmse(calibrated - real_delta)
    calculated_vs_zero = (calculated_zero - calculated_calibrated) / calculated_zero
    calculated_vs_raw = (calculated_raw - calculated_calibrated) / calculated_raw
    calculated_direction = direction_mean(calibrated, real_delta)

    stored = result["analysis"]["forms"]["per_horizon_scalar"]["differential"][
        "overall"
    ]
    assert math.isclose(stored["zero_difference_rmse"], calculated_zero, abs_tol=1e-12)
    assert math.isclose(stored["raw_rmse"], calculated_raw, abs_tol=1e-12)
    assert math.isclose(stored["differential_rmse"], calculated_calibrated, abs_tol=1e-12)
    assert math.isclose(
        stored["relative_improvement_over_zero"], calculated_vs_zero, abs_tol=1e-12
    )
    assert math.isclose(stored["improvement_over_raw"], calculated_vs_raw, abs_tol=1e-12)
    assert math.isclose(
        stored["direction_cosine"]["mean"], calculated_direction, abs_tol=1e-12
    )

    assert math.isclose(calculated_zero, 0.020335207741131414, abs_tol=1e-12)
    assert math.isclose(calculated_raw, 0.018707141912585698, abs_tol=1e-12)
    assert math.isclose(calculated_calibrated, 0.013184314212118499, abs_tol=1e-12)
    assert math.isclose(calculated_vs_zero, 0.35165087173164294, abs_tol=1e-12)
    assert math.isclose(calculated_vs_raw, 0.29522562699711913, abs_tol=1e-12)

    for pair_index, pair_name in enumerate(("A0-A1", "A0-A2", "A1-A2")):
        pair_real = real_delta[:, pair_index]
        pair_calibrated = calibrated[:, pair_index]
        pair_zero = rmse(pair_real)
        pair_error = rmse(pair_calibrated - pair_real)
        pair_improvement = (pair_zero - pair_error) / pair_zero
        pair_stored = result["analysis"]["forms"]["per_horizon_scalar"][
            "differential"
        ]["per_pair"][pair_name]
        assert math.isclose(pair_stored["differential_rmse"], pair_error, abs_tol=1e-12)
        assert math.isclose(
            pair_stored["relative_improvement_over_zero"],
            pair_improvement,
            abs_tol=1e-12,
        )
        assert pair_improvement > 0


def test_groupwise_partial_pass_and_mechanical_decision() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    real_delta, raw_delta = load_deltas()
    group_results = result["analysis"]["forms"]["groupwise_scalar"]

    recomputed = {}
    for group, indices in GROUPS.items():
        alpha = freeze["coefficients"]["groupwise_scalar"][group]
        real = real_delta[..., indices]
        calibrated = raw_delta[..., indices] * alpha
        zero = eligible_vector_rmse(real, real)
        error = eligible_vector_rmse(calibrated - real, real)
        improvement = (zero - error) / zero
        direction = direction_mean(calibrated, real)
        sign, coverage = sign_accuracy(calibrated, real)
        recomputed[group] = {
            "improvement": improvement,
            "direction": direction,
            "sign": sign,
            "coverage": coverage,
        }
        stored = group_results["differential"]["groups"][group]
        stored_sign = group_results["sign"]["groups"][group]
        assert math.isclose(
            stored["relative_improvement_over_zero"], improvement, abs_tol=1e-12
        )
        assert math.isclose(stored["direction_cosine"]["mean"], direction, abs_tol=1e-12)
        assert math.isclose(stored_sign["accuracy"], sign, abs_tol=1e-12)
        assert math.isclose(stored_sign["eligible_fraction"], coverage, abs_tol=1e-12)

    assert recomputed["ee_pos"]["improvement"] >= 0.05
    assert recomputed["ee_pos"]["sign"] >= 0.55
    assert recomputed["joints"]["improvement"] >= 0.05
    assert recomputed["joints"]["sign"] >= 0.55
    assert recomputed["gripper"]["improvement"] < 0.0
    assert recomputed["gripper"]["sign"] < 0.55

    decision = result["analysis"]["decision"]
    assert decision["primary_gate_pass"]["full_gripper_validity"] is False
    assert all(
        passed
        for name, passed in decision["primary_gate_pass"].items()
        if name != "full_gripper_validity"
    )
    assert decision["partial_valid_groups"] == ["ee_pos", "joints"]
    assert decision["partial_group_checks"]["gripper"] == {
        "zero_improvement": False,
        "direction": True,
        "sign": False,
    }
    assert result["verdict"] == "PARTIAL_PHYSICAL_MEDIATOR_PASS"
    assert result["wm_v2c_authorization"] == "BLOCK_WM_V2C_PHYSICAL_MEDIATOR"
    assert result["mechanism_labels"] == [
        "EE_JOINT_MEDIATOR_VALID_GRIPPER_INVALID"
    ]


def test_report_states_the_block_without_overclaiming() -> None:
    report = REPORT_PATH.read_text()
    assert "Final verdict: `PARTIAL_PHYSICAL_MEDIATOR_PASS`" in report
    assert "WM-v2C authorization: `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`" in report
    assert "`EE_JOINT_MEDIATOR_VALID_GRIPPER_INVALID`" in report
    assert "gripper | 0.419146 | -6.17%" in report
    assert "This experiment does not design or train WM-v2C." in report
