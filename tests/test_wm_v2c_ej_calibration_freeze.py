#!/usr/bin/env python3
"""Immutable train-only gradient-calibration checks for WM-v2C-EJ."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/wm_v2c_ej_calibration_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_train_only_calibration_is_frozen_before_training() -> None:
    freeze = json.loads(FREEZE.read_text())
    result_path = ROOT / freeze["calibration_result"]
    result = json.loads(result_path.read_text())
    assert freeze["status"] == "FROZEN_BEFORE_TRAINING"
    assert result["status"] == "COMPLETE_BEFORE_TRAINING"
    assert sha256(result_path) == freeze["calibration_result_sha256"]
    assert sha256(ROOT / freeze["protocol"]) == freeze["protocol_sha256"]
    assert sha256(ROOT / freeze["config"]) == freeze["config_sha256"]
    assert result["data"]["split"] == "train demos 0-44 only"
    assert set(record["demo_id"] for record in result["data"]["identities"]) <= set(range(45))
    assert len(result["data"]["identities"]) == 1024
    assert result["data"]["identity_sha256"] == freeze["train_sample_identity_sha256"]
    assert result["model"]["optimizer_updates"] == freeze["optimizer_updates"] == 0
    assert result["model"]["initial_max_abs_physical_visual_residual"] == 0.0
    assert result["validation_metrics_used"] is False
    assert result["matched_counterfactual_used"] is False
    assert freeze["validation_metrics_used"] is False
    assert freeze["matched_counterfactual_used"] is False
    calibration = result["gradient_calibration"]
    assert calibration["g_z"] == freeze["g_z"]
    assert calibration["g_ej"] == freeze["g_ej"]
    assert calibration["final_lambda_ej"] == freeze["final_lambda_ej"]
    assert freeze["clip_range"][0] <= freeze["final_lambda_ej"] <= freeze["clip_range"][1]
    assert freeze["lambda_res"] == 0.001
    assert freeze["gripper_adapter_input"] is False

