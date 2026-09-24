#!/usr/bin/env python3
"""Training completion and pre-diagnosis checkpoint freeze checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/wm_v2c_ej_checkpoint_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_checkpoint_is_frozen_before_matched_diagnosis() -> None:
    freeze = json.loads(FREEZE.read_text())
    assert freeze["status"] == "FROZEN_BEFORE_MATCHED_COUNTERFACTUAL_DIAGNOSIS"
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("last_checkpoint", "last_checkpoint_sha256"),
        ("metrics", "metrics_sha256"),
        ("run_metadata", "run_metadata_sha256"),
        ("best_validation", "best_validation_sha256"),
        ("config", "config_sha256"),
        ("calibration_freeze", "calibration_freeze_sha256"),
    ):
        assert sha256(ROOT / freeze[path_key]) == freeze[hash_key]
    checkpoint = torch.load(ROOT / freeze["checkpoint"], map_location="cpu", weights_only=False)
    metadata = json.loads((ROOT / freeze["run_metadata"]).read_text())
    best = json.loads((ROOT / freeze["best_validation"]).read_text())
    assert checkpoint["epoch"] == best["epoch"] == freeze["epoch"] == 15
    assert checkpoint["global_step"] == best["global_step"] == freeze["global_step"]
    assert checkpoint["best_validation_total_loss"] == freeze["best_validation_total_loss"]
    assert checkpoint["best_validation_visual_cosine_error"] == (
        freeze["best_validation_visual_cosine_error"]
    )
    assert best["ej_delta_smooth_l1"] == freeze["best_validation_ej_smooth_l1"]
    assert metadata["parameters"] == freeze["parameters"] == 901516
    assert metadata["completed_epochs"] == freeze["completed_epochs"] == 15
    assert metadata["nan_inf_audit"] == freeze["nan_inf_audit"] == "pass"
    assert metadata["matched_counterfactual_used_for_selection"] is False
    assert freeze["matched_counterfactual_read_before_freeze"] is False
    assert freeze["matched_counterfactual_used_for_selection"] is False
    assert freeze["init_40_49_accessed"] is False

