#!/usr/bin/env python3
"""Independent WM-v2B train-only gradient-calibration freeze audit."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_world_model_v2b import model_from_config, tensor_state_sha256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    freeze_path = ROOT / "configs/wm_v2b_calibration_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    calibration_path = ROOT / freeze["calibration_result"]
    calibration = json.loads(calibration_path.read_text())
    protocol_path = ROOT / freeze["protocol"]
    config_path = ROOT / freeze["config"]
    assert freeze["status"] == "FROZEN_BEFORE_TRAINING"
    assert sha256(protocol_path) == freeze["protocol_sha256"]
    assert sha256(config_path) == freeze["config_sha256"]
    assert sha256(calibration_path) == freeze["calibration_result_sha256"]
    assert calibration["status"] == "COMPLETE_BEFORE_TRAINING"
    assert calibration["source_commit"] == freeze["calibration_source_commit"]
    assert calibration["validation_metrics_used"] is False
    assert freeze["validation_metrics_used"] is False
    assert freeze["optimizer_updates"] == 0

    identities = calibration["data"]["identities"]
    assert len(identities) == 1024
    assert len({record["dataset_index"] for record in identities}) == 1024
    assert {record["demo_id"] for record in identities}.issubset(set(range(45)))
    expected = np.random.default_rng(220501).permutation(52381)[:1024]
    assert [record["dataset_index"] for record in identities] == expected.tolist()
    assert calibration["data"]["identity_sha256"] == freeze["identity_sha256"]

    g_z = float(np.mean([batch["g_z"] for batch in calibration["batches"]]))
    g_p = float(np.mean([batch["g_p"] for batch in calibration["batches"]]))
    assert math.isclose(g_z, freeze["g_z"], rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(g_p, freeze["g_p"], rel_tol=0.0, abs_tol=1e-15)
    raw = g_z / g_p
    final = float(np.clip(raw, *freeze["clip_range"]))
    assert math.isclose(raw, freeze["unclipped_lambda_p"], rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(final, freeze["final_lambda_p"], rel_tol=0.0, abs_tol=1e-15)
    assert freeze["was_clipped"] is False

    config = yaml.safe_load(config_path.read_text())
    import torch

    torch.manual_seed(11)
    model = model_from_config(config)
    assert tensor_state_sha256(model) == calibration["model"]["initial_state_sha256"]
    assert calibration["model"]["parameters"] == 826444
    print(
        {
            "status": "PASS",
            "g_z": g_z,
            "g_p": g_p,
            "lambda_p": final,
            "identity_sha256": freeze["identity_sha256"],
        }
    )


if __name__ == "__main__":
    main()
