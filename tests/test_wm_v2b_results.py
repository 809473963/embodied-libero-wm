#!/usr/bin/env python3
"""Result-level provenance, metric, and gate audit for WM-v2B seed 11."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/wm_v2b_future_proprio_protocol.json"
RESULT_PATH = ROOT / "results/world_model_v2b/seed11/diagnosis.json"
REPORT_PATH = ROOT / "docs/wm_v2b_future_proprio_auxiliary_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    assert result["experiment_version"] == protocol["protocol_version"]
    assert result["status"] == "COMPLETE"
    assert result["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert result["protocol"]["preregistered_commit"] == (
        "fdafce8d649a48d19ebeb721f4c43a0c961f0504"
    )
    assert result["protocol"]["training_source_commit"] == (
        "801809f880517cebbe6c5d7ebc8b58a4d5198048"
    )
    for commit in (
        result["protocol"]["preregistered_commit"],
        result["protocol"]["training_source_commit"],
    ):
        assert subprocess.check_output(
            ["git", "cat-file", "-t", commit], cwd=ROOT, text=True
        ).strip() == "commit"

    model = result["model"]
    checkpoint_path = ROOT / model["checkpoint"]
    assert sha256(checkpoint_path) == model["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["source_commit"] == result["protocol"]["training_source_commit"]
    assert checkpoint["epoch"] == model["checkpoint_epoch"] == 14
    assert checkpoint["global_step"] == model["checkpoint_global_step"] == 2870
    assert checkpoint["lambda_p"] == model["lambda_p"] == 0.42692670353890133
    assert model["parameters"] == model["backbone_parameters"] + model[
        "auxiliary_head_parameters"
    ] == 826444
    assert all(
        torch.isfinite(value).all() for value in checkpoint["model_state"].values()
    )
    freeze_path = ROOT / "configs/wm_v2b_calibration_freeze.json"
    assert sha256(freeze_path) == model["calibration_freeze_sha256"]

    assert result["scope"] == {
        "train_demos": "0-44",
        "validation_demos": "45-49",
        "train_samples": 52381,
        "validation_samples": 5869,
        "tasks": 10,
        "simulator_or_controller_rollouts": 0,
        "success_metric_used": False,
        "future_proprio_is_input": False,
    }
    sidecar_path = ROOT / result["sample_artifact"]["path"]
    assert sha256(sidecar_path) == result["sample_artifact"]["sha256"]
    with np.load(sidecar_path) as samples:
        assert {name: list(samples[name].shape) for name in samples.files} == result[
            "sample_artifact"
        ]["arrays"]
        task = samples["task_id"].astype(np.int64)
        demo = samples["demo_id"].astype(np.int64)
        timestep = samples["start_timestep"].astype(np.int64)
        assert set(task.tolist()) == set(range(10))
        assert set(demo.tolist()) == set(range(45, 50))
        assert len(set(zip(task.tolist(), demo.tolist(), timestep.tolist()))) == 5869
        assert all(
            np.isfinite(samples[name]).all()
            for name in samples.files
            if samples[name].dtype.kind in "fiu"
        )

        z_error = samples["global_z_error"].astype(np.float64)
        p_error = samples["global_p_error"].astype(np.float64)
        z_separation = samples["candidate_z_separation"].astype(np.float64)
        p_separation = samples["candidate_p_separation"].astype(np.float64)
        p_rmse = samples["correct_p_rmse"].astype(np.float64)
        assert math.isclose(
            float(z_error[0].mean()),
            result["visual_prediction"]["correct_action_error"],
            abs_tol=2e-9,
        )
        assert math.isclose(
            float(p_error[0].mean()),
            result["proprio_prediction"]["correct_action_error"],
            abs_tol=2e-9,
        )
        assert math.isclose(
            float(z_separation.mean() / z_error[0].mean()),
            result["visual_local_candidate_sensitivity"]["separation_to_error_ratio"],
            abs_tol=5e-10,
        )
        assert math.isclose(
            float(p_separation.mean() / p_rmse.mean()),
            result["proprio_local_candidate_sensitivity"][
                "separation_to_correct_prediction_rmse_ratio"
            ],
            abs_tol=1e-7,
        )
        assert math.isclose(
            float(samples["visual_direction_pair_win"].mean()),
            result["visual_local_directional_validity"]["pairwise_ranking_accuracy"],
            abs_tol=5e-8,
        )
        assert math.isclose(
            float(samples["proprio_direction_pair_win"].mean()),
            result["proprio_local_directional_validity"]["pairwise_ranking_accuracy"],
            abs_tol=5e-8,
        )

    decision = result["decision"]
    assert decision["gate_pass"] == {
        "prediction_quality": True,
        "visual_global": True,
        "visual_local": False,
        "visual_direction": True,
        "proprio_prediction": True,
        "proprio_action": True,
        "proprio_local": True,
        "proprio_direction": True,
    }
    assert result["verdict"] == decision["verdict"] == (
        "PHYSICAL_SIGNAL_LEARNED_VISUAL_TRANSFER_INSUFFICIENT"
    )
    assert decision["seed23_seed47_authorized"] is False
    assert result["seed23_seed47_trained"] is False
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
    assert finite_tree(result)

    report = REPORT_PATH.read_text()
    assert "PENDING" not in report
    assert (
        "Final verdict: `PHYSICAL_SIGNAL_LEARNED_VISUAL_TRANSFER_INSUFFICIENT`"
        in report
    )
    assert "seed23/47 are not authorized" in report.lower()
    assert "No simulator or controller rollout was run" in report
    print(
        {
            "status": "PASS",
            "verdict": result["verdict"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "sidecar_sha256": result["sample_artifact"]["sha256"],
        }
    )


if __name__ == "__main__":
    main()
