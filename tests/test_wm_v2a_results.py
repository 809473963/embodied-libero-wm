#!/usr/bin/env python3
"""Result-level provenance and gate audit for WM-v2A seed 11."""

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
PROTOCOL_PATH = ROOT / "configs/wm_v2a_local_dynamics_protocol.json"
RESULT_PATH = ROOT / "results/world_model_v2a/seed11/diagnosis.json"
REPORT_PATH = ROOT / "docs/wm_v2a_local_dynamics_resolution_audit.md"


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
    preregistered_commit = result["protocol"]["preregistered_commit"]
    assert preregistered_commit == "d2767d8488cb71c6050f3626ea2c527729fc0a0f"
    assert subprocess.check_output(
        ["git", "cat-file", "-t", preregistered_commit], cwd=ROOT, text=True
    ).strip() == "commit"

    model = result["model"]
    checkpoint_path = ROOT / model["checkpoint"]
    assert sha256(checkpoint_path) == model["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["source_commit"] == preregistered_commit
    assert checkpoint["epoch"] == model["checkpoint_epoch"] == 13
    assert checkpoint["global_step"] == model["checkpoint_global_step"] == 2665
    assert checkpoint["config_sha256"] == protocol["experiment"]["config_sha256"]
    assert checkpoint["d13_store_sha256"] == protocol["data"]["store_sha256"]
    assert checkpoint["loss_contract"] == model["loss_contract"]
    assert model["parameters"] == 792000
    assert model["lambda_delta"] == 1.0
    assert all(
        torch.isfinite(value).all() for value in checkpoint["model_state"].values()
    )

    assert result["scope"] == {
        "train_demos": "0-44",
        "validation_demos": "45-49",
        "train_samples": 52381,
        "validation_samples": 5869,
        "tasks": 10,
        "simulator_or_controller_rollouts": 0,
        "success_metric_used": False,
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

        global_error = samples["global_error"].astype(np.float64)
        candidate_separation = samples["candidate_separation"].astype(np.float64)
        direction_win = samples["direction_pair_win"].astype(np.float64)
        direction_spearman = samples["direction_within_spearman"].astype(np.float64)
        correct_error = float(global_error[0].mean())
        within_delta = float((global_error[2] - global_error[0]).mean())
        within_accuracy = float(np.mean(global_error[0] < global_error[2]))
        local_ratio = float(candidate_separation.mean() / correct_error)
        assert math.isclose(
            correct_error,
            result["prediction_quality"]["correct_action_error"],
            abs_tol=2e-9,
        )
        within = result["global_action_diagnostics"]["comparisons"]["shuffle_task"]
        assert math.isclose(within_delta, within["delta"], abs_tol=2e-9)
        assert math.isclose(
            within_accuracy, within["true_action_win_accuracy"], abs_tol=1e-12
        )
        assert math.isclose(
            local_ratio,
            result["local_candidate_sensitivity"][
                "separation_to_ordinary_error_ratio"
            ],
            abs_tol=5e-10,
        )
        directional = result["local_directional_validity"]
        assert math.isclose(
            float(direction_win.mean()),
            directional["pairwise_ranking_accuracy"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            float(direction_spearman.mean()),
            directional["mean_within_sample_spearman"],
            abs_tol=2e-9,
        )

    decision = result["decision"]
    assert decision["gate_pass"] == {
        "prediction_quality": True,
        "global_action_conditioning": True,
        "local_sensitivity": False,
        "local_directional_validity": True,
    }
    assert result["verdict"] == decision["verdict"] == (
        "DELTA_OBJECTIVE_INSUFFICIENT"
    )
    assert decision["seed23_seed47_authorized"] is False
    assert result["seed23_seed47_trained"] is False
    assert result["wm_v2b_implemented"] is False
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
    assert finite_tree(result)

    report = REPORT_PATH.read_text()
    assert "PENDING" not in report
    assert "Final verdict: `DELTA_OBJECTIVE_INSUFFICIENT`" in report
    assert "seed23/47 are not authorized" in report.lower()
    assert "No simulator or controller rollout was run" in report
    print(
        {
            "status": "PASS",
            "verdict": result["verdict"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "sample_sha256": result["sample_artifact"]["sha256"],
        }
    )


if __name__ == "__main__":
    main()
