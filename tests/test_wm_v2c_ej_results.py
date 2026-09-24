#!/usr/bin/env python3
"""Independent artifact and metric back-checks for WM-v2C-EJ."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/world_model_v2c_ej/seed11/diagnosis.json"
SIDECAR = ROOT / "artifacts/stage3/wm_v2c_ej_seed11_diagnosis/samples.npz"
BRANCHES = ROOT / "artifacts/stage3/local_candidate_counterfactual/branches.npz"
REPORT = ROOT / "docs/wm_v2c_ej_physical_visual_coupling_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(
        1.0
        - np.sum(left * right, axis=-1)
        / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)),
        0.0,
        None,
    )


def pairwise(value: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            cosine_distance(value[:, 0], value[:, 1]),
            cosine_distance(value[:, 0], value[:, 2]),
            cosine_distance(value[:, 1], value[:, 2]),
        ],
        axis=1,
    )


def test_artifact_hashes_identity_boundaries_and_finiteness() -> None:
    result = json.loads(RESULT.read_text())
    assert sha256(RESULT) == "39bd7716b8e939da10b4ae953c9496833c9c216d5bae764ee80062e4d2f454dc"
    assert sha256(SIDECAR) == "133557d9cf9559115bb11b0eb1786e9eaf439d8e769b28b796675a4a2d051865"
    assert result["verdict"] == "PHYSICAL_TO_VISUAL_TRANSFER_STILL_INSUFFICIENT"
    assert result["authorization"] == "BLOCK_RERANKER_REDIAGNOSIS"
    assert result["nan_inf_audit"] == "pass"
    with np.load(SIDECAR) as arrays:
        assert arrays["task_id"].shape == (5869,)
        assert set(arrays["demo_id"].tolist()) == set(range(45, 50))
        assert arrays["matched_task_id"].shape == (40,)
        assert set(arrays["matched_task_id"].tolist()) == set(range(10))
        assert set(arrays["matched_init_state_index"].tolist()) == {30, 35}
        assert set(arrays["matched_planning_index"].tolist()) == {0, 8}
        assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
    assert result["scope"]["init_40_49_accessed"] is False
    assert result["scope"]["simulator_rerun"] is False
    assert result["scope"]["controller_success_used"] is False
    assert not any(result["forbidden_activity_audit"].values())


def test_demo_validation_and_matched_metrics_recompute() -> None:
    result = json.loads(RESULT.read_text())
    with np.load(SIDECAR) as arrays, np.load(BRANCHES) as branches:
        global_error = arrays["global_error"]
        assert np.isclose(
            global_error[0].mean(),
            result["ordinary_visual_prediction"]["overall"],
            atol=2e-8,
        )
        assert np.isclose(
            global_error[2].mean() / global_error[0].mean() - 1.0,
            result["global_action_conditioning"]["within_task_shuffle"][
                "relative_delta"
            ],
            atol=2e-8,
        )
        predictions = arrays["matched_predictions"][0].astype(np.float64)
        real = branches["real_z"].astype(np.float64)
        predicted_separation = pairwise(predictions)
        real_separation = pairwise(real)
        error = cosine_distance(predictions, real)
        transfer = predicted_separation.mean() / real_separation.mean()
        full = result["matched_counterfactual"]["modes"]["full_ej"]
        assert np.isclose(transfer, full["counterfactual_transfer_ratio"], atol=1e-12)
        assert np.isclose(error.mean(), full["candidate_specific_error"]["overall"]["mean"], atol=1e-12)
        assert np.allclose(
            error.mean(axis=(0, 1)),
            full["candidate_specific_error"]["per_horizon"],
            atol=1e-12,
        )
        assert np.isclose(
            arrays["residual_pair_l2"].mean() / arrays["base_pair_l2"].mean(),
            result["physical_visual_contribution"]["production_candidate_sensitivity"][
                "adapter_pair_l2_to_base_pair_l2"
            ],
            atol=1e-8,
        )
        assert np.max(np.abs(np.linalg.norm(predictions, axis=-1) - 1.0)) < 1e-5


def test_mechanical_gate_and_candidate_cross_tables() -> None:
    result = json.loads(RESULT.read_text())
    decision = result["decision"]
    assert decision["gate_pass"] == {
        "A_ordinary_visual": True,
        "B_global_action": True,
        "C_local_predicted_real_separation": False,
        "D_candidate_specific_prediction": True,
        "E_visual_delta_direction": False,
        "F_residual_non_degeneracy": True,
    }
    assert decision["controller_authorized"] is False
    assert decision["seed23_seed47_authorized"] is False
    with np.load(SIDECAR) as arrays:
        error = arrays["full_ej_candidate_error"]
        task = arrays["matched_task_id"]
        assert error.shape == (40, 3, 8)
        candidate_horizon = error.mean(axis=0)
        assert np.isclose(candidate_horizon[0, 0], 0.01062768, atol=1e-7)
        assert np.isclose(candidate_horizon[2, 7], 0.01645484, atol=1e-7)
        candidate_task = np.stack(
            [error[task == task_id].mean(axis=(0, 2)) for task_id in range(10)]
        )
        assert np.isclose(candidate_task[0, 0], 0.01704413, atol=1e-7)
        assert np.isclose(candidate_task[9, 2], 0.01647814, atol=1e-7)
    report = REPORT.read_text()
    assert "Candidate-specific errors by horizon" in report
    assert "Candidate-specific errors by task" in report
    assert "A0-A1 direction is `0.0093`" in report
    assert "PHYSICAL_TO_VISUAL_TRANSFER_STILL_INSUFFICIENT" in report

