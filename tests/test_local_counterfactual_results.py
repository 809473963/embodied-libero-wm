#!/usr/bin/env python3
"""Independent result audit for the matched local-counterfactual branches."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/local_candidate_counterfactual_protocol.json"
RESULT_PATH = ROOT / "results/local_candidate_counterfactual/diagnosis.json"
REPORT_PATH = ROOT / "docs/local_candidate_counterfactual_dynamics_audit.md"
FINAL_RESULT_SHA256 = "e3fede26f63f480df815c9a6bc78a658cfd15f7439f135194318e1002ef5f517"
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
MODEL_NAMES = (
    "wm_v1_d15_ensemble",
    "wm_v2a_seed11",
    "wm_v2b_seed11",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def pairwise(value: np.ndarray, metric: str) -> np.ndarray:
    rows = []
    for left, right in PAIR_INDICES:
        difference = value[:, left].astype(np.float64) - value[:, right].astype(
            np.float64
        )
        if metric == "cosine":
            a = value[:, left].astype(np.float64)
            b = value[:, right].astype(np.float64)
            similarity = np.sum(a * b, axis=-1) / np.maximum(
                np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12
            )
            rows.append(1.0 - similarity)
        elif metric == "l2":
            rows.append(np.linalg.norm(difference, axis=-1))
        elif metric == "rmse":
            rows.append(np.sqrt(np.mean(difference**2, axis=-1)))
        else:  # pragma: no cover - test helper misuse
            raise ValueError(metric)
    return np.stack(rows, axis=1)


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def test_counterfactual_result_scope_metrics_and_gate() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    assert sha256(RESULT_PATH) == FINAL_RESULT_SHA256
    assert result["status"] == "COMPLETE"
    assert result["experiment_version"] == (
        "local-candidate-counterfactual-dynamics-v1.2"
    )
    assert result["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert result["scope"] == {
        "suite": "libero_spatial",
        "tasks": 10,
        "init_state_indices": [30, 35],
        "planning_indices": [0, 8],
        "branch_points": 40,
        "candidate_branches": 120,
        "branch_horizon": 8,
        "rewards_recorded": False,
        "done_recorded": False,
        "success_recorded": False,
        "init_state_indices_read_or_run": [30, 35],
    }

    without_self_hash = copy.deepcopy(result)
    expected_before_hash = without_self_hash["artifacts"].pop(
        "result_sha256_before_hash_field"
    )
    payload = (
        json.dumps(without_self_hash, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == expected_before_hash

    identities = result["branch_identities"]
    observed = {
        (row["task_id"], row["init_state_index"], row["planning_index"])
        for row in identities
    }
    expected = {
        (task, init_state, plan)
        for task in range(10)
        for init_state in (30, 35)
        for plan in (0, 8)
    }
    assert observed == expected
    assert all(row["prefix_audit"]["pass"] for row in identities)
    assert all(row["prefix_audit"]["rgb_exact"] for row in identities)
    assert all(
        row["prefix_audit"]["planning_observation"]["exact"]
        for row in identities
    )
    assert all(
        row["prefix_audit"]["simulator_state_max_abs_difference"] == 0.0
        for row in identities
    )
    assert all(
        min(row["prefix_audit"]["clip_cosine_to_branch0"]) >= 0.999999
        for row in identities
    )

    sidecar_path = ROOT / result["artifacts"]["sidecar"]
    assert sha256(sidecar_path) == result["artifacts"]["sidecar_sha256"]
    with np.load(sidecar_path) as samples:
        assert {name: list(samples[name].shape) for name in samples.files} == result[
            "artifacts"
        ]["sidecar_arrays"]
        assert all(
            np.isfinite(samples[name]).all()
            for name in samples.files
            if samples[name].dtype.kind in "fiu"
        )
        assert np.array_equal(samples["candidates"], samples["executed_actions"])
        assert set(samples["task_id"].tolist()) == set(range(10))
        assert set(samples["init_state_index"].tolist()) == {30, 35}
        assert set(samples["planning_index"].tolist()) == {0, 8}
        assert set(samples["prefix_lengths"].tolist()) == {0, 64}

        candidate_hashes = [
            array_sha256(candidate) for candidate in samples["candidates"]
        ]
        assert candidate_hashes == result["analysis"]["candidate_hashes"]["values"]
        manifest_payload = json.dumps(
            candidate_hashes, sort_keys=True, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(manifest_payload).hexdigest() == result["analysis"][
            "candidate_hashes"
        ]["ordered_manifest_sha256"]

        real_z = samples["real_z"]
        real_visual = pairwise(real_z, "cosine")
        assert np.allclose(
            real_visual, samples["real_visual_cosine"], atol=2e-8, rtol=0
        )
        assert np.allclose(
            pairwise(real_z, "l2"), samples["real_visual_l2"], atol=2e-8, rtol=0
        )

        checkpoint = torch.load(
            ROOT / protocol["frozen_upstream"]["wm_v2b_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        p_mean = np.asarray(
            checkpoint["normalization"]["proprio"]["mean"], dtype=np.float64
        )
        p_std = np.asarray(
            checkpoint["normalization"]["proprio"]["std"], dtype=np.float64
        )
        standardized_p = (samples["real_proprio"].astype(np.float64) - p_mean) / p_std
        assert np.allclose(
            pairwise(standardized_p, "rmse"),
            samples["real_proprio_standardized_rmse"],
            atol=2e-8,
            rtol=0,
        )

        rgb = samples["real_rgb"].astype(np.float64)
        pixel_mae = np.stack(
            [
                np.mean(np.abs(rgb[:, a] - rgb[:, b]), axis=(-1, -2, -3))
                for a, b in PAIR_INDICES
            ],
            axis=1,
        )
        assert np.allclose(pixel_mae, samples["pixel_mae"], atol=3e-7, rtol=0)

        real_mean = float(real_visual.mean())
        analysis = result["analysis"]
        assert math.isclose(
            real_mean,
            analysis["real_counterfactual_dynamics"]["visual_cosine"]["mean"],
            abs_tol=2e-10,
        )
        visual_reference = protocol["decision_rules"]["reference_values"][
            "d15_ordinary_validation_visual_error"
        ]
        assert math.isclose(
            real_mean / visual_reference,
            analysis["real_counterfactual_dynamics"][
                "real_visual_to_d15_validation_error"
            ],
            abs_tol=2e-9,
        )

        validation_refs = {
            "wm_v1_d15_ensemble": "d15_ordinary_validation_visual_error",
            "wm_v2a_seed11": "wm_v2a_ordinary_validation_visual_error",
            "wm_v2b_seed11": "wm_v2b_ordinary_validation_visual_error",
        }
        for model in MODEL_NAMES:
            predicted = samples[f"{model}_predicted_visual_separation"]
            error = samples[f"{model}_candidate_visual_error"]
            transfer = analysis["model_transfer"][model]["transfer"]
            assert math.isclose(
                float(predicted.mean()) / real_mean,
                transfer["ratio_of_means"],
                abs_tol=2e-9,
            )
            alt = float(error[:, 1:].mean())
            a0 = float(error[:, 0].mean())
            gate = analysis["decision"]["prediction_failure_by_model"][model]
            reference = protocol["decision_rules"]["reference_values"][
                validation_refs[model]
            ]
            assert math.isclose(
                alt / reference,
                gate["alternative_to_validation_error"],
                abs_tol=2e-7,
            )
            assert math.isclose(
                alt / a0,
                gate["alternative_to_matched_A0_error"],
                abs_tol=2e-7,
            )
            assert math.isclose(
                float((error[:, 1:] > error[:, [0]]).mean()),
                gate["alternative_worse_than_A0_fraction"],
                abs_tol=1e-12,
            )
            assert gate["fail"] is False

        correlation = spearmanr(
            samples["real_proprio_standardized_rmse"].ravel(),
            samples["real_visual_cosine"].ravel(),
        ).statistic
        assert math.isclose(
            float(correlation),
            analysis["proprio_visual_correlation"]["overall"]["spearman"],
            abs_tol=1e-12,
        )

    decision = result["analysis"]["decision"]
    assert decision["real_visual_meaningful"] is True
    assert decision["real_physical_meaningful"] is True
    assert decision["wm_v2b_predicted_below_real_horizons"] == 8
    assert decision["visual_transfer_failure_confirmed"] is True
    assert decision["local_clip_counterfactual_limit"] is False
    assert decision["verdicts"] == ["VISUAL_TRANSFER_FAILURE_CONFIRMED"]
    assert result["verdict"] == decision["final_verdict"] == (
        "VISUAL_TRANSFER_FAILURE_CONFIRMED"
    )
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
    assert finite_tree(result)

    report = REPORT_PATH.read_text()
    assert "Final verdict: `VISUAL_TRANSFER_FAILURE_CONFIRMED`" in report
    assert "Preflight correction history" in report
    assert "0--29 and 40--49 were not run" in report
    assert "No reward, done, success" in report
    assert FINAL_RESULT_SHA256 in report
