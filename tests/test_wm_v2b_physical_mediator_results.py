#!/usr/bin/env python3
"""Independent result audit for WM-v2B physical-mediator fidelity."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/wm_v2b_counterfactual_physical_mediator_protocol.json"
)
RESULT_PATH = ROOT / "results/world_model_v2b_physical_mediator/diagnosis.json"
REPORT_PATH = ROOT / "docs/wm_v2b_counterfactual_physical_mediator_audit.md"
FINAL_RESULT_SHA256 = (
    "07bc7dba47a71b917c98c2be541527641157c66742f0314b29cf8db37aebd39d"
)
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))


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


def pairwise_delta(value: np.ndarray) -> np.ndarray:
    return np.stack(
        [value[:, left] - value[:, right] for left, right in PAIR_INDICES],
        axis=1,
    )


def test_result_provenance_scope_and_source_integrity() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    assert sha256(RESULT_PATH) == FINAL_RESULT_SHA256
    assert result["status"] == "COMPLETE"
    assert result["experiment_version"] == protocol["protocol_version"]
    assert result["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert result["protocol"]["execution_source_commit"] == (
        "f2df2ba5d5a80042fe068c607510d3f7be9fe15f"
    )
    assert subprocess.check_output(
        ["git", "cat-file", "-t", result["protocol"]["execution_source_commit"]],
        cwd=ROOT,
        text=True,
    ).strip() == "commit"

    without_self_hash = copy.deepcopy(result)
    before_hash = without_self_hash["artifacts"].pop(
        "result_sha256_before_hash_field"
    )
    payload = (
        json.dumps(without_self_hash, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == before_hash

    assert result["scope"] == protocol["scope"]
    assert result["scope"]["simulator_runs"] == 0
    assert result["scope"]["success_reward_done_used"] is False
    assert result["scope"]["init_state_indices"] == [30, 35]
    assert result["scope"]["planning_indices"] == [0, 8]
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
    assert finite_tree(result)

    frozen = protocol["frozen_upstream"]
    for path_key, hash_key in (
        ("counterfactual_result", "counterfactual_result_sha256"),
        ("counterfactual_report", "counterfactual_report_sha256"),
        ("matched_branch_sidecar", "matched_branch_sidecar_sha256"),
        ("wm_v2b_result", "wm_v2b_result_sha256"),
        ("wm_v2b_validation_sidecar", "wm_v2b_validation_sidecar_sha256"),
        ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
        ("wm_v2b_config", "wm_v2b_config_sha256"),
        ("wm_v2b_protocol", "wm_v2b_protocol_sha256"),
    ):
        assert sha256(ROOT / frozen[path_key]) == frozen[hash_key]


def test_absolute_and_counterfactual_metrics_recompute_from_frozen_sources() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    frozen = protocol["frozen_upstream"]
    checkpoint = torch.load(
        ROOT / frozen["wm_v2b_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    p_mean = np.asarray(
        checkpoint["normalization"]["proprio"]["mean"], dtype=np.float64
    )
    p_std = np.asarray(
        checkpoint["normalization"]["proprio"]["std"], dtype=np.float64
    )

    with np.load(ROOT / frozen["wm_v2b_validation_sidecar"]) as validation:
        loss = validation["correct_p_element_error"].astype(np.float64)
        validation_absolute = np.where(
            loss < 0.5, np.sqrt(2.0 * loss), loss + 0.5
        )
        assert np.allclose(
            np.sqrt(np.mean(validation_absolute**2, axis=-1)),
            validation["correct_p_rmse"],
            atol=2e-7,
            rtol=0,
        )

    with np.load(ROOT / frozen["matched_branch_sidecar"]) as branches:
        real = (branches["real_proprio"].astype(np.float64) - p_mean) / p_std
        predicted = branches[
            "pred_proprio_wm_v2b_standardized"
        ].astype(np.float64)
        branch_absolute = np.abs(predicted - real)
        real_delta = pairwise_delta(real)
        predicted_delta = pairwise_delta(predicted)
        task_id = branches["task_id"].astype(np.int64)
        init_state = branches["init_state_index"].astype(np.int64)
        planning_index = branches["planning_index"].astype(np.int64)

    derived_path = ROOT / result["artifacts"]["derived_sidecar"]
    assert sha256(derived_path) == result["artifacts"]["derived_sidecar_sha256"]
    with np.load(derived_path) as derived:
        assert {name: list(derived[name].shape) for name in derived.files} == result[
            "artifacts"
        ]["derived_sidecar_arrays"]
        assert np.array_equal(derived["task_id"], task_id)
        assert np.array_equal(derived["init_state_index"], init_state)
        assert np.array_equal(derived["planning_index"], planning_index)
        assert np.allclose(
            derived["validation_absolute_error"],
            validation_absolute,
            atol=2e-7,
            rtol=0,
        )
        assert np.allclose(
            derived["branch_absolute_error"], branch_absolute, atol=2e-7, rtol=0
        )
        assert np.allclose(derived["real_delta"], real_delta, atol=2e-7, rtol=0)
        assert np.allclose(
            derived["predicted_delta"], predicted_delta, atol=2e-7, rtol=0
        )
        assert all(
            np.isfinite(derived[name]).all()
            for name in derived.files
            if derived[name].dtype.kind in "fiu"
        )

    absolute = result["analysis"]["absolute_proprio_error"]
    assert math.isclose(
        float(np.sqrt(np.mean(validation_absolute**2))),
        absolute["validation"]["overall"]["rmse"],
        abs_tol=2e-10,
    )
    assert math.isclose(
        float(np.mean(validation_absolute)),
        absolute["validation"]["overall"]["mae"],
        abs_tol=2e-10,
    )
    assert math.isclose(
        float(np.sqrt(np.mean(branch_absolute**2))),
        absolute["branch_pooled"]["overall"]["rmse"],
        abs_tol=2e-10,
    )

    real_norm = np.linalg.norm(real_delta, axis=-1)
    predicted_norm = np.linalg.norm(predicted_delta, axis=-1)
    eligible = real_norm >= protocol["metric_contract"][
        "minimum_real_norm_for_direction_and_ratio"
    ]
    numerator = np.sum(predicted_delta * real_delta, axis=-1)
    direction = numerator / np.maximum(predicted_norm * real_norm, 1e-12)
    differential_rmse = float(
        np.sqrt(np.mean((predicted_delta[eligible] - real_delta[eligible]) ** 2))
    )
    zero_rmse = float(np.sqrt(np.mean(real_delta[eligible] ** 2)))
    improvement = (zero_rmse - differential_rmse) / zero_rmse
    overall = result["analysis"]["counterfactual_differential"]["overall"]
    assert math.isclose(
        float(direction[eligible].mean()),
        overall["direction_cosine"]["mean"],
        abs_tol=2e-10,
    )
    assert math.isclose(
        float(predicted_norm[eligible].mean() / real_norm[eligible].mean()),
        overall["magnitude_ratio"]["ratio_of_means"],
        abs_tol=2e-10,
    )
    assert math.isclose(differential_rmse, overall["differential_rmse"], abs_tol=2e-10)
    assert math.isclose(zero_rmse, overall["zero_difference_rmse"], abs_tol=2e-10)
    assert math.isclose(
        improvement, overall["relative_improvement_over_zero"], abs_tol=2e-10
    )

    within = []
    ordering = []
    tolerance = protocol["metric_contract"]["ranking_tie_tolerance"]
    comparisons = ((0, 1), (0, 2), (1, 2))
    for branch in range(40):
        for horizon in range(8):
            pred_row = predicted_norm[branch, :, horizon]
            real_row = real_norm[branch, :, horizon]
            pred_rank = rankdata(pred_row, method="average")
            real_rank = rankdata(real_row, method="average")
            within.append(float(np.corrcoef(pred_rank, real_rank)[0, 1]))
            for left, right in comparisons:
                pred_difference = pred_row[left] - pred_row[right]
                real_difference = real_row[left] - real_row[right]
                if (
                    abs(pred_difference) <= tolerance
                    or abs(real_difference) <= tolerance
                ):
                    ordering.append(0.5)
                else:
                    ordering.append(
                        float(np.sign(pred_difference) == np.sign(real_difference))
                    )
    ranking = result["analysis"]["separation_ranking"]["overall"]
    assert math.isclose(
        float(np.mean(within)),
        ranking["within_context_spearman"]["mean"],
        abs_tol=2e-10,
    )
    assert math.isclose(
        float(np.mean(ordering)),
        ranking["pairwise_ordering_accuracy"]["mean"],
        abs_tol=2e-10,
    )


def test_preregistered_gate_and_report_conclusion() -> None:
    result = json.loads(RESULT_PATH.read_text())
    decision = result["analysis"]["decision"]
    assert decision["gate_pass"] == {
        "differential_improvement": False,
        "direction_alignment": True,
        "ranking_relation": True,
        "no_candidate_specific_collapse": False,
    }
    assert decision["candidate_pair_collapse_checks"] == {
        "A0-A1": {
            "relative_improvement_positive": False,
            "direction_positive": True,
            "magnitude_ratio_at_least_0_05": True,
        },
        "A0-A2": {
            "relative_improvement_positive": True,
            "direction_positive": True,
            "magnitude_ratio_at_least_0_05": True,
        },
        "A1-A2": {
            "relative_improvement_positive": False,
            "direction_positive": True,
            "magnitude_ratio_at_least_0_05": True,
        },
    }
    assert result["verdict"] == decision["authorization_verdict"] == (
        "BLOCK_WM_V2C_PHYSICAL_MEDIATOR"
    )
    assert result["mechanism_labels"] == [
        "PHYSICAL_MEDIATOR_COUNTERFACTUAL_FAIL"
    ]
    assert "PHYSICAL_MEDIATOR_RELATIVE_SIGNAL_VALID" not in result[
        "mechanism_labels"
    ]

    report = REPORT_PATH.read_text()
    normalized_report = " ".join(report.lower().split())
    assert "PENDING" not in report
    assert "Authorization verdict: `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`" in report
    assert "direction and ranking signal is real but insufficient" in normalized_report
    assert "must not be interpreted as mediator accuracy" in normalized_report
    assert "No simulator, inference, training, controller" in report
