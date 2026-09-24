#!/usr/bin/env python3
"""Synthetic mechanical-decision checks for WM-v2C-EJ."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_world_model_v2c_ej import make_decision


def inputs() -> tuple[dict, dict, dict, dict, dict]:
    protocol = json.loads((ROOT / "configs/wm_v2c_ej_protocol.json").read_text())
    ordinary = {
        "relative_change_vs_d14": 0.01,
        "overall": 0.009,
        "copy_current": 0.015,
    }
    comparison = {
        "relative_delta": 0.2,
        "delta_bootstrap": {"ci95_lower": 0.001},
        "true_action_win_accuracy": 0.7,
        "accuracy_bootstrap": {"ci95_lower": 0.6},
    }
    global_action = {
        "within_task_shuffle": comparison,
        "temporal_permutation": comparison,
        "magnitude_distance_spearman": 0.95,
    }
    direction = {
        "overall": {"mean": 0.2},
        "per_pair": {"A0-A1": {"mean": 0.2}, "A0-A2": {"mean": 0.15}},
        "bootstrap": {
            "overall_ci95": [0.1, 0.3],
            "per_pair_ci95": {"A0-A1": [0.1, 0.3], "A0-A2": [0.05, 0.25]},
        },
    }
    matched = {
        "full_ej": {
            "counterfactual_transfer_ratio": 0.05,
            "multiple_vs_wm_v2b": 3.5,
            "candidate_specific_error": {"relative_change_vs_wm_v2b": 0.01},
            "visual_delta_direction": direction,
        }
    }
    contribution = {
        "demo_validation_correct_action": {
            "residual_to_base_l2": {"mean": 0.2, "p95": 0.6},
            "final_latent_max_unit_norm_error": 1e-6,
        },
        "production_candidate_sensitivity": {
            "adapter_pair_l2_to_base_pair_l2": 0.1
        },
    }
    return protocol, ordinary, global_action, matched, contribution


def test_pass_authorizes_only_reranker_rediagnosis() -> None:
    decision = make_decision(*inputs())
    assert decision["verdict"] == "WM_V2C_EJ_PASS"
    assert decision["authorization"] == "GO_RERANKER_REDIAGNOSIS"
    assert decision["controller_authorized"] is False
    assert decision["legacy_full_12d_mediator_verdict"] == (
        "BLOCK_WM_V2C_PHYSICAL_MEDIATOR"
    )
    assert decision["legacy_reranker_verdict"] == "RERANKER_SCORE_BLOCK"


def test_failure_precedence_is_mechanical() -> None:
    protocol, ordinary, global_action, matched, contribution = inputs()
    ordinary["relative_change_vs_d14"] = 0.06
    assert make_decision(
        protocol, ordinary, global_action, matched, contribution
    )["verdict"] == "PREDICTION_QUALITY_REGRESSION"

    protocol, ordinary, global_action, matched, contribution = inputs()
    contribution["production_candidate_sensitivity"][
        "adapter_pair_l2_to_base_pair_l2"
    ] = 0.01
    assert make_decision(
        protocol, ordinary, global_action, matched, contribution
    )["verdict"] == "PHYSICAL_VISUAL_ADAPTER_NOT_USING_LOCAL_SIGNAL"

    protocol, ordinary, global_action, matched, contribution = inputs()
    matched["full_ej"]["candidate_specific_error"][
        "relative_change_vs_wm_v2b"
    ] = 0.06
    assert make_decision(
        protocol, ordinary, global_action, matched, contribution
    )["verdict"] == "LOCAL_VISUAL_SEPARATION_ARTIFACT"

    protocol, ordinary, global_action, matched, contribution = inputs()
    matched["full_ej"]["counterfactual_transfer_ratio"] = 0.02
    matched["full_ej"]["multiple_vs_wm_v2b"] = 1.4
    assert make_decision(
        protocol, ordinary, global_action, matched, contribution
    )["verdict"] == "PHYSICAL_TO_VISUAL_TRANSFER_STILL_INSUFFICIENT"

