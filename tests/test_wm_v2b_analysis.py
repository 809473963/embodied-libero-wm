#!/usr/bin/env python3
"""Synthetic WM-v2B preregistered verdict-path checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_world_model_v2b import decide


def comparison(relative: float = 0.2, accuracy: float = 0.7) -> dict:
    return {
        "relative_delta": relative,
        "delta_bootstrap": {"ci95_lower": 0.001},
        "true_action_win_accuracy": accuracy,
        "accuracy_bootstrap": {"ci95_lower": 0.6},
    }


def direction() -> dict:
    return {
        "pairwise_ranking_accuracy": 0.56,
        "pairwise_accuracy_ci95": [0.53, 0.59],
        "mean_within_sample_spearman": 0.1,
        "spearman_ci95": [0.05, 0.15],
    }


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/wm_v2b_future_proprio_protocol.json").read_text()
    )
    prediction = {
        "relative_error_change_vs_d14_seed11": 0.01,
        "correct_action_error": 0.009,
        "copy_current_error": 0.015,
    }
    visual_local = {
        "separation_to_error_ratio": 0.012,
        "improvement_multiple_vs_wm_v1": 2.5,
    }
    proprio_prediction = {
        "correct_action_error": 0.02,
        "zero_delta_baseline_error": 0.04,
    }
    proprio_local = {"separation_to_correct_prediction_rmse_ratio": 0.08}
    args = (
        protocol,
        prediction,
        comparison(),
        visual_local,
        direction(),
        proprio_prediction,
        comparison(),
        proprio_local,
        direction(),
    )
    passed = decide(*args)
    assert passed["verdict"] == "WM_V2B_LOCAL_SENSITIVITY_PASS"
    assert passed["seed23_seed47_authorized"] is True

    bad_visual = dict(visual_local, separation_to_error_ratio=0.006)
    physical_only = decide(*args[:3], bad_visual, *args[4:])
    assert physical_only["verdict"] == (
        "PHYSICAL_SIGNAL_LEARNED_VISUAL_TRANSFER_INSUFFICIENT"
    )

    bad_proprio = dict(proprio_prediction, correct_action_error=0.05)
    no_aux = decide(*args[:5], bad_proprio, *args[6:])
    assert no_aux["verdict"] == "PROPRIO_AUX_ACTION_SIGNAL_FAIL"

    bad_prediction = dict(prediction, relative_error_change_vs_d14_seed11=0.06)
    regression = decide(protocol, bad_prediction, *args[2:])
    assert regression["verdict"] == "PREDICTION_QUALITY_REGRESSION"

    weak_physical = {"separation_to_correct_prediction_rmse_ratio": 0.01}
    insufficient = decide(
        *args[:3], bad_visual, *args[4:7], weak_physical, args[8]
    )
    assert insufficient["verdict"] == "WM_V2B_AUXILIARY_INSUFFICIENT"
    print({"status": "PASS", "decision_paths": 5})


if __name__ == "__main__":
    main()
