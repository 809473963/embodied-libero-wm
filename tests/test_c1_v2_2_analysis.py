#!/usr/bin/env python3
"""Synthetic construction, bootstrap, and decision checks for C1-v2.2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_wm_action_conditioning import (
    BootstrapPlan,
    comparison_summary,
    construct_conditions,
    decide,
    shuffled_sources,
)


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/c1_v2_2_wm_action_conditioning.json").read_text()
    )
    task = np.repeat(np.arange(10, dtype=np.int64), 10)
    demo = np.tile(np.repeat(np.arange(45, 50, dtype=np.int64), 2), 10)
    timestep = np.tile(np.arange(10, dtype=np.int64), 10)
    raw = np.arange(100 * 8 * 7, dtype=np.float32).reshape(100, 8, 7) / 1000.0
    data = {
        "task_id": task,
        "demo_id": demo,
        "start_timestep": timestep,
        "actions": raw,
    }
    within, cross = shuffled_sources(task, 220201, 220202)
    assert np.all(within != np.arange(100))
    assert np.all(task[within] == task)
    assert np.all(task[cross] != task)

    conditions, within_again, cross_again = construct_conditions(data, protocol)
    assert np.array_equal(within, within_again)
    assert np.array_equal(cross, cross_again)
    assert np.array_equal(conditions["zero"], np.zeros_like(raw))
    assert np.array_equal(conditions["reverse"], raw[:, ::-1])
    assert np.array_equal(
        conditions["temporal_shuffle"], raw[:, [2, 5, 0, 7, 3, 1, 6, 4]]
    )
    assert np.array_equal(conditions["repeat_first"], np.repeat(raw[:, :1], 8, 1))
    assert np.allclose(conditions["scale_1p5"], raw * 1.5)

    true_error = np.full((100, 8), 0.01, dtype=np.float64)
    wrong_error = true_error + 0.001
    bootstrap = BootstrapPlan(task, demo, replicates=1000, seed=7)
    comparison = comparison_summary(true_error, wrong_error, task, bootstrap)
    assert comparison["delta"] > 0.0
    assert comparison["true_action_win_accuracy"] == 1.0
    assert comparison["delta_bootstrap"]["ci95_lower"] > 0.0
    assert comparison["accuracy_bootstrap"]["ci95_lower"] == 1.0

    comparisons = {
        "shuffle_task": comparison,
        "temporal_shuffle": comparison,
    }
    magnitude = {
        "scales": [
            {
                "scale": 0.0,
                "separation_from_1x": {"to_ordinary_wm_error_overall": 0.2},
            }
        ],
        "distance_from_1x_monotonicity": {"spearman": 1.0},
    }
    representation = {
        "real_displacement_to_ordinary_error_ratio": 2.0,
        "per_horizon": [{"median": 0.02} for _ in range(8)],
    }
    passed = decide(protocol, comparisons, magnitude, representation)
    assert passed["action_signal_pass"] is True
    assert passed["verdicts"] == [
        "WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY"
    ]

    weak = json.loads(json.dumps(comparison))
    weak["relative_delta"] = 0.0
    weak["true_action_win_accuracy"] = 0.5
    weak["delta_bootstrap"]["ci95_lower"] = -0.001
    weak["accuracy_bootstrap"]["ci95_lower"] = 0.48
    for record in weak["per_task"].values():
        record["relative_delta"] = 0.0
        record["true_action_win_accuracy"] = 0.5
    weak_magnitude = {
        "scales": [
            {
                "scale": 0.0,
                "separation_from_1x": {"to_ordinary_wm_error_overall": 0.001},
            }
        ],
        "distance_from_1x_monotonicity": {"spearman": 0.5},
    }
    failed = decide(
        protocol,
        {"shuffle_task": weak, "temporal_shuffle": weak},
        weak_magnitude,
        representation,
    )
    assert failed["verdicts"] == ["WM_ACTION_IGNORING"]
    print(
        {
            "status": "PASS",
            "whole_chunk_shuffle": True,
            "bootstrap_ci": [
                comparison["delta_bootstrap"]["ci95_lower"],
                comparison["delta_bootstrap"]["ci95_upper"],
            ],
            "decision_paths": 2,
        }
    )


if __name__ == "__main__":
    main()
