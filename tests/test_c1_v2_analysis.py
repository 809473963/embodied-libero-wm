#!/usr/bin/env python3
"""Synthetic checks for C1-v2 distribution statistics and frozen gate logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_c1_v2_rebase import (
    build_distribution_table,
    gate_decision,
    statistics,
)


def main() -> None:
    values = np.arange(1.0, 101.0)
    stats = statistics(values)
    assert stats["count"] == 100
    assert stats["median"] == 50.5
    distributions = {
        name: {
            horizon: {
                "mismatch": values / 1000.0,
                "uncertainty": values / 10000.0,
            }
            for horizon in (2, 4, 6, 8)
        }
        for name in ("expert", "c0_v1", "c0_v2", "c1_v2")
    }
    rows, nested = build_distribution_table(distributions, 0.029, 0.004)
    assert len(rows) == 32
    assert nested["expert"]["2"]["mismatch"]["d20_v1_exceed_count"] == 71

    protocol = json.loads((ROOT / "configs/c1_v2_rebase_protocol.json").read_text())
    c0 = {
        "successes": 45,
        "nonzero_tasks": 7,
        "planned_action_clipping_ratio": 0.04,
    }
    c1 = {
        "successes": 40,
        "nonzero_tasks": 6,
        "planned_action_clipping_ratio": 0.05,
    }
    reranking = {
        "replacement_count": 101,
        "selected_score_ge_default_fraction": 1.0,
        "positive_score_gain_fraction": 1.0,
        "selected_observed_mismatch_nonworse_horizon_count": 2,
    }
    checks, verdict = gate_decision(protocol, c0, c1, reranking, True, True)
    assert all(checks.values())
    assert verdict == "GO_D20_V2"
    c1["successes"] = 35
    checks, verdict = gate_decision(protocol, c0, c1, reranking, True, True)
    assert not checks["c1_v2_total_successes_min"]
    assert verdict == "BLOCK_C1_V2_RERANKER"
    _, verdict = gate_decision(protocol, c0, c1, reranking, False, True)
    assert verdict == "BLOCK_WM_INTERFACE"
    print({"status": "PASS", "distribution_rows": len(rows), "gate_paths": 3})


if __name__ == "__main__":
    main()
