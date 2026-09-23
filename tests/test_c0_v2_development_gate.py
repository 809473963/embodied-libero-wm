#!/usr/bin/env python3
"""Synthetic boundary tests for the preregistered C0-v2 GO gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_c0_v2_development import gate_evaluation


def summary(successes: int, nonzero_tasks: int, clipping: float = 0.0) -> dict[str, float | int]:
    return {
        "successes": successes,
        "nonzero_tasks": nonzero_tasks,
        "clipped_element_ratio": clipping,
    }


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/c0_v2_causal_alignment_protocol.json").read_text()
    )
    criteria = protocol["development_gate"]["go_requires_all"]
    checks, measured, go = gate_evaluation(summary(10, 3), summary(20, 5, 0.1), criteria)
    assert go and all(checks.values())
    assert measured["success_improvement"] == 10
    assert measured["nonzero_task_improvement"] == 2

    checks, _, go = gate_evaluation(summary(10, 3), summary(19, 5), criteria)
    assert not go and not checks["c0_v2_total_successes_min"]
    checks, _, go = gate_evaluation(summary(5, 4), summary(20, 5), criteria)
    assert not go and not checks["nonzero_task_improvement_over_c0_v1_min"]
    checks, _, go = gate_evaluation(summary(10, 3), summary(20, 5, 0.100001), criteria)
    assert not go and not checks["planned_action_element_clipping_ratio_max"]
    print({"status": "PASS", "boundary_go": True, "failed_conditions_checked": 3})


if __name__ == "__main__":
    main()
