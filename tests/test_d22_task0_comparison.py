#!/usr/bin/env python3
"""D22 same-protocol and high-frequency-replanning assertions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_d22_task0 import build_comparison


def main() -> int:
    result = build_comparison(
        PROJECT_ROOT / "results/c0/seed11/summary.json",
        PROJECT_ROOT / "results/c1/seed11/task0_d19.json",
        PROJECT_ROOT / "results/c2/seed11/task0_d22.json",
    )
    assert result["protocol"]["episode_identity_aligned"]
    assert all(result["protocol"]["action_clip_source_checks"].values())
    assert len(set(result["protocol"]["checkpoint_hashes"].values())) == 1
    assert len(result["paired_episode_results"]) == 20
    assert all(item["episodes"] == 20 for item in result["methods"].values())
    assert all(item["success_rate"] == 0.0 for item in result["methods"].values())
    assert result["c2_behavior"]["zero_action_plans"] == 0
    assert result["c2_behavior"]["all_environment_actions_accounted_for"]
    assert result["c2_behavior"]["early_replan_fraction_of_plans"] > 0.9
    assert result["status"] == "BLOCK_D23_HIGH_FREQUENCY_REPLANNING"
    assert not result["decision"]["d23_authorized"]
    print(
        {
            "status": result["status"],
            "success_rates": {
                key: value["success_rate"] for key, value in result["methods"].items()
            },
            "early_replan_fraction": result["c2_behavior"][
                "early_replan_fraction_of_plans"
            ],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
