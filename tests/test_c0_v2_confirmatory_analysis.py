#!/usr/bin/env python3
"""Synthetic checks for paired confirmatory outcome accounting."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_c0_v2_confirmatory import exact_two_sided_mcnemar, paired_outcomes


def result(left: list[bool], right: list[bool] | None = None) -> dict:
    values = left if right is None else right
    return {
        "tasks": [
            {
                "task_id": 0,
                "episode_results": [
                    {
                        "seed": 40000 + index,
                        "init_state_index": 40 + index,
                        "success": success,
                    }
                    for index, success in enumerate(values)
                ],
            }
        ]
    }


def main() -> None:
    v1 = result([True, True, False, False])
    v2 = result([], [True, False, True, False])
    paired = paired_outcomes(v1, v2)
    assert paired["both_success"] == 1
    assert paired["c0_v1_only"] == 1
    assert paired["c0_v2_only"] == 1
    assert paired["neither"] == 1
    assert paired["exact_two_sided_mcnemar_p"] == 1.0
    assert exact_two_sided_mcnemar(0, 10) == 2 / 1024
    print({"status": "PASS", "paired_accounting": paired})


if __name__ == "__main__":
    main()
