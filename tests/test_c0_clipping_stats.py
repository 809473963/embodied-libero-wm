#!/usr/bin/env python3
"""Verify evaluator clipping accounting without changing the clip rule itself."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_c0_all import add_clipping_stats, aggregate_clipping_stats, empty_clipping_stats, finalized_clipping_stats
from src.policies.c0_actor_policy import C0ActorPolicy


def main() -> int:
    raw = np.array(
        [
            [[-1.2, -1.0, -0.5, 0.0, 0.5, 1.0, 1.1]],
            [[-2.0, -0.1, 0.1, 0.2, 0.3, 1.5, 2.0]],
        ],
        dtype=np.float32,
    )
    per_chunk = C0ActorPolicy._clipping_statistics(raw)
    assert [item["elements_outside_minus_one_to_one"] for item in per_chunk] == [2, 3]
    total = empty_clipping_stats()
    for item in per_chunk:
        add_clipping_stats(total, item)
    episode = finalized_clipping_stats(total)
    assert episode["planned_action_elements"] == 14
    assert episode["elements_outside_minus_one_to_one"] == 5
    assert episode["clipped_element_ratio"] == 5 / 14
    run = aggregate_clipping_stats([{"action_clipping": episode}])
    assert run["episodes_with_clipping_stats"] == 1
    assert run["elements_outside_minus_one_to_one"] == 5
    print({"episode": episode, "run": run})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
