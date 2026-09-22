#!/usr/bin/env python3
"""D16 paired episode-bootstrap and formal-gate tests."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_world_model_transitions import decide_gate, paired_episode_bootstrap


def main() -> int:
    positive = {task_id: [0.01 + task_id * 1e-4] * 5 for task_id in range(10)}
    first = paired_episode_bootstrap(positive, replicates=1000, seed=1601)
    second = paired_episode_bootstrap(positive, replicates=1000, seed=1601)
    assert first == second
    assert first["ci95_lower"] > 0
    passing = decide_gate(0.01, [0.001 * (index + 1) for index in range(8)], first["ci95_lower"])
    assert passing["pass"] and all(passing["checks"].values())
    failing = decide_gate(0.01, [0.001, -0.001, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008], first["ci95_lower"])
    assert not failing["pass"] and not failing["checks"]["every_horizon_improves"]
    print({"bootstrap": first, "passing_gate": passing, "failing_gate": failing})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
