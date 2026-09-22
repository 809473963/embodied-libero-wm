#!/usr/bin/env python3
"""D17 deterministic stratified-tail selection test."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_wm_sample_review import pearson, select_stratified_tails


def main() -> int:
    rows = []
    for task_id in range(10):
        for demo_id in range(45, 50):
            for timestep in range(3):
                error = task_id + (demo_id - 45) * 0.1 + timestep * 0.01
                rows.append(
                    {
                        "task_id": task_id,
                        "demo_id": demo_id,
                        "start_timestep": timestep,
                        "learned_cosine_error": [error] * 8,
                    }
                )
    correct, incorrect = select_stratified_tails(rows, per_task=2)
    assert len(correct) == len(incorrect) == 20
    for selected in (correct, incorrect):
        for task_id in range(10):
            task_rows = [row for row in selected if row["task_id"] == task_id]
            assert len(task_rows) == 2
            assert len({row["demo_id"] for row in task_rows}) == 2
    assert all(row["demo_id"] in {45, 46} for row in correct)
    assert all(row["demo_id"] in {48, 49} for row in incorrect)
    assert pearson([1.0, 2.0], [3.0, 3.0]) is None
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) > 0.999
    print({"correct": len(correct), "incorrect": len(incorrect), "task_coverage": 10})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
