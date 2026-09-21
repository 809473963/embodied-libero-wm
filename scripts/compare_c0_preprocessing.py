#!/usr/bin/env python3
"""Compare two same-seed C0 episode records without hiding regressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result["episodes"] != 20 or len(result["episode_results"]) != 20:
        raise ValueError(f"{path}: expected 20 episode records")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    for field in ("task_suite", "task_id", "episodes", "max_steps"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"protocol mismatch for {field}")
    before = {row["episode_index"]: row for row in baseline["episode_results"]}
    after = {row["episode_index"]: row for row in candidate["episode_results"]}
    if set(before) != set(after):
        raise ValueError("episode indices differ")
    improved = []
    regressed = []
    unchanged = []
    for index in sorted(before):
        old, new = before[index], after[index]
        if old["success"] == new["success"]:
            unchanged.append(index)
        elif new["success"]:
            improved.append(index)
        else:
            regressed.append(index)
    result = {
        "schema_version": 1,
        "baseline": {"file": str(args.baseline), "successes": baseline["successes"], "success_rate": baseline["success_rate"]},
        "candidate": {"file": str(args.candidate), "successes": candidate["successes"], "success_rate": candidate["success_rate"]},
        "delta_successes": candidate["successes"] - baseline["successes"],
        "improved_episode_indices": improved,
        "regressed_episode_indices": regressed,
        "unchanged_episode_indices": unchanged,
        "interpretation": "Same task, 20 fixed seeds, fixed init-state indices, max steps, and official success API. This is a preprocessing ablation only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
