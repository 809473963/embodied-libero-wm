#!/usr/bin/env python3
"""Evaluate the precommitted C0-v2 confirmatory acceptance criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from scripts.analyze_c0_v2_development import finite_tree, gate_evaluation, policy_summary


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_holdout_result(
    result: dict[str, Any],
    label: str,
    rollout: dict[str, Any],
    rollout_hash: str,
    expected_checkpoint_hash: str,
) -> None:
    expected = {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "policy_label": label,
        "partition": "confirmatory_holdout",
        "protocol_sha256": rollout_hash,
        "checkpoint_sha256": expected_checkpoint_hash,
        "suite": rollout["suite"],
        "episodes_per_task": 10,
        "complete": True,
        "episodes": 100,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        raise ValueError(f"{label} holdout result mismatch: {mismatches}")
    if result["completed_task_ids"] != rollout["task_ids"]:
        raise ValueError(f"{label} holdout task IDs differ")
    partition = rollout["partitions"]["confirmatory_holdout"]
    for task in result["tasks"]:
        task_id = int(task["task_id"])
        episodes = task["episode_results"]
        if [episode["init_state_index"] for episode in episodes] != partition["init_state_indices"]:
            raise ValueError(f"{label} task {task_id}: init-state sequence differs")
        if [episode["seed"] for episode in episodes] != partition["task_seeds"][str(task_id)]:
            raise ValueError(f"{label} task {task_id}: seed sequence differs")
    if not finite_tree(result):
        raise FloatingPointError(f"{label} holdout contains NaN/Inf")


def exact_two_sided_mcnemar(v1_only: int, v2_only: int) -> float:
    discordant = v1_only + v2_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(v1_only, v2_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_outcomes(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    counts = {"both_success": 0, "c0_v1_only": 0, "c0_v2_only": 0, "neither": 0}
    for v1_task, v2_task in zip(v1["tasks"], v2["tasks"]):
        if v1_task["task_id"] != v2_task["task_id"]:
            raise ValueError("paired task order differs")
        for left, right in zip(v1_task["episode_results"], v2_task["episode_results"]):
            if (left["seed"], left["init_state_index"]) != (
                right["seed"],
                right["init_state_index"],
            ):
                raise ValueError("paired episode identity differs")
            key = (
                "both_success"
                if left["success"] and right["success"]
                else "c0_v1_only"
                if left["success"]
                else "c0_v2_only"
                if right["success"]
                else "neither"
            )
            counts[key] += 1
    counts["exact_two_sided_mcnemar_p"] = exact_two_sided_mcnemar(
        counts["c0_v1_only"], counts["c0_v2_only"]
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-protocol",
        type=Path,
        default=ROOT / "configs/c0_v2_confirmatory_analysis_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/c0_v2/confirmatory_comparison.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    analysis = json.loads(args.analysis_protocol.read_text())
    rollout_path = ROOT / analysis["source_rollout_protocol"]
    rollout = json.loads(rollout_path.read_text())
    rollout_hash = sha256(rollout_path)
    if rollout_hash != analysis["source_rollout_protocol_sha256"]:
        raise ValueError("rollout protocol hash differs")
    v1_path = ROOT / analysis["c0_v1_result"]
    v2_path = ROOT / analysis["c0_v2_result"]
    v1_raw = json.loads(v1_path.read_text())
    v2_raw = json.loads(v2_path.read_text())
    validate_holdout_result(
        v1_raw,
        "c0_v1",
        rollout,
        rollout_hash,
        analysis["frozen_c0_v1_checkpoint_sha256"],
    )
    validate_holdout_result(
        v2_raw,
        "c0_v2",
        rollout,
        rollout_hash,
        analysis["frozen_c0_v2_checkpoint_sha256"],
    )
    v1 = policy_summary(v1_raw)
    v2 = policy_summary(v2_raw)
    criteria = analysis["acceptance_requires_all"]
    checks, measured, accepted = gate_evaluation(v1, v2, criteria)
    verdict_key = "all_acceptance_checks_pass" if accepted else "any_acceptance_check_fails"
    result = {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "partition": "confirmatory_holdout",
        "analysis_protocol": str(args.analysis_protocol.resolve()),
        "analysis_protocol_sha256": sha256(args.analysis_protocol),
        "rollout_protocol_sha256": rollout_hash,
        "development_gate_commit": analysis["development_gate_commit"],
        "development_gate_decision_sha256": analysis["development_gate_decision_sha256"],
        "c0_v1_result_sha256": sha256(v1_path),
        "c0_v2_result_sha256": sha256(v2_path),
        "c0_v1": v1,
        "c0_v2": v2,
        "delta": {
            "successes": v2["successes"] - v1["successes"],
            "macro_success_rate": v2["macro_success_rate"] - v1["macro_success_rate"],
            "nonzero_tasks": v2["nonzero_tasks"] - v1["nonzero_tasks"],
            "mean_steps": v2["mean_steps"] - v1["mean_steps"],
            "clipped_element_ratio": v2["clipped_element_ratio"]
            - v1["clipped_element_ratio"],
        },
        "paired_outcomes": paired_outcomes(v1_raw, v2_raw),
        "acceptance_criteria": criteria,
        "acceptance_checks": checks,
        "acceptance_measured": measured,
        "accepted": accepted,
        "verdict": analysis["verdict_mapping"][verdict_key],
        "holdout_may_not_be_used_for_tuning": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "c0_v1_successes": v1["successes"],
                "c0_v2_successes": v2["successes"],
                "c0_v1_nonzero_tasks": v1["nonzero_tasks"],
                "c0_v2_nonzero_tasks": v2["nonzero_tasks"],
                "paired_outcomes": result["paired_outcomes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
