#!/usr/bin/env python3
"""Apply the preregistered C0-v2 development gate before any holdout rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen gate output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def validate_result(
    result: dict[str, Any],
    label: str,
    protocol: dict[str, Any],
    protocol_hash: str,
) -> None:
    expected = {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "policy_label": label,
        "partition": "controller_development",
        "protocol_sha256": protocol_hash,
        "suite": protocol["suite"],
        "episodes_per_task": 10,
        "complete": True,
        "episodes": 100,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        raise ValueError(f"{label} development result mismatch: {mismatches}")
    if result["completed_task_ids"] != protocol["task_ids"]:
        raise ValueError(f"{label} task IDs differ from protocol")
    partition = protocol["partitions"]["controller_development"]
    expected_indices = partition["init_state_indices"]
    for task in result["tasks"]:
        task_id = int(task["task_id"])
        episodes = task["episode_results"]
        if [episode["init_state_index"] for episode in episodes] != expected_indices:
            raise ValueError(f"{label} task {task_id} init states differ")
        if [episode["seed"] for episode in episodes] != partition["task_seeds"][str(task_id)]:
            raise ValueError(f"{label} task {task_id} seeds differ")
    if not finite_tree(result):
        raise FloatingPointError(f"{label} development result contains NaN/Inf")


def policy_summary(result: dict[str, Any]) -> dict[str, Any]:
    tasks = [
        {
            "task_id": int(task["task_id"]),
            "successes": int(task["successes"]),
            "episodes": int(task["episodes"]),
            "success_rate": float(task["success_rate"]),
            "mean_steps": float(task["mean_steps"]),
            "clipped_element_ratio": float(task["action_clipping"]["clipped_element_ratio"]),
        }
        for task in result["tasks"]
    ]
    return {
        "successes": int(result["successes"]),
        "episodes": int(result["episodes"]),
        "macro_success_rate": float(result["macro_success_rate"]),
        "nonzero_tasks": int(result["nonzero_tasks"]),
        "mean_steps": float(result["mean_steps"]),
        "clipped_element_ratio": float(result["action_clipping"]["clipped_element_ratio"]),
        "tasks": tasks,
    }


def gate_evaluation(
    c0_v1: dict[str, Any], c0_v2: dict[str, Any], criteria: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any], bool]:
    measured = {
        "c0_v1_successes": c0_v1["successes"],
        "c0_v2_successes": c0_v2["successes"],
        "success_improvement": c0_v2["successes"] - c0_v1["successes"],
        "c0_v1_nonzero_tasks": c0_v1["nonzero_tasks"],
        "c0_v2_nonzero_tasks": c0_v2["nonzero_tasks"],
        "nonzero_task_improvement": c0_v2["nonzero_tasks"] - c0_v1["nonzero_tasks"],
        "c0_v2_clipped_element_ratio": c0_v2["clipped_element_ratio"],
    }
    checks = {
        "c0_v2_total_successes_min": c0_v2["successes"]
        >= int(criteria["c0_v2_total_successes_min"]),
        "c0_v2_nonzero_tasks_min": c0_v2["nonzero_tasks"]
        >= int(criteria["c0_v2_nonzero_tasks_min"]),
        "success_improvement_over_c0_v1_min": measured["success_improvement"]
        >= int(criteria["success_improvement_over_c0_v1_min"]),
        "nonzero_task_improvement_over_c0_v1_min": measured["nonzero_task_improvement"]
        >= int(criteria["nonzero_task_improvement_over_c0_v1_min"]),
        "finite_actions_and_metrics": True,
        "planned_action_element_clipping_ratio_max": c0_v2["clipped_element_ratio"]
        <= float(criteria["planned_action_element_clipping_ratio_max"]),
    }
    return checks, measured, all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/c0_v2_causal_alignment_protocol.json",
    )
    parser.add_argument(
        "--c0-v1",
        type=Path,
        default=ROOT / "results/c0_v2/controller_development/c0_v1.json",
    )
    parser.add_argument(
        "--c0-v2",
        type=Path,
        default=ROOT / "results/c0_v2/controller_development/c0_v2.json",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=ROOT / "results/c0_v2/development_comparison.json",
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=ROOT / "configs/c0_v2_holdout_gate_decision.json",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    v1_raw = json.loads(args.c0_v1.read_text())
    v2_raw = json.loads(args.c0_v2.read_text())
    validate_result(v1_raw, "c0_v1", protocol, protocol_hash)
    validate_result(v2_raw, "c0_v2", protocol, protocol_hash)
    if v1_raw["checkpoint_sha256"] != protocol["frozen_inputs"]["c0_v1_checkpoint_sha256"]:
        raise ValueError("C0-v1 result did not use the frozen checkpoint")
    v1 = policy_summary(v1_raw)
    v2 = policy_summary(v2_raw)
    criteria = protocol["development_gate"]["go_requires_all"]
    checks, measured, go = gate_evaluation(v1, v2, criteria)
    comparison = {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "partition": "controller_development",
        "protocol_sha256": protocol_hash,
        "holdout_observed": False,
        "c0_v1_result": str(args.c0_v1.resolve()),
        "c0_v1_result_sha256": sha256(args.c0_v1),
        "c0_v2_result": str(args.c0_v2.resolve()),
        "c0_v2_result_sha256": sha256(args.c0_v2),
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
        "gate_criteria": criteria,
        "gate_checks": checks,
        "gate_measured": measured,
        "gate_pass": go,
    }
    decision = {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "protocol_sha256": protocol_hash,
        "decision": "GO_CONFIRMATORY_HOLDOUT" if go else "NO_GO_CONFIRMATORY_HOLDOUT",
        "holdout_observed_at_decision": False,
        "frozen_c0_v2_checkpoint": v2_raw["checkpoint"],
        "frozen_c0_v2_checkpoint_sha256": v2_raw["checkpoint_sha256"],
        "development_comparison_sha256": None,
        "gate_criteria": criteria,
        "gate_checks": checks,
        "gate_measured": measured,
    }
    atomic_json(args.comparison_output, comparison)
    decision["development_comparison_sha256"] = sha256(args.comparison_output)
    atomic_json(args.gate_output, decision)
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "c0_v1_successes": v1["successes"],
                "c0_v2_successes": v2["successes"],
                "c0_v1_nonzero_tasks": v1["nonzero_tasks"],
                "c0_v2_nonzero_tasks": v2["nonzero_tasks"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
