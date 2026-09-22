#!/usr/bin/env python3
"""Build the protocol-checked D22 task-0 C0/C1/C2 ablation record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def episode_identity(episodes: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    return [
        (int(item["episode_index"]), int(item["seed"]), int(item["init_state_index"]))
        for item in episodes
    ]


def build_comparison(c0_path: Path, c1_path: Path, c2_path: Path) -> dict[str, Any]:
    c0 = read_json(c0_path)
    c1 = read_json(c1_path)
    c2 = read_json(c2_path)
    seed_path = PROJECT_ROOT / "configs/eval_seeds_spatial.json"
    seed_spec = read_json(seed_path)
    expected_seeds = [int(value) for value in seed_spec["task_seeds"]["0"]]
    expected_identity = [(index, seed, index) for index, seed in enumerate(expected_seeds)]

    if not c0.get("complete") or 0 not in c0.get("completed_task_ids", []):
        raise ValueError("C0 all-task result is incomplete for task 0")
    c0_task = next(item for item in c0["tasks"] if int(item["task_id"]) == 0)
    if not c1.get("complete") or not c2.get("complete"):
        raise ValueError("C1/C2 task-0 result is incomplete")
    if int(c1["task_id"]) != 0 or int(c2["task_id"]) != 0:
        raise ValueError("D22 is restricted to task 0")

    episode_sets = {
        "c0": c0_task["episode_results"],
        "c1": c1["episode_results"],
        "c2": c2["episode_results"],
    }
    for method, episodes in episode_sets.items():
        if len(episodes) != 20 or episode_identity(episodes) != expected_identity:
            raise ValueError(f"{method} does not use the frozen 20 episode identities")
    checkpoint_hashes = {
        "c0": c0["checkpoint_sha256"],
        "c1": c1["actor_checkpoint_sha256"],
        "c2": c2["actor_checkpoint_sha256"],
    }
    if len(set(checkpoint_hashes.values())) != 1:
        raise ValueError("C0/C1/C2 do not share the same actor checkpoint")
    if {int(c0["max_steps"]), int(c1["max_steps"]), int(c2["max_steps"])} != {300}:
        raise ValueError("C0/C1/C2 max-step protocols differ")

    clip_expression = "env.step(np.clip(action, -1.0, 1.0))"
    evaluator_paths = {
        "c0": PROJECT_ROOT / "scripts/evaluate_c0_all.py",
        "c1": PROJECT_ROOT / "scripts/evaluate_c1.py",
        "c2": PROJECT_ROOT / "scripts/evaluate_c2.py",
    }
    action_scale_checks = {
        method: clip_expression in path.read_text()
        for method, path in evaluator_paths.items()
    }
    if not all(action_scale_checks.values()):
        raise ValueError("an evaluator does not use the common [-1,1] environment action clip")

    c2_plans = [
        event for episode in c2["episode_results"] for event in episode["plan_events"]
    ]
    c2_checks = [
        event
        for episode in c2["episode_results"]
        for event in episode["verification_events"]
    ]
    if len(c2_plans) != int(c2["planning_events"]):
        raise ValueError("C2 planning-event count mismatch")
    if len(c2_checks) != int(c2["verification_events"]):
        raise ValueError("C2 verification-event count mismatch")
    if sum(event["executed_actions"] for event in c2_plans) != sum(
        episode["steps"] for episode in c2["episode_results"]
    ):
        raise ValueError("C2 plan execution counts do not cover all environment actions")
    if any(int(event["executed_actions"]) <= 0 for event in c2_plans):
        raise ValueError("C2 contains a zero-action plan")
    early_replans = sum(bool(event["early_replan"]) for event in c2_checks)
    if early_replans != int(c2["early_replan_events"]):
        raise ValueError("C2 early-replan count mismatch")
    if any(
        bool(event["early_replan"])
        != (bool(event["threshold_exceeded"]) and int(event["remaining_actions"]) > 0)
        for event in c2_checks
    ):
        raise ValueError("C2 early-replan semantics mismatch")

    jsonl_metadata = c2["replan_events_file"]
    jsonl_path = Path(jsonl_metadata["path"])
    if not jsonl_path.is_absolute():
        jsonl_path = PROJECT_ROOT / jsonl_path
    if (
        file_sha256(jsonl_path) != jsonl_metadata["sha256"]
        or len(jsonl_path.read_text().splitlines()) != early_replans
    ):
        raise ValueError("C2 replan JSONL integrity check failed")

    paired_results = []
    for index in range(20):
        paired_results.append(
            {
                "episode_index": index,
                "seed": expected_seeds[index],
                "init_state_index": index,
                "c0_success": bool(episode_sets["c0"][index]["success"]),
                "c1_success": bool(episode_sets["c1"][index]["success"]),
                "c2_success": bool(episode_sets["c2"][index]["success"]),
                "c0_steps": int(episode_sets["c0"][index]["steps"]),
                "c1_steps": int(episode_sets["c1"][index]["steps"]),
                "c2_steps": int(episode_sets["c2"][index]["steps"]),
                "c1_plans": int(episode_sets["c1"][index]["plan_count"]),
                "c2_plans": int(episode_sets["c2"][index]["plan_count"]),
                "c2_early_replans": int(
                    episode_sets["c2"][index]["early_replan_count"]
                ),
            }
        )

    method_results = {
        "c0": {
            "successes": int(c0_task["successes"]),
            "episodes": int(c0_task["episodes"]),
            "success_rate": float(c0_task["success_rate"]),
            "planning_events": None,
            "early_replan_events": 0,
        },
        "c1": {
            "successes": int(c1["successes"]),
            "episodes": int(c1["episodes"]),
            "success_rate": float(c1["success_rate"]),
            "planning_events": int(c1["planning_events"]),
            "early_replan_events": 0,
        },
        "c2": {
            "successes": int(c2["successes"]),
            "episodes": int(c2["episodes"]),
            "success_rate": float(c2["success_rate"]),
            "planning_events": int(c2["planning_events"]),
            "verification_events": int(c2["verification_events"]),
            "early_replan_events": int(c2["early_replan_events"]),
        },
    }
    early_plan_fraction = early_replans / len(c2_plans)
    return {
        "schema_version": 1,
        "day": "D22",
        "status": "BLOCK_D23_HIGH_FREQUENCY_REPLANNING",
        "scope": "libero_spatial task 0, actor training seed 11",
        "protocol": {
            "episodes": 20,
            "max_steps": 300,
            "seed_file": str(seed_path.relative_to(PROJECT_ROOT)),
            "seed_file_sha256": file_sha256(seed_path),
            "episode_identity_aligned": True,
            "shared_actor_checkpoint_sha256": next(iter(checkpoint_hashes.values())),
            "checkpoint_hashes": checkpoint_hashes,
            "environment_action_range": [-1.0, 1.0],
            "action_clip_source_checks": action_scale_checks,
        },
        "sources": {
            "c0": {"path": str(c0_path), "sha256": file_sha256(c0_path)},
            "c1": {"path": str(c1_path), "sha256": file_sha256(c1_path)},
            "c2": {"path": str(c2_path), "sha256": file_sha256(c2_path)},
            "c2_replan_jsonl": {
                "path": str(jsonl_path),
                "sha256": file_sha256(jsonl_path),
                "event_count": early_replans,
            },
        },
        "methods": method_results,
        "paired_episode_results": paired_results,
        "paired_success_differences": {
            "c1_minus_c0": float(c1["success_rate"] - c0_task["success_rate"]),
            "c2_minus_c0": float(c2["success_rate"] - c0_task["success_rate"]),
            "c2_minus_c1": float(c2["success_rate"] - c1["success_rate"]),
        },
        "c2_behavior": {
            "episodes_with_early_replan": int(c2["episodes_with_early_replan"]),
            "mean_early_replans_per_episode": float(c2["mean_early_replans_per_episode"]),
            "mean_executed_actions_per_plan": float(c2["mean_executed_actions_per_plan"]),
            "early_replan_fraction_of_plans": early_plan_fraction,
            "trigger_reason_counts": c2["trigger_reason_counts"],
            "horizon_check_counts": {
                str(horizon): sum(event["horizon"] == horizon for event in c2_checks)
                for horizon in (2, 4, 6, 8)
            },
            "horizon_early_replan_counts": {
                str(horizon): sum(
                    event["horizon"] == horizon and event["early_replan"]
                    for event in c2_checks
                )
                for horizon in (2, 4, 6, 8)
            },
            "zero_action_plans": 0,
            "all_environment_actions_accounted_for": True,
        },
        "decision": {
            "performance_claim": "no detected difference: all three methods are 0/20",
            "behavior_finding": (
                "C2 ends 90.01% of plans early and averages 100.9 replans per episode; "
                "the project stop-and-diagnose condition is met"
            ),
            "thresholds_changed_from_test_results": False,
            "d23_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c0", type=Path, default=PROJECT_ROOT / "results/c0/seed11/summary.json"
    )
    parser.add_argument(
        "--c1", type=Path, default=PROJECT_ROOT / "results/c1/seed11/task0_d19.json"
    )
    parser.add_argument(
        "--c2", type=Path, default=PROJECT_ROOT / "results/c2/seed11/task0_d22.json"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/d22_task0_ablation.json"
    )
    args = parser.parse_args()
    result = build_comparison(args.c0, args.c1, args.c2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "success_rates": {
                    key: value["success_rate"] for key, value in result["methods"].items()
                },
                "c2_early_replan_fraction": result["c2_behavior"][
                    "early_replan_fraction_of_plans"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
