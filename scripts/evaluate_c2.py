#!/usr/bin/env python3
"""D21 C2 evaluator with h=2 verification and explicit replan JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scripts.evaluate_c0_all import (
    disable_camera_observable,
    make_env,
    policy_observation,
    refresh_policy_observation,
    trusted_init_states,
)
from src.policies.c2_verify_replan_policy import C2VerifyReplanPolicy
from src.world_models.normalization import file_sha256


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_replan_jsonl(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as destination:
        for episode in episodes:
            for event in episode["verification_events"]:
                if event["early_replan"]:
                    destination.write(
                        json.dumps(
                            {
                                "episode_index": episode["episode_index"],
                                "seed": episode["seed"],
                                **event,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    os.replace(temporary, path)


def attach_replan_artifact(
    result: dict[str, Any], path: Path, episodes: list[dict[str, Any]]
) -> None:
    expected_count = sum(
        event["early_replan"]
        for episode in episodes
        for event in episode["verification_events"]
    )
    result["replan_events_file"] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "event_count": expected_count,
    }


def summarize(
    checkpoint: Path,
    config: Path,
    seed_file: Path,
    task: Any,
    task_id: int,
    max_steps: int,
    episodes_target: int,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    plan_events = [event for episode in episodes for event in episode["plan_events"]]
    verification_events = [
        event for episode in episodes for event in episode["verification_events"]
    ]
    replan_events = [event for event in verification_events if event["early_replan"]]
    triggers = Counter(
        "both"
        if event["mismatch_exceeded"] and event["uncertainty_exceeded"]
        else "mismatch"
        if event["mismatch_exceeded"]
        else "uncertainty"
        for event in replan_events
    )
    return {
        "schema_version": 1,
        "method": "c2_prediction_verify_replan",
        "actor_checkpoint": str(checkpoint),
        "actor_checkpoint_sha256": file_sha256(checkpoint),
        "c2_config": str(config),
        "c2_config_sha256": file_sha256(config),
        "threshold_selection_sha256": file_sha256(
            PROJECT_ROOT / "results/threshold_selection.json"
        ),
        "seed_file": str(seed_file),
        "task_suite": "libero_spatial",
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "max_steps": max_steps,
        "episodes_target": episodes_target,
        "completed_episode_indices": [episode["episode_index"] for episode in episodes],
        "complete": len(episodes) == episodes_target,
        "episodes": len(episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "success_rate": (
            sum(episode["success"] for episode in episodes) / len(episodes) if episodes else 0.0
        ),
        "planning_events": len(plan_events),
        "verification_events": len(verification_events),
        "early_replan_events": len(replan_events),
        "episodes_with_early_replan": sum(
            any(event["early_replan"] for event in episode["verification_events"])
            for episode in episodes
        ),
        "mean_early_replans_per_episode": (
            len(replan_events) / len(episodes) if episodes else 0.0
        ),
        "trigger_reason_counts": {
            "mismatch_only": triggers["mismatch"],
            "uncertainty_only": triggers["uncertainty"],
            "both": triggers["both"],
        },
        "mean_executed_actions_per_plan": (
            float(np.mean([event["executed_actions"] for event in plan_events]))
            if plan_events
            else None
        ),
        "episode_results": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "results/c0/seed11/best.pt"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/c2_verify_replan.yaml"
    )
    parser.add_argument(
        "--seed-file", type=Path, default=PROJECT_ROOT / "configs/eval_seeds_spatial.json"
    )
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--episode-limit", type=int, choices=range(1, 21), default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replan-events", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_spec = json.loads(args.seed_file.read_text())
    seeds = [int(value) for value in seed_spec["task_seeds"][str(args.task_id)]][
        : args.episode_limit
    ]
    suite = benchmark.get_benchmark_dict()[seed_spec["suite"]]()
    task = suite.get_task(args.task_id)
    init_states = trusted_init_states(suite, args.task_id)
    episodes: list[dict[str, Any]] = []
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")
        existing = json.loads(args.output.read_text())
        expected = {
            "method": "c2_prediction_verify_replan",
            "actor_checkpoint_sha256": file_sha256(args.checkpoint),
            "c2_config_sha256": file_sha256(args.config),
            "threshold_selection_sha256": file_sha256(
                PROJECT_ROOT / "results/threshold_selection.json"
            ),
            "task_id": args.task_id,
            "max_steps": args.max_steps,
            "episodes_target": args.episode_limit,
        }
        mismatches = [key for key, value in expected.items() if existing.get(key) != value]
        if mismatches:
            raise ValueError(f"incompatible C2 resume fields: {mismatches}")
        episodes = existing["episode_results"]
        if existing["completed_episode_indices"] != list(range(len(episodes))):
            raise ValueError("C2 resume episodes are not a contiguous prefix")
    elif args.resume:
        raise FileNotFoundError(args.output)
    if len(episodes) == len(seeds):
        atomic_replan_jsonl(args.replan_events, episodes)
        result = summarize(
            args.checkpoint,
            args.config,
            args.seed_file,
            task,
            args.task_id,
            args.max_steps,
            len(seeds),
            episodes,
        )
        attach_replan_artifact(result, args.replan_events, episodes)
        atomic_json(args.output, result)
        print(
            json.dumps(
                {
                    "resume_complete": True,
                    "episodes": len(episodes),
                    "replan_events_sha256": result["replan_events_file"]["sha256"],
                }
            )
        )
        return 0

    policy = C2VerifyReplanPolicy(args.checkpoint, device=args.device, config=args.config)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    for episode_index in range(len(episodes), len(seeds)):
        env: OffScreenRenderEnv = make_env(bddl_file)
        plan_events: list[dict[str, Any]] = []
        verification_events: list[dict[str, Any]] = []
        try:
            env.seed(seeds[episode_index])
            env.reset()
            observation = env.set_init_state(init_states[episode_index])
            disable_camera_observable(env)
            policy.reset(seed=seeds[episode_index])
            chunk = np.empty((0, 7), dtype=np.float32)
            chunk_index = 8
            current_plan: dict[str, Any] | None = None
            success = False
            for step in range(1, args.max_steps + 1):
                if chunk_index >= 8:
                    observation = refresh_policy_observation(env)
                    chunk, current_plan = policy.plan_chunk(
                        policy_observation(observation), task.language
                    )
                    current_plan["episode_step_before_chunk"] = step - 1
                    current_plan["executed_full_chunk"] = None
                    current_plan["early_replan"] = None
                    current_plan["executed_actions"] = 0
                    plan_events.append(current_plan)
                    chunk_index = 0
                action = np.asarray(chunk[chunk_index], dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("C2 selected chunk contains an invalid action")
                observation, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
                chunk_index += 1
                assert current_plan is not None
                current_plan["executed_actions"] = chunk_index
                if env.check_success():
                    success = True
                    current_plan["executed_full_chunk"] = chunk_index == 8
                    current_plan["early_replan"] = False
                    current_plan["termination"] = "task_success"
                    break
                if chunk_index % 2 == 0:
                    observation = refresh_policy_observation(env)
                    verification = policy.verify(
                        policy_observation(observation),
                        horizon=chunk_index,
                        remaining_actions=8 - chunk_index,
                    )
                    verification.update(
                        {
                            "plan_index": current_plan["plan_index"],
                            "episode_step_after_check": step,
                        }
                    )
                    verification_events.append(verification)
                    if verification["early_replan"]:
                        current_plan["executed_full_chunk"] = False
                        current_plan["early_replan"] = True
                        current_plan["termination"] = "threshold_trigger"
                        chunk_index = 8
                    elif chunk_index == 8:
                        current_plan["executed_full_chunk"] = True
                        current_plan["early_replan"] = False
                        current_plan["termination"] = "chunk_complete"
            if not success and current_plan is not None and current_plan["executed_full_chunk"] is None:
                current_plan["executed_full_chunk"] = chunk_index == 8
                current_plan["early_replan"] = False
                current_plan["termination"] = "episode_step_limit"
        finally:
            env.close()
        episodes.append(
            {
                "episode_index": episode_index,
                "seed": seeds[episode_index],
                "init_state_index": episode_index,
                "success": success,
                "steps": step,
                "plan_count": len(plan_events),
                "verification_count": len(verification_events),
                "early_replan_count": sum(
                    event["early_replan"] for event in verification_events
                ),
                "plan_events": plan_events,
                "verification_events": verification_events,
            }
        )
        atomic_replan_jsonl(args.replan_events, episodes)
        result = summarize(
            args.checkpoint,
            args.config,
            args.seed_file,
            task,
            args.task_id,
            args.max_steps,
            len(seeds),
            episodes,
        )
        attach_replan_artifact(result, args.replan_events, episodes)
        atomic_json(args.output, result)
        print(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "success": success,
                    "steps": step,
                    "plans": len(plan_events),
                    "verification_events": len(verification_events),
                    "early_replans": episodes[-1]["early_replan_count"],
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "episodes",
                    "successes",
                    "planning_events",
                    "verification_events",
                    "early_replan_events",
                    "mean_early_replans_per_episode",
                    "trigger_reason_counts",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
