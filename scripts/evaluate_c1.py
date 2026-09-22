#!/usr/bin/env python3
"""D19 single-task C1 evaluation with immutable episodes and plan logs."""

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
from src.policies.c1_rerank_policy import C1RerankPolicy
from src.world_models.normalization import file_sha256


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def summarize(
    actor_checkpoint: Path,
    config: Path,
    seed_file: Path,
    task_id: int,
    task: Any,
    max_steps: int,
    episodes_target: int,
    episode_results: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    events = [event for episode in episode_results for event in episode["plan_events"]]
    counts = Counter(event["selected_candidate"] for event in events)
    return {
        "schema_version": 1,
        "method": "c1_actor_wm_pre_execution_rerank",
        "actor_checkpoint": str(actor_checkpoint),
        "actor_checkpoint_sha256": file_sha256(actor_checkpoint),
        "c1_config": str(config),
        "c1_config_sha256": file_sha256(config),
        "seed_file": str(seed_file),
        "task_suite": "libero_spatial",
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "max_steps": max_steps,
        "episodes_target": episodes_target,
        "candidate_count": 3,
        "in_execution_verification": False,
        "early_replan": False,
        "completed_episode_indices": [item["episode_index"] for item in episode_results],
        "complete": complete,
        "episodes": len(episode_results),
        "successes": sum(item["success"] for item in episode_results),
        "success_rate": (
            sum(item["success"] for item in episode_results) / len(episode_results)
            if episode_results
            else 0.0
        ),
        "planning_events": len(events),
        "selected_candidate_counts": {str(index): counts.get(index, 0) for index in range(3)},
        "nonbase_selection_fraction": (
            (counts.get(1, 0) + counts.get(2, 0)) / len(events) if events else 0.0
        ),
        "mean_planning_ms": (
            float(np.mean([event["planning_ms"] for event in events])) if events else None
        ),
        "episode_results": episode_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "results/c0/seed11/best.pt"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/c1_rerank.yaml")
    parser.add_argument(
        "--seed-file", type=Path, default=PROJECT_ROOT / "configs/eval_seeds_spatial.json"
    )
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--episode-limit", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--output", type=Path, required=True)
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
    episode_results: list[dict[str, Any]] = []
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")
        existing = json.loads(args.output.read_text())
        expected = {
            "method": "c1_actor_wm_pre_execution_rerank",
            "actor_checkpoint_sha256": file_sha256(args.checkpoint),
            "c1_config_sha256": file_sha256(args.config),
            "task_id": args.task_id,
            "max_steps": args.max_steps,
            "episodes_target": args.episode_limit,
        }
        mismatches = [key for key, value in expected.items() if existing.get(key) != value]
        if mismatches:
            raise ValueError(f"incompatible C1 resume fields: {mismatches}")
        episode_results = existing["episode_results"]
        if existing["completed_episode_indices"] != list(range(len(episode_results))):
            raise ValueError("C1 resume episodes are not a contiguous prefix")
    elif args.resume:
        raise FileNotFoundError(args.output)

    if len(episode_results) == len(seeds):
        result = summarize(
            args.checkpoint,
            args.config,
            args.seed_file,
            args.task_id,
            task,
            args.max_steps,
            len(seeds),
            episode_results,
            complete=True,
        )
        print(json.dumps({"resume_complete": True, "episodes": len(episode_results)}))
        return 0

    policy = C1RerankPolicy(args.checkpoint, device=args.device, config=args.config)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    for episode_index in range(len(episode_results), len(seeds)):
        env: OffScreenRenderEnv = make_env(bddl_file)
        plan_events: list[dict[str, Any]] = []
        try:
            env.seed(seeds[episode_index])
            env.reset()
            observation = env.set_init_state(init_states[episode_index])
            disable_camera_observable(env)
            policy.reset(seed=seeds[episode_index])
            chunk = np.empty((0, 7), dtype=np.float32)
            chunk_index = 8
            success = False
            for step in range(1, args.max_steps + 1):
                if chunk_index >= 8:
                    observation = refresh_policy_observation(env)
                    chunk, diagnostics = policy.plan_chunk(
                        policy_observation(observation), task.language
                    )
                    diagnostics["episode_step_before_chunk"] = step - 1
                    plan_events.append(diagnostics)
                    chunk_index = 0
                action = np.asarray(chunk[chunk_index], dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("C1 selected chunk contains an invalid action")
                observation, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
                chunk_index += 1
                if env.check_success():
                    success = True
                    break
        finally:
            env.close()
        episode_results.append(
            {
                "episode_index": episode_index,
                "seed": seeds[episode_index],
                "init_state_index": episode_index,
                "success": success,
                "steps": step,
                "plan_count": len(plan_events),
                "plan_events": plan_events,
            }
        )
        result = summarize(
            args.checkpoint,
            args.config,
            args.seed_file,
            args.task_id,
            task,
            args.max_steps,
            len(seeds),
            episode_results,
            complete=len(episode_results) == len(seeds),
        )
        write_result(args.output, result)
        print(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "success": success,
                    "steps": step,
                    "plans": len(plan_events),
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
                    "success_rate",
                    "planning_events",
                    "selected_candidate_counts",
                    "nonbase_selection_fraction",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
