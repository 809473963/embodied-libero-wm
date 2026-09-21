#!/usr/bin/env python3
"""Evaluate a C0 checkpoint on all Spatial tasks with batched independent rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from src.policies.c0_actor_policy import C0ActorPolicy, create_policy


def trusted_init_states(task_suite: object, task_id: int) -> object:
    original_load = torch.load

    def load_with_trusted_default(*load_args: object, **load_kwargs: object) -> object:
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = load_with_trusted_default  # type: ignore[assignment]
    try:
        return task_suite.get_task_init_states(task_id)  # type: ignore[attr-defined]
    finally:
        torch.load = original_load  # type: ignore[assignment]


def policy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.ascontiguousarray(observation["agentview_image"]),
        "robot0_eef_pos": np.ascontiguousarray(observation["robot0_eef_pos"]),
        "robot0_gripper_qpos": np.ascontiguousarray(observation["robot0_gripper_qpos"]),
        "robot0_joint_pos": np.ascontiguousarray(observation["robot0_joint_pos"]),
    }


def make_env(bddl_file: Path) -> OffScreenRenderEnv:
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
    )


def disable_camera_observable(env: OffScreenRenderEnv) -> None:
    """Avoid rendering RGB on the seven intervening open-loop actions."""

    env.env._observables["agentview_image"].set_active(False)


def refresh_policy_observation(env: OffScreenRenderEnv) -> dict[str, np.ndarray]:
    """Render one current RGB frame plus current robot proprioception for planning."""

    camera = env.env._observables["agentview_image"]
    camera.set_active(True)
    env._update_observables(force=True)
    observation = env.env._get_observations()
    camera.set_active(False)
    return observation


def partial_summary(
    checkpoint: Path,
    seed_spec: dict[str, Any],
    max_steps: int,
    parallel_envs: int,
    task_results: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    macro_success_rate = float(np.mean([item["success_rate"] for item in task_results]))
    return {
        "schema_version": 1,
        "method": "c0_actor_only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "task_suite": seed_spec["suite"],
        "seed_file": seed_spec["_path"],
        "episodes_per_task": seed_spec["episodes_per_task"],
        "max_steps": max_steps,
        "parallel_envs": parallel_envs,
        "completed_task_ids": [item["task_id"] for item in task_results],
        "complete": complete,
        "tasks": task_results,
        "macro_success_rate": macro_success_rate,
        "successes": sum(item["successes"] for item in task_results),
        "episodes": sum(item["episodes"] for item in task_results),
    }


def evaluate_task(
    policy: C0ActorPolicy,
    task_suite: object,
    task_id: int,
    seeds: list[int],
    max_steps: int,
    parallel_envs: int,
) -> dict[str, Any]:
    task = task_suite.get_task(task_id)  # type: ignore[attr-defined]
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = trusted_init_states(task_suite, task_id)
    if len(init_states) < len(seeds):
        raise ValueError(f"task {task_id}: insufficient official initial states")
    episode_results = []
    horizon = policy.model.action_horizon

    for batch_start in range(0, len(seeds), parallel_envs):
        episode_indices = list(range(batch_start, min(batch_start + parallel_envs, len(seeds))))
        envs = [make_env(bddl_file) for _ in episode_indices]
        try:
            observations = []
            for env, episode_index in zip(envs, episode_indices):
                env.seed(int(seeds[episode_index]))
                env.reset()
                observations.append(env.set_init_state(init_states[episode_index]))
                disable_camera_observable(env)
            action_chunks: list[np.ndarray | None] = [None] * len(envs)
            chunk_indices = [horizon] * len(envs)
            active = [True] * len(envs)
            episode_steps = [0] * len(envs)
            for _ in range(max_steps):
                plan_indices = [
                    index
                    for index in range(len(envs))
                    if active[index] and chunk_indices[index] >= horizon
                ]
                if plan_indices:
                    for index in plan_indices:
                        observations[index] = refresh_policy_observation(envs[index])
                    new_chunks = policy.plan_batch(
                        [policy_observation(observations[index]) for index in plan_indices],
                        [task.language] * len(plan_indices),
                    )
                    for index, chunk in zip(plan_indices, new_chunks):
                        action_chunks[index] = chunk
                        chunk_indices[index] = 0
                if not any(active):
                    break
                for index, env in enumerate(envs):
                    if not active[index]:
                        continue
                    action = np.asarray(action_chunks[index][chunk_indices[index]], dtype=np.float32)  # type: ignore[index]
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("C0 policy produced an invalid action")
                    observations[index], _, _, _ = env.step(np.clip(action, -1.0, 1.0))
                    chunk_indices[index] += 1
                    episode_steps[index] += 1
                    if env.check_success():
                        active[index] = False
                        episode_results.append(
                            {
                                "episode_index": episode_indices[index],
                                "seed": int(seeds[episode_indices[index]]),
                                "init_state_index": episode_indices[index],
                                "success": True,
                                "steps": episode_steps[index],
                            }
                        )
                if not any(active):
                    break
            for index, is_active in enumerate(active):
                if is_active:
                    episode_results.append(
                        {
                            "episode_index": episode_indices[index],
                            "seed": int(seeds[episode_indices[index]]),
                            "init_state_index": episode_indices[index],
                            "success": False,
                            "steps": episode_steps[index],
                        }
                    )
        finally:
            for env in envs:
                env.close()

    episode_results.sort(key=lambda item: item["episode_index"])
    successes = sum(item["success"] for item in episode_results)
    print(
        json.dumps({"task_id": task_id, "successes": successes, "episodes": len(episode_results)}),
        flush=True,
    )
    return {
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "episodes": len(episode_results),
        "successes": successes,
        "success_rate": successes / len(episode_results),
        "episode_results": episode_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, default=PROJECT_ROOT / "configs/eval_seeds_spatial.json")
    parser.add_argument("--max-steps", type=int, default=300)
    # Keep the default single-stream inference path aligned with the D10
    # evaluator.  Batched FP16 inference is opt-in because it can change a
    # borderline open-loop rollout through harmless numerical rounding.
    parser.add_argument("--parallel-envs", type=int, default=1)
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(10))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a compatible partial --output at its next unfinished task",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.parallel_envs < 1:
        raise ValueError("--parallel-envs must be positive")
    seed_spec = json.loads(args.seed_file.read_text())
    seed_spec["_path"] = str(args.seed_file.resolve())
    suite = benchmark.get_benchmark_dict()[seed_spec["suite"]]()
    task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
    task_results: list[dict[str, Any]] = []
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(
                f"refusing to overwrite existing result: {args.output}; use --resume to continue it"
            )
        existing = json.loads(args.output.read_text())
        expected_metadata = {
            "schema_version": 1,
            "method": "c0_actor_only",
            "checkpoint_sha256": checkpoint_sha256,
            "task_suite": seed_spec["suite"],
            "seed_file": seed_spec["_path"],
            "episodes_per_task": seed_spec["episodes_per_task"],
            "max_steps": args.max_steps,
            "parallel_envs": args.parallel_envs,
        }
        mismatches = [
            key
            for key, expected in expected_metadata.items()
            if existing.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "resume result is incompatible with the requested protocol: "
                + ", ".join(mismatches)
            )
        completed_ids = existing.get("completed_task_ids")
        existing_tasks = existing.get("tasks")
        if not isinstance(completed_ids, list) or not isinstance(existing_tasks, list):
            raise ValueError("resume result has no valid completed task list")
        if completed_ids != task_ids[: len(completed_ids)]:
            raise ValueError("resume result does not contain a prefix of the requested task ids")
        if [task.get("task_id") for task in existing_tasks] != completed_ids:
            raise ValueError("resume result task records do not match its completed task ids")
        if len(completed_ids) > len(task_ids):
            raise ValueError("resume result contains more tasks than requested")
        if existing.get("complete") and len(completed_ids) != len(task_ids):
            raise ValueError("resume result is marked complete but omits requested tasks")
        task_results = existing_tasks
    elif args.resume:
        raise FileNotFoundError(f"cannot resume because result does not exist: {args.output}")

    if len(task_results) == len(task_ids):
        result = partial_summary(
            args.checkpoint,
            seed_spec,
            args.max_steps,
            args.parallel_envs,
            task_results,
            complete=True,
        )
        print(json.dumps({key: result[key] for key in ("episodes", "successes", "macro_success_rate")}, indent=2))
        return 0

    policy = create_policy(checkpoint=str(args.checkpoint), device=args.device)
    for task_position in range(len(task_results), len(task_ids)):
        task_id = task_ids[task_position]
        seeds = [int(seed) for seed in seed_spec["task_seeds"][str(task_id)]]
        if len(seeds) != seed_spec["episodes_per_task"]:
            raise ValueError(f"task {task_id}: evaluation seed count differs from protocol")
        task_results.append(evaluate_task(policy, suite, task_id, seeds, args.max_steps, args.parallel_envs))
        result = partial_summary(
            args.checkpoint,
            seed_spec,
            args.max_steps,
            args.parallel_envs,
            task_results,
            complete=task_position == len(task_ids) - 1,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".part")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, args.output)
    print(
        json.dumps(
            {key: result[key] for key in ("episodes", "successes", "macro_success_rate")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"all-task C0 evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
