#!/usr/bin/env python3
"""Evaluate one trained C0 checkpoint on the immutable LIBERO-Spatial seeds."""

from __future__ import annotations

import argparse
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
from src.policies.c0_actor_policy import create_policy


def trusted_init_states(task_suite: object, task_id: int) -> object:
    """Load fixed, checked-in task initial states under PyTorch 2.6+."""

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
    """Pass exactly the C0 online contract, without simulator/oracle fields."""

    return {
        "agentview_image": np.ascontiguousarray(observation["agentview_image"]),
        "robot0_eef_pos": np.ascontiguousarray(observation["robot0_eef_pos"]),
        "robot0_gripper_qpos": np.ascontiguousarray(observation["robot0_gripper_qpos"]),
        "robot0_joint_pos": np.ascontiguousarray(observation["robot0_joint_pos"]),
    }


def checked_action(action: Any) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError("C0 adapter did not produce a finite 7-D action")
    return np.clip(array, -1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True, choices=range(10))
    parser.add_argument("--seed-file", type=Path, default=PROJECT_ROOT / "configs/eval_seeds_spatial.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_spec = json.loads(args.seed_file.read_text())
    seeds = seed_spec["task_seeds"][str(args.task_id)]
    if len(seeds) != seed_spec["episodes_per_task"]:
        raise ValueError("evaluation seed count differs from protocol")
    task_suite = benchmark.get_benchmark_dict()[seed_spec["suite"]]()
    task = task_suite.get_task(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = trusted_init_states(task_suite, args.task_id)
    if len(init_states) < len(seeds):
        raise ValueError("task has fewer initial states than requested episodes")
    policy = create_policy(checkpoint=str(args.checkpoint), device=args.device)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
    )
    episode_results = []
    try:
        for episode_index, seed in enumerate(seeds):
            env.seed(int(seed))
            env.reset()
            policy.reset()
            # The fixed seeds choose the matching first 20 official task init
            # states. No test state is inspected or supplied to the policy.
            observation = env.set_init_state(init_states[episode_index])
            success = False
            for step in range(1, args.max_steps + 1):
                action = checked_action(policy.act(policy_observation(observation), task.language))
                observation, _, _, _ = env.step(action)
                if env.check_success():
                    success = True
                    break
            episode_results.append(
                {
                    "episode_index": episode_index,
                    "seed": int(seed),
                    "init_state_index": episode_index,
                    "success": success,
                    "steps": step,
                }
            )
            print(json.dumps(episode_results[-1]), flush=True)
    finally:
        env.close()
    successes = sum(item["success"] for item in episode_results)
    result = {
        "schema_version": 1,
        "method": "c0_actor_only",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "task_suite": seed_spec["suite"],
        "task_id": args.task_id,
        "task_name": task.name,
        "task_language": task.language,
        "seed_file": str(args.seed_file),
        "episodes": len(episode_results),
        "max_steps": args.max_steps,
        "successes": successes,
        "success_rate": successes / len(episode_results),
        "episode_results": episode_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("task_id", "episodes", "successes", "success_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"C0 evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
