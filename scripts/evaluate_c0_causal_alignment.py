#!/usr/bin/env python3
"""Evaluate frozen C0 variants on preregistered causal-alignment partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
from scripts.evaluate_c0_all import (
    add_clipping_stats,
    aggregate_clipping_stats,
    empty_clipping_stats,
    finalized_clipping_stats,
    make_env,
    policy_observation,
    trusted_init_states,
)
from src.policies.c0_actor_policy import C0ActorPolicy, create_policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def disable_camera_observables(env: OffScreenRenderEnv) -> None:
    disabled = []
    for name, observable in env.env._observables.items():
        if name.endswith("_image"):
            observable.set_active(False)
            disabled.append(name)
    if "agentview_image" not in disabled:
        raise KeyError("environment has no agentview_image observable")


def refresh_policy_observation(env: OffScreenRenderEnv) -> dict[str, np.ndarray]:
    camera = env.env._observables["agentview_image"]
    camera.set_active(True)
    env._update_observables(force=True)
    observation = env.env._get_observations()
    camera.set_active(False)
    return observation


def validate_checkpoint(
    checkpoint: Path, policy_label: str, protocol: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_hash = sha256(checkpoint)
    if policy_label == "c0_v1":
        frozen = protocol["frozen_inputs"]
        if checkpoint.resolve() != (PROJECT_ROOT / frozen["c0_v1_checkpoint"]).resolve():
            raise ValueError("C0-v1 evaluation must use the preregistered checkpoint path")
        if checkpoint_hash != frozen["c0_v1_checkpoint_sha256"]:
            raise ValueError("frozen C0-v1 checkpoint hash changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("method") != "c0_actor_only":
        raise ValueError("checkpoint is not compatible with the frozen C0 architecture")
    if policy_label == "c0_v2":
        supervision = state.get("supervision_contract", {})
        if supervision.get("action_start_offset") != 1:
            raise ValueError("C0-v2 checkpoint does not declare causal action offset 1")
        if supervision.get("horizon") != 8:
            raise ValueError("C0-v2 checkpoint does not declare H=8")
    return {"checkpoint_sha256": checkpoint_hash, "checkpoint_state": state}


def require_committed_holdout_go(
    partition: str,
    policy_label: str,
    gate_decision_path: Path | None,
    protocol_sha256: str,
    checkpoint_sha256: str,
) -> None:
    if partition != "confirmatory_holdout":
        return
    if gate_decision_path is None:
        raise ValueError("confirmatory_holdout requires --gate-decision")
    relative = gate_decision_path.resolve().relative_to(PROJECT_ROOT)
    subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    decision = json.loads(gate_decision_path.read_text())
    expected = {
        "protocol_sha256": protocol_sha256,
        "decision": "GO_CONFIRMATORY_HOLDOUT",
    }
    if policy_label == "c0_v2":
        expected["frozen_c0_v2_checkpoint_sha256"] = checkpoint_sha256
    mismatches = [key for key, value in expected.items() if decision.get(key) != value]
    if mismatches:
        raise ValueError(f"holdout gate decision mismatch: {mismatches}")


def evaluate_task(
    policy: C0ActorPolicy,
    suite: Any,
    task_id: int,
    seeds: list[int],
    init_state_indices: list[int],
    max_steps: int,
    parallel_envs: int,
) -> dict[str, Any]:
    if len(seeds) != len(init_state_indices):
        raise ValueError("seed and init-state counts differ")
    task = suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = trusted_init_states(suite, task_id)
    if max(init_state_indices) >= len(init_states):
        raise ValueError(f"task {task_id}: insufficient official initial states")
    episode_results: list[dict[str, Any]] = []
    horizon = policy.model.action_horizon

    for batch_start in range(0, len(seeds), parallel_envs):
        episode_positions = list(range(batch_start, min(batch_start + parallel_envs, len(seeds))))
        envs = [make_env(bddl_file) for _ in episode_positions]
        try:
            observations = []
            for env, position in zip(envs, episode_positions):
                env.seed(seeds[position])
                env.reset()
                observations.append(env.set_init_state(init_states[init_state_indices[position]]))
                disable_camera_observables(env)
            chunks: list[np.ndarray | None] = [None] * len(envs)
            chunk_indices = [horizon] * len(envs)
            active = [True] * len(envs)
            episode_steps = [0] * len(envs)
            clipping = [empty_clipping_stats() for _ in envs]
            for _ in range(max_steps):
                planning = [
                    index
                    for index in range(len(envs))
                    if active[index] and chunk_indices[index] >= horizon
                ]
                if planning:
                    for index in planning:
                        observations[index] = refresh_policy_observation(envs[index])
                    new_chunks, new_clipping = policy.plan_batch_with_clipping_stats(
                        [policy_observation(observations[index]) for index in planning],
                        [task.language] * len(planning),
                    )
                    for index, chunk, stats in zip(planning, new_chunks, new_clipping):
                        chunks[index] = chunk
                        chunk_indices[index] = 0
                        add_clipping_stats(clipping[index], stats)
                for index, env in enumerate(envs):
                    if not active[index]:
                        continue
                    action = np.asarray(chunks[index][chunk_indices[index]], dtype=np.float32)  # type: ignore[index]
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("C0 policy produced an invalid action")
                    observations[index], _, _, _ = env.step(np.clip(action, -1.0, 1.0))
                    chunk_indices[index] += 1
                    episode_steps[index] += 1
                    if env.check_success():
                        active[index] = False
                        position = episode_positions[index]
                        episode_results.append(
                            {
                                "episode_index": position,
                                "seed": seeds[position],
                                "init_state_index": init_state_indices[position],
                                "success": True,
                                "steps": episode_steps[index],
                                "action_clipping": finalized_clipping_stats(clipping[index]),
                            }
                        )
                if not any(active):
                    break
            for index, is_active in enumerate(active):
                if is_active:
                    position = episode_positions[index]
                    episode_results.append(
                        {
                            "episode_index": position,
                            "seed": seeds[position],
                            "init_state_index": init_state_indices[position],
                            "success": False,
                            "steps": episode_steps[index],
                            "action_clipping": finalized_clipping_stats(clipping[index]),
                        }
                    )
        finally:
            for env in envs:
                env.close()

    episode_results.sort(key=lambda item: item["episode_index"])
    successes = sum(bool(item["success"]) for item in episode_results)
    return {
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "episodes": len(episode_results),
        "successes": successes,
        "success_rate": successes / len(episode_results),
        "mean_steps": float(np.mean([item["steps"] for item in episode_results])),
        "episode_results": episode_results,
        "action_clipping": aggregate_clipping_stats(episode_results),
    }


def build_result(
    protocol_path: Path,
    protocol: dict[str, Any],
    partition: str,
    policy_label: str,
    checkpoint: Path,
    checkpoint_hash: str,
    parallel_envs: int,
    requested_task_ids: list[int],
    tasks: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    episodes = [episode for task in tasks for episode in task["episode_results"]]
    return {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "policy_label": policy_label,
        "partition": partition,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "suite": protocol["suite"],
        "episodes_per_task": protocol["episodes_per_task"],
        "max_steps": protocol["max_steps"],
        "parallel_envs": parallel_envs,
        "requested_task_ids": requested_task_ids,
        "completed_task_ids": [task["task_id"] for task in tasks],
        "complete": complete,
        "tasks": tasks,
        "episodes": len(episodes),
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "macro_success_rate": float(np.mean([task["success_rate"] for task in tasks])),
        "mean_steps": float(np.mean([episode["steps"] for episode in episodes])),
        "nonzero_tasks": sum(task["successes"] > 0 for task in tasks),
        "action_clipping": aggregate_clipping_stats(episodes),
    }


def validate_resume(
    existing: dict[str, Any],
    partition: str,
    policy_label: str,
    protocol_hash: str,
    checkpoint_hash: str,
    parallel_envs: int,
    requested_task_ids: list[int],
) -> None:
    expected = {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "policy_label": policy_label,
        "partition": partition,
        "protocol_sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hash,
        "parallel_envs": parallel_envs,
    }
    mismatches = [key for key, value in expected.items() if existing.get(key) != value]
    if mismatches:
        raise ValueError(f"incompatible resume result: {mismatches}")
    existing_requested = existing.get("requested_task_ids")
    if existing_requested is not None and existing_requested != requested_task_ids:
        raise ValueError("resume requested task IDs differ")
    if existing_requested is None and requested_task_ids != list(range(10)):
        raise ValueError("legacy partial result may only resume the full protocol task list")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-label", choices=("c0_v1", "c0_v2"), required=True)
    parser.add_argument(
        "--partition",
        choices=("controller_development", "confirmatory_holdout"),
        required=True,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/c0_v2_causal_alignment_protocol.json",
    )
    parser.add_argument("--gate-decision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-envs", type=int, default=1)
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(10))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.parallel_envs < 1:
        raise ValueError("--parallel-envs must be positive")
    protocol = json.loads(args.protocol.read_text())
    if protocol["protocol_version"] != "c0-v2-causal-alignment-v1":
        raise ValueError("unexpected causal-alignment protocol")
    checkpoint_record = validate_checkpoint(args.checkpoint, args.policy_label, protocol)
    checkpoint_hash = checkpoint_record["checkpoint_sha256"]
    protocol_hash = sha256(args.protocol)
    require_committed_holdout_go(
        args.partition, args.policy_label, args.gate_decision, protocol_hash, checkpoint_hash
    )
    partition = protocol["partitions"][args.partition]
    protocol_task_ids = [int(value) for value in protocol["task_ids"]]
    task_ids = args.task_ids if args.task_ids is not None else protocol_task_ids
    if len(set(task_ids)) != len(task_ids) or task_ids != sorted(task_ids):
        raise ValueError("--task-ids must be unique and sorted")
    if not set(task_ids).issubset(protocol_task_ids):
        raise ValueError("--task-ids leave the preregistered protocol")
    tasks: list[dict[str, Any]] = []
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")
        existing = json.loads(args.output.read_text())
        validate_resume(
            existing,
            args.partition,
            args.policy_label,
            protocol_hash,
            checkpoint_hash,
            args.parallel_envs,
            task_ids,
        )
        tasks = existing["tasks"]
        if [task["task_id"] for task in tasks] != task_ids[: len(tasks)]:
            raise ValueError("resume tasks are not a prefix of the protocol")
    elif args.resume:
        raise FileNotFoundError(f"cannot resume missing output {args.output}")

    suite = benchmark.get_benchmark_dict()[protocol["suite"]]()
    if len(tasks) < len(task_ids):
        policy = create_policy(checkpoint=str(args.checkpoint), device=args.device)
        for position in range(len(tasks), len(task_ids)):
            task_id = task_ids[position]
            task = evaluate_task(
                policy,
                suite,
                task_id,
                [int(value) for value in partition["task_seeds"][str(task_id)]],
                [int(value) for value in partition["init_state_indices"]],
                int(protocol["max_steps"]),
                args.parallel_envs,
            )
            tasks.append(task)
            result = build_result(
                args.protocol,
                protocol,
                args.partition,
                args.policy_label,
                args.checkpoint,
                checkpoint_hash,
                args.parallel_envs,
                task_ids,
                tasks,
                complete=position == len(task_ids) - 1,
            )
            atomic_json(args.output, result)
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "successes": task["successes"],
                        "episodes": task["episodes"],
                        "mean_steps": task["mean_steps"],
                    }
                ),
                flush=True,
            )
    result = build_result(
        args.protocol,
        protocol,
        args.partition,
        args.policy_label,
        args.checkpoint,
        checkpoint_hash,
        args.parallel_envs,
        task_ids,
        tasks,
        complete=len(tasks) == len(task_ids),
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("episodes", "successes", "macro_success_rate", "nonzero_tasks", "mean_steps")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
