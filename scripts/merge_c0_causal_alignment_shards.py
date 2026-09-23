#!/usr/bin/env python3
"""Merge disjoint protocol-valid C0 development shards without losing raw episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_clipping(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "planned_chunks",
        "planned_action_elements",
        "elements_below_minus_one",
        "elements_above_one",
        "elements_outside_minus_one_to_one",
    )
    sums = {field: sum(int(episode["action_clipping"][field]) for episode in episodes) for field in fields}
    elements = sums["planned_action_elements"]
    minima = [episode["action_clipping"]["raw_min"] for episode in episodes]
    maxima = [episode["action_clipping"]["raw_max"] for episode in episodes]
    return {
        "episodes_with_clipping_stats": len(episodes),
        **sums,
        "clipped_element_ratio": sums["elements_outside_minus_one_to_one"] / elements,
        "raw_min": min(float(value) for value in minima if value is not None),
        "raw_max": max(float(value) for value in maxima if value is not None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-label", required=True, choices=("c0_v1", "c0_v2"))
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/c0_v2_causal_alignment_protocol.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    expected_task_ids = [int(value) for value in protocol["task_ids"]]
    task_map: dict[int, dict[str, Any]] = {}
    checkpoint_hash: str | None = None
    checkpoint_path: str | None = None
    shards = []
    partition = protocol["partitions"]["controller_development"]
    for path in args.inputs:
        result = json.loads(path.read_text())
        expected = {
            "schema_version": 1,
            "experiment": "c0-v2-causal-alignment",
            "policy_label": args.policy_label,
            "partition": "controller_development",
            "protocol_sha256": protocol_hash,
        }
        mismatches = [key for key, value in expected.items() if result.get(key) != value]
        if mismatches:
            raise ValueError(f"{path}: incompatible fields {mismatches}")
        if checkpoint_hash is None:
            checkpoint_hash = result["checkpoint_sha256"]
            checkpoint_path = result["checkpoint"]
        elif result["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("shards use different checkpoints")
        for task in result["tasks"]:
            task_id = int(task["task_id"])
            if task_id in task_map:
                raise ValueError(f"duplicate task {task_id}")
            episodes = task["episode_results"]
            if [episode["init_state_index"] for episode in episodes] != partition["init_state_indices"]:
                raise ValueError(f"task {task_id}: init-state sequence differs")
            if [episode["seed"] for episode in episodes] != partition["task_seeds"][str(task_id)]:
                raise ValueError(f"task {task_id}: seed sequence differs")
            task_map[task_id] = task
        shards.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "task_ids": [int(task["task_id"]) for task in result["tasks"]],
                "parallel_envs": result["parallel_envs"],
            }
        )
    if sorted(task_map) != expected_task_ids:
        raise ValueError(f"merged task IDs {sorted(task_map)} != {expected_task_ids}")
    tasks = [task_map[task_id] for task_id in expected_task_ids]
    episodes = [episode for task in tasks for episode in task["episode_results"]]
    if len(episodes) != 100:
        raise ValueError(f"merged episode count {len(episodes)} != 100")
    merged = {
        "schema_version": 1,
        "experiment": "c0-v2-causal-alignment",
        "policy_label": args.policy_label,
        "partition": "controller_development",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_hash,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_hash,
        "suite": protocol["suite"],
        "episodes_per_task": 10,
        "max_steps": protocol["max_steps"],
        "parallel_envs": "sharded; see execution_shards",
        "requested_task_ids": expected_task_ids,
        "completed_task_ids": expected_task_ids,
        "complete": True,
        "tasks": tasks,
        "episodes": 100,
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "macro_success_rate": float(np.mean([task["success_rate"] for task in tasks])),
        "mean_steps": float(np.mean([episode["steps"] for episode in episodes])),
        "nonzero_tasks": sum(int(task["successes"]) > 0 for task in tasks),
        "action_clipping": aggregate_clipping(episodes),
        "execution_shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "successes": merged["successes"],
                "nonzero_tasks": merged["nonzero_tasks"],
                "episodes": merged["episodes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
