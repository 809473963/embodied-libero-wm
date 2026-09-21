#!/usr/bin/env python3
"""Audit the official LIBERO-Spatial HDF5 layout without exporting training data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# LIBERO creates its default config interactively at import time. Point every
# standalone invocation at the project-local, already reviewed config first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))

from libero.libero import benchmark


REQUIRED_OBSERVATIONS = {
    "agentview_rgb": (128, 128, 3),
    "ee_pos": (3,),
    "ee_ori": (3,),
    "gripper_states": (2,),
    "joint_states": (7,),
}


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_task(
    path: Path, task_id: int, task_name: str, task_language: str, horizon: int
) -> dict[str, Any]:
    action_min = np.full(7, np.inf, dtype=np.float64)
    action_max = np.full(7, -np.inf, dtype=np.float64)
    transition_count = 0
    chunk_count = 0
    demo_lengths: list[int] = []

    with h5py.File(path, "r") as h5_file:
        require("data" in h5_file, f"{path}: missing /data")
        data = h5_file["data"]
        demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))
        require(len(demo_names) == 50, f"{path}: expected 50 demos, got {len(demo_names)}")
        require(int(data.attrs["num_demos"]) == 50, f"{path}: bad num_demos attribute")

        for demo_id, demo_name in enumerate(demo_names):
            require(demo_name == f"demo_{demo_id}", f"{path}: unexpected demo order")
            demo = data[demo_name]
            for field in ("actions", "dones", "obs", "rewards", "states"):
                require(field in demo, f"{path}/{demo_name}: missing {field}")

            actions = demo["actions"]
            num_steps = int(actions.shape[0])
            require(actions.shape == (num_steps, 7), f"{path}/{demo_name}: bad action shape")
            require(demo["dones"].shape == (num_steps,), f"{path}/{demo_name}: bad dones")
            require(demo["rewards"].shape == (num_steps,), f"{path}/{demo_name}: bad rewards")
            require(demo["states"].shape == (num_steps, 92), f"{path}/{demo_name}: bad states")

            observations = demo["obs"]
            for field, trailing_shape in REQUIRED_OBSERVATIONS.items():
                require(field in observations, f"{path}/{demo_name}: missing obs/{field}")
                expected_shape = (num_steps, *trailing_shape)
                require(
                    observations[field].shape == expected_shape,
                    f"{path}/{demo_name}: obs/{field} has {observations[field].shape}, expected {expected_shape}",
                )
            require(
                observations["agentview_rgb"].dtype == np.dtype("uint8"),
                f"{path}/{demo_name}: agentview_rgb must be uint8",
            )

            action_values = actions[...]
            require(np.isfinite(action_values).all(), f"{path}/{demo_name}: non-finite action")
            action_min = np.minimum(action_min, action_values.min(axis=0))
            action_max = np.maximum(action_max, action_values.max(axis=0))
            transition_count += num_steps
            chunk_count += max(0, num_steps - horizon + 1)
            demo_lengths.append(num_steps)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "task_language": task_language,
        "file": str(path),
        "demonstrations": len(demo_lengths),
        "transitions": transition_count,
        "action_chunk_samples": chunk_count,
        "demo_length": {
            "min": min(demo_lengths),
            "max": max(demo_lengths),
            "mean": float(np.mean(demo_lengths)),
        },
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    require(args.horizon > 0, "--horizon must be positive")

    project_root = PROJECT_ROOT
    dataset_dir = args.dataset_dir or project_root / "data/LIBERO/datasets/libero_spatial"
    require(dataset_dir.is_dir(), f"dataset directory not found: {dataset_dir}")
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    task_results = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        hdf5_path = dataset_dir / f"{task.name}_demo.hdf5"
        require(hdf5_path.is_file(), f"missing task data: {hdf5_path}")
        task_results.append(
            audit_task(hdf5_path, task_id, task.name, task.language, args.horizon)
        )

    action_min = np.min([item["action_min"] for item in task_results], axis=0)
    action_max = np.max([item["action_max"] for item in task_results], axis=0)
    result: dict[str, Any] = {
        "schema_version": 1,
        "suite": "libero_spatial",
        "libero_commit": git_commit(project_root / "third_party/LIBERO"),
        "dataset_dir": str(dataset_dir.relative_to(project_root)),
        "action_chunk_horizon": args.horizon,
        "expected_observation_schema": {
            field: list(shape) for field, shape in REQUIRED_OBSERVATIONS.items()
        },
        "split": {
            "train_demo_ids": list(range(45)),
            "validation_demo_ids": list(range(45, 50)),
            "unit": "whole demonstration",
        },
        "tasks": task_results,
        "totals": {
            "tasks": len(task_results),
            "demonstrations": sum(item["demonstrations"] for item in task_results),
            "transitions": sum(item["transitions"] for item in task_results),
            "action_chunk_samples": sum(
                item["action_chunk_samples"] for item in task_results
            ),
            "action_min": action_min.tolist(),
            "action_max": action_max.tolist(),
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["totals"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"data audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
