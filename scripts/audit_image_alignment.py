#!/usr/bin/env python3
"""Verify official demonstration RGB has the same orientation as live LIBERO RGB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    dataset = (
        PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial" / f"{task.name}_demo.hdf5"
    )
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    comparisons = []
    try:
        env.reset()
        with h5py.File(dataset, "r") as source:
            demo = source["data/demo_0"]
            stored = demo["obs/agentview_rgb"][0].astype(np.float32)
            for state_index in (0, 1, 2):
                live = env.regenerate_obs_from_state(demo["states"][state_index])["agentview_image"].astype(np.float32)
                comparisons.append(
                    {
                        "state_index": state_index,
                        "mae_direct": float(np.abs(stored - live).mean()),
                        "mae_vertical_flip": float(np.abs(np.flipud(stored) - live).mean()),
                    }
                )
    finally:
        env.close()
    result = {
        "schema_version": 1,
        "task_id": args.task_id,
        "task_name": task.name,
        "comparisons": comparisons,
        "selected_preprocessing": "uint8 RGB; vertical_flip=false",
        "passed": all(item["mae_direct"] < item["mae_vertical_flip"] for item in comparisons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"image alignment audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
