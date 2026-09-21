#!/usr/bin/env python3
"""Exercise the trained-C0 adapter with contract-shaped data, not oracle state."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark
from src.policies.c0_actor_policy import create_policy


def main() -> int:
    checkpoint = PROJECT_ROOT / "results/c0/task0_seed11_upright/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"run the C0 backward probe first: {checkpoint}")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    source_path = (
        PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial" / f"{task.name}_demo.hdf5"
    )
    with h5py.File(source_path, "r") as source:
        observations = source["data/demo_0/obs"]
        observation = {
            "agentview_image": observations["agentview_rgb"][0],
            "robot0_eef_pos": observations["ee_pos"][0],
            "robot0_gripper_qpos": observations["gripper_states"][0],
            "robot0_joint_pos": observations["joint_states"][0],
        }
    policy = create_policy(checkpoint=str(checkpoint), device="cuda")
    first_action = policy.act(observation, task.language)
    second_action = policy.act(observation, task.language)
    assert first_action.shape == (7,)
    assert second_action.shape == (7,)
    assert np.isfinite(first_action).all() and np.isfinite(second_action).all()
    assert len(policy._chunk) == 8
    print(
        {
            "first_action": first_action.round(6).tolist(),
            "second_action": second_action.round(6).tolist(),
            "action_chunk_horizon": len(policy._chunk),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
