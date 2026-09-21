#!/usr/bin/env python3
"""Shape, index, split, boundary, and oracle tests for D13 data only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark
from src.d13 import WorldModelDataset


STORE = PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
SOURCE_DIR = PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial"
HORIZON = 8


def direct_back_check(dataset: WorldModelDataset, store: h5py.File, task_id: int, demo_id: int, k: int) -> None:
    sample = dataset[dataset.find_index(task_id, demo_id, k)]
    group = store[f"tasks/task_{task_id}/demos/demo_{demo_id}"]
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    source_path = SOURCE_DIR / f"{suite.get_task(task_id).name}_demo.hdf5"
    with h5py.File(source_path, "r") as source:
        source_actions = source[f"data/demo_{demo_id}/actions"][k + 1 : k + HORIZON + 1].astype(np.float32)
    assert torch.equal(sample["z0"], torch.from_numpy(group["vision"][k].astype(np.float32)))
    assert np.array_equal(sample["actions"][0].numpy(), source_actions[0])  # action[k+1]
    assert np.array_equal(sample["actions"][7].numpy(), source_actions[7])  # action[k+8]
    assert torch.equal(sample["z_future"][0], torch.from_numpy(group["vision"][k + 1].astype(np.float32)))
    assert torch.equal(sample["z_future"][7], torch.from_numpy(group["vision"][k + 8].astype(np.float32)))
    assert sample["task_id"].item() == task_id
    assert sample["demo_id"].item() == demo_id
    assert sample["start_timestep"].item() == k


def main() -> int:
    if not STORE.is_file():
        raise FileNotFoundError(f"run scripts/extract_world_model_features.py first: {STORE}")
    train = WorldModelDataset(STORE, split="train", horizon=HORIZON)
    validation = WorldModelDataset(STORE, split="validation", horizon=HORIZON)
    try:
        sample = train[0]
        assert sample["z0"].shape == (512,)
        assert sample["proprio0"].shape == (12,)
        assert sample["language"].shape == (512,)
        assert sample["actions"].shape == (8, 7)
        assert sample["z_future"].shape == (8, 512)
        assert all(value.dtype == torch.float32 for key, value in sample.items() if key in {"z0", "proprio0", "language", "actions", "z_future"})
        batch = next(iter(DataLoader(train, batch_size=4, shuffle=False)))
        assert batch["z0"].shape == (4, 512)
        assert batch["proprio0"].shape == (4, 12)
        assert batch["language"].shape == (4, 512)
        assert batch["actions"].shape == (4, 8, 7)
        assert batch["z_future"].shape == (4, 8, 512)
        train_episodes = {(record.task_id, record.demo_id) for record in train.records}
        validation_episodes = {(record.task_id, record.demo_id) for record in validation.records}
        assert not (train_episodes & validation_episodes)
        assert all(demo_id <= 44 for _, demo_id in train_episodes)
        assert all(demo_id >= 45 for _, demo_id in validation_episodes)
        with h5py.File(STORE, "r") as store:
            # Multiple tasks, demos, and timesteps; all reads are group-local.
            direct_back_check(train, store, 0, 0, 0)
            direct_back_check(train, store, 4, 17, 5)
            direct_back_check(validation, store, 6, 47, 5)
            group = store["tasks/task_8/demos/demo_44"]
            last_k = int(group.attrs["observation_count"]) - HORIZON - 1
            direct_back_check(train, store, 8, 44, last_k)
            last = train[train.find_index(8, 44, last_k)]
            assert last["actions"].shape == (HORIZON, 7)
            assert last["z_future"].shape == (HORIZON, 512)
            # A D13 group has only policy-visible trajectory fields.
            forbidden = {"states", "robot_states", "rewards", "dones", "success"}
            leaked = []
            store.visititems(lambda name, _: leaked.append(name) if name.rsplit("/", 1)[-1] in forbidden else None)
            assert not leaked, leaked
        copy_pred = WorldModelDataset.copy_current_state_prediction(sample["z0"])
        assert copy_pred.shape == (HORIZON, 512)
        assert torch.equal(copy_pred, sample["z0"].unsqueeze(0).repeat(HORIZON, 1))
        print({"train_samples": len(train), "validation_samples": len(validation), "batch_z_future": list(batch["z_future"].shape), "index_checks": 4})
    finally:
        train.close()
        validation.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
