#!/usr/bin/env python3
"""Shape, count, causal-index, boundary, and normalization tests for C0-v2."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark
from src.c0 import CausalActorDataset, compute_causal_actor_normalization


STORE = ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
MANIFEST = ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/feature_manifest.json"
SOURCE = ROOT / "data/LIBERO/datasets/libero_spatial"


def back_check(dataset: CausalActorDataset, task_id: int, demo_id: int, k: int) -> None:
    sample = dataset[dataset.find_index(task_id, demo_id, k)]
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    path = SOURCE / f"{suite.get_task(task_id).name}_demo.hdf5"
    with h5py.File(path, "r") as source, h5py.File(STORE, "r") as store:
        demo = source[f"data/demo_{demo_id}"]
        group = store[f"tasks/task_{task_id}/demos/demo_{demo_id}"]
        assert torch.equal(
            sample["image_features"], torch.from_numpy(group["vision"][k].astype(np.float32))
        )
        assert torch.equal(
            sample["proprio_raw"], torch.from_numpy(group["proprio"][k].astype(np.float32))
        )
        assert np.array_equal(
            sample["action_chunks"][0].numpy(),
            np.asarray(demo["actions"][k + 1], dtype=np.float32),
        )
        assert np.array_equal(
            sample["action_chunks"][7].numpy(),
            np.asarray(demo["actions"][k + 8], dtype=np.float32),
        )
        assert sample["task_id"].item() == task_id
        assert sample["demo_id"].item() == demo_id
        assert sample["start_timestep"].item() == k


def main() -> None:
    train = CausalActorDataset(STORE, "train")
    validation = CausalActorDataset(STORE, "validation")
    try:
        assert len(train) == 52381
        assert len(validation) == 5869
        sample = train[0]
        assert sample["image_features"].shape == (512,)
        assert sample["language_features"].shape == (512,)
        assert sample["proprio_raw"].shape == (12,)
        assert sample["action_chunks"].shape == (8, 7)
        batch = next(iter(DataLoader(train, batch_size=4, shuffle=False)))
        assert batch["image_features"].shape == (4, 512)
        assert batch["proprio_raw"].shape == (4, 12)
        assert batch["action_chunks"].shape == (4, 8, 7)

        back_check(train, 0, 0, 0)
        back_check(train, 4, 17, 5)
        back_check(validation, 6, 47, 5)
        with h5py.File(STORE, "r") as store:
            count = int(store["tasks/task_8/demos/demo_44"].attrs["observation_count"])
        last_k = count - 9
        back_check(train, 8, 44, last_k)
        last_record = train.records[train.find_index(8, 44, last_k)]
        assert last_record.start_timestep + 8 == last_record.observation_count - 1

        train_episodes = {(record.task_id, record.demo_id) for record in train.records}
        validation_episodes = {(record.task_id, record.demo_id) for record in validation.records}
        assert train_episodes.isdisjoint(validation_episodes)
        normalization = compute_causal_actor_normalization(STORE, MANIFEST)
        assert normalization["proprio"]["count"] == 55981
        assert normalization["action"]["count"] == 55531
        assert "actions[1:]" in normalization["action"]["rule"]
        print(
            {
                "status": "PASS",
                "train_samples": len(train),
                "validation_samples": len(validation),
                "index_back_checks": 4,
                "proprio_count": normalization["proprio"]["count"],
                "action_count": normalization["action"]["count"],
            }
        )
    finally:
        train.close()
        validation.close()


if __name__ == "__main__":
    main()
