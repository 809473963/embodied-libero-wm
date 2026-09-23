"""C0-v2 dataset with the isolated causal action-label offset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


TRAIN_DEMO_IDS = frozenset(range(45))
VALIDATION_DEMO_IDS = frozenset(range(45, 50))
CONTRACT_VERSION = "c0-v2-causal-alignment-v1"
HORIZON = 8


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CausalActorIndex:
    task_id: int
    demo_id: int
    start_timestep: int
    observation_count: int


class CausalActorDataset(Dataset[dict[str, torch.Tensor]]):
    """Yield ``obs[k] -> actions[k+1:k+9]`` without crossing episodes."""

    def __init__(self, path: str | Path, split: str, in_memory: bool = False) -> None:
        self.path = Path(path)
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.split = split
        self._file: h5py.File | None = None
        self._cached: dict[str, np.ndarray] | None = None
        selected = TRAIN_DEMO_IDS if split == "train" else VALIDATION_DEMO_IDS
        records: list[CausalActorIndex] = []
        with h5py.File(self.path, "r") as store:
            if str(store.attrs.get("world_model_data_contract")) != "d13-v1":
                raise ValueError("C0-v2 requires the frozen D13 observation-aligned store")
            if int(store.attrs.get("horizon", -1)) != HORIZON:
                raise ValueError("C0-v2 freezes H=8")
            language = store["task_metadata/language_features"][:].astype(np.float32)
            if language.shape != (10, 512):
                raise ValueError(f"unexpected language shape {language.shape}")
            self.language_features = torch.from_numpy(language)
            for task_group in store["tasks"].values():
                task_id = int(task_group.attrs["task_id"])
                for demo_group in task_group["demos"].values():
                    demo_id = int(demo_group.attrs["demo_id"])
                    if demo_id not in selected:
                        continue
                    count = int(demo_group.attrs["observation_count"])
                    expected = {
                        "vision": (count, 512),
                        "proprio": (count, 12),
                        "actions": (count, 7),
                        "timestep": (count,),
                    }
                    for name, shape in expected.items():
                        if demo_group[name].shape != shape:
                            raise ValueError(
                                f"episode ({task_id},{demo_id}) {name} shape differs: "
                                f"{demo_group[name].shape} != {shape}"
                            )
                    # Legal k is 0 <= k <= T-9, yielding T-8 starts.
                    for k in range(count - HORIZON):
                        records.append(CausalActorIndex(task_id, demo_id, k, count))
            self.records = tuple(records)
            if not self.records:
                raise ValueError(f"no C0-v2 {split} samples")
            if in_memory:
                image = np.empty((len(records), 512), dtype=np.float32)
                proprio = np.empty((len(records), 12), dtype=np.float32)
                action = np.empty((len(records), HORIZON, 7), dtype=np.float32)
                task_ids = np.empty(len(records), dtype=np.int64)
                demo_ids = np.empty(len(records), dtype=np.int64)
                timesteps = np.empty(len(records), dtype=np.int64)
                for index, record in enumerate(records):
                    group = store[
                        f"tasks/task_{record.task_id}/demos/demo_{record.demo_id}"
                    ]
                    k = record.start_timestep
                    image[index] = group["vision"][k].astype(np.float32)
                    proprio[index] = group["proprio"][k].astype(np.float32)
                    action[index] = group["actions"][k + 1 : k + HORIZON + 1].astype(
                        np.float32
                    )
                    task_ids[index] = record.task_id
                    demo_ids[index] = record.demo_id
                    timesteps[index] = k
                self._cached = {
                    "image_features": image,
                    "proprio_raw": proprio,
                    "action_chunks": action,
                    "task_id": task_ids,
                    "demo_id": demo_ids,
                    "timestep": timesteps,
                }

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.records)

    def find_index(self, task_id: int, demo_id: int, start_timestep: int) -> int:
        target = (task_id, demo_id, start_timestep)
        for index, record in enumerate(self.records):
            if (record.task_id, record.demo_id, record.start_timestep) == target:
                return index
        raise KeyError(target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if self._cached is None:
            group = self._handle()[
                f"tasks/task_{record.task_id}/demos/demo_{record.demo_id}"
            ]
            k = record.start_timestep
            image = group["vision"][k].astype(np.float32)
            proprio = group["proprio"][k].astype(np.float32)
            actions = group["actions"][k + 1 : k + HORIZON + 1].astype(np.float32)
            task_id = record.task_id
            demo_id = record.demo_id
            timestep = k
        else:
            image = self._cached["image_features"][index]
            proprio = self._cached["proprio_raw"][index]
            actions = self._cached["action_chunks"][index]
            task_id = int(self._cached["task_id"][index])
            demo_id = int(self._cached["demo_id"][index])
            timestep = int(self._cached["timestep"][index])
        if actions.shape != (HORIZON, 7):
            raise RuntimeError(f"record {record} crossed an episode boundary")
        return {
            "image_features": torch.from_numpy(image),
            "language_features": self.language_features[task_id],
            "proprio_raw": torch.from_numpy(proprio),
            "action_chunks": torch.from_numpy(actions),
            "task_id": torch.tensor(task_id, dtype=torch.int64),
            "demo_id": torch.tensor(demo_id, dtype=torch.int64),
            "start_timestep": torch.tensor(timestep, dtype=torch.int64),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()


class _Moments:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_square = np.zeros(dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("normalization input has invalid shape or non-finite values")
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.total_square += np.square(values).sum(axis=0)

    def result(self, minimum_std: float) -> dict[str, Any]:
        mean = self.total / self.count
        raw_std = np.sqrt(
            np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        )
        return {
            "mean": mean.tolist(),
            "std": np.maximum(raw_std, minimum_std).tolist(),
            "raw_std": raw_std.tolist(),
            "count": self.count,
        }


def compute_causal_actor_normalization(
    store_path: str | Path,
    feature_manifest_path: str | Path,
    minimum_std: float = 1.0e-6,
) -> dict[str, Any]:
    """Compute train-only unique-row statistics, never overlapping windows."""

    store_path = Path(store_path)
    manifest_path = Path(feature_manifest_path)
    proprio = _Moments(12)
    actions = _Moments(7)
    episode_count = 0
    with h5py.File(store_path, "r") as store:
        if str(store.attrs.get("world_model_data_contract")) != "d13-v1":
            raise ValueError("normalization requires the frozen D13 store")
        for task_group in store["tasks"].values():
            for demo_group in task_group["demos"].values():
                if int(demo_group.attrs["demo_id"]) not in TRAIN_DEMO_IDS:
                    continue
                proprio.update(demo_group["proprio"][:])
                actions.update(demo_group["actions"][1:])
                episode_count += 1
    if episode_count != 450:
        raise ValueError(f"expected 450 train episodes, got {episode_count}")
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "store_sha256": file_sha256(store_path),
        "feature_manifest_sha256": file_sha256(manifest_path),
        "train_episode_count": episode_count,
        "minimum_std": minimum_std,
        "proprio": {
            **proprio.result(minimum_std),
            "rule": "train demos 0-44; every episode proprio observation row exactly once; population mean/std",
        },
        "action": {
            **actions.result(minimum_std),
            "rule": "train demos 0-44; every episode actions[1:] row exactly once; action[0] excluded; population mean/std",
        },
    }
