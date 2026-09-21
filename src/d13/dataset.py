"""Observation-aligned D13 world-model Dataset without any dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


TRAIN_DEMO_IDS = frozenset(range(45))
VALIDATION_DEMO_IDS = frozenset(range(45, 50))
CONTRACT_VERSION = "d13-v1"


@dataclass(frozen=True)
class WorldModelIndex:
    """One legal observation-dynamics start within exactly one episode."""

    task_id: int
    demo_id: int
    start_timestep: int
    observation_count: int


class WorldModelDataset(Dataset[dict[str, torch.Tensor]]):
    """Yield `(z[k], p[k], language, a[k+1:k+H+1]) -> z[k+1:k+H+1]`.

    The index is constructed per `(task, demo)` group, so future observations
    can never cross an episode boundary or be inferred from global row order.
    """

    def __init__(self, path: str | Path, split: str, horizon: int = 8) -> None:
        self.path = Path(path)
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if horizon != 8:
            raise ValueError("D13-v1 freezes horizon=8")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.split = split
        self.horizon = horizon
        self._file: h5py.File | None = None
        self.records: tuple[WorldModelIndex, ...]
        self.language_features: torch.Tensor

        selected_demo_ids = TRAIN_DEMO_IDS if split == "train" else VALIDATION_DEMO_IDS
        records: list[WorldModelIndex] = []
        with h5py.File(self.path, "r") as store:
            if str(store.attrs.get("world_model_data_contract")) != CONTRACT_VERSION:
                raise ValueError(f"{self.path}: not a {CONTRACT_VERSION} store")
            if int(store.attrs.get("horizon", -1)) != horizon:
                raise ValueError(f"{self.path}: horizon differs from requested {horizon}")
            language = store["task_metadata/language_features"][:].astype(np.float32)
            if language.ndim != 2 or language.shape != (10, 512):
                raise ValueError(f"{self.path}: expected language_features [10,512]")
            self.language_features = torch.from_numpy(language)
            if "tasks" not in store:
                raise ValueError(f"{self.path}: missing tasks group")
            for task_group in store["tasks"].values():
                task_id = int(task_group.attrs["task_id"])
                if task_id < 0 or task_id >= len(language):
                    raise ValueError(f"{self.path}: invalid task_id={task_id}")
                for demo_group in task_group["demos"].values():
                    demo_id = int(demo_group.attrs["demo_id"])
                    if demo_id not in TRAIN_DEMO_IDS | VALIDATION_DEMO_IDS:
                        raise ValueError(f"{self.path}: invalid demo_id={demo_id}")
                    self._validate_episode(demo_group, task_id, demo_id)
                    if demo_id not in selected_demo_ids:
                        continue
                    count = int(demo_group.attrs["observation_count"])
                    # `k + horizon` is the final future target/action index.
                    for start_timestep in range(count - horizon):
                        records.append(WorldModelIndex(task_id, demo_id, start_timestep, count))
        if not records:
            raise ValueError(f"{self.path}: no {split} D13 samples")
        self.records = tuple(records)

    @staticmethod
    def _validate_episode(group: h5py.Group, task_id: int, demo_id: int) -> None:
        required = {"vision", "proprio", "actions", "timestep"}
        if set(group.keys()) != required:
            raise ValueError(f"episode ({task_id}, {demo_id}) has unexpected fields {sorted(group.keys())}")
        count = int(group.attrs["observation_count"])
        if count <= 8:
            raise ValueError(f"episode ({task_id}, {demo_id}) is too short for H=8")
        expected_shapes = {
            "vision": (count, 512),
            "proprio": (count, 12),
            "actions": (count, 7),
            "timestep": (count,),
        }
        for name, shape in expected_shapes.items():
            if group[name].shape != shape:
                raise ValueError(f"episode ({task_id}, {demo_id}) {name}: {group[name].shape} != {shape}")
        if not np.array_equal(group["timestep"][:], np.arange(count, dtype=np.int32)):
            raise ValueError(f"episode ({task_id}, {demo_id}) has noncanonical timestep values")

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

    def record_at(self, index: int) -> WorldModelIndex:
        return self.records[index]

    def find_index(self, task_id: int, demo_id: int, start_timestep: int) -> int:
        target = (task_id, demo_id, start_timestep)
        for index, record in enumerate(self.records):
            if (record.task_id, record.demo_id, record.start_timestep) == target:
                return index
        raise KeyError(f"no {self.split} record {target}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        group = self._handle()[f"tasks/task_{record.task_id}/demos/demo_{record.demo_id}"]
        k = record.start_timestep
        stop = k + self.horizon + 1
        # stop is exclusive, so slices below contain exactly H future entries.
        z0 = group["vision"][k].astype(np.float32)
        proprio0 = group["proprio"][k].astype(np.float32)
        actions = group["actions"][k + 1 : stop].astype(np.float32)
        z_future = group["vision"][k + 1 : stop].astype(np.float32)
        if actions.shape != (self.horizon, 7) or z_future.shape != (self.horizon, 512):
            raise RuntimeError(f"record {record} crossed an episode boundary")
        return {
            "z0": torch.from_numpy(z0),
            "proprio0": torch.from_numpy(proprio0),
            "language": self.language_features[record.task_id],
            "actions": torch.from_numpy(actions),
            "z_future": torch.from_numpy(z_future),
            "task_id": torch.tensor(record.task_id, dtype=torch.int64),
            "demo_id": torch.tensor(record.demo_id, dtype=torch.int64),
            "start_timestep": torch.tensor(k, dtype=torch.int64),
        }

    @staticmethod
    def copy_current_state_prediction(z0: torch.Tensor, horizon: int = 8) -> torch.Tensor:
        """The D16 comparison baseline, not a learned world-model prediction."""

        if horizon != 8:
            raise ValueError("D13-v1 freezes horizon=8")
        if z0.shape != (512,):
            raise ValueError(f"expected z0 [512], got {tuple(z0.shape)}")
        return z0.unsqueeze(0).repeat(horizon, 1)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
