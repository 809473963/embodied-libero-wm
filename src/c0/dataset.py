"""Policy-safe PyTorch access to the frozen C0 HDF5 feature store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazily open an HDF5 split so DataLoader workers do not share a handle."""

    def __init__(
        self, path: str | Path, task_id: int | None = None, in_memory: bool = False
    ) -> None:
        self.path = Path(path)
        self._file: h5py.File | None = None
        self._cached: dict[str, np.ndarray] | None = None
        self.task_id_filter = task_id
        with h5py.File(self.path, "r") as feature_file:
            required = {
                "image_features",
                "proprio_raw",
                "action_chunks",
                "task_id",
                "demo_id",
                "timestep",
                "task_metadata",
            }
            missing = required - set(feature_file.keys())
            if missing:
                raise ValueError(f"{self.path}: missing fields {sorted(missing)}")
            all_task_ids = feature_file["task_id"][:]
            if task_id is None:
                self.indices = np.arange(len(all_task_ids), dtype=np.int64)
            else:
                self.indices = np.flatnonzero(all_task_ids == task_id).astype(np.int64)
                if not len(self.indices):
                    raise ValueError(f"{self.path}: no rows for task_id={task_id}")
            self.length = len(self.indices)
            self.horizon = int(feature_file["action_chunks"].shape[1])
            self.action_dim = int(feature_file["action_chunks"].shape[2])
            self.proprio_dim = int(feature_file["proprio_raw"].shape[1])
            self.language_features = torch.from_numpy(
                feature_file["task_metadata/language_features"][:].astype(np.float32)
            )
            if in_memory:
                self._cached = {
                    key: (
                        feature_file[key][:]
                        if task_id is None
                        else feature_file[key][self.indices]
                    )
                    for key in ("image_features", "proprio_raw", "action_chunks", "task_id")
                }

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        source_index = int(self.indices[index])
        if self._cached is None:
            feature_file = self._handle()
            image_features = feature_file["image_features"][source_index]
            proprio_raw = feature_file["proprio_raw"][source_index]
            action_chunks = feature_file["action_chunks"][source_index]
            task_id = int(feature_file["task_id"][source_index])
        else:
            image_features = self._cached["image_features"][index]
            proprio_raw = self._cached["proprio_raw"][index]
            action_chunks = self._cached["action_chunks"][index]
            task_id = int(self._cached["task_id"][index])
        return {
            "image_features": torch.from_numpy(image_features.astype(np.float32)),
            "language_features": self.language_features[task_id],
            "proprio_raw": torch.from_numpy(proprio_raw.astype(np.float32)),
            "action_chunks": torch.from_numpy(action_chunks.astype(np.float32)),
            "task_id": torch.tensor(task_id, dtype=torch.int64),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()


def _selected_values(
    data: h5py.Dataset, indices: np.ndarray | None, chunk_rows: int
) -> list[np.ndarray]:
    if indices is not None:
        # FeatureDataset keeps indices sorted, which h5py requires for fancy
        # indexing. A single selected task is only about 5k rows.
        return [data[indices]]
    return [data[start : start + chunk_rows] for start in range(0, data.shape[0], chunk_rows)]


def compute_proprio_stats(
    path: str | Path, indices: np.ndarray | None = None, chunk_rows: int = 4096
) -> dict[str, list[float]]:
    """Compute proprio normalization from selected training rows only."""

    source_path = Path(path)
    with h5py.File(source_path, "r") as feature_file:
        data = feature_file["proprio_raw"]
        count = 0
        total = np.zeros(data.shape[1], dtype=np.float64)
        total_squared = np.zeros(data.shape[1], dtype=np.float64)
        for rows in _selected_values(data, indices, chunk_rows):
            values = rows.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{source_path}: non-finite proprioception")
            count += values.shape[0]
            total += values.sum(axis=0)
            total_squared += np.square(values).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}


def compute_action_stats(
    path: str | Path, indices: np.ndarray | None = None, chunk_rows: int = 4096
) -> dict[str, list[float]]:
    """Compute action normalization from selected training action chunks only."""

    source_path = Path(path)
    with h5py.File(source_path, "r") as feature_file:
        data = feature_file["action_chunks"]
        count = 0
        total = np.zeros(data.shape[-1], dtype=np.float64)
        total_squared = np.zeros(data.shape[-1], dtype=np.float64)
        for rows in _selected_values(data, indices, chunk_rows):
            values = rows.astype(np.float64).reshape(-1, data.shape[-1])
            if not np.isfinite(values).all():
                raise ValueError(f"{source_path}: non-finite action")
            count += values.shape[0]
            total += values.sum(axis=0)
            total_squared += np.square(values).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}
