"""Auditable train-only normalization for the D14 predictor."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from src.d13.dataset import CONTRACT_VERSION, TRAIN_DEMO_IDS


PROPRIO_RULE = (
    "train demos 0-44: every episode proprio observation row exactly once; "
    "population mean/std; no overlapping WM-window expansion"
)
ACTION_RULE = (
    "train demos 0-44: every episode actions[1:] row exactly once; action[0] excluded; "
    "population mean/std; no overlapping WM-window expansion"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Moments:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.count = 0
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_square = np.zeros(dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        if values64.ndim != 2 or values64.shape[1] != self.dimension:
            raise ValueError(f"expected [N,{self.dimension}], got {values64.shape}")
        if not np.isfinite(values64).all():
            raise ValueError("normalization source contains NaN/Inf")
        self.count += int(values64.shape[0])
        self.total += values64.sum(axis=0)
        self.total_square += np.square(values64).sum(axis=0)

    def result(self, minimum_std: float) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("cannot compute moments from zero rows")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        raw_std = np.sqrt(variance)
        std = np.maximum(raw_std, minimum_std)
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "raw_std": raw_std.tolist(),
            "count": self.count,
        }


def compute_world_model_normalization(
    store_path: str | Path,
    data_contract_path: str | Path,
    minimum_std: float = 1.0e-6,
) -> dict[str, Any]:
    """Compute statistics from unique train episode rows, never WM samples."""

    store_path = Path(store_path)
    data_contract_path = Path(data_contract_path)
    proprio = _Moments(12)
    actions = _Moments(7)
    episode_count = 0
    with h5py.File(store_path, "r") as store:
        if str(store.attrs.get("world_model_data_contract")) != CONTRACT_VERSION:
            raise ValueError(f"{store_path}: expected {CONTRACT_VERSION}")
        for task_group in store["tasks"].values():
            for demo_group in task_group["demos"].values():
                demo_id = int(demo_group.attrs["demo_id"])
                if demo_id not in TRAIN_DEMO_IDS:
                    continue
                proprio.update(demo_group["proprio"][:])
                actions.update(demo_group["actions"][1:])
                episode_count += 1
    result = {
        "schema_version": 1,
        "data_contract_version": CONTRACT_VERSION,
        "store_sha256": file_sha256(store_path),
        "data_contract_sha256": file_sha256(data_contract_path),
        "train_episode_count": episode_count,
        "minimum_std": float(minimum_std),
        "z0": {
            "rule": "frozen CLIP image latent is already L2-normalized; no standardization"
        },
        "language": {
            "rule": "frozen CLIP text latent is already L2-normalized; no standardization"
        },
        "proprio": {**proprio.result(minimum_std), "rule": PROPRIO_RULE},
        "action": {**actions.result(minimum_std), "rule": ACTION_RULE},
    }
    if result["train_episode_count"] != 450:
        raise ValueError(f"expected 450 train episodes, got {result['train_episode_count']}")
    return result


def normalization_tensors(
    normalization: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proprio_mean = torch.tensor(
        normalization["proprio"]["mean"], dtype=torch.float32, device=device
    ).view(1, -1)
    proprio_std = torch.tensor(
        normalization["proprio"]["std"], dtype=torch.float32, device=device
    ).view(1, -1)
    action_mean = torch.tensor(
        normalization["action"]["mean"], dtype=torch.float32, device=device
    ).view(1, 1, -1)
    action_std = torch.tensor(
        normalization["action"]["std"], dtype=torch.float32, device=device
    ).view(1, 1, -1)
    return proprio_mean, proprio_std, action_mean, action_std


def apply_world_model_normalization(
    proprio0: torch.Tensor,
    actions: torch.Tensor,
    normalization: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize only proprio/actions; z0/language remain untouched."""

    tensors = normalization_tensors(normalization, proprio0.device)
    proprio_mean, proprio_std, action_mean, action_std = tensors
    if actions.device != proprio0.device:
        raise ValueError("proprio0 and actions must be on the same device")
    return (proprio0 - proprio_mean) / proprio_std, (actions - action_mean) / action_std
