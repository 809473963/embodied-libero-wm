"""WM-v2B future-proprio auxiliary dataset, model, and losses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.d13 import WorldModelDataset

from .latent_dynamics import LatentDynamicsModel


PROPRIO_GROUPS = {
    "ee_pos": slice(0, 3),
    "gripper": slice(3, 5),
    "joint": slice(5, 12),
}


class FutureProprioWorldModelDataset(WorldModelDataset):
    """Add raw future proprio targets without changing the frozen D13 inputs."""

    def __init__(self, path: str | Path, split: str, horizon: int = 8) -> None:
        super().__init__(path, split, horizon)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        record = self.records[index]
        group = self._handle()[
            f"tasks/task_{record.task_id}/demos/demo_{record.demo_id}"
        ]
        k = record.start_timestep
        stop = k + self.horizon + 1
        proprio_future = group["proprio"][k + 1 : stop].astype(np.float32)
        if proprio_future.shape != (self.horizon, 12):
            raise RuntimeError(f"record {record} crossed an episode boundary")
        sample["proprio_future"] = torch.from_numpy(proprio_future)
        return sample


def standardized_proprio_delta_target(
    proprio0: torch.Tensor,
    proprio_future: torch.Tensor,
    normalization: dict[str, Any],
) -> torch.Tensor:
    """Return `(p_future-mean)/std - (p0-mean)/std` as [B,H,12]."""

    if proprio0.ndim != 2 or proprio0.shape[-1] != 12:
        raise ValueError(f"expected proprio0 [B,12], got {tuple(proprio0.shape)}")
    if proprio_future.shape != (proprio0.shape[0], 8, 12):
        raise ValueError(
            f"expected proprio_future [B,8,12], got {tuple(proprio_future.shape)}"
        )
    mean = torch.tensor(
        normalization["proprio"]["mean"],
        dtype=torch.float32,
        device=proprio0.device,
    ).view(1, 1, 12)
    std = torch.tensor(
        normalization["proprio"]["std"],
        dtype=torch.float32,
        device=proprio0.device,
    ).view(1, 1, 12)
    p0_normalized = (proprio0.float().unsqueeze(1) - mean) / std
    future_normalized = (proprio_future.float() - mean) / std
    return future_normalized - p0_normalized


def proprio_delta_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Return SmoothL1 future-proprio-delta error [B,H], averaged over 12-D."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [B,H,12]")
    if prediction.shape[-2:] != (8, 12):
        raise ValueError(f"expected [B,8,12], got {tuple(prediction.shape)}")
    return F.smooth_l1_loss(
        prediction.float(), target.float(), beta=beta, reduction="none"
    ).mean(dim=-1)


class LatentDynamicsProprioAuxModel(LatentDynamicsModel):
    """Exact D14 backbone plus a small head on the shared GRU output."""

    def __init__(self, proprio_head_hidden_dim: int = 128, **kwargs: int) -> None:
        super().__init__(**kwargs)
        gru_hidden_dim = int(kwargs.get("gru_hidden_dim", 256))
        self.proprio_head_hidden_dim = int(proprio_head_hidden_dim)
        self.proprio_delta_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, self.proprio_head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.proprio_head_hidden_dim, self.proprio_dim),
        )

    def forward(
        self,
        z0: torch.Tensor,
        proprio0: torch.Tensor,
        language: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z0.ndim != 2 or z0.shape[-1] != self.latent_dim:
            raise ValueError(f"expected z0 [B,{self.latent_dim}], got {tuple(z0.shape)}")
        batch_size = z0.shape[0]
        if proprio0.shape != (batch_size, self.proprio_dim):
            raise ValueError(
                f"expected proprio0 [B,{self.proprio_dim}], got {tuple(proprio0.shape)}"
            )
        if language.shape != (batch_size, self.latent_dim):
            raise ValueError(
                f"expected language [B,{self.latent_dim}], got {tuple(language.shape)}"
            )
        if actions.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError(
                f"expected actions [B,{self.horizon},{self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )

        language_projection = self.language_projection(language)
        fused_state = torch.cat(
            (
                self.z_projection(z0),
                self.proprio_projection(proprio0),
                language_projection,
            ),
            dim=-1,
        )
        initial_hidden = self.initial_hidden(fused_state).unsqueeze(0)
        recurrent_input = torch.cat(
            (
                self.action_projection(actions),
                language_projection.unsqueeze(1).expand(-1, self.horizon, -1),
            ),
            dim=-1,
        )
        recurrent_output, _ = self.gru(recurrent_input, initial_hidden)
        delta_z = self.delta_head(recurrent_output)
        pred_z = F.normalize(
            z0.unsqueeze(1) + delta_z, p=2, dim=-1, eps=1.0e-8
        )
        pred_p_delta = self.proprio_delta_head(recurrent_output)
        return pred_z, pred_p_delta

    def shared_gru_parameters(self) -> tuple[nn.Parameter, ...]:
        """The exact shared parameter set used by gradient-scale calibration."""

        return tuple(self.gru.parameters())
