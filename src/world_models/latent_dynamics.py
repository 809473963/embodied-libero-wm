"""D14 single-model latent forward predictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
    )


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return one cosine-distance value for every batch/horizon pair."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"prediction and target must share [B,H,D], got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    # Compute the primary metric in float32 even when the forward pass uses AMP.
    return 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)


class LatentDynamicsModel(nn.Module):
    """Predict eight future CLIP latents from one observation and causal actions."""

    def __init__(
        self,
        latent_dim: int = 512,
        proprio_dim: int = 12,
        action_dim: int = 7,
        horizon: int = 8,
        z_projection_dim: int = 256,
        proprio_projection_dim: int = 64,
        language_projection_dim: int = 128,
        action_projection_dim: int = 64,
        gru_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if horizon != 8:
            raise ValueError("D13-v1/D14 freeze horizon=8")
        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.z_projection = _projection(latent_dim, z_projection_dim)
        self.proprio_projection = _projection(proprio_dim, proprio_projection_dim)
        self.language_projection = _projection(latent_dim, language_projection_dim)
        self.initial_hidden = nn.Sequential(
            nn.Linear(
                z_projection_dim + proprio_projection_dim + language_projection_dim,
                gru_hidden_dim,
            ),
            nn.LayerNorm(gru_hidden_dim),
            nn.GELU(),
        )
        self.action_projection = _projection(action_dim, action_projection_dim)
        self.gru = nn.GRU(
            input_size=action_projection_dim + language_projection_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.delta_head = nn.Linear(gru_hidden_dim, latent_dim)
        # Start close to the copy-current diagnostic while retaining gradients
        # through the recurrent path from the first optimizer step.
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        z0: torch.Tensor,
        proprio0: torch.Tensor,
        language: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
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
                f"expected actions [B,{self.horizon},{self.action_dim}], got {tuple(actions.shape)}"
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
        return F.normalize(z0.unsqueeze(1) + delta_z, p=2, dim=-1, eps=1.0e-8)
