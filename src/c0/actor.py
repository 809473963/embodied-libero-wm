"""Small action-chunk actor operating on frozen CLIP features."""

from __future__ import annotations

import torch
from torch import nn


class C0Actor(nn.Module):
    """Fuse frozen vision/language features and robot proprioception.

    The module receives already frozen 512-D CLIP features. It intentionally
    contains no object state, reward, success signal, or future observation.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        proprio_dim: int = 15,
        fusion_dim: int = 256,
        action_horizon: int = 8,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.vision_encoder = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU()
        )
        self.language_encoder = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU()
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
        )
        fused_dim = fusion_dim * 2 + fusion_dim // 2
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, action_horizon * action_dim),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        language_features: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        if image_features.ndim != 2 or image_features.shape[-1] != self.vision_encoder[0].in_features:
            raise ValueError(f"unexpected image feature shape: {tuple(image_features.shape)}")
        if language_features.shape != image_features.shape:
            raise ValueError("image and language feature shapes must match")
        if proprioception.ndim != 2 or proprioception.shape[0] != image_features.shape[0]:
            raise ValueError("proprioception batch shape must match image features")
        fused = torch.cat(
            (
                self.vision_encoder(image_features),
                self.language_encoder(language_features),
                self.proprio_encoder(proprioception),
            ),
            dim=-1,
        )
        return self.action_head(fused).view(
            image_features.shape[0], self.action_horizon, self.action_dim
        )
