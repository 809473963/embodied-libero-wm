"""Constrained probes for counterfactual physical-to-visual sufficiency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class FrozenContextExtractor(nn.Module):
    """Expose the unchanged WM-v2B pre-action hidden representation.

    The wrapped modules consume only the current visual latent, normalized
    current proprioception, and language.  No action or future value enters
    this computation.
    """

    def __init__(self, world_model: nn.Module) -> None:
        super().__init__()
        self.z_projection = world_model.z_projection
        self.proprio_projection = world_model.proprio_projection
        self.language_projection = world_model.language_projection
        self.initial_hidden = world_model.initial_hidden

    def forward(
        self,
        z0: torch.Tensor,
        proprio0: torch.Tensor,
        language: torch.Tensor,
    ) -> torch.Tensor:
        if z0.ndim != 2 or z0.shape[-1] != 512:
            raise ValueError(f"expected z0 [B,512], got {tuple(z0.shape)}")
        if proprio0.shape != (z0.shape[0], 12):
            raise ValueError(f"expected proprio0 [B,12], got {tuple(proprio0.shape)}")
        if language.shape != (z0.shape[0], 512):
            raise ValueError(f"expected language [B,512], got {tuple(language.shape)}")
        language_projection = self.language_projection(language)
        fused = torch.cat(
            (
                self.z_projection(z0),
                self.proprio_projection(proprio0),
                language_projection,
            ),
            dim=-1,
        )
        return self.initial_hidden(fused)


class ContextConditionedPhysicalVisualProbe(nn.Module):
    """Low-rank scene-gated map from a 10-D physical delta to a visual delta."""

    def __init__(
        self,
        physical_dim: int = 10,
        context_dim: int = 256,
        hidden_dim: int = 64,
        visual_dim: int = 512,
    ) -> None:
        super().__init__()
        self.physical = nn.Linear(physical_dim, hidden_dim, bias=False)
        self.context_gate = nn.Linear(context_dim, hidden_dim, bias=True)
        self.visual = nn.Linear(hidden_dim, visual_dim, bias=False)

    def forward(
        self, context: torch.Tensor, physical_delta: torch.Tensor
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != 256:
            raise ValueError(f"expected context [B,256], got {tuple(context.shape)}")
        if physical_delta.shape != (context.shape[0], 10):
            raise ValueError(
                f"expected physical_delta [B,10], got {tuple(physical_delta.shape)}"
            )
        physical_feature = self.physical(physical_delta)
        context_gate = torch.sigmoid(self.context_gate(context))
        return self.visual(physical_feature * context_gate)


@dataclass(frozen=True)
class RidgeFit:
    weight: np.ndarray
    bias: np.ndarray
    input_mean: np.ndarray


def fit_ridge(
    inputs: np.ndarray,
    targets: np.ndarray,
    regularization: float,
    *,
    bias: bool,
) -> RidgeFit:
    """Fit mean-loss ridge regression, optionally with an unpenalized bias."""

    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("ridge inputs/targets must be aligned rank-2 arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError("ridge inputs contain NaN/Inf")
    if regularization <= 0:
        raise ValueError("ridge regularization must be positive")

    if bias:
        input_mean = x.mean(axis=0)
        target_mean = y.mean(axis=0)
        centered_x = x - input_mean
        centered_y = y - target_mean
    else:
        input_mean = np.zeros(x.shape[1], dtype=np.float64)
        target_mean = np.zeros(y.shape[1], dtype=np.float64)
        centered_x = x
        centered_y = y
    gram = centered_x.T @ centered_x / len(centered_x)
    cross = centered_x.T @ centered_y / len(centered_x)
    weight = np.linalg.solve(
        gram + regularization * np.eye(x.shape[1], dtype=np.float64), cross
    )
    fitted_bias = target_mean - input_mean @ weight if bias else target_mean
    if not np.isfinite(weight).all() or not np.isfinite(fitted_bias).all():
        raise FloatingPointError("ridge fit contains NaN/Inf")
    return RidgeFit(weight=weight, bias=fitted_bias, input_mean=input_mean)


def predict_ridge(inputs: np.ndarray, fit: RidgeFit) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float64) @ fit.weight + fit.bias
    if not np.isfinite(values).all():
        raise FloatingPointError("ridge prediction contains NaN/Inf")
    return values
