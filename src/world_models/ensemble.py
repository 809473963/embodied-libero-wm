"""D15 ensemble prediction and epistemic-disagreement definition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .latent_dynamics import LatentDynamicsModel


def ensemble_statistics(predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate member latents and calculate mean pairwise cosine distance.

    Args:
        predictions: L2-normalized member predictions shaped ``[M,B,H,D]``.

    Returns:
        ``(mean_prediction [B,H,D], uncertainty [B,H])``. Uncertainty is
        epistemic disagreement, not predictive error or a calibrated failure
        probability.
    """

    if predictions.ndim != 4 or predictions.shape[0] < 2:
        raise ValueError(f"expected [M>=2,B,H,D], got {tuple(predictions.shape)}")
    if not torch.isfinite(predictions).all():
        raise ValueError("ensemble predictions contain NaN/Inf")
    mean_prediction = F.normalize(predictions.mean(dim=0), p=2, dim=-1, eps=1.0e-8)
    pairwise_distances: list[torch.Tensor] = []
    for first in range(predictions.shape[0]):
        for second in range(first + 1, predictions.shape[0]):
            pairwise_distances.append(
                1.0
                - F.cosine_similarity(
                    predictions[first].float(), predictions[second].float(), dim=-1
                )
            )
    uncertainty = torch.stack(pairwise_distances, dim=0).mean(dim=0).clamp_min(0.0)
    return mean_prediction, uncertainty


class LatentDynamicsEnsemble(nn.Module):
    """A fixed-size collection of independently trained D14 predictors."""

    def __init__(self, members: Sequence[LatentDynamicsModel]) -> None:
        super().__init__()
        if len(members) not in {2, 3}:
            raise ValueError("D15-v1 requires two or three ensemble members")
        self.members = nn.ModuleList(members)

    def forward(
        self,
        z0: torch.Tensor,
        proprio0: torch.Tensor,
        language: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [member(z0, proprio0, language, actions) for member in self.members], dim=0
        )

    def predict(
        self,
        z0: torch.Tensor,
        proprio0: torch.Tensor,
        language: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        member_predictions = self(z0, proprio0, language, actions)
        mean_prediction, uncertainty = ensemble_statistics(member_predictions)
        return mean_prediction, uncertainty, member_predictions
