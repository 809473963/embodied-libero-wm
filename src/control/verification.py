"""Pure prediction-verification decision used by C2."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def verification_decision(
    predicted_latent: torch.Tensor,
    actual_latent: torch.Tensor,
    uncertainty: float,
    mismatch_threshold: float,
    uncertainty_threshold: float,
    remaining_actions: int,
) -> dict[str, Any]:
    if predicted_latent.shape != (1, 512) or actual_latent.shape != (1, 512):
        raise ValueError("predicted and actual latent must be [1,512]")
    if remaining_actions < 0 or remaining_actions > 6:
        raise ValueError("remaining_actions must be between zero and six at a C2 check")
    mismatch = float(
        1.0
        - F.cosine_similarity(
            predicted_latent.float(), actual_latent.float(), dim=-1
        ).item()
    )
    values = torch.tensor(
        [mismatch, uncertainty, mismatch_threshold, uncertainty_threshold],
        dtype=torch.float64,
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("verification value is NaN/Inf")
    mismatch_exceeded = mismatch > float(mismatch_threshold)
    uncertainty_exceeded = float(uncertainty) > float(uncertainty_threshold)
    threshold_exceeded = mismatch_exceeded or uncertainty_exceeded
    return {
        "mismatch": mismatch,
        "uncertainty": float(uncertainty),
        "mismatch_threshold": float(mismatch_threshold),
        "uncertainty_threshold": float(uncertainty_threshold),
        "mismatch_exceeded": mismatch_exceeded,
        "uncertainty_exceeded": uncertainty_exceeded,
        "threshold_exceeded": threshold_exceeded,
        "remaining_actions": remaining_actions,
        "early_replan": threshold_exceeded and remaining_actions > 0,
    }
