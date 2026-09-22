"""Pure D19 candidate generation and pre-execution scoring functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_antithetic_candidates(
    base_chunk: torch.Tensor,
    noise_std: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return base, +noise, and -noise action chunks, clipped to LIBERO range."""

    if base_chunk.shape != (8, 7):
        raise ValueError(f"expected base chunk [8,7], got {tuple(base_chunk.shape)}")
    if noise_std.shape != (7,) or torch.any(noise_std < 0):
        raise ValueError("noise_std must be nonnegative [7]")
    # Generate on CPU for device-independent deterministic candidate streams.
    noise = torch.randn((8, 7), generator=generator, dtype=torch.float32)
    noise = noise * noise_std.detach().float().cpu()
    # Fixed three-tap smoothing avoids high-frequency per-step jitter. Edge
    # values are replicated; the gripper remains unchanged because std[6]=0.
    padded = torch.cat((noise[:1], noise, noise[-1:]), dim=0)
    noise = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    noise = noise.to(device=base_chunk.device, dtype=base_chunk.dtype)
    return torch.stack(
        (
            base_chunk,
            torch.clamp(base_chunk + noise, -1.0, 1.0),
            torch.clamp(base_chunk - noise, -1.0, 1.0),
        ),
        dim=0,
    )


def score_candidates(
    z0: torch.Tensor,
    language: torch.Tensor,
    candidates: torch.Tensor,
    predictions: torch.Tensor,
    uncertainty: torch.Tensor,
    action_std: torch.Tensor,
    goal_progress_weight: float = 1.0,
    uncertainty_penalty_weight: float = 0.25,
    base_deviation_penalty_weight: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Score three candidates without any future or simulator-only input."""

    if z0.shape != (1, 512) or language.shape != (1, 512):
        raise ValueError("z0 and language must be [1,512]")
    if candidates.shape != (3, 8, 7):
        raise ValueError(f"expected candidates [3,8,7], got {tuple(candidates.shape)}")
    if predictions.shape != (3, 8, 512) or uncertainty.shape != (3, 8):
        raise ValueError("predictions/uncertainty must be [3,8,512] and [3,8]")
    if action_std.numel() != 7:
        raise ValueError("action_std must contain seven dimensions")
    if not all(
        torch.isfinite(value).all()
        for value in (z0, language, candidates, predictions, uncertainty, action_std)
    ):
        raise ValueError("candidate scoring input contains NaN/Inf")
    current_similarity = F.cosine_similarity(z0.float(), language.float(), dim=-1)[0]
    final_similarity = F.cosine_similarity(
        predictions[:, -1].float(), language.expand(3, -1).float(), dim=-1
    )
    goal_progress = final_similarity - current_similarity
    mean_uncertainty = uncertainty.float().mean(dim=-1)
    normalized_delta = (
        candidates - candidates[0:1]
    ) / action_std.reshape(1, 1, 7).clamp_min(1.0e-6)
    base_deviation = torch.square(normalized_delta.float()).mean(dim=(1, 2))
    score = (
        float(goal_progress_weight) * goal_progress
        - float(uncertainty_penalty_weight) * mean_uncertainty
        - float(base_deviation_penalty_weight) * base_deviation
    )
    return {
        "score": score,
        "goal_progress": goal_progress,
        "mean_uncertainty": mean_uncertainty,
        "base_deviation": base_deviation,
        "final_language_similarity": final_similarity,
    }
