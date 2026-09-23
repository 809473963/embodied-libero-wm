"""C1-v2 candidate generation with an explicit action-space interface trace."""

from __future__ import annotations

import torch


def prepare_candidate_action_interface(
    actor_normalized_chunk: torch.Tensor,
    actor_action_mean: torch.Tensor,
    actor_action_std: torch.Tensor,
    wm_action_mean: torch.Tensor,
    wm_action_std: torch.Tensor,
    noise_std: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Map actor-normalized output to raw candidates and then WM-normalized input.

    The returned tensors make every semantic boundary inspectable. ``candidates``
    are physical LIBERO actions clipped to ``[-1, 1]``. Only
    ``wm_normalized_candidates`` may be supplied to the world model.
    """

    if actor_normalized_chunk.shape != (8, 7):
        raise ValueError(
            f"expected actor_normalized_chunk [8,7], got {tuple(actor_normalized_chunk.shape)}"
        )
    vectors = {
        "actor_action_mean": actor_action_mean,
        "actor_action_std": actor_action_std,
        "wm_action_mean": wm_action_mean,
        "wm_action_std": wm_action_std,
        "noise_std": noise_std,
    }
    flattened: dict[str, torch.Tensor] = {}
    for name, value in vectors.items():
        if value.numel() != 7:
            raise ValueError(f"{name} must contain seven values")
        flattened[name] = value.reshape(7).to(
            device=actor_normalized_chunk.device,
            dtype=actor_normalized_chunk.dtype,
        )
    if torch.any(flattened["actor_action_std"] <= 0):
        raise ValueError("actor action std must be positive")
    if torch.any(flattened["wm_action_std"] <= 0):
        raise ValueError("world-model action std must be positive")
    if torch.any(flattened["noise_std"] < 0):
        raise ValueError("candidate noise std must be nonnegative")
    if not all(torch.isfinite(value).all() for value in flattened.values()):
        raise ValueError("action-interface statistics contain NaN/Inf")
    if not torch.isfinite(actor_normalized_chunk).all():
        raise ValueError("actor-normalized chunk contains NaN/Inf")

    actor_denormalized = (
        actor_normalized_chunk * flattened["actor_action_std"]
        + flattened["actor_action_mean"]
    )
    actor_default = actor_denormalized.clamp(-1.0, 1.0)

    # Match the frozen C1-v1 antithetic, smoothed noise construction exactly.
    noise = torch.randn((8, 7), generator=generator, dtype=torch.float32)
    noise = noise * flattened["noise_std"].detach().float().cpu()
    padded = torch.cat((noise[:1], noise, noise[-1:]), dim=0)
    noise = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    noise = noise.to(device=actor_normalized_chunk.device, dtype=actor_normalized_chunk.dtype)

    candidate_preclip = torch.stack(
        (actor_default, actor_default + noise, actor_default - noise), dim=0
    )
    candidates = candidate_preclip.clamp(-1.0, 1.0)
    wm_normalized_candidates = (
        candidates - flattened["wm_action_mean"].reshape(1, 1, 7)
    ) / flattened["wm_action_std"].reshape(1, 1, 7)
    outputs = {
        "actor_normalized": actor_normalized_chunk,
        "actor_denormalized_preclip": actor_denormalized,
        "actor_default": actor_default,
        "candidate_noise": noise,
        "candidate_preclip": candidate_preclip,
        "candidates": candidates,
        "wm_normalized_candidates": wm_normalized_candidates,
    }
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("action interface produced NaN/Inf")
    return outputs
