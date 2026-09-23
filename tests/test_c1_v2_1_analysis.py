#!/usr/bin/env python3
"""Synthetic unit checks for C1-v2.1 candidate and ranking diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_c1_v2_candidates import (
    action_quality,
    action_separation,
    decide,
    generate_base_noise,
    make_candidates,
    ranking_quality,
)
from src.control import prepare_candidate_action_interface


def main() -> None:
    checkpoint = torch.load(
        ROOT / "results/c0_v2/seed11/best.pt", map_location="cpu", weights_only=False
    )
    action = checkpoint["normalization"]["action"]
    mean = np.asarray(action["mean"], dtype=np.float32)
    std = np.asarray(action["std"], dtype=np.float32)
    normalization = {
        "actor_action_mean": mean,
        "actor_action_std": std,
        "wm_action_mean": mean,
        "wm_action_std": std,
    }
    noise_std = np.asarray([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.0], dtype=np.float32)
    actor_normalized = np.linspace(-1.0, 1.0, 56, dtype=np.float32).reshape(1, 8, 7)
    base_noise = generate_base_noise(1, 1901, noise_std)
    preclip, candidates, wm_normalized = make_candidates(
        actor_normalized, base_noise, 1.0, normalization
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1901)
    reference = prepare_candidate_action_interface(
        torch.from_numpy(actor_normalized[0]),
        torch.from_numpy(mean),
        torch.from_numpy(std),
        torch.from_numpy(mean),
        torch.from_numpy(std),
        torch.from_numpy(noise_std),
        generator,
    )
    assert np.allclose(preclip[0], reference["candidate_preclip"].numpy(), atol=1e-7)
    assert np.allclose(candidates[0], reference["candidates"].numpy(), atol=1e-7)
    assert np.allclose(
        wm_normalized[0], reference["wm_normalized_candidates"].numpy(), atol=1e-6
    )

    target = candidates[:, 1].copy()
    quality, arrays = action_quality(candidates, target, std, np.asarray([0]))
    assert quality["oracle_best_fraction"]["1"] == 1.0
    assert quality["best_of_3_raw_action_mse"] == 0.0
    separation, _ = action_separation(candidates, std)
    assert separation["mean_pairwise_normalized_chunk_l2"] > 0.0

    scores = np.asarray([[0.0, 2.0, 1.0], [2.0, 1.0, 0.0]], dtype=np.float64)
    errors = np.asarray([[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float64)
    ranking = ranking_quality(scores, errors)
    assert ranking["top_score_equals_oracle_best_fraction"] == 1.0
    assert ranking["default_alternative_pairwise_ranking_accuracy"] == 1.0
    assert ranking["mean_within_sample_spearman"] == 1.0

    protocol = json.loads((ROOT / "configs/c1_v2_1_candidate_diagnosis.json").read_text())
    decision = decide(
        protocol,
        {
            "best_of_3_raw_mse_relative_gain": 0.2,
            "alternative_oracle_best_fraction": 0.5,
        },
        {
            "top_score_equals_oracle_best_fraction": 0.8,
            "default_alternative_pairwise_ranking_accuracy": 0.8,
            "mean_within_sample_spearman": 0.5,
            "replacement_margin_vs_oracle_improvement": {
                "spearman": {"value": 0.3}
            },
        },
        {"mean_pairwise_normalized_chunk_l2": 1.0},
        {
            "mean_pairwise_prediction_cosine_separation_to_ordinary_error_ratio": 0.1
        },
    )
    assert decision["blockers"] == ["NO_MECHANISM_BLOCK"]
    assert decision["new_c1_controller_design_allowed"] is True
    print(
        {
            "status": "PASS",
            "generator_matches_deployment": True,
            "synthetic_ranking_accuracy": ranking[
                "top_score_equals_oracle_best_fraction"
            ],
        }
    )


if __name__ == "__main__":
    main()
