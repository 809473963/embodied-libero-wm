#!/usr/bin/env python3
"""WM-v2A objective, real-batch, and preregistered-decision checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_world_model_v2a import decide
from scripts.train_world_model import make_model
from scripts.train_world_model_v2a import delta_error, update_best_validation
from src.d13 import WorldModelDataset
from src.world_models import apply_world_model_normalization, cosine_distance


STORE = ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
CHECKPOINT = ROOT / "results/world_model/d14_single_seed11/best.pt"


def synthetic_objective_test(config: dict) -> None:
    torch.manual_seed(11)
    model = make_model(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 792000
    optimizer = AdamW(model.parameters(), lr=3.0e-4)
    z0 = F.normalize(torch.randn(2, 512), dim=-1)
    proprio = torch.randn(2, 12)
    language = F.normalize(torch.randn(2, 512), dim=-1)
    actions = torch.randn(2, 8, 7)
    target = F.normalize(torch.randn(2, 8, 512), dim=-1)
    prediction = model(z0, proprio, language, actions)
    assert prediction.shape == (2, 8, 512)
    assert torch.isfinite(prediction).all()
    assert torch.allclose(
        torch.linalg.vector_norm(prediction, dim=-1),
        torch.ones(2, 8),
        atol=1.0e-5,
    )

    future = cosine_distance(prediction, target)
    delta = delta_error(prediction, z0, target, beta=1.0)
    direct_delta = F.smooth_l1_loss(
        prediction - z0[:, None, :],
        target - z0[:, None, :],
        beta=1.0,
        reduction="none",
    ).mean(dim=-1)
    assert future.shape == delta.shape == (2, 8)
    assert torch.allclose(delta, direct_delta, atol=1.0e-7, rtol=1.0e-6)
    loss = future.mean() + float(config["loss"]["lambda_delta"]) * delta.mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients if g is not None)
    optimizer.step()

    best_total, best_future, improved = update_best_validation(
        0.02, 0.019, 0.021, 0.01
    )
    assert (best_total, best_future, improved) == (0.02, 0.019, False)
    best_total, best_future, improved = update_best_validation(
        best_total, best_future, 0.02, 0.018
    )
    assert (best_total, best_future, improved) == (0.02, 0.018, True)


def real_batch_test(config: dict) -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    normalization = checkpoint["normalization"]
    dataset = WorldModelDataset(STORE, split="train", horizon=8)
    try:
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
        proprio, actions = apply_world_model_normalization(
            batch["proprio0"], batch["actions"], normalization
        )
        model = make_model(config)
        prediction = model(batch["z0"], proprio, batch["language"], actions)
        assert prediction.shape == batch["z_future"].shape == (4, 8, 512)
        future = cosine_distance(prediction, batch["z_future"])
        delta = delta_error(prediction, batch["z0"], batch["z_future"], 1.0)
        loss = future.mean() + delta.mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert normalization["proprio"]["count"] == 55981
        assert normalization["action"]["count"] == 55531
    finally:
        dataset.close()


def decision_test(protocol: dict) -> None:
    prediction = {
        "relative_error_change_vs_d14_seed11": 0.01,
        "correct_action_error": 0.009,
        "copy_current_error": 0.015,
    }
    within = {
        "relative_delta": 0.02,
        "delta_bootstrap": {"ci95_lower": 0.001},
        "true_action_win_accuracy": 0.7,
        "accuracy_bootstrap": {"ci95_lower": 0.6},
    }
    local = {
        "separation_to_ordinary_error_ratio": 0.012,
        "improvement_multiple_vs_wm_v1": 2.5,
    }
    directional = {
        "pairwise_ranking_accuracy": 0.55,
        "pairwise_accuracy_ci95": [0.52, 0.58],
        "mean_within_sample_spearman": 0.08,
        "spearman_ci95": [0.02, 0.12],
    }
    passed = decide(protocol, prediction, within, local, directional)
    assert passed["verdict"] == "WM_V2A_LOCAL_SENSITIVITY_PASS"
    assert passed["seed23_seed47_authorized"] is True

    bad_direction = dict(directional, mean_within_sample_spearman=0.0)
    assert decide(protocol, prediction, within, local, bad_direction)["verdict"] == (
        "LOCAL_SENSITIVITY_ARTIFACT"
    )
    bad_local = dict(local, separation_to_ordinary_error_ratio=0.009)
    assert decide(protocol, prediction, within, bad_local, directional)["verdict"] == (
        "DELTA_OBJECTIVE_INSUFFICIENT"
    )
    bad_prediction = dict(prediction, relative_error_change_vs_d14_seed11=0.051)
    assert decide(protocol, bad_prediction, within, local, directional)["verdict"] == (
        "PREDICTION_QUALITY_REGRESSION"
    )


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/world_model_v2a_seed11.yaml").read_text())
    protocol = json.loads(
        (ROOT / "configs/wm_v2a_local_dynamics_protocol.json").read_text()
    )
    synthetic_objective_test(config)
    real_batch_test(config)
    decision_test(protocol)
    print(
        {
            "status": "PASS",
            "parameters": 792000,
            "lambda_delta": config["loss"]["lambda_delta"],
            "decision_paths": 4,
        }
    )


if __name__ == "__main__":
    main()
