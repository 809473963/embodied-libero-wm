#!/usr/bin/env python3
"""A real one-batch C0 forward/backward probe against the generated features."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.c0 import C0Actor, FeatureDataset, compute_proprio_stats


def main() -> int:
    feature_path = PROJECT_ROOT / "artifacts/stage1/features/clip_vit_b32_h8_p12_upright/train.hdf5"
    dataset = FeatureDataset(feature_path)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    model = C0Actor(
        feature_dim=512,
        proprio_dim=dataset.proprio_dim,
        fusion_dim=256,
        action_horizon=dataset.horizon,
        action_dim=dataset.action_dim,
    )
    stats = compute_proprio_stats(feature_path)
    proprio_mean = torch.tensor(stats["mean"]).view(1, -1)
    proprio_std = torch.tensor(stats["std"]).clamp_min(1e-6).view(1, -1)
    prediction = model(
        batch["image_features"],
        batch["language_features"],
        (batch["proprio_raw"] - proprio_mean) / proprio_std,
    )
    assert prediction.shape == (2, 8, 7), prediction.shape
    loss = torch.nn.functional.mse_loss(prediction, batch["action_chunks"])
    assert torch.isfinite(loss), loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    dataset.close()
    print({"shape": list(prediction.shape), "loss": float(loss)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
