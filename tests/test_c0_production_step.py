#!/usr/bin/env python3
"""One production-equivalent C0 optimization step on canonical frozen features."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_c0 import normalized_targets
from src.c0 import C0Actor, FeatureDataset, compute_action_stats, compute_proprio_stats


def autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager[None]:
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def main() -> int:
    feature_path = PROJECT_ROOT / "artifacts/stage1/features/clip_vit_b32_h8_p12_upright/train.hdf5"
    if not feature_path.is_file():
        raise FileNotFoundError(f"canonical D12 feature store is required: {feature_path}")
    config = yaml.safe_load((PROJECT_ROOT / "configs/c0_actor.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    dataset = FeatureDataset(feature_path)
    try:
        batch = next(iter(DataLoader(dataset, batch_size=int(config["optimization"]["batch_size"]), shuffle=False)))
        action_stats = compute_action_stats(feature_path)
        proprio_stats = compute_proprio_stats(feature_path)
        action_mean = torch.tensor(action_stats["mean"], dtype=torch.float32, device=device).view(1, 1, 7)
        action_std = torch.tensor(action_stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6).view(1, 1, 7)
        proprio_mean = torch.tensor(proprio_stats["mean"], dtype=torch.float32, device=device).view(1, -1)
        proprio_std = torch.tensor(proprio_stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6).view(1, -1)
        model = C0Actor(
            feature_dim=int(config["model"]["feature_dim"]),
            proprio_dim=dataset.proprio_dim,
            fusion_dim=int(config["model"]["fusion_dim"]),
            action_horizon=dataset.horizon,
            action_dim=dataset.action_dim,
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["optimization"]["learning_rate"]),
            weight_decay=float(config["optimization"]["weight_decay"]),
        )
        amp_enabled = device.type == "cuda" and str(config["optimization"]["precision"]).lower() == "fp16"
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp_enabled):
            output = model(
                batch["image_features"],
                batch["language_features"],
                (batch["proprio_raw"] - proprio_mean) / proprio_std,
            )
            target = normalized_targets(batch["action_chunks"], action_mean, action_std)
            loss = nn.SmoothL1Loss()(output, target)
        assert output.shape == (int(config["optimization"]["batch_size"]), 8, 7), output.shape
        assert torch.isfinite(output).all()
        assert torch.isfinite(target).all()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))
        assert torch.isfinite(torch.tensor(grad_norm))
        scaler.step(optimizer)
        scaler.update()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        print(
            {
                "device": str(device),
                "amp_enabled": amp_enabled,
                "batch_shape": {key: list(value.shape) for key, value in batch.items()},
                "output_shape": list(output.shape),
                "loss": float(loss.detach()),
                "gradient_tensors": len(gradients),
                "gradient_norm_before_clip": grad_norm,
            }
        )
    finally:
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
