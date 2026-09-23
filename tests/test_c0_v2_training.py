#!/usr/bin/env python3
"""One real C0-v2 batch must support finite forward/backward/update."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_c0_v2 import normalization_tensors, update_best
from src.c0 import C0Actor, CausalActorDataset, compute_causal_actor_normalization


STORE = ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
MANIFEST = ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/feature_manifest.json"


def main() -> None:
    first_best, first_improved = update_best(float("inf"), 0.25)
    assert first_best == 0.25 and first_improved
    later_best, later_improved = update_best(first_best, 0.30)
    assert later_best == 0.25 and not later_improved
    dataset = CausalActorDataset(STORE, "train")
    try:
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        normalization = compute_causal_actor_normalization(STORE, MANIFEST)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = C0Actor(
            feature_dim=512, proprio_dim=12, fusion_dim=256, action_horizon=8, action_dim=7
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
        tensors = normalization_tensors(normalization, device)
        proprio_mean, proprio_std, action_mean, action_std = tensors
        image = batch["image_features"].to(device)
        language = batch["language_features"].to(device)
        proprio = batch["proprio_raw"].to(device)
        actions = batch["action_chunks"].to(device)
        prediction = model(image, language, (proprio - proprio_mean) / proprio_std)
        target = (actions - action_mean) / action_std
        loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        assert prediction.shape == (2, 8, 7)
        assert torch.isfinite(prediction).all() and torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
        assert np.isfinite(float(loss))
        print(
            {
                "status": "PASS",
                "input": {"image": [2, 512], "language": [2, 512], "proprio": [2, 12]},
                "target": [2, 8, 7],
                "output": list(prediction.shape),
                "loss": float(loss),
            }
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
