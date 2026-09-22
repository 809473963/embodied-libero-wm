#!/usr/bin/env python3
"""D14 synthetic and real D13-batch forward/backward checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.d13 import WorldModelDataset
from src.world_models import (
    LatentDynamicsModel,
    apply_world_model_normalization,
    compute_world_model_normalization,
    cosine_distance,
)
from scripts.train_world_model import update_best_validation


STORE = PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
CONTRACT = PROJECT_ROOT / "docs/world_model_data_contract.md"


def assert_finite_gradients(model: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def synthetic_test() -> dict[str, object]:
    torch.manual_seed(11)
    model = LatentDynamicsModel()
    optimizer = AdamW(model.parameters(), lr=3.0e-4)
    z0 = F.normalize(torch.randn(2, 512), dim=-1)
    proprio0 = torch.randn(2, 12)
    language = F.normalize(torch.randn(2, 512), dim=-1)
    actions = torch.randn(2, 8, 7)
    target = F.normalize(torch.randn(2, 8, 512), dim=-1)
    output = model(z0, proprio0, language, actions)
    assert output.shape == (2, 8, 512)
    assert torch.isfinite(output).all()
    assert torch.allclose(torch.linalg.vector_norm(output, dim=-1), torch.ones(2, 8), atol=1.0e-5)
    loss = cosine_distance(output, target).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert_finite_gradients(model)
    optimizer.step()
    historical_best, improved = update_best_validation(0.05, 0.06)
    assert historical_best == 0.05 and not improved
    historical_best, improved = update_best_validation(historical_best, 0.04)
    assert historical_best == 0.04 and improved
    return {"shape": list(output.shape), "loss": float(loss.detach())}


def real_batch_test() -> dict[str, object]:
    if not STORE.is_file():
        raise FileNotFoundError(f"missing frozen D13 store: {STORE}")
    normalization = compute_world_model_normalization(STORE, CONTRACT)
    assert normalization["proprio"]["count"] == 55981
    assert normalization["action"]["count"] == 55531
    dataset = WorldModelDataset(STORE, split="train", horizon=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
        assert batch["z0"].shape == (4, 512)
        assert batch["proprio0"].shape == (4, 12)
        assert batch["language"].shape == (4, 512)
        assert batch["actions"].shape == (4, 8, 7)
        assert batch["z_future"].shape == (4, 8, 512)
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"z0", "proprio0", "language", "actions", "z_future"}
        }
        normalized_proprio, normalized_actions = apply_world_model_normalization(
            tensor_batch["proprio0"], tensor_batch["actions"], normalization
        )
        model = LatentDynamicsModel().to(device)
        optimizer = AdamW(model.parameters(), lr=3.0e-4)
        output = model(
            tensor_batch["z0"],
            normalized_proprio,
            tensor_batch["language"],
            normalized_actions,
        )
        assert output.shape == tensor_batch["z_future"].shape == (4, 8, 512)
        loss = cosine_distance(output, tensor_batch["z_future"]).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert_finite_gradients(model)
        optimizer.step()
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        return {
            "device": str(device),
            "shape": list(output.shape),
            "loss": float(loss.detach()),
            "proprio_count": normalization["proprio"]["count"],
            "action_count": normalization["action"]["count"],
            "gpu_peak_memory_bytes": peak,
        }
    finally:
        dataset.close()


def main() -> int:
    print({"synthetic": synthetic_test(), "real_batch": real_batch_test()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
