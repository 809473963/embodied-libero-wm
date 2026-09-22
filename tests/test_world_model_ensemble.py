#!/usr/bin/env python3
"""D15 ensemble shape, disagreement, and real-checkpoint checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_world_model_ensemble import load_ensemble
from src.world_models import (
    LatentDynamicsEnsemble,
    LatentDynamicsModel,
    ensemble_statistics,
)


CHECKPOINTS = [
    PROJECT_ROOT / "results/world_model/d14_single_seed11/best.pt",
    PROJECT_ROOT / "results/world_model/d15_member_seed23/best.pt",
    PROJECT_ROOT / "results/world_model/d15_member_seed47/best.pt",
]


def synthetic_test() -> dict[str, object]:
    torch.manual_seed(15)
    batch = 2
    inputs = (
        F.normalize(torch.randn(batch, 512), dim=-1),
        torch.randn(batch, 12),
        F.normalize(torch.randn(batch, 512), dim=-1),
        torch.randn(batch, 8, 7),
    )
    ensemble = LatentDynamicsEnsemble([LatentDynamicsModel() for _ in range(3)])
    mean_prediction, uncertainty, predictions = ensemble.predict(*inputs)
    assert predictions.shape == (3, 2, 8, 512)
    assert mean_prediction.shape == (2, 8, 512)
    assert uncertainty.shape == (2, 8)
    assert torch.isfinite(predictions).all() and torch.isfinite(uncertainty).all()
    assert (uncertainty >= 0).all() and float(uncertainty.max()) > 0
    assert torch.allclose(
        torch.linalg.vector_norm(mean_prediction, dim=-1), torch.ones(2, 8), atol=1.0e-5
    )
    identical = F.normalize(torch.randn(2, 8, 512), dim=-1).repeat(3, 1, 1, 1)
    _, identical_uncertainty = ensemble_statistics(identical)
    assert torch.allclose(identical_uncertainty, torch.zeros_like(identical_uncertainty), atol=2e-7)
    return {
        "member_shape": list(predictions.shape),
        "mean_shape": list(mean_prediction.shape),
        "uncertainty_shape": list(uncertainty.shape),
        "uncertainty_mean": float(uncertainty.mean()),
    }


def checkpoint_test() -> dict[str, object]:
    if not all(path.is_file() for path in CHECKPOINTS):
        raise FileNotFoundError("train all three D15 members before the real-checkpoint test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble, contract, members = load_ensemble(CHECKPOINTS, device)
    z0 = F.normalize(torch.randn(2, 512, device=device), dim=-1)
    proprio = torch.randn(2, 12, device=device)
    language = F.normalize(torch.randn(2, 512, device=device), dim=-1)
    actions = torch.randn(2, 8, 7, device=device)
    with torch.inference_mode():
        mean_prediction, uncertainty, predictions = ensemble.predict(
            z0, proprio, language, actions
        )
    assert predictions.shape == (3, 2, 8, 512)
    assert mean_prediction.shape == (2, 8, 512)
    assert uncertainty.shape == (2, 8)
    assert torch.isfinite(predictions).all() and torch.isfinite(uncertainty).all()
    assert [member["seed"] for member in members] == [11, 23, 47]
    return {
        "device": str(device),
        "seeds": [member["seed"] for member in members],
        "store_sha256": contract["d13_store_sha256"],
        "uncertainty_mean": float(uncertainty.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    result = {"synthetic": synthetic_test()}
    if not args.synthetic_only:
        result["real_checkpoints"] = checkpoint_test()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
