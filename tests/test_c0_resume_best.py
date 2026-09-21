#!/usr/bin/env python3
"""Regression: a resumed worse validation result must not replace best.pt."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_c0 import checkpoint_payload, update_best_validation
from src.c0 import C0Actor


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    torch.manual_seed(0)
    model = C0Actor(feature_dim=8, proprio_dim=3, fusion_dim=4, action_horizon=2, action_dim=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    normalization = {
        "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
        "proprio": {"mean": [0.0] * 3, "std": [1.0] * 3},
    }
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        historical_best = 0.125
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch=4,
            global_step=12,
            config_hash="config-hash",
            feature_manifest_hash="feature-hash",
            normalization=normalization,
            task_id=None,
            best_validation=historical_best,
        )
        best_path = output_dir / "best.pt"
        resume_path = output_dir / "last.pt"
        torch.save(payload, best_path)
        torch.save(payload, resume_path)
        before = digest(best_path)

        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        restored_best = float(resumed["best_validation"])
        retained_best, improved = update_best_validation(restored_best, current_validation=0.250)
        assert restored_best == historical_best
        assert retained_best == historical_best
        assert not improved
        if improved:  # Mirrors the production best.pt write guard.
            torch.save(resumed, best_path)
        assert digest(best_path) == before, "worse resumed validation overwrote historical best.pt"

    print({"historical_best": historical_best, "worse_validation": 0.250, "best_checkpoint_preserved": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
