#!/usr/bin/env python3
"""Audit the completed C0-v2 checkpoint against its training evidence."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def main() -> None:
    run = ROOT / "results/c0_v2/seed11"
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    validations = [row for row in rows if row["event"] == "validation"]
    skips = [row for row in rows if row["event"] == "skipped_optimizer_step"]
    assert len(validations) == 50
    assert all(finite_tree(row) for row in rows)
    best = min(validations, key=lambda row: row["raw_action_mse"])
    assert checkpoint["best_validation"] == best["raw_action_mse"]
    assert checkpoint["epoch"] == best["epoch"] + 1
    assert checkpoint["global_step"] == best["global_step"]
    assert checkpoint["supervision_contract"] == {
        "version": "c0-v2-causal-alignment-v1",
        "pairing": "obs[k] -> actions[k+1:k+9]",
        "action_start_offset": 1,
        "horizon": 8,
    }
    normalization = checkpoint["normalization"]
    assert normalization["proprio"]["count"] == 55981
    assert normalization["action"]["count"] == 55531
    assert "actions[1:]" in normalization["action"]["rule"]
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["train_samples"] == 52381
    assert metadata["validation_samples"] == 5869
    assert metadata["feature_store_sha256"] == normalization["store_sha256"]
    assert metadata["feature_manifest_sha256"] == normalization["feature_manifest_sha256"]
    assert checkpoint["config_sha256"] == sha256(run / "config.yaml")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", checkpoint["source_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    print(
        {
            "status": "PASS",
            "checkpoint_sha256": sha256(run / "best.pt"),
            "best_epoch": best["epoch"],
            "best_validation_raw_mse": best["raw_action_mse"],
            "validation_events": len(validations),
            "amp_skipped_steps": len(skips),
            "source_commit": checkpoint["source_commit"],
        }
    )


if __name__ == "__main__":
    main()
