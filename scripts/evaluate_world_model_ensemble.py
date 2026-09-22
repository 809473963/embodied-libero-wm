#!/usr/bin/env python3
"""Evaluate the D15 ensemble and write an auditable checkpoint manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.d13 import WorldModelDataset
from src.world_models import (
    LatentDynamicsEnsemble,
    LatentDynamicsModel,
    apply_world_model_normalization,
    cosine_distance,
)
from src.world_models.normalization import file_sha256


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def make_model(model_config: dict[str, Any]) -> LatentDynamicsModel:
    return LatentDynamicsModel(**{key: int(value) for key, value in model_config.items()})


def load_ensemble(
    checkpoint_paths: list[Path], device: torch.device
) -> tuple[LatentDynamicsEnsemble, dict[str, Any], list[dict[str, Any]]]:
    members: list[LatentDynamicsModel] = []
    records: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    seen_seeds: set[int] = set()
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        seed = int(checkpoint["config"]["run"]["seed"])
        if seed in seen_seeds:
            raise ValueError(f"ensemble member seed repeated: {seed}")
        seen_seeds.add(seed)
        comparable = {
            "model": checkpoint["config"]["model"],
            "d13_data_contract_version": checkpoint["d13_data_contract_version"],
            "d13_store_sha256": checkpoint["d13_store_sha256"],
            "d13_data_contract_sha256": checkpoint["d13_data_contract_sha256"],
            "normalization": checkpoint["normalization"],
        }
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError(f"ensemble member {path} does not share data/model contract")
        model = make_model(checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        members.append(model)
        records.append(
            {
                "seed": seed,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "best_validation": float(checkpoint["best_validation"]),
                "epoch": int(checkpoint["epoch"]),
                "global_step": int(checkpoint["global_step"]),
                "source_commit": checkpoint["source_commit"],
                "config_sha256": checkpoint["config_sha256"],
            }
        )
    assert reference is not None
    ensemble = LatentDynamicsEnsemble(members).to(device).eval()
    return ensemble, reference, records


def pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double()
    second = second.double()
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.sqrt(torch.square(first).sum() * torch.square(second).sum())
    if denominator == 0:
        return float("nan")
    return float((first * second).sum() / denominator)


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/world_model_ensemble.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/wm/d15_ensemble"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["contract_version"] != "d15-v1" or int(config["member_count"]) != 3:
        raise ValueError("D15-v1 evaluation requires the frozen three-member ensemble")
    checkpoint_paths = [PROJECT_ROOT / path for path in config["member_checkpoints"]]
    if not all(path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("one or more D15 ensemble checkpoints are missing")
    device = torch.device(args.device)
    ensemble, contract, member_records = load_ensemble(checkpoint_paths, device)
    expected_seeds = [int(value) for value in config["member_seeds"]]
    if [record["seed"] for record in member_records] != expected_seeds:
        raise ValueError("checkpoint member seeds/order differ from ensemble config")
    normalization = contract["normalization"]
    store_path = PROJECT_ROOT / normalization["store_path"] if "store_path" in normalization else PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
    dataset = WorldModelDataset(store_path, split="validation", horizon=8)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ensemble_error_sum = torch.zeros(8, dtype=torch.float64)
    member_error_sum = torch.zeros(3, 8, dtype=torch.float64)
    uncertainties: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    sample_count = 0
    inference_seconds = 0.0
    try:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            proprio, actions = apply_world_model_normalization(
                batch["proprio0"], batch["actions"], normalization
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            mean_prediction, uncertainty, member_predictions = ensemble.predict(
                batch["z0"], proprio, batch["language"], actions
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            ensemble_error = cosine_distance(mean_prediction, batch["z_future"])
            if not torch.isfinite(ensemble_error).all() or not torch.isfinite(uncertainty).all():
                raise FloatingPointError("ensemble validation produced NaN/Inf")
            ensemble_error_sum += ensemble_error.double().sum(dim=0).cpu()
            for member_index in range(3):
                member_error_sum[member_index] += cosine_distance(
                    member_predictions[member_index], batch["z_future"]
                ).double().sum(dim=0).cpu()
            uncertainties.append(uncertainty.cpu())
            errors.append(ensemble_error.cpu())
            sample_count += int(ensemble_error.shape[0])
    finally:
        dataset.close()

    uncertainty = torch.cat(uncertainties, dim=0)
    prediction_error = torch.cat(errors, dim=0)
    if uncertainty.shape != (sample_count, 8):
        raise RuntimeError("uncertainty aggregation shape mismatch")
    per_horizon_error = ensemble_error_sum / sample_count
    per_horizon_uncertainty = uncertainty.double().mean(dim=0)
    flattened_uncertainty = uncertainty.flatten()
    flattened_error = prediction_error.flatten()
    quantiles = torch.quantile(
        flattened_uncertainty.double(), torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    )
    if not math.isfinite(pearson(flattened_uncertainty, flattened_error)):
        raise FloatingPointError("uncertainty/error correlation is undefined")
    validation = {
        "sample_count": sample_count,
        "ensemble_overall_cosine_error": float(per_horizon_error.mean()),
        "ensemble_per_horizon_cosine_error": per_horizon_error.tolist(),
        "members": [
            {
                **record,
                "validation_overall_cosine_error_recomputed": float(
                    (member_error_sum[index] / sample_count).mean()
                ),
                "validation_per_horizon_cosine_error_recomputed": (
                    member_error_sum[index] / sample_count
                ).tolist(),
            }
            for index, record in enumerate(member_records)
        ],
        "uncertainty_definition": config["aggregation"]["uncertainty"],
        "uncertainty_formula": config["aggregation"]["uncertainty_formula"],
        "uncertainty_semantics": "epistemic member disagreement; not error or calibrated failure probability",
        "uncertainty_mean": float(flattened_uncertainty.double().mean()),
        "uncertainty_std": float(flattened_uncertainty.double().std(unbiased=False)),
        "uncertainty_per_horizon_mean": per_horizon_uncertainty.tolist(),
        "uncertainty_quantiles": {
            key: float(value)
            for key, value in zip(
                ["min", "p50", "p90", "p95", "p99", "max"], quantiles.tolist()
            )
        },
        "uncertainty_error_pearson_diagnostic": pearson(
            flattened_uncertainty, flattened_error
        ),
        "inference_seconds": inference_seconds,
        "inference_ms_per_sample": 1000.0 * inference_seconds / sample_count,
        "gpu_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "gpu_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "nan_inf_audit": "pass",
    }
    manifest = {
        "schema_version": 1,
        "contract_version": "d15-v1",
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(args.config),
        "member_count": len(member_records),
        "members": member_records,
        "d13_data_contract_version": contract["d13_data_contract_version"],
        "d13_store_sha256": contract["d13_store_sha256"],
        "d13_data_contract_sha256": contract["d13_data_contract_sha256"],
        "normalization": normalization,
        "aggregation": config["aggregation"],
    }
    write_json(args.output_dir / "ensemble_manifest.json", manifest)
    write_json(args.output_dir / "ensemble_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
