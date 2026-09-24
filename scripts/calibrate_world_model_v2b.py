#!/usr/bin/env python3
"""One-shot train-only shared-GRU gradient calibration for WM-v2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_world_model import set_seed
from src.world_models import (
    FutureProprioWorldModelDataset,
    LatentDynamicsProprioAuxModel,
    apply_world_model_normalization,
    compute_world_model_normalization,
    cosine_distance,
    proprio_delta_error,
    standardized_proprio_delta_target,
)
from src.world_models.normalization import file_sha256


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def model_from_config(config: dict[str, Any]) -> LatentDynamicsProprioAuxModel:
    values = {key: int(value) for key, value in config["model"].items()}
    head = values.pop("proprio_head_hidden_dim")
    return LatentDynamicsProprioAuxModel(
        proprio_head_hidden_dim=head, **values
    )


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def gradient_norm(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]
) -> float:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("calibration gradient contains NaN/Inf")
    squared = sum(gradient.double().square().sum() for gradient in gradients)
    return float(torch.sqrt(squared))


def identity_sha256(records: list[dict[str, int]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/world_model_v2b_seed11.yaml"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2b_future_proprio_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/world_model_v2b/seed11/gradient_calibration.json",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    protocol = json.loads(args.protocol.read_text())
    calibration = protocol["calibration"]
    if protocol["status"] != "PREREGISTERED_BEFORE_CALIBRATION_AND_TRAINING":
        raise ValueError("WM-v2B protocol was not preregistered")
    if file_sha256(args.config) != protocol["experiment"]["config_sha256"]:
        raise ValueError("WM-v2B config differs from preregistration")
    if config["loss"]["lambda_p"] != "from_frozen_gradient_calibration":
        raise ValueError("lambda_p must come only from frozen calibration")
    if calibration["device"] != "cpu float32":
        raise ValueError("calibration device changed")

    set_seed(int(calibration["model_seed"]))
    store = ROOT / config["data"]["store"]
    normalization = compute_world_model_normalization(
        store,
        ROOT / config["data"]["data_contract"],
        minimum_std=float(config["normalization"]["minimum_std"]),
    )
    dataset = FutureProprioWorldModelDataset(store, "train", 8)
    if len(dataset) != 52381:
        raise ValueError("WM-v2B train sample count changed")
    selection_seed = int(calibration["selection_seed"])
    count = int(calibration["sample_count"])
    batch_size = int(calibration["batch_size"])
    selected_indices = np.random.default_rng(selection_seed).permutation(len(dataset))[
        :count
    ]
    if len(np.unique(selected_indices)) != count:
        raise RuntimeError("calibration indices are not unique")

    model = model_from_config(config).cpu().float().train()
    if sum(parameter.numel() for parameter in model.parameters()) != 826444:
        raise ValueError("WM-v2B parameter count differs from preregistration")
    initial_state_hash = tensor_state_sha256(model)
    shared = model.shared_gru_parameters()
    identities: list[dict[str, int]] = []
    batches: list[dict[str, Any]] = []
    try:
        for batch_index, start in enumerate(range(0, count, batch_size)):
            indices = selected_indices[start : start + batch_size]
            samples = [dataset[int(index)] for index in indices]
            batch = default_collate(samples)
            if not set(batch["demo_id"].tolist()).issubset(set(range(45))):
                raise ValueError("calibration accessed a non-train demo")
            batch_identities = [
                {
                    "dataset_index": int(indices[position]),
                    "task_id": int(batch["task_id"][position]),
                    "demo_id": int(batch["demo_id"][position]),
                    "start_timestep": int(batch["start_timestep"][position]),
                }
                for position in range(len(indices))
            ]
            identities.extend(batch_identities)
            normalized_proprio, normalized_actions = apply_world_model_normalization(
                batch["proprio0"], batch["actions"], normalization
            )
            target_p = standardized_proprio_delta_target(
                batch["proprio0"], batch["proprio_future"], normalization
            )

            pred_z, _ = model(
                batch["z0"], normalized_proprio, batch["language"], normalized_actions
            )
            visual_loss = cosine_distance(pred_z, batch["z_future"]).mean()
            g_z = gradient_norm(visual_loss, shared)

            _, pred_p = model(
                batch["z0"], normalized_proprio, batch["language"], normalized_actions
            )
            proprio_loss = proprio_delta_error(pred_p, target_p, beta=1.0).mean()
            g_p = gradient_norm(proprio_loss, shared)
            if not all(np.isfinite(value) and value > 0 for value in (g_z, g_p)):
                raise FloatingPointError("invalid calibration gradient norm")
            batches.append(
                {
                    "batch": batch_index,
                    "first_dataset_index": int(indices[0]),
                    "last_dataset_index": int(indices[-1]),
                    "identity_sha256": identity_sha256(batch_identities),
                    "visual_loss": float(visual_loss),
                    "proprio_loss": float(proprio_loss),
                    "g_z": g_z,
                    "g_p": g_p,
                }
            )
    finally:
        dataset.close()

    g_z = float(np.mean([record["g_z"] for record in batches]))
    g_p = float(np.mean([record["g_p"] for record in batches]))
    raw_lambda = g_z / g_p
    clip_min, clip_max = map(float, calibration["clip_range"])
    final_lambda = float(np.clip(raw_lambda, clip_min, clip_max))
    result = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "COMPLETE_BEFORE_TRAINING",
        "source_commit": source_commit(),
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": file_sha256(args.protocol),
        },
        "config": {
            "path": str(args.config.relative_to(ROOT)),
            "sha256": file_sha256(args.config),
        },
        "data": {
            "split": "train demos 0-44 only",
            "store_sha256": normalization["store_sha256"],
            "normalization": normalization,
            "sample_count": count,
            "batch_count": len(batches),
            "batch_size": batch_size,
            "selection_seed": selection_seed,
            "identity_sha256": identity_sha256(identities),
            "identities": identities,
        },
        "model": {
            "seed": int(calibration["model_seed"]),
            "parameters": 826444,
            "shared_parameter_set": "model.gru.parameters() only",
            "initial_state_sha256": initial_state_hash,
            "optimizer_updates": 0,
            "device": "cpu",
            "dtype": "float32",
        },
        "batches": batches,
        "gradient_calibration": {
            "aggregation": calibration["g_z"].split(" of ")[0],
            "g_z": g_z,
            "g_p": g_p,
            "unclipped_lambda_p": raw_lambda,
            "clip_range": [clip_min, clip_max],
            "final_lambda_p": final_lambda,
            "was_clipped": final_lambda != raw_lambda,
        },
        "validation_metrics_used": false,
        "nan_inf_audit": "pass",
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "g_z": g_z,
                "g_p": g_p,
                "unclipped_lambda_p": raw_lambda,
                "final_lambda_p": final_lambda,
                "identity_sha256": result["data"]["identity_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
