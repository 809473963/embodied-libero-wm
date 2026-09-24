#!/usr/bin/env python3
"""One-shot train-only gradient calibration for WM-v2C-EJ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_world_model import set_seed
from scripts.wm_v2c_ej_common import (
    assert_initial_visual_parity,
    identity_sha256,
    load_initialized_model,
    load_protocol_and_config,
    source_commit,
    tensor_state_sha256,
)
from src.world_models import (
    FutureProprioWorldModelDataset,
    apply_world_model_normalization,
    compute_world_model_normalization,
    cosine_distance,
    ej_delta_error,
    standardized_proprio_delta_target,
)
from src.world_models.normalization import file_sha256


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def gradient_norm(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]
) -> float:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("calibration gradient contains NaN/Inf")
    squared = sum(gradient.double().square().sum() for gradient in gradients)
    return float(torch.sqrt(squared))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/world_model_v2c_ej_seed11.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2c_ej_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/world_model_v2c_ej/seed11/gradient_calibration.json",
    )
    args = parser.parse_args()

    protocol, config = load_protocol_and_config(args.protocol, args.config)
    calibration = protocol["gradient_calibration"]
    if config["loss"]["lambda_ej"] != "from_frozen_train_only_gradient_calibration":
        raise ValueError("lambda_EJ must come only from frozen calibration")
    if float(config["loss"]["lambda_res"]) != float(protocol["loss"]["lambda_res"]):
        raise ValueError("lambda_res differs from preregistration")
    if calibration["device"] != "cpu float32":
        raise ValueError("gradient calibration device changed")

    set_seed(int(config["run"]["seed"]))
    model, checkpoint_normalization, source_checkpoint = load_initialized_model(
        config, protocol
    )
    model.cpu().float().train()
    initial_state_hash = tensor_state_sha256(model)
    store = ROOT / config["data"]["store"]
    normalization = compute_world_model_normalization(
        store,
        ROOT / config["data"]["data_contract"],
        minimum_std=float(config["normalization"]["minimum_std"]),
    )
    if normalization != checkpoint_normalization:
        raise ValueError("recomputed normalization differs from frozen WM-v2B")
    dataset = FutureProprioWorldModelDataset(store, "train", 8)
    if len(dataset) != 52381:
        raise ValueError("WM-v2C-EJ train sample count changed")
    count = int(calibration["sample_count"])
    batch_size = int(calibration["batch_size"])
    selection_seed = int(calibration["selection_seed"])
    selected_indices = np.random.default_rng(selection_seed).permutation(len(dataset))[
        :count
    ]
    if len(np.unique(selected_indices)) != count:
        raise RuntimeError("gradient-calibration indices are not unique")

    shared = tuple(model.gru.parameters())
    identities: list[dict[str, int]] = []
    batches: list[dict[str, Any]] = []
    max_initial_residual = 0.0
    try:
        for batch_index, start in enumerate(range(0, count, batch_size)):
            indices = selected_indices[start : start + batch_size]
            samples = [dataset[int(index)] for index in indices]
            batch = default_collate(samples)
            if not set(batch["demo_id"].tolist()).issubset(set(range(45))):
                raise ValueError("gradient calibration accessed a non-train demo")
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
            components = model.forward_with_components(
                batch["z0"],
                normalized_proprio,
                batch["language"],
                normalized_actions,
            )
            max_initial_residual = max(
                max_initial_residual, assert_initial_visual_parity(components)
            )
            visual_loss = cosine_distance(
                components["pred_z"], batch["z_future"]
            ).mean()
            g_z = gradient_norm(visual_loss, shared)

            # Use an independent forward graph for the second loss, matching
            # the frozen WM-v2B gradient-calibration procedure. This avoids
            # retaining one graph while keeping both norms fully independent.
            ej_components = model.forward_with_components(
                batch["z0"],
                normalized_proprio,
                batch["language"],
                normalized_actions,
            )
            assert_initial_visual_parity(ej_components)
            ej_loss = ej_delta_error(
                ej_components["pred_proprio_delta"], target_p, beta=1.0
            ).mean()
            g_ej = gradient_norm(ej_loss, shared)
            if not all(np.isfinite(value) and value > 0 for value in (g_z, g_ej)):
                raise FloatingPointError("invalid gradient-calibration norm")
            batches.append(
                {
                    "batch": batch_index,
                    "first_dataset_index": int(indices[0]),
                    "last_dataset_index": int(indices[-1]),
                    "identity_sha256": identity_sha256(batch_identities),
                    "visual_loss": float(visual_loss),
                    "ej_loss": float(ej_loss),
                    "g_z": g_z,
                    "g_ej": g_ej,
                }
            )
    finally:
        dataset.close()

    g_z = float(np.mean([record["g_z"] for record in batches]))
    g_ej = float(np.mean([record["g_ej"] for record in batches]))
    raw_lambda = g_z / g_ej
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
            "evaluation_or_matched_branches_used": False,
        },
        "model": {
            "seed": int(config["run"]["seed"]),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "wm_v2b_checkpoint": config["initialization"]["checkpoint"],
            "wm_v2b_checkpoint_sha256": file_sha256(
                ROOT / config["initialization"]["checkpoint"]
            ),
            "wm_v2b_checkpoint_epoch": int(source_checkpoint["epoch"]),
            "shared_parameter_set": "model.gru.parameters() only",
            "initial_state_sha256": initial_state_hash,
            "initial_max_abs_physical_visual_residual": max_initial_residual,
            "optimizer_updates": 0,
            "device": "cpu",
            "dtype": "float32",
            "gripper_adapter_input": False,
        },
        "batches": batches,
        "gradient_calibration": {
            "aggregation": "arithmetic mean of four per-batch gradient norms",
            "g_z": g_z,
            "g_ej": g_ej,
            "unclipped_lambda_ej": raw_lambda,
            "clip_range": [clip_min, clip_max],
            "final_lambda_ej": final_lambda,
            "was_clipped": final_lambda != raw_lambda,
        },
        "lambda_res": float(config["loss"]["lambda_res"]),
        "validation_metrics_used": False,
        "matched_counterfactual_used": False,
        "nan_inf_audit": "pass",
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "g_z": g_z,
                "g_ej": g_ej,
                "unclipped_lambda_ej": raw_lambda,
                "final_lambda_ej": final_lambda,
                "identity_sha256": result["data"]["identity_sha256"],
                "initial_state_sha256": initial_state_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
