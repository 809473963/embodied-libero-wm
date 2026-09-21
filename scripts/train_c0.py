#!/usr/bin/env python3
"""Train or resume the small frozen-feature C0 actor baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.c0 import C0Actor, FeatureDataset, compute_action_stats, compute_proprio_stats


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def normalized_targets(
    actions: torch.Tensor, action_mean: torch.Tensor, action_std: torch.Tensor
) -> torch.Tensor:
    return (actions - action_mean) / action_std


@torch.inference_mode()
def evaluate(
    model: C0Actor,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = model(
            batch["image_features"],
            batch["language_features"],
            (batch["proprio_raw"] - proprio_mean) / proprio_std,
        )
        predicted_actions = prediction * action_std + action_mean
        squared_error += float(torch.square(predicted_actions - batch["action_chunks"]).sum())
        count += int(np.prod(batch["action_chunks"].shape))
    model.train()
    return {"raw_action_mse": squared_error / count}


def checkpoint_payload(
    model: C0Actor,
    optimizer: AdamW,
    epoch: int,
    global_step: int,
    config_hash: str,
    feature_manifest_hash: str,
    normalization: dict[str, Any],
    task_id: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "c0_actor_only",
        "epoch": epoch,
        "global_step": global_step,
        "config_sha256": config_hash,
        "feature_manifest_sha256": feature_manifest_hash,
        "normalization": normalization,
        "task_id": task_id,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": rng_state(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/c0_actor.yaml")
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "artifacts/stage1/features/clip_vit_b32_h8_p12_upright")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--task-id", type=int, choices=range(10))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-memory", action="store_true", help="cache selected frozen features in RAM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = yaml.safe_load(args.config.read_text())
    config_hash = file_sha256(args.config)
    manifest_path = args.features_dir / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    feature_manifest_hash = file_sha256(manifest_path)
    train_path = args.features_dir / "train.hdf5"
    validation_path = args.features_dir / "validation.hdf5"
    train_set = FeatureDataset(train_path, task_id=args.task_id, in_memory=args.in_memory)
    validation_set = FeatureDataset(validation_path, task_id=args.task_id, in_memory=args.in_memory)
    if (train_set.horizon, train_set.action_dim, train_set.proprio_dim) != (
        validation_set.horizon,
        validation_set.action_dim,
        validation_set.proprio_dim,
    ):
        raise ValueError("train/validation feature schemas differ")
    data_config = config["data"]
    if int(data_config["action_chunk_horizon"]) != train_set.horizon:
        raise ValueError("config action horizon differs from feature store")
    if int(data_config["action_dim"]) != train_set.action_dim:
        raise ValueError("config action dim differs from feature store")
    if int(data_config["proprioception_dim"]) != train_set.proprio_dim:
        raise ValueError("config proprioception dim differs from feature store")

    set_seed(args.seed)
    optimization = config["optimization"]
    batch_size = int(optimization["batch_size"])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model_config = config["model"]
    model = C0Actor(
        feature_dim=int(model_config["feature_dim"]),
        proprio_dim=train_set.proprio_dim,
        fusion_dim=int(model_config["fusion_dim"]),
        action_horizon=train_set.horizon,
        action_dim=train_set.action_dim,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    selected_indices = train_set.indices if args.task_id is not None else None
    action_statistics = compute_action_stats(train_path, selected_indices)
    normalization = {
        "action": {
            "mean": action_statistics["mean"],
            "std": [max(float(value), 1e-6) for value in action_statistics["std"]],
        },
        "proprio": compute_proprio_stats(train_path, selected_indices),
    }
    normalization["proprio"]["std"] = [
        max(float(value), 1e-6) for value in normalization["proprio"]["std"]
    ]
    accumulation = int(optimization["gradient_accumulation_steps"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    amp_enabled = device.type == "cuda" and str(optimization["precision"]).lower() == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    loss_function: nn.Module = nn.SmoothL1Loss()
    epoch = 0
    global_step = 0
    best_validation = float("inf")

    if args.resume:
        # RNG state must remain a CPU ByteTensor; mapping a whole checkpoint to
        # CUDA makes torch.set_rng_state reject it. load_state_dict moves model
        # and optimizer tensors to their parameter devices afterward.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("resume config hash differs")
        if checkpoint["feature_manifest_sha256"] != feature_manifest_hash:
            raise ValueError("resume feature manifest hash differs")
        if checkpoint.get("task_id") != args.task_id:
            raise ValueError("resume task_id differs")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng_state(checkpoint["rng_state"])
        epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        normalization = checkpoint["normalization"]

    action_mean = torch.tensor(normalization["action"]["mean"], device=device).view(1, 1, -1)
    action_std = torch.tensor(normalization["action"]["std"], device=device).view(1, 1, -1)
    proprio_mean = torch.tensor(normalization["proprio"]["mean"], device=device).view(1, -1)
    proprio_std = torch.tensor(normalization["proprio"]["std"], device=device).view(1, -1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output_dir / "config.yaml"
    if not run_config_path.exists():
        shutil.copy2(args.config, run_config_path)
    write_json(
        args.output_dir / "run_metadata.json",
        {
            "method": "c0_actor_only",
            "seed": args.seed,
            "task_id": args.task_id,
            "in_memory": args.in_memory,
            "train_samples": len(train_set),
            "validation_samples": len(validation_set),
            "device": str(device),
            "config_sha256": config_hash,
            "feature_manifest_sha256": feature_manifest_hash,
            "feature_manifest": str(manifest_path),
            "normalization": normalization,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    max_epochs = int(optimization["max_epochs"])
    started = time.perf_counter()
    reached_limit = False
    for current_epoch in range(epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(
                    batch["image_features"],
                    batch["language_features"],
                    (batch["proprio_raw"] - proprio_mean) / proprio_std,
                )
                loss = loss_function(
                    prediction,
                    normalized_targets(batch["action_chunks"], action_mean, action_std),
                ) / accumulation
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation != 0 and batch_index + 1 != len(train_loader):
                continue
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            metrics = {
                "event": "train_step",
                "epoch": current_epoch,
                "global_step": global_step,
                "normalized_huber_loss": float(loss.detach()) * accumulation,
                "gradient_norm": gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(json.dumps(metrics) + "\n")
            if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
                reached_limit = True
                break
        validation = evaluate(
            model, validation_loader, device, proprio_mean, proprio_std, action_mean, action_std
        )
        summary = {"event": "validation", "epoch": current_epoch, "global_step": global_step, **validation}
        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(summary) + "\n")
        payload = checkpoint_payload(
            model, optimizer, current_epoch + 1, global_step, config_hash, feature_manifest_hash, normalization, args.task_id
        )
        torch.save(payload, args.output_dir / "last.pt")
        if validation["raw_action_mse"] < best_validation:
            best_validation = validation["raw_action_mse"]
            torch.save(payload, args.output_dir / "best.pt")
        print(json.dumps(summary), flush=True)
        if reached_limit:
            break
    train_set.close()
    validation_set.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"C0 training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
