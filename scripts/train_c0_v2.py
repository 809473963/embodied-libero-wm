#!/usr/bin/env python3
"""Train the single-factor C0-v2 causal-label actor."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
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

from src.c0 import C0Actor, CausalActorDataset, compute_causal_actor_normalization
from src.c0.causal_dataset import CONTRACT_VERSION, file_sha256


EXPECTED_TRAIN_SAMPLES = 52381
EXPECTED_VALIDATION_SAMPLES = 5869


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, content: dict[str, Any]) -> None:
    with path.open("a") as destination:
        destination.write(json.dumps(content, ensure_ascii=False) + "\n")


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def update_best(historical: float, current: float) -> tuple[float, bool]:
    # Positive infinity is the intentional before-first-validation sentinel.
    # Reject NaN/-inf and any non-finite measured validation value.
    if np.isnan(historical) or historical == float("-inf") or not np.isfinite(current):
        raise ValueError("best/current validation must be finite")
    improved = current < historical
    return (current if improved else historical), improved


def normalization_tensors(
    normalization: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proprio_mean = torch.tensor(
        normalization["proprio"]["mean"], dtype=torch.float32, device=device
    ).view(1, -1)
    proprio_std = torch.tensor(
        normalization["proprio"]["std"], dtype=torch.float32, device=device
    ).view(1, -1)
    action_mean = torch.tensor(
        normalization["action"]["mean"], dtype=torch.float32, device=device
    ).view(1, 1, -1)
    action_std = torch.tensor(
        normalization["action"]["std"], dtype=torch.float32, device=device
    ).view(1, 1, -1)
    return proprio_mean, proprio_std, action_mean, action_std


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def evaluate(
    model: C0Actor,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    proprio_mean, proprio_std, action_mean, action_std = tensors
    total_squared = 0.0
    total_count = 0
    task_squared = np.zeros(10, dtype=np.float64)
    task_count = np.zeros(10, dtype=np.int64)
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = model(
            batch["image_features"],
            batch["language_features"],
            (batch["proprio_raw"] - proprio_mean) / proprio_std,
        )
        prediction = prediction * action_std + action_mean
        squared = torch.square(prediction - batch["action_chunks"])
        if not torch.isfinite(squared).all():
            raise FloatingPointError("validation contains NaN/Inf")
        total_squared += float(squared.sum())
        total_count += squared.numel()
        per_sample_sum = squared.sum(dim=(1, 2)).double().cpu().numpy()
        task_ids = batch["task_id"].cpu().numpy()
        for task_id in range(10):
            selected = task_ids == task_id
            task_squared[task_id] += float(per_sample_sum[selected].sum())
            task_count[task_id] += int(selected.sum()) * 8 * 7
    model.train()
    if total_count != int(task_count.sum()) or np.any(task_count == 0):
        raise RuntimeError("validation per-task accounting mismatch")
    return {
        "raw_action_mse": total_squared / total_count,
        "per_task_raw_action_mse": {
            str(task_id): float(task_squared[task_id] / task_count[task_id])
            for task_id in range(10)
        },
        "samples": total_count // (8 * 7),
    }


def checkpoint_payload(
    model: C0Actor,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation: float,
    config: dict[str, Any],
    config_hash: str,
    normalization: dict[str, Any],
    commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        # Keep adapter compatibility; supervision_contract distinguishes v2.
        "method": "c0_actor_only",
        "experiment": "c0_v2_causal_alignment",
        "supervision_contract": {
            "version": CONTRACT_VERSION,
            "pairing": "obs[k] -> actions[k+1:k+9]",
            "action_start_offset": 1,
            "horizon": 8,
        },
        "epoch": epoch,
        "global_step": global_step,
        "best_validation": float(best_validation),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": rng_state(),
        "config": config,
        "config_sha256": config_hash,
        "source_commit": commit,
        "feature_store_sha256": normalization["store_sha256"],
        "feature_manifest_sha256": normalization["feature_manifest_sha256"],
        "normalization": normalization,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/c0_actor_v2_causal.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/c0_v2/seed11"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    config_hash = file_sha256(args.config)
    data = config["data"]
    if data["contract_version"] != CONTRACT_VERSION:
        raise ValueError("configuration differs from C0-v2 contract")
    if int(data["action_chunk_horizon"]) != 8 or int(data["action_start_offset"]) != 1:
        raise ValueError("C0-v2 requires H=8 and action offset 1")
    store_path = PROJECT_ROOT / data["store"]
    manifest_path = PROJECT_ROOT / data["feature_manifest"]
    normalization = compute_causal_actor_normalization(
        store_path,
        manifest_path,
        minimum_std=float(config["normalization"]["minimum_std"]),
    )
    if normalization["proprio"]["count"] != 55981:
        raise ValueError("unexpected unique train proprio count")
    if normalization["action"]["count"] != 55531:
        raise ValueError("unexpected unique causal action count")

    train_set = CausalActorDataset(store_path, "train", in_memory=args.in_memory)
    validation_set = CausalActorDataset(store_path, "validation", in_memory=args.in_memory)
    if len(train_set) != EXPECTED_TRAIN_SAMPLES or len(validation_set) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(
            f"BLOCK_DATA_INDEXING: samples {len(train_set)}/{len(validation_set)} "
            f"!= {EXPECTED_TRAIN_SAMPLES}/{EXPECTED_VALIDATION_SAMPLES}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = int(config["run"]["seed"])
    set_seed(seed)
    optimization = config["optimization"]
    loader_args = {
        "batch_size": int(optimization["batch_size"]),
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    model_config = config["model"]
    model = C0Actor(
        feature_dim=int(model_config["feature_dim"]),
        proprio_dim=int(data["proprioception_dim"]),
        fusion_dim=int(model_config["fusion_dim"]),
        action_horizon=8,
        action_dim=int(data["action_dim"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    amp_enabled = device.type == "cuda" and str(optimization["precision"]).lower() == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    tensors = normalization_tensors(normalization, device)
    loss_function: nn.Module = nn.SmoothL1Loss()
    commit = source_commit()
    start_epoch = 0
    global_step = 0
    best_validation = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        required = {
            "best_validation",
            "normalization",
            "source_commit",
            "scaler_state",
            "supervision_contract",
        }
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f"unsafe resume checkpoint missing {sorted(missing)}")
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("resume config hash differs")
        if checkpoint["normalization"] != normalization:
            raise ValueError("resume normalization differs")
        if checkpoint["supervision_contract"]["action_start_offset"] != 1:
            raise ValueError("resume checkpoint is not causal C0-v2")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied_config = args.output_dir / "config.yaml"
    if copied_config.exists() and file_sha256(copied_config) != config_hash:
        raise ValueError("output directory contains a different config")
    if not copied_config.exists():
        shutil.copy2(args.config, copied_config)
    atomic_json(
        args.output_dir / "run_metadata.json",
        {
            "method": "c0_actor_only",
            "experiment": "c0_v2_causal_alignment",
            "supervision_contract": "obs[k] -> actions[k+1:k+9]",
            "action_start_offset": 1,
            "seed": seed,
            "source_commit": commit,
            "device": str(device),
            "in_memory": args.in_memory,
            "train_samples": len(train_set),
            "validation_samples": len(validation_set),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "config_sha256": config_hash,
            "feature_store": str(store_path),
            "feature_store_sha256": normalization["store_sha256"],
            "feature_manifest": str(manifest_path),
            "feature_manifest_sha256": normalization["feature_manifest_sha256"],
            "normalization": normalization,
        },
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    accumulation = int(optimization["gradient_accumulation_steps"])
    max_epochs = int(optimization["max_epochs"])
    gradient_clip = float(optimization["gradient_clip_norm"])
    proprio_mean, proprio_std, action_mean, action_std = tensors
    started = time.perf_counter()
    reached_limit = False
    for epoch in range(start_epoch, max_epochs):
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
                target = (batch["action_chunks"] - action_mean) / action_std
                loss = loss_function(prediction, target) / accumulation
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation != 0 and batch_index + 1 != len(train_loader):
                continue
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss produced NaN/Inf")
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            skipped_overflow = not np.isfinite(gradient_norm)
            if skipped_overflow:
                if not amp_enabled or scale_after >= scale_before:
                    raise FloatingPointError("non-finite gradient was not handled by AMP scaler")
                append_jsonl(
                    metrics_path,
                    {
                        "event": "skipped_optimizer_step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "reason": "amp_nonfinite_gradient",
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                continue
            global_step += 1
            append_jsonl(
                metrics_path,
                {
                    "event": "train_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "normalized_huber_loss": float(loss.detach()) * accumulation,
                    "gradient_norm": gradient_norm,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
                reached_limit = True
                break
        validation = evaluate(model, validation_loader, device, tensors)
        best_validation, improved = update_best(
            best_validation, float(validation["raw_action_mse"])
        )
        summary = {
            "event": "validation",
            "epoch": epoch,
            "global_step": global_step,
            **validation,
            "best_validation": best_validation,
        }
        append_jsonl(metrics_path, summary)
        payload = checkpoint_payload(
            model,
            optimizer,
            scaler,
            epoch + 1,
            global_step,
            best_validation,
            config,
            config_hash,
            normalization,
            commit,
        )
        torch.save(payload, args.output_dir / "last.pt")
        if improved:
            torch.save(payload, args.output_dir / "best.pt")
        print(json.dumps(summary), flush=True)
        if reached_limit:
            break
    train_set.close()
    validation_set.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"C0-v2 training failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
