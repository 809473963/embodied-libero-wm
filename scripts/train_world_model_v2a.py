#!/usr/bin/env python3
"""Train WM-v2A seed 11 with objective-only latent-delta supervision."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_world_model import (
    make_model,
    move_batch,
    restore_rng_state,
    rng_state,
    set_seed,
)
from src.d13 import WorldModelDataset
from src.d13.dataset import CONTRACT_VERSION
from src.world_models import (
    LatentDynamicsModel,
    apply_world_model_normalization,
    compute_world_model_normalization,
    cosine_distance,
)
from src.world_models.normalization import file_sha256


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, content: dict[str, Any]) -> None:
    with path.open("a") as destination:
        destination.write(json.dumps(content, ensure_ascii=False) + "\n")


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def delta_error(
    prediction: torch.Tensor,
    z0: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return SmoothL1 latent-delta error as [B,H], averaged over D=512."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target must share [B,H,D]")
    if z0.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError("z0 must be [B,D]")
    predicted_delta = prediction.float() - z0.float().unsqueeze(1)
    target_delta = target.float() - z0.float().unsqueeze(1)
    return F.smooth_l1_loss(
        predicted_delta, target_delta, beta=beta, reduction="none"
    ).mean(dim=-1)


def update_best_validation(
    best_total: float,
    best_future: float,
    current_total: float,
    current_future: float,
) -> tuple[float, float, bool]:
    values = (best_total, best_future, current_total, current_future)
    if not all(np.isfinite(value) or np.isinf(value) for value in values):
        raise ValueError("validation metric is NaN")
    improved = current_total < best_total or (
        current_total == best_total and current_future < best_future
    )
    return (
        current_total if improved else best_total,
        current_future if improved else best_future,
        improved,
    )


@torch.inference_mode()
def evaluate(
    model: LatentDynamicsModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    normalization: dict[str, Any],
    lambda_delta: float,
    delta_beta: float,
    diagnostic_sample_count: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    future_sum = torch.zeros(8, dtype=torch.float64)
    delta_sum = torch.zeros(8, dtype=torch.float64)
    copy_sum = torch.zeros(8, dtype=torch.float64)
    count = 0
    diagnostics: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        proprio, actions = apply_world_model_normalization(
            batch["proprio0"], batch["actions"], normalization
        )
        prediction = model(batch["z0"], proprio, batch["language"], actions)
        future = cosine_distance(prediction, batch["z_future"])
        delta = delta_error(prediction, batch["z0"], batch["z_future"], delta_beta)
        copy = cosine_distance(
            batch["z0"].unsqueeze(1).expand(-1, 8, -1), batch["z_future"]
        )
        if not all(torch.isfinite(value).all() for value in (future, delta, copy)):
            raise FloatingPointError("WM-v2A validation contains NaN/Inf")
        future_sum += future.double().sum(dim=0).cpu()
        delta_sum += delta.double().sum(dim=0).cpu()
        copy_sum += copy.double().sum(dim=0).cpu()
        remaining = diagnostic_sample_count - len(diagnostics)
        for index in range(min(max(remaining, 0), future.shape[0])):
            diagnostics.append(
                {
                    "task_id": int(batch["task_id"][index]),
                    "demo_id": int(batch["demo_id"][index]),
                    "start_timestep": int(batch["start_timestep"][index]),
                    "future_cosine_error": future[index].cpu().tolist(),
                    "delta_smooth_l1": delta[index].cpu().tolist(),
                    "copy_cosine_error": copy[index].cpu().tolist(),
                }
            )
        count += int(future.shape[0])
    future_h = future_sum / count
    delta_h = delta_sum / count
    copy_h = copy_sum / count
    future_overall = float(future_h.mean())
    delta_overall = float(delta_h.mean())
    if was_training:
        model.train()
    return {
        "sample_count": count,
        "future_cosine_error": future_overall,
        "future_per_horizon": future_h.tolist(),
        "delta_smooth_l1": delta_overall,
        "delta_per_horizon": delta_h.tolist(),
        "total_loss": future_overall + lambda_delta * delta_overall,
        "copy_cosine_error": float(copy_h.mean()),
        "copy_per_horizon": copy_h.tolist(),
        "diagnostic_samples": diagnostics,
    }


def checkpoint_payload(
    model: LatentDynamicsModel,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_total: float,
    best_future: float,
    config: dict[str, Any],
    config_hash: str,
    normalization: dict[str, Any],
    commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "wm_v2a_local_dynamics_resolution",
        "version": config["run"]["version"],
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_total_loss": float(best_total),
        "best_validation_future_cosine_error": float(best_future),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": rng_state(),
        "config": config,
        "config_sha256": config_hash,
        "d13_data_contract_version": CONTRACT_VERSION,
        "d13_store_sha256": normalization["store_sha256"],
        "d13_data_contract_sha256": normalization["data_contract_sha256"],
        "normalization": normalization,
        "loss_contract": config["loss"],
        "source_commit": commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/world_model_v2a_seed11.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/world_model_v2a/seed11"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if config["run"]["method"] != "wm_v2a_local_dynamics_resolution":
        raise ValueError("unexpected WM-v2A method")
    if float(config["loss"]["lambda_delta"]) != 1.0:
        raise ValueError("WM-v2A-v1 preregisters lambda_delta=1.0")
    if config["validation"]["checkpoint_selection"] != "minimum_validation_total_loss":
        raise ValueError("checkpoint selection changed")
    config_hash = file_sha256(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = int(config["run"]["seed"])
    if seed != 11:
        raise ValueError("first WM-v2A experiment must use seed 11")
    set_seed(seed)
    data_config = config["data"]
    if data_config["contract_version"] != CONTRACT_VERSION or int(
        data_config["horizon"]
    ) != 8:
        raise ValueError("WM-v2A changes the frozen D13 contract")
    store_path = ROOT / data_config["store"]
    normalization = compute_world_model_normalization(
        store_path,
        ROOT / data_config["data_contract"],
        minimum_std=float(config["normalization"]["minimum_std"]),
    )
    if normalization["proprio"]["count"] != 55981:
        raise ValueError("unexpected proprio normalization count")
    if normalization["action"]["count"] != 55531:
        raise ValueError("unexpected action normalization count")

    train_set = WorldModelDataset(store_path, "train", 8)
    validation_set = WorldModelDataset(store_path, "validation", 8)
    if len(train_set) != 52381 or len(validation_set) != 5869:
        raise ValueError("WM-v2A split/sample count changed")
    optimization = config["optimization"]
    num_workers = (
        int(optimization["num_workers"])
        if args.num_workers is None
        else args.num_workers
    )
    loader_args = {
        "batch_size": int(optimization["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    model = make_model(config).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 792000:
        raise ValueError("WM-v2A architecture differs from D14")
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    amp_enabled = device.type == "cuda" and str(optimization["precision"]).lower() == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    lambda_delta = float(config["loss"]["lambda_delta"])
    delta_beta = float(config["loss"]["delta_beta"])
    commit = source_commit()
    start_epoch = 0
    global_step = 0
    best_total = float("inf")
    best_future = float("inf")

    if args.resume is None and (args.output_dir / "best.pt").exists():
        raise FileExistsError("WM-v2A output exists; use explicit --resume")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        required = {
            "best_validation_total_loss",
            "best_validation_future_cosine_error",
            "normalization",
            "source_commit",
            "scaler_state",
            "loss_contract",
        }
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f"unsafe resume missing {sorted(missing)}")
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("resume config hash differs")
        if checkpoint["d13_store_sha256"] != normalization["store_sha256"]:
            raise ValueError("resume D13 store differs")
        if checkpoint["normalization"] != normalization:
            raise ValueError("resume normalization differs")
        if checkpoint["loss_contract"] != config["loss"]:
            raise ValueError("resume loss contract differs")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_total = float(checkpoint["best_validation_total_loss"])
        best_future = float(checkpoint["best_validation_future_cosine_error"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied_config = args.output_dir / "config.yaml"
    if not copied_config.exists():
        shutil.copy2(args.config, copied_config)
    elif file_sha256(copied_config) != config_hash:
        raise ValueError("output directory has a different config")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_metadata = {
        "method": config["run"]["method"],
        "version": config["run"]["version"],
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_enabled": amp_enabled,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_commit": commit,
        "config_sha256": config_hash,
        "normalization": normalization,
        "data_pairing": data_config["pairing"],
        "loss_contract": config["loss"],
        "single_changed_factor": "latent-delta SmoothL1 objective term",
    }
    write_json(args.output_dir / "run_metadata.json", run_metadata)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, int(optimization["max_epochs"])):
            model.train()
            future_sum = 0.0
            delta_sum = 0.0
            item_count = 0
            gradient_norm_sum = 0.0
            for batch in train_loader:
                batch = move_batch(batch, device)
                proprio, actions = apply_world_model_normalization(
                    batch["proprio0"], batch["actions"], normalization
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    prediction = model(
                        batch["z0"], proprio, batch["language"], actions
                    )
                    future = cosine_distance(prediction, batch["z_future"])
                    delta = delta_error(
                        prediction, batch["z0"], batch["z_future"], delta_beta
                    )
                    loss = future.mean() + lambda_delta * delta.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("WM-v2A training loss is NaN/Inf")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimization["gradient_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("WM-v2A gradient norm is NaN/Inf")
                scaler.step(optimizer)
                scaler.update()
                batch_items = int(future.numel())
                future_sum += float(future.detach().sum())
                delta_sum += float(delta.detach().sum())
                item_count += batch_items
                gradient_norm_sum += float(gradient_norm)
                global_step += 1

            validation = evaluate(
                model,
                validation_loader,
                device,
                normalization,
                lambda_delta,
                delta_beta,
                int(config["validation"]["diagnostic_sample_count"]),
            )
            train_future = future_sum / item_count
            train_delta = delta_sum / item_count
            summary = {
                "event": "validation",
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_future_cosine_error": train_future,
                "train_delta_smooth_l1": train_delta,
                "train_total_loss": train_future + lambda_delta * train_delta,
                "mean_unclipped_gradient_norm": gradient_norm_sum / len(train_loader),
                "elapsed_seconds": time.perf_counter() - started,
                **{
                    key: value
                    for key, value in validation.items()
                    if key != "diagnostic_samples"
                },
            }
            append_jsonl(metrics_path, summary)
            best_total, best_future, improved = update_best_validation(
                best_total,
                best_future,
                float(validation["total_loss"]),
                float(validation["future_cosine_error"]),
            )
            payload = checkpoint_payload(
                model,
                optimizer,
                scaler,
                epoch + 1,
                global_step,
                best_total,
                best_future,
                config,
                config_hash,
                normalization,
                commit,
            )
            torch.save(payload, args.output_dir / "last.pt")
            if improved:
                torch.save(payload, args.output_dir / "best.pt")
                write_json(
                    args.output_dir / "best_validation.json",
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "source_commit": commit,
                        "d13_store_sha256": normalization["store_sha256"],
                        "selection_metric": "validation_total_loss",
                        **validation,
                    },
                )
            print(
                json.dumps(
                    {
                        **summary,
                        "historical_best_total": best_total,
                        "historical_best_future": best_future,
                    }
                ),
                flush=True,
            )
    finally:
        train_set.close()
        validation_set.close()

    run_metadata.update(
        {
            "completed_epochs": int(optimization["max_epochs"]),
            "global_step": global_step,
            "best_validation_total_loss": best_total,
            "best_checkpoint_future_cosine_error": best_future,
            "wall_time_seconds": time.perf_counter() - started,
            "gpu_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "gpu_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            ),
            "nan_inf_audit": "pass",
        }
    )
    write_json(args.output_dir / "run_metadata.json", run_metadata)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"WM-v2A training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
