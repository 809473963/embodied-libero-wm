#!/usr/bin/env python3
"""Train or safely resume the D14 single latent forward predictor."""

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
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


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


def update_best_validation(historical_best: float, current: float) -> tuple[float, bool]:
    if not np.isfinite(current):
        raise ValueError("validation cosine error is NaN/Inf")
    improved = current < historical_best
    return (current if improved else historical_best), improved


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_model(config: dict[str, Any]) -> LatentDynamicsModel:
    return LatentDynamicsModel(**{key: int(value) for key, value in config["model"].items()})


@torch.inference_mode()
def evaluate(
    model: LatentDynamicsModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    normalization: dict[str, Any],
    diagnostic_sample_count: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    learned_sum = torch.zeros(8, dtype=torch.float64)
    copy_sum = torch.zeros(8, dtype=torch.float64)
    sample_count = 0
    diagnostics: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        normalized_proprio, normalized_actions = apply_world_model_normalization(
            batch["proprio0"], batch["actions"], normalization
        )
        prediction = model(
            batch["z0"], normalized_proprio, batch["language"], normalized_actions
        )
        learned_error = cosine_distance(prediction, batch["z_future"])
        copy_prediction = batch["z0"].unsqueeze(1).expand(-1, 8, -1)
        copy_error = cosine_distance(copy_prediction, batch["z_future"])
        if not torch.isfinite(learned_error).all() or not torch.isfinite(copy_error).all():
            raise FloatingPointError("validation produced NaN/Inf")
        learned_sum += learned_error.double().sum(dim=0).cpu()
        copy_sum += copy_error.double().sum(dim=0).cpu()
        remaining = diagnostic_sample_count - len(diagnostics)
        if remaining > 0:
            for index in range(min(remaining, learned_error.shape[0])):
                diagnostics.append(
                    {
                        "task_id": int(batch["task_id"][index]),
                        "demo_id": int(batch["demo_id"][index]),
                        "start_timestep": int(batch["start_timestep"][index]),
                        "learned_cosine_error": learned_error[index].cpu().tolist(),
                        "copy_cosine_error": copy_error[index].cpu().tolist(),
                    }
                )
        sample_count += int(learned_error.shape[0])
    learned = learned_sum / sample_count
    copy = copy_sum / sample_count
    if was_training:
        model.train()
    return {
        "sample_count": sample_count,
        "overall_cosine_error": float(learned.mean()),
        "learned_per_horizon": learned.tolist(),
        "copy_overall_cosine_error": float(copy.mean()),
        "copy_per_horizon": copy.tolist(),
        "copy_minus_learned_per_horizon": (copy - learned).tolist(),
        "diagnostic_samples": diagnostics,
    }


def checkpoint_payload(
    model: LatentDynamicsModel,
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
        "schema_version": 1,
        "method": "d14_single_latent_forward_predictor",
        "epoch": epoch,
        "global_step": global_step,
        "best_validation": float(best_validation),
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
        "source_commit": commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/world_model.yaml")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/world_model/d14_single_seed11"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    config_hash = file_sha256(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = int(config["run"]["seed"])
    set_seed(seed)
    data_config = config["data"]
    if data_config["contract_version"] != CONTRACT_VERSION:
        raise ValueError("configuration differs from frozen D13-v1")
    if int(data_config["horizon"]) != 8:
        raise ValueError("D14 freezes H=8")
    store_path = PROJECT_ROOT / data_config["store"]
    contract_path = PROJECT_ROOT / data_config["data_contract"]
    normalization = compute_world_model_normalization(
        store_path,
        contract_path,
        minimum_std=float(config["normalization"]["minimum_std"]),
    )
    if normalization["proprio"]["count"] != 55981:
        raise ValueError("unexpected unique train proprio count")
    if normalization["action"]["count"] != 55531:
        raise ValueError("unexpected causal train action count")

    train_set = WorldModelDataset(store_path, split="train", horizon=8)
    validation_set = WorldModelDataset(store_path, split="validation", horizon=8)
    optimization = config["optimization"]
    num_workers = (
        int(optimization["num_workers"]) if args.num_workers is None else args.num_workers
    )
    loader_arguments = {
        "batch_size": int(optimization["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_arguments)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_arguments)
    model = make_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    amp_enabled = device.type == "cuda" and str(optimization["precision"]).lower() == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    commit = source_commit()
    start_epoch = 0
    global_step = 0
    best_validation = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        required = {"best_validation", "normalization", "source_commit", "scaler_state"}
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f"unsafe legacy resume checkpoint missing {sorted(missing)}")
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("resume config hash differs")
        if checkpoint["d13_data_contract_version"] != CONTRACT_VERSION:
            raise ValueError("resume D13 contract differs")
        if checkpoint["d13_store_sha256"] != normalization["store_sha256"]:
            raise ValueError("resume D13 store hash differs")
        if checkpoint["normalization"] != normalization:
            raise ValueError("resume normalization differs")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output_dir / "config.yaml"
    if not run_config_path.exists():
        shutil.copy2(args.config, run_config_path)
    elif file_sha256(run_config_path) != config_hash:
        raise ValueError("output directory already contains a different config")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_metadata = {
        "method": "d14_single_latent_forward_predictor",
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_enabled": amp_enabled,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_commit": commit,
        "d13_source_commit": "e37073e588505b4461d11dc81b174032ff297f2d",
        "d13_source_tag": "d13-wm-data-v1",
        "config_sha256": config_hash,
        "normalization": normalization,
        "data_pairing": data_config["pairing"],
    }
    write_json(args.output_dir / "run_metadata.json", run_metadata)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, int(optimization["max_epochs"])):
            model.train()
            train_sum = 0.0
            train_count = 0
            gradient_norm_sum = 0.0
            for batch in train_loader:
                batch = move_batch(batch, device)
                normalized_proprio, normalized_actions = apply_world_model_normalization(
                    batch["proprio0"], batch["actions"], normalization
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    prediction = model(
                        batch["z0"],
                        normalized_proprio,
                        batch["language"],
                        normalized_actions,
                    )
                    per_item_error = cosine_distance(prediction, batch["z_future"])
                    loss = per_item_error.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("training loss is NaN/Inf")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimization["gradient_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("gradient norm is NaN/Inf")
                scaler.step(optimizer)
                scaler.update()
                batch_items = int(per_item_error.numel())
                train_sum += float(per_item_error.detach().sum())
                train_count += batch_items
                gradient_norm_sum += float(gradient_norm)
                global_step += 1

            validation = evaluate(
                model,
                validation_loader,
                device,
                normalization,
                int(config["validation"]["diagnostic_sample_count"]),
            )
            summary = {
                "event": "validation",
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_overall_cosine_error": train_sum / train_count,
                "mean_unclipped_gradient_norm": gradient_norm_sum / len(train_loader),
                "elapsed_seconds": time.perf_counter() - started,
                **{key: value for key, value in validation.items() if key != "diagnostic_samples"},
            }
            append_jsonl(metrics_path, summary)
            best_validation, improved = update_best_validation(
                best_validation, float(validation["overall_cosine_error"])
            )
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
                write_json(
                    args.output_dir / "best_validation.json",
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "source_commit": commit,
                        "d13_store_sha256": normalization["store_sha256"],
                        **validation,
                    },
                )
            print(json.dumps({**summary, "historical_best": best_validation}), flush=True)
    finally:
        train_set.close()
        validation_set.close()

    run_metadata.update(
        {
            "completed_epochs": int(optimization["max_epochs"]),
            "global_step": global_step,
            "best_validation_overall_cosine_error": best_validation,
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
        print(f"D14 training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
