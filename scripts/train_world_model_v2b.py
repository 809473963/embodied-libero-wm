#!/usr/bin/env python3
"""Train WM-v2B seed 11 with a frozen-calibrated future-proprio objective."""

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

from scripts.calibrate_world_model_v2b import model_from_config, tensor_state_sha256
from scripts.train_world_model import (
    move_batch,
    restore_rng_state,
    rng_state,
    set_seed,
)
from src.d13.dataset import CONTRACT_VERSION
from src.world_models import (
    FutureProprioWorldModelDataset,
    PROPRIO_GROUPS,
    apply_world_model_normalization,
    compute_world_model_normalization,
    cosine_distance,
    proprio_delta_error,
    standardized_proprio_delta_target,
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


def update_best_validation(
    best_total: float,
    best_visual: float,
    current_total: float,
    current_visual: float,
) -> tuple[float, float, bool]:
    values = (best_total, best_visual, current_total, current_visual)
    if not all(np.isfinite(value) or np.isinf(value) for value in values):
        raise ValueError("validation metric is NaN/Inf")
    improved = current_total < best_total or (
        current_total == best_total and current_visual < best_visual
    )
    return (
        current_total if improved else best_total,
        current_visual if improved else best_visual,
        improved,
    )


def load_calibration_freeze(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], float]:
    freeze_path = ROOT / config["loss"]["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    calibration_path = ROOT / freeze["calibration_result"]
    calibration = json.loads(calibration_path.read_text())
    if file_sha256(calibration_path) != freeze["calibration_result_sha256"]:
        raise ValueError("gradient calibration result hash changed")
    if file_sha256(config_path) != freeze["config_sha256"]:
        raise ValueError("training config differs from calibration freeze")
    if calibration["status"] != "COMPLETE_BEFORE_TRAINING":
        raise ValueError("gradient calibration was not completed before training")
    value = float(freeze["final_lambda_p"])
    if value != float(calibration["gradient_calibration"]["final_lambda_p"]):
        raise ValueError("frozen lambda differs from calibration")
    if not 0.05 <= value <= 20.0:
        raise ValueError("frozen lambda is outside preregistered clip range")
    return freeze, calibration, value


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    normalization: dict[str, Any],
    lambda_p: float,
    beta: float,
    diagnostic_count: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    visual_sum = torch.zeros(8, dtype=torch.float64)
    proprio_sum = torch.zeros(8, dtype=torch.float64)
    proprio_dim_sum = torch.zeros(12, dtype=torch.float64)
    zero_delta_sum = torch.zeros(8, dtype=torch.float64)
    copy_sum = torch.zeros(8, dtype=torch.float64)
    count = 0
    diagnostics: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        normalized_proprio, normalized_actions = apply_world_model_normalization(
            batch["proprio0"], batch["actions"], normalization
        )
        target_p = standardized_proprio_delta_target(
            batch["proprio0"], batch["proprio_future"], normalization
        )
        pred_z, pred_p = model(
            batch["z0"], normalized_proprio, batch["language"], normalized_actions
        )
        visual = cosine_distance(pred_z, batch["z_future"])
        proprio_element = F.smooth_l1_loss(
            pred_p.float(), target_p.float(), beta=beta, reduction="none"
        )
        proprio = proprio_element.mean(dim=-1)
        zero_delta = F.smooth_l1_loss(
            torch.zeros_like(target_p), target_p.float(), beta=beta, reduction="none"
        ).mean(dim=-1)
        copy = cosine_distance(
            batch["z0"].unsqueeze(1).expand(-1, 8, -1), batch["z_future"]
        )
        if not all(
            torch.isfinite(value).all()
            for value in (visual, proprio, proprio_element, zero_delta, copy)
        ):
            raise FloatingPointError("WM-v2B validation contains NaN/Inf")
        visual_sum += visual.double().sum(dim=0).cpu()
        proprio_sum += proprio.double().sum(dim=0).cpu()
        proprio_dim_sum += proprio_element.double().sum(dim=(0, 1)).cpu()
        zero_delta_sum += zero_delta.double().sum(dim=0).cpu()
        copy_sum += copy.double().sum(dim=0).cpu()
        remaining = diagnostic_count - len(diagnostics)
        for index in range(min(max(remaining, 0), visual.shape[0])):
            diagnostics.append(
                {
                    "task_id": int(batch["task_id"][index]),
                    "demo_id": int(batch["demo_id"][index]),
                    "start_timestep": int(batch["start_timestep"][index]),
                    "visual_cosine_error": visual[index].cpu().tolist(),
                    "proprio_delta_smooth_l1": proprio[index].cpu().tolist(),
                }
            )
        count += int(visual.shape[0])
    visual_h = visual_sum / count
    proprio_h = proprio_sum / count
    proprio_dim = proprio_dim_sum / (count * 8)
    zero_h = zero_delta_sum / count
    copy_h = copy_sum / count
    visual_overall = float(visual_h.mean())
    proprio_overall = float(proprio_h.mean())
    if was_training:
        model.train()
    return {
        "sample_count": count,
        "visual_cosine_error": visual_overall,
        "visual_per_horizon": visual_h.tolist(),
        "proprio_delta_smooth_l1": proprio_overall,
        "proprio_per_horizon": proprio_h.tolist(),
        "proprio_per_dimension": proprio_dim.tolist(),
        "proprio_groups": {
            name: float(proprio_dim[group].mean())
            for name, group in PROPRIO_GROUPS.items()
        },
        "total_loss": visual_overall + lambda_p * proprio_overall,
        "zero_delta_proprio_error": float(zero_h.mean()),
        "zero_delta_proprio_per_horizon": zero_h.tolist(),
        "copy_cosine_error": float(copy_h.mean()),
        "copy_per_horizon": copy_h.tolist(),
        "diagnostic_samples": diagnostics,
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_total: float,
    best_visual: float,
    config: dict[str, Any],
    config_hash: str,
    normalization: dict[str, Any],
    calibration_freeze: dict[str, Any],
    calibration_freeze_hash: str,
    lambda_p: float,
    commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "wm_v2b_future_proprio_auxiliary",
        "version": config["run"]["version"],
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_total_loss": float(best_total),
        "best_validation_visual_cosine_error": float(best_visual),
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
        "calibration_freeze": calibration_freeze,
        "calibration_freeze_sha256": calibration_freeze_hash,
        "lambda_p": lambda_p,
        "source_commit": commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/world_model_v2b_seed11.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/world_model_v2b/seed11"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if config["run"]["method"] != "wm_v2b_future_proprio_auxiliary":
        raise ValueError("unexpected WM-v2B method")
    config_hash = file_sha256(args.config)
    freeze, calibration, lambda_p = load_calibration_freeze(config, args.config)
    freeze_path = ROOT / config["loss"]["calibration_freeze"]
    freeze_hash = file_sha256(freeze_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = int(config["run"]["seed"])
    if seed != 11:
        raise ValueError("first WM-v2B experiment must use seed 11")
    set_seed(seed)
    data = config["data"]
    if data["contract_version"] != CONTRACT_VERSION or int(data["horizon"]) != 8:
        raise ValueError("WM-v2B changes the frozen D13 contract")
    store = ROOT / data["store"]
    normalization = compute_world_model_normalization(
        store,
        ROOT / data["data_contract"],
        minimum_std=float(config["normalization"]["minimum_std"]),
    )
    if normalization["proprio"]["count"] != 55981:
        raise ValueError("unexpected proprio normalization count")
    if normalization["action"]["count"] != 55531:
        raise ValueError("unexpected action normalization count")
    train_set = FutureProprioWorldModelDataset(store, "train", 8)
    validation_set = FutureProprioWorldModelDataset(store, "validation", 8)
    if len(train_set) != 52381 or len(validation_set) != 5869:
        raise ValueError("WM-v2B split/sample count changed")
    optimization = config["optimization"]
    workers = (
        int(optimization["num_workers"])
        if args.num_workers is None
        else args.num_workers
    )
    loader_args = {
        "batch_size": int(optimization["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    model = model_from_config(config)
    if tensor_state_sha256(model) != calibration["model"]["initial_state_sha256"]:
        raise ValueError("training initial model differs from gradient calibration")
    model.to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 826444:
        raise ValueError("WM-v2B architecture differs from preregistration")
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    amp_enabled = device.type == "cuda" and str(optimization["precision"]).lower() == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    beta = float(config["loss"]["proprio_beta"])
    commit = source_commit()
    start_epoch = 0
    global_step = 0
    best_total = float("inf")
    best_visual = float("inf")

    if args.resume is None and (args.output_dir / "best.pt").exists():
        raise FileExistsError("WM-v2B output exists; use explicit --resume")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        required = {
            "best_validation_total_loss",
            "best_validation_visual_cosine_error",
            "normalization",
            "calibration_freeze_sha256",
            "lambda_p",
            "scaler_state",
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
        if checkpoint["calibration_freeze_sha256"] != freeze_hash:
            raise ValueError("resume calibration freeze differs")
        if float(checkpoint["lambda_p"]) != lambda_p:
            raise ValueError("resume lambda_p differs")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_total = float(checkpoint["best_validation_total_loss"])
        best_visual = float(checkpoint["best_validation_visual_cosine_error"])

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
        "backbone_parameters": 792000,
        "auxiliary_head_parameters": 34444,
        "source_commit": commit,
        "config_sha256": config_hash,
        "normalization": normalization,
        "data_pairing": data["pairing"],
        "auxiliary_target": data["auxiliary_target"],
        "future_proprio_is_input": False,
        "loss_contract": config["loss"],
        "calibration_freeze_sha256": freeze_hash,
        "gradient_calibration": calibration["gradient_calibration"],
        "lambda_p": lambda_p,
    }
    write_json(args.output_dir / "run_metadata.json", run_metadata)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, int(optimization["max_epochs"])):
            model.train()
            visual_sum = 0.0
            proprio_sum = 0.0
            item_count = 0
            gradient_norm_sum = 0.0
            for batch in train_loader:
                batch = move_batch(batch, device)
                normalized_proprio, normalized_actions = apply_world_model_normalization(
                    batch["proprio0"], batch["actions"], normalization
                )
                target_p = standardized_proprio_delta_target(
                    batch["proprio0"], batch["proprio_future"], normalization
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    pred_z, pred_p = model(
                        batch["z0"],
                        normalized_proprio,
                        batch["language"],
                        normalized_actions,
                    )
                    visual = cosine_distance(pred_z, batch["z_future"])
                    proprio = proprio_delta_error(pred_p, target_p, beta=beta)
                    loss = visual.mean() + lambda_p * proprio.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("WM-v2B training loss is NaN/Inf")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimization["gradient_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("WM-v2B gradient norm is NaN/Inf")
                scaler.step(optimizer)
                scaler.update()
                items = int(visual.numel())
                visual_sum += float(visual.detach().sum())
                proprio_sum += float(proprio.detach().sum())
                item_count += items
                gradient_norm_sum += float(gradient_norm)
                global_step += 1

            validation = evaluate(
                model,
                validation_loader,
                device,
                normalization,
                lambda_p,
                beta,
                int(config["validation"]["diagnostic_sample_count"]),
            )
            train_visual = visual_sum / item_count
            train_proprio = proprio_sum / item_count
            summary = {
                "event": "validation",
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_visual_cosine_error": train_visual,
                "train_proprio_delta_smooth_l1": train_proprio,
                "train_total_loss": train_visual + lambda_p * train_proprio,
                "mean_unclipped_gradient_norm": gradient_norm_sum / len(train_loader),
                "elapsed_seconds": time.perf_counter() - started,
                **{
                    key: value
                    for key, value in validation.items()
                    if key != "diagnostic_samples"
                },
            }
            append_jsonl(metrics_path, summary)
            best_total, best_visual, improved = update_best_validation(
                best_total,
                best_visual,
                float(validation["total_loss"]),
                float(validation["visual_cosine_error"]),
            )
            payload = checkpoint_payload(
                model,
                optimizer,
                scaler,
                epoch + 1,
                global_step,
                best_total,
                best_visual,
                config,
                config_hash,
                normalization,
                freeze,
                freeze_hash,
                lambda_p,
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
                        "selection_metric": "validation_total_loss",
                        "lambda_p": lambda_p,
                        **validation,
                    },
                )
            print(
                json.dumps(
                    {
                        **summary,
                        "historical_best_total": best_total,
                        "historical_best_visual": best_visual,
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
            "best_checkpoint_visual_cosine_error": best_visual,
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
    except Exception as error:
        print(f"WM-v2B training failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
