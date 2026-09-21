#!/usr/bin/env python3
"""Validation-only C0 multi-task diagnostic; it neither trains nor evaluates envs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark  # noqa: E402
from src.c0 import C0Actor  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[C0Actor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    output_dim = int(state["action_head.4.weight"].shape[0])
    model = C0Actor(
        feature_dim=int(state["vision_encoder.0.weight"].shape[1]),
        proprio_dim=int(state["proprio_encoder.0.weight"].shape[1]),
        fusion_dim=int(state["vision_encoder.0.weight"].shape[0]),
        action_horizon=output_dim // 7,
        action_dim=7,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def initial_accumulator() -> dict[str, Any]:
    return {
        "rows": 0,
        "action_count_per_dimension": 0,
        "squared_error": 0.0,
        "target_sum": np.zeros(7, dtype=np.float64),
        "target_square_sum": np.zeros(7, dtype=np.float64),
        "target_min": np.full(7, np.inf, dtype=np.float64),
        "target_max": np.full(7, -np.inf, dtype=np.float64),
        "normalized_sum": np.zeros(7, dtype=np.float64),
        "normalized_square_sum": np.zeros(7, dtype=np.float64),
        "normalized_min": np.full(7, np.inf, dtype=np.float64),
        "normalized_max": np.full(7, -np.inf, dtype=np.float64),
        "prediction_sum": np.zeros(7, dtype=np.float64),
        "prediction_square_sum": np.zeros(7, dtype=np.float64),
        "error_sum": np.zeros(7, dtype=np.float64),
        "error_square_sum": np.zeros(7, dtype=np.float64),
    }


def distribution(sum_: np.ndarray, square_sum: np.ndarray, min_: np.ndarray, max_: np.ndarray, count: int) -> dict[str, list[float]]:
    mean = sum_ / count
    variance = np.maximum(square_sum / count - np.square(mean), 0.0)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "min": min_.tolist(), "max": max_.tolist()}


def finalize(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = int(accumulator["action_count_per_dimension"])
    if count == 0:
        raise ValueError("empty task accumulator")
    prediction_mean = accumulator["prediction_sum"] / count
    target_mean = accumulator["target_sum"] / count
    error_mean = accumulator["error_sum"] / count
    return {
        "feature_rows": int(accumulator["rows"]),
        "action_count_per_dimension": count,
        "raw_action_mse": float(accumulator["squared_error"] / (count * 7)),
        "target_raw_distribution": distribution(
            accumulator["target_sum"], accumulator["target_square_sum"], accumulator["target_min"], accumulator["target_max"], count
        ),
        "target_normalized_distribution": distribution(
            accumulator["normalized_sum"], accumulator["normalized_square_sum"], accumulator["normalized_min"], accumulator["normalized_max"], count
        ),
        "prediction_raw_mean": prediction_mean.tolist(),
        "shared_actor_raw_mean_bias_prediction_minus_target": (prediction_mean - target_mean).tolist(),
        "shared_actor_raw_mean_bias_l2": float(np.linalg.norm(prediction_mean - target_mean)),
        "shared_actor_raw_error_rmse_per_dimension": np.sqrt(accumulator["error_square_sum"] / count).tolist(),
    }


@torch.inference_mode()
def collect_split(
    path: Path,
    model: C0Actor,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    normalization = checkpoint["normalization"]
    action_mean_np = np.asarray(normalization["action"]["mean"], dtype=np.float32).reshape(1, 1, 7)
    action_std_np = np.asarray(normalization["action"]["std"], dtype=np.float32).reshape(1, 1, 7)
    proprio_mean = torch.tensor(normalization["proprio"]["mean"], dtype=torch.float32, device=device).view(1, -1)
    proprio_std = torch.tensor(normalization["proprio"]["std"], dtype=torch.float32, device=device).view(1, -1)
    action_mean = torch.tensor(action_mean_np, dtype=torch.float32, device=device)
    action_std = torch.tensor(action_std_np, dtype=torch.float32, device=device)
    accumulators: dict[int, dict[str, Any]] = defaultdict(initial_accumulator)
    with h5py.File(path, "r") as feature_file:
        language_features = feature_file["task_metadata/language_features"][:].astype(np.float32)
        rows = int(feature_file["task_id"].shape[0])
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            task_ids = feature_file["task_id"][start:stop].astype(np.int64)
            images = feature_file["image_features"][start:stop].astype(np.float32)
            proprio = feature_file["proprio_raw"][start:stop].astype(np.float32)
            target = feature_file["action_chunks"][start:stop].astype(np.float32)
            image_tensor = torch.from_numpy(images).to(device)
            proprio_tensor = torch.from_numpy(proprio).to(device)
            language_tensor = torch.from_numpy(language_features[task_ids]).to(device)
            prediction = model(image_tensor, language_tensor, (proprio_tensor - proprio_mean) / proprio_std)
            prediction_raw = (prediction * action_std + action_mean).cpu().numpy()
            normalized_target = (target - action_mean_np) / action_std_np
            for task_id in np.unique(task_ids):
                mask = task_ids == task_id
                targets = target[mask].astype(np.float64)
                normalized_targets = normalized_target[mask].astype(np.float64)
                predictions = prediction_raw[mask].astype(np.float64)
                errors = predictions - targets
                acc = accumulators[int(task_id)]
                acc["rows"] += int(mask.sum())
                acc["action_count_per_dimension"] += int(np.prod(targets.shape[:-1]))
                acc["squared_error"] += float(np.square(errors).sum())
                for prefix, values in (("target", targets), ("normalized", normalized_targets)):
                    flattened = values.reshape(-1, 7)
                    acc[f"{prefix}_sum"] += flattened.sum(axis=0)
                    acc[f"{prefix}_square_sum"] += np.square(flattened).sum(axis=0)
                    acc[f"{prefix}_min"] = np.minimum(acc[f"{prefix}_min"], flattened.min(axis=0))
                    acc[f"{prefix}_max"] = np.maximum(acc[f"{prefix}_max"], flattened.max(axis=0))
                flattened_prediction = predictions.reshape(-1, 7)
                flattened_error = errors.reshape(-1, 7)
                acc["prediction_sum"] += flattened_prediction.sum(axis=0)
                acc["prediction_square_sum"] += np.square(flattened_prediction).sum(axis=0)
                acc["error_sum"] += flattened_error.sum(axis=0)
                acc["error_square_sum"] += np.square(flattened_error).sum(axis=0)
    return {task_id: finalize(acc) for task_id, acc in sorted(accumulators.items())}


def overall(per_task: dict[int, dict[str, Any]]) -> dict[str, float]:
    total_elements = sum(item["action_count_per_dimension"] * 7 for item in per_task.values())
    weighted_mse = sum(item["raw_action_mse"] * item["action_count_per_dimension"] * 7 for item in per_task.values()) / total_elements
    macro_mse = float(np.mean([item["raw_action_mse"] for item in per_task.values()]))
    return {"weighted_raw_action_mse": float(weighted_mse), "macro_task_raw_action_mse": macro_mse}


def language_audit(feature_path: Path) -> dict[str, Any]:
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    with h5py.File(feature_path, "r") as feature_file:
        task_ids = feature_file["task_metadata/task_id"][:].astype(np.int64)
        names = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in feature_file["task_metadata/task_name"][:]]
        languages = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in feature_file["task_metadata/task_language"][:]]
        features = feature_file["task_metadata/language_features"][:].astype(np.float32)
    expected = [(suite.get_task(int(task_id)).name, suite.get_task(int(task_id)).language) for task_id in task_ids]
    mapping = [
        {
            "task_id": int(task_id),
            "stored_name": names[row],
            "stored_language": languages[row],
            "official_name": expected[row][0],
            "official_language": expected[row][1],
            "mapping_matches_official": names[row] == expected[row][0] and languages[row] == expected[row][1],
        }
        for row, task_id in enumerate(task_ids)
    ]
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~np.eye(len(cosine), dtype=bool)]
    return {
        "tasks": len(mapping),
        "all_task_ids_unique": len(set(int(item) for item in task_ids)) == len(task_ids),
        "all_languages_unique": len(set(languages)) == len(languages),
        "all_mapping_matches_official": all(item["mapping_matches_official"] for item in mapping),
        "off_diagonal_cosine_min": float(off_diagonal.min()),
        "off_diagonal_cosine_max": float(off_diagonal.max()),
        "off_diagonal_cosine_mean": float(off_diagonal.mean()),
        "mapping": mapping,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "artifacts/stage1/features/clip_vit_b32_h8_p12_upright")
    parser.add_argument("--multitask-checkpoint", type=Path, default=PROJECT_ROOT / "results/c0/seed11/best.pt")
    parser.add_argument("--single-task0-checkpoint", type=Path, default=PROJECT_ROOT / "results/c0/task0_seed11_upright/best.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/stage1/c0_validation_diagnostic.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (args.multitask_checkpoint, args.single_task0_checkpoint, args.features_dir / "train.hdf5", args.features_dir / "validation.hdf5"):
        if not path.is_file():
            raise FileNotFoundError(path)

    multi_model, multi_checkpoint = build_model(args.multitask_checkpoint, device)
    single_model, single_checkpoint = build_model(args.single_task0_checkpoint, device)
    multi_train = collect_split(args.features_dir / "train.hdf5", multi_model, multi_checkpoint, device, args.batch_size)
    multi_validation = collect_split(args.features_dir / "validation.hdf5", multi_model, multi_checkpoint, device, args.batch_size)
    single_train = collect_split(args.features_dir / "train.hdf5", single_model, single_checkpoint, device, args.batch_size)
    single_validation = collect_split(args.features_dir / "validation.hdf5", single_model, single_checkpoint, device, args.batch_size)
    if set(multi_train) != set(range(10)) or set(multi_validation) != set(range(10)):
        raise ValueError("expected all ten tasks in canonical feature store")

    task0_comparison = {
        "multi_task_train": multi_train[0],
        "multi_task_validation": multi_validation[0],
        "single_task0_train": single_train[0],
        "single_task0_validation": single_validation[0],
        "validation_raw_mse_ratio_multi_over_single": float(multi_validation[0]["raw_action_mse"] / single_validation[0]["raw_action_mse"]),
    }
    result = {
        "schema_version": 1,
        "mode": "validation_only_offline_diagnostic",
        "prohibited_actions": ["no environment evaluation", "no hyperparameter update", "no training", "no checkpoint overwrite"],
        "features": {
            "directory": str(args.features_dir.relative_to(PROJECT_ROOT)),
            "train_sha256": sha256(args.features_dir / "train.hdf5"),
            "validation_sha256": sha256(args.features_dir / "validation.hdf5"),
        },
        "checkpoints": {
            "multitask": {"path": str(args.multitask_checkpoint.relative_to(PROJECT_ROOT)), "sha256": sha256(args.multitask_checkpoint), "epoch": int(multi_checkpoint["epoch"]), "selection_metric": "overall validation raw_action_mse"},
            "single_task0": {"path": str(args.single_task0_checkpoint.relative_to(PROJECT_ROOT)), "sha256": sha256(args.single_task0_checkpoint), "epoch": int(single_checkpoint["epoch"])},
        },
        "language_and_task_mapping": language_audit(args.features_dir / "train.hdf5"),
        "multitask": {
            "train_per_task": multi_train,
            "validation_per_task": multi_validation,
            "train_overall": overall(multi_train),
            "validation_overall": overall(multi_validation),
            "selection_limitation": "The saved best checkpoint was selected by weighted overall raw MSE. Per-task checkpoint history was not saved, so this audit can quantify dispersion at the selected checkpoint but cannot claim which task-specific epoch would have been best.",
        },
        "task0_single_vs_multitask": task0_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation_overall": result["multitask"]["validation_overall"],
                "validation_per_task_raw_mse": {task_id: values["raw_action_mse"] for task_id, values in multi_validation.items()},
                "task0_multi_vs_single_ratio": task0_comparison["validation_raw_mse_ratio_multi_over_single"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"C0 validation diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
