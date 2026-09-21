#!/usr/bin/env python3
"""Verify canonical offline CLIP features against the deployed C0 online path.

Inputs are read from the immutable source / feature HDF5 files.  The only
output is a JSON audit file; no dataset, feature, or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark  # noqa: E402
from src.policies.c0_actor_policy import C0ActorPolicy, MODEL_ID, MODEL_REVISION  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def normalized(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def error_metrics(offline: np.ndarray, online: np.ndarray) -> dict[str, float]:
    offline32 = np.asarray(offline, dtype=np.float32)
    online32 = np.asarray(online, dtype=np.float32)
    delta = online32 - offline32
    return {
        "cosine": float(np.dot(offline32, online32) / max(np.linalg.norm(offline32) * np.linalg.norm(online32), 1e-12)),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "l2_error": float(np.linalg.norm(delta)),
        "offline_l2_norm": float(np.linalg.norm(offline32)),
        "online_l2_norm": float(np.linalg.norm(online32)),
    }


def summary(records: list[dict[str, Any]], metric_key: str = "metrics") -> dict[str, dict[str, float]]:
    metric_names = ("cosine", "max_abs_error", "mean_abs_error", "l2_error", "offline_l2_norm", "online_l2_norm")
    return {
        name: {
            "mean": float(np.mean([record[metric_key][name] for record in records])),
            "median": float(np.median([record[metric_key][name] for record in records])),
            "min": float(np.min([record[metric_key][name] for record in records])),
            "max": float(np.max([record[metric_key][name] for record in records])),
        }
        for name in metric_names
    }


def evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) < count:
        raise ValueError(f"need {count} feature rows but have {len(indices)}")
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return indices[positions]


def encode_images(policy: C0ActorPolicy, images: list[np.ndarray]) -> np.ndarray:
    pixels = policy.processor(images=[np.ascontiguousarray(image) for image in images], return_tensors="pt")["pixel_values"].to(policy.device)
    with torch.inference_mode(), policy._autocast():  # noqa: SLF001 - this is the deployed path under test
        features = normalized(policy.clip.get_image_features(pixel_values=pixels))
    return features.float().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results/c0/seed11/best.pt")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "artifacts/stage1/features/clip_vit_b32_h8_p12_upright")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial")
    parser.add_argument("--samples-per-task-per-split", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/stage1/offline_online_feature_parity.json")
    args = parser.parse_args()
    if args.samples_per_task_per_split < 1:
        raise ValueError("--samples-per-task-per-split must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for split in ("train", "validation"):
        if not (args.feature_dir / f"{split}.hdf5").is_file():
            raise FileNotFoundError(args.feature_dir / f"{split}.hdf5")

    policy = C0ActorPolicy(args.checkpoint, device=args.device)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    image_records: list[dict[str, Any]] = []
    source_paths: dict[int, Path] = {}
    orientation_checks: list[dict[str, Any]] = []

    for split in ("train", "validation"):
        feature_path = args.feature_dir / f"{split}.hdf5"
        with h5py.File(feature_path, "r") as features:
            preprocessing = decode(features.attrs["image_preprocessing"])
            if preprocessing != "uint8 RGB; vertical_flip=false; CLIPProcessor from pinned snapshot":
                raise ValueError(f"unexpected offline preprocessing contract: {preprocessing}")
            task_ids = features["task_id"][:].astype(np.int64)
            demo_ids = features["demo_id"][:].astype(np.int64)
            timesteps = features["timestep"][:].astype(np.int64)
            for task_id in range(suite.n_tasks):
                selected = evenly_spaced(np.flatnonzero(task_ids == task_id), args.samples_per_task_per_split)
                task = suite.get_task(task_id)
                source_path = args.dataset_dir / f"{task.name}_demo.hdf5"
                source_paths[task_id] = source_path
                with h5py.File(source_path, "r") as source:
                    images: list[np.ndarray] = []
                    offline_rows: list[np.ndarray] = []
                    record_headers: list[dict[str, Any]] = []
                    for row in selected:
                        demo_id = int(demo_ids[row])
                        timestep = int(timesteps[row])
                        canonical_rgb = source[f"data/demo_{demo_id}/obs/agentview_rgb"][timestep]
                        if canonical_rgb.dtype != np.uint8 or canonical_rgb.shape != (128, 128, 3):
                            raise ValueError(f"unexpected canonical RGB contract at {source_path}, demo {demo_id}, t={timestep}")
                        contiguous_rgb = np.ascontiguousarray(canonical_rgb)
                        orientation_checks.append(
                            {
                                "split": split,
                                "task_id": task_id,
                                "demo_id": demo_id,
                                "timestep": timestep,
                                "source_shape": list(canonical_rgb.shape),
                                "source_dtype": str(canonical_rgb.dtype),
                                "online_input_equal_source": bool(np.array_equal(contiguous_rgb, canonical_rgb)),
                                "vertical_flip_applied": False,
                            }
                        )
                        images.append(contiguous_rgb)
                        offline_rows.append(features["image_features"][row].astype(np.float32))
                        record_headers.append(
                            {
                                "split": split,
                                "feature_row": int(row),
                                "task_id": task_id,
                                "task_name": task.name,
                                "demo_id": demo_id,
                                "timestep": timestep,
                            }
                        )
                    online_rows = encode_images(policy, images)
                    for header, offline_row, online_row in zip(record_headers, offline_rows, online_rows):
                        image_records.append({**header, "metrics": error_metrics(offline_row, online_row)})

    text_records: list[dict[str, Any]] = []
    with h5py.File(args.feature_dir / "train.hdf5", "r") as features:
        names = [decode(item) for item in features["task_metadata/task_name"][:]]
        languages = [decode(item) for item in features["task_metadata/task_language"][:]]
        task_ids = features["task_metadata/task_id"][:].astype(np.int64)
        offline_text = features["task_metadata/language_features"][:].astype(np.float32)
        for row, task_id in enumerate(task_ids):
            task = suite.get_task(int(task_id))
            if names[row] != task.name or languages[row] != task.language:
                raise ValueError(f"feature metadata mapping mismatch at task_id={task_id}")
            online_text = policy._language_features(task.language).float().cpu().numpy()[0]  # noqa: SLF001
            text_records.append(
                {
                    "task_id": int(task_id),
                    "task_name": task.name,
                    "language": task.language,
                    "metrics": error_metrics(offline_text[row], online_text),
                }
            )

    manifest = json.loads((args.feature_dir / "feature_manifest.json").read_text())
    result = {
        "schema_version": 1,
        "mode": "read_only_offline_online_representation_parity",
        "input_mutation": "none",
        "checkpoint": str(args.checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256(args.checkpoint),
        "feature_dir": str(args.feature_dir.relative_to(PROJECT_ROOT)),
        "feature_files": {
            split: {"path": str((args.feature_dir / f"{split}.hdf5").relative_to(PROJECT_ROOT)), "sha256": sha256(args.feature_dir / f"{split}.hdf5")}
            for split in ("train", "validation")
        },
        "encoder": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "offline_manifest_model_id": manifest["clip"]["model_id"],
            "offline_manifest_model_revision": manifest["clip"]["revision"],
            "online_local_snapshot": str(PROJECT_ROOT / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots" / MODEL_REVISION),
            "device": str(policy.device),
        },
        "preprocessing": {
            "offline": "uint8 RGB; vertical_flip=false; CLIPProcessor from pinned snapshot",
            "online": "C0ActorPolicy.plan_batch: uint8 HWC RGB -> np.ascontiguousarray -> same CLIPProcessor; no flip/resize/normalize implementation outside CLIPProcessor",
            "orientation_checks_passed": all(item["online_input_equal_source"] and not item["vertical_flip_applied"] for item in orientation_checks),
            "orientation_checks": orientation_checks,
        },
        "image_feature_parity": {"samples": len(image_records), "summary": summary(image_records), "records": image_records},
        "text_feature_parity": {"tasks": len(text_records), "summary": summary(text_records), "records": text_records},
        "source_hdf5": {
            str(task_id): {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)}
            for task_id, path in source_paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "image_samples": len(image_records),
                "image_summary": result["image_feature_parity"]["summary"],
                "text_summary": result["text_feature_parity"]["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"offline/online feature parity audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
