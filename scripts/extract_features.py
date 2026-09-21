#!/usr/bin/env python3
"""Extract frozen CLIP features from the official LIBERO-Spatial demonstrations.

The generated HDF5 files contain policy-visible inputs and action labels only.
They intentionally contain no MuJoCo state, reward, success, object pose, or
future visual observations.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

from huggingface_hub import snapshot_download
from libero.libero import benchmark
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
TRAIN_DEMO_IDS = tuple(range(45))
VALIDATION_DEMO_IDS = tuple(range(45, 50))
# Keep offline / online control inputs identical. See docs/data_contract.md for
# why the public 3-D ee_ori field is not part of C0 yet.
PROPRIO_FIELDS = ("ee_pos", "gripper_states", "joint_states")


@dataclass(frozen=True)
class DemoRecord:
    task_id: int
    task_name: str
    task_language: str
    path: Path
    demo_id: int
    transitions: int


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def hdf5_string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def collect_records(
    dataset_dir: Path, demo_ids: tuple[int, ...], horizon: int
) -> tuple[list[DemoRecord], list[dict[str, Any]], dict[str, Any]]:
    """Return fixed-order records and train-only action normalization statistics."""

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    records: list[DemoRecord] = []
    task_metadata: list[dict[str, Any]] = []
    action_count = 0
    action_sum = np.zeros(7, dtype=np.float64)
    action_sum_squared = np.zeros(7, dtype=np.float64)
    action_min = np.full(7, np.inf, dtype=np.float64)
    action_max = np.full(7, -np.inf, dtype=np.float64)

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        path = dataset_dir / f"{task.name}_demo.hdf5"
        require(path.is_file(), f"missing task data: {path}")
        task_metadata.append(
            {"task_id": task_id, "task_name": task.name, "task_language": task.language}
        )
        with h5py.File(path, "r") as h5_file:
            data = h5_file["data"]
            for demo_id in demo_ids:
                demo = data[f"demo_{demo_id}"]
                actions = demo["actions"]
                transitions = int(actions.shape[0])
                require(
                    transitions >= horizon,
                    f"{path}/demo_{demo_id} is shorter than action horizon {horizon}",
                )
                require(actions.shape[1:] == (7,), f"{path}/demo_{demo_id}: action dim")
                for field, expected_tail in {
                    "agentview_rgb": (128, 128, 3),
                    "ee_pos": (3,),
                    "ee_ori": (3,),
                    "gripper_states": (2,),
                    "joint_states": (7,),
                }.items():
                    require(
                        demo["obs"][field].shape == (transitions, *expected_tail),
                        f"{path}/demo_{demo_id}: bad obs/{field} shape",
                    )
                records.append(
                    DemoRecord(
                        task_id=task_id,
                        task_name=task.name,
                        task_language=task.language,
                        path=path,
                        demo_id=demo_id,
                        transitions=transitions,
                    )
                )

                # Only train records are passed here for normalization. It is
                # deliberately computed on raw actions, not test/validation.
                values = actions[...].astype(np.float64, copy=False)
                require(np.isfinite(values).all(), f"{path}/demo_{demo_id}: non-finite action")
                action_count += transitions
                action_sum += values.sum(axis=0)
                action_sum_squared += np.square(values).sum(axis=0)
                action_min = np.minimum(action_min, values.min(axis=0))
                action_max = np.maximum(action_max, values.max(axis=0))

    mean = action_sum / action_count
    variance = np.maximum(action_sum_squared / action_count - np.square(mean), 0.0)
    return records, task_metadata, {
        "count": action_count,
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "min": action_min.tolist(),
        "max": action_max.tolist(),
    }


def load_clip(device: torch.device) -> tuple[CLIPModel, CLIPProcessor, Path]:
    """Fetch the exact official snapshot through the current proxy, then load locally."""

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=["*.json", "vocab.json", "merges.txt", "pytorch_model.bin"],
        )
    )
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor, snapshot


def normalized_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def encode_languages(
    model: CLIPModel, processor: CLIPProcessor, task_metadata: list[dict[str, Any]], device: torch.device
) -> np.ndarray:
    language_batch = [str(item["task_language"]) for item in task_metadata]
    encoded = processor(text=language_batch, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode(), autocast_context(device):
        features = normalized_features(model.get_text_features(**encoded))
    return features.float().cpu().numpy().astype(np.float16)


def create_feature_file(
    path: Path,
    sample_count: int,
    task_metadata: list[dict[str, Any]],
    language_features: np.ndarray,
    split_name: str,
    horizon: int,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(path, "w")
    h5_file.attrs.update(
        {
            "schema_version": 1,
            "data_contract_version": 2,
            "split": split_name,
            "action_chunk_horizon": horizon,
            "vision_model_id": MODEL_ID,
            "vision_model_revision": MODEL_REVISION,
            "image_preprocessing": "uint8 RGB; vertical_flip=false; CLIPProcessor from pinned snapshot",
            "policy_visible_fields": "agentview_rgb, task_language, ee_pos, gripper_states, joint_states",
            "forbidden_fields": "states, robot_states, rewards, dones, object poses, init_state, model_file, eye_in_hand_rgb, ee_ori",
        }
    )
    compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    h5_file.create_dataset("image_features", (sample_count, 512), dtype="f2", **compression)
    h5_file.create_dataset("proprio_raw", (sample_count, 12), dtype="f4", **compression)
    h5_file.create_dataset("action_chunks", (sample_count, horizon, 7), dtype="f4", **compression)
    h5_file.create_dataset("task_id", (sample_count,), dtype="u1", **compression)
    h5_file.create_dataset("demo_id", (sample_count,), dtype="u1", **compression)
    h5_file.create_dataset("timestep", (sample_count,), dtype="i4", **compression)
    metadata = h5_file.create_group("task_metadata")
    metadata.create_dataset("task_id", data=np.asarray([item["task_id"] for item in task_metadata], dtype=np.uint8))
    metadata.create_dataset("task_name", data=hdf5_string_array([str(item["task_name"]) for item in task_metadata]))
    metadata.create_dataset("task_language", data=hdf5_string_array([str(item["task_language"]) for item in task_metadata]))
    metadata.create_dataset("language_features", data=language_features, **compression)
    return h5_file


def source_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_split(
    records: list[DemoRecord],
    task_metadata: list[dict[str, Any]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    output_path: Path,
    split_name: str,
    horizon: int,
    batch_size: int,
) -> dict[str, Any]:
    sample_count = sum(record.transitions - horizon + 1 for record in records)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    require(not temporary_path.exists(), f"incomplete output already exists: {temporary_path}")
    require(not output_path.exists(), f"output already exists: {output_path}")
    language_features = encode_languages(model, processor, task_metadata, device)
    written = 0
    total_batches = sum(
        (record.transitions - horizon + batch_size) // batch_size for record in records
    )
    current_batch = 0

    try:
        with create_feature_file(
            temporary_path,
            sample_count,
            task_metadata,
            language_features,
            split_name,
            horizon,
        ) as feature_file:
            for record in records:
                with h5py.File(record.path, "r") as source:
                    demo = source["data"][f"demo_{record.demo_id}"]
                    observations = demo["obs"]
                    actions = demo["actions"]
                    usable = record.transitions - horizon + 1
                    for start in range(0, usable, batch_size):
                        stop = min(start + batch_size, usable)
                        count = stop - start
                        images = observations["agentview_rgb"][start:stop]
                        # Official demonstrations and the live LIBERO
                        # observation already share the same upright RGB
                        # convention. Do not invert frozen-CLIP inputs.
                        images = np.ascontiguousarray(images)
                        # transformers 4.21 treats a rank-4 NumPy array as one
                        # image; pass individual HWC arrays explicitly.
                        pixel_values = processor(
                            images=list(images), return_tensors="pt"
                        )["pixel_values"].to(device)
                        with torch.inference_mode(), autocast_context(device):
                            image_features = normalized_features(
                                model.get_image_features(pixel_values=pixel_values)
                            )
                        proprio = np.concatenate(
                            [observations[field][start:stop] for field in PROPRIO_FIELDS], axis=1
                        ).astype(np.float32, copy=False)
                        action_chunks = np.stack(
                            [actions[t : t + horizon] for t in range(start, stop)], axis=0
                        ).astype(np.float32, copy=False)
                        destination = slice(written, written + count)
                        feature_file["image_features"][destination] = image_features.float().cpu().numpy().astype(np.float16)
                        feature_file["proprio_raw"][destination] = proprio
                        feature_file["action_chunks"][destination] = action_chunks
                        feature_file["task_id"][destination] = record.task_id
                        feature_file["demo_id"][destination] = record.demo_id
                        feature_file["timestep"][destination] = np.arange(start, stop, dtype=np.int32)
                        written += count
                        current_batch += 1
                        if current_batch % 100 == 0 or current_batch == total_batches:
                            print(
                                f"[{split_name}] batches {current_batch}/{total_batches}; "
                                f"samples {written}/{sample_count}",
                                flush=True,
                            )
            require(written == sample_count, f"{split_name}: wrote {written}, expected {sample_count}")
            feature_file.attrs["samples_written"] = written
            feature_file.flush()
        os.replace(temporary_path, output_path)
    except Exception:
        # Keep .part for inspection/recovery instead of silently deleting a
        # partially generated artifact.
        raise

    return {
        "split": split_name,
        "file": source_relative(output_path),
        "sha256": sha256(output_path),
        "bytes": output_path.stat().st_size,
        "samples": sample_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/stage1/features/clip_vit_b32_h8_p12_upright")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    require(args.horizon > 0, "--horizon must be positive")
    require(args.batch_size > 0, "--batch-size must be positive")
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA was requested but is unavailable")

    train_records, task_metadata, action_statistics = collect_records(
        args.dataset_dir, TRAIN_DEMO_IDS, args.horizon
    )
    validation_records, validation_task_metadata, _ = collect_records(
        args.dataset_dir, VALIDATION_DEMO_IDS, args.horizon
    )
    require(task_metadata == validation_task_metadata, "task metadata differs between splits")
    model, processor, model_snapshot = load_clip(device)

    output_dir = args.output_dir
    manifest_path = output_dir / "feature_manifest.json"
    require(not manifest_path.exists(), f"feature manifest already exists: {manifest_path}")
    train_result = write_split(
        train_records, task_metadata, model, processor, device,
        output_dir / "train.hdf5", "train", args.horizon, args.batch_size,
    )
    validation_result = write_split(
        validation_records, task_metadata, model, processor, device,
        output_dir / "validation.hdf5", "validation", args.horizon, args.batch_size,
    )
    manifest = {
        "schema_version": 1,
        "data_contract_version": 2,
        "libero_commit": git_commit(PROJECT_ROOT / "third_party/LIBERO"),
        "dataset_dir": source_relative(args.dataset_dir),
        "clip": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot": source_relative(model_snapshot),
            "projection_dim": 512,
            "frozen": True,
        },
        "image_preprocessing": "uint8 RGB; vertical_flip=false; CLIPProcessor from pinned snapshot",
        "proprio_raw_fields": list(PROPRIO_FIELDS),
        "proprio_raw_dim": 12,
        "action_dim": 7,
        "action_chunk_horizon": args.horizon,
        "action_statistics_train_only": action_statistics,
        "split": {"train_demo_ids": list(TRAIN_DEMO_IDS), "validation_demo_ids": list(VALIDATION_DEMO_IDS)},
        "device": str(device),
        "files": [train_result, validation_result],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"feature extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
