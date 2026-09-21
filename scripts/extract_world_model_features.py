#!/usr/bin/env python3
"""Build the D13 observation-aligned feature store; no world model is trained."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark  # noqa: E402
from src.policies.c0_actor_policy import MODEL_ID, MODEL_REVISION  # noqa: E402


HORIZON = 8
TRAIN_DEMO_IDS = tuple(range(45))
VALIDATION_DEMO_IDS = tuple(range(45, 50))
PROPRIO_FIELDS = ("ee_pos", "gripper_states", "joint_states")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_clip(device: torch.device) -> tuple[CLIPModel, CLIPProcessor, Path]:
    """Load exactly the locally pinned C0 snapshot; never substitute a revision."""

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=PROJECT_ROOT / ".cache/huggingface/hub",
            allow_patterns=["*.json", "vocab.json", "merges.txt", "pytorch_model.bin"],
            local_files_only=True,
        )
    )
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor, snapshot


def encode_images(
    model: CLIPModel, processor: CLIPProcessor, images: np.ndarray, device: torch.device
) -> np.ndarray:
    """Match C0: upright uint8 RGB -> CLIPProcessor -> normalized 512-D feature."""

    batch = processor(images=[np.ascontiguousarray(image) for image in images], return_tensors="pt")
    pixels = batch["pixel_values"].to(device)
    with torch.inference_mode(), autocast(device):
        features = normalized(model.get_image_features(pixel_values=pixels))
    return features.float().cpu().numpy().astype(np.float16)


def encode_languages(
    model: CLIPModel, processor: CLIPProcessor, languages: list[str], device: torch.device
) -> np.ndarray:
    tokens = processor(text=languages, padding=True, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.inference_mode(), autocast(device):
        features = normalized(model.get_text_features(**tokens))
    return features.float().cpu().numpy().astype(np.float16)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_hashes() -> dict[str, str]:
    manifest = json.loads((PROJECT_ROOT / "artifacts/stage0/data_manifest.json").read_text())
    return {entry["name"]: entry["sha256"] for entry in manifest["files"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="replace only this script's existing output store")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    output_dir = args.output_dir.resolve()
    store_path = output_dir / "observations.hdf5"
    manifest_path = output_dir / "feature_manifest.json"
    if (store_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite D13 output {output_dir}; pass --overwrite after checking hashes")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_store = store_path.with_suffix(store_path.suffix + ".part")
    if temporary_store.exists():
        temporary_store.unlink()

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    model, processor, snapshot = load_clip(device)
    task_metadata = [
        {"task_id": task_id, "task_name": suite.get_task(task_id).name, "task_language": suite.get_task(task_id).language}
        for task_id in range(suite.n_tasks)
    ]
    language_features = encode_languages(model, processor, [item["task_language"] for item in task_metadata], device)
    source_file_hashes = source_hashes()
    task_summaries: list[dict[str, Any]] = []
    compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

    try:
        with h5py.File(temporary_store, "w") as store:
            store.attrs.update(
                {
                    "schema_version": 1,
                    "world_model_data_contract": "d13-v1",
                    "horizon": HORIZON,
                    "store_kind": "observation_aligned_episode_trajectory",
                    "vision_model_id": MODEL_ID,
                    "vision_model_revision": MODEL_REVISION,
                    "image_preprocessing": "uint8 RGB; vertical_flip=false; CLIPProcessor from pinned snapshot",
                    "vision_latent": "512-D L2-normalized CLIP image projection; stored float16, dataset float32",
                    "proprio_fields": ",".join(PROPRIO_FIELDS),
                    "forbidden_fields": "states,robot_states,object_pose,rewards,dones,success,future_proprio,evaluation_episode",
                }
            )
            metadata_group = store.create_group("task_metadata")
            metadata_group.create_dataset("task_id", data=np.arange(suite.n_tasks, dtype=np.uint8))
            string_type = h5py.string_dtype(encoding="utf-8")
            metadata_group.create_dataset("task_name", data=np.asarray([item["task_name"] for item in task_metadata], dtype=string_type))
            metadata_group.create_dataset("task_language", data=np.asarray([item["task_language"] for item in task_metadata], dtype=string_type))
            metadata_group.create_dataset("language_features", data=language_features, **compression)
            tasks_group = store.create_group("tasks")

            for metadata in task_metadata:
                task_id = int(metadata["task_id"])
                source_path = args.dataset_dir / f"{metadata['task_name']}_demo.hdf5"
                require(source_path.is_file(), f"missing official source HDF5: {source_path}")
                expected_hash = source_file_hashes.get(source_path.name)
                require(expected_hash is not None, f"source hash missing from stage0 manifest: {source_path.name}")
                task_group = tasks_group.create_group(f"task_{task_id}")
                task_group.attrs.update(
                    {
                        "task_id": task_id,
                        "task_name": metadata["task_name"],
                        "task_language": metadata["task_language"],
                        "source_hdf5_relative_path": str(source_path.relative_to(PROJECT_ROOT)),
                        "source_hdf5_sha256": expected_hash,
                    }
                )
                demos_group = task_group.create_group("demos")
                demo_summaries: list[dict[str, Any]] = []
                with h5py.File(source_path, "r") as source:
                    for demo_id in range(50):
                        source_demo = source[f"data/demo_{demo_id}"]
                        observations = source_demo["obs"]
                        observation_count = int(observations["agentview_rgb"].shape[0])
                        require(observation_count == int(source_demo["actions"].shape[0]), f"{source_path}/demo_{demo_id}: obs/actions length mismatch")
                        require(observation_count > HORIZON, f"{source_path}/demo_{demo_id}: too short for H={HORIZON}")
                        require(observations["agentview_rgb"].shape == (observation_count, 128, 128, 3), f"{source_path}/demo_{demo_id}: RGB schema")
                        for field, width in zip(PROPRIO_FIELDS, (3, 2, 7)):
                            require(observations[field].shape == (observation_count, width), f"{source_path}/demo_{demo_id}: {field} schema")
                        demo_group = demos_group.create_group(f"demo_{demo_id}")
                        split = "train" if demo_id in TRAIN_DEMO_IDS else "validation"
                        demo_group.attrs.update(
                            {
                                "task_id": task_id,
                                "demo_id": demo_id,
                                "split": split,
                                "observation_count": observation_count,
                                "world_model_sample_count": observation_count - HORIZON,
                            }
                        )
                        vision_dataset = demo_group.create_dataset("vision", (observation_count, 512), dtype="f2", **compression)
                        for start in range(0, observation_count, args.batch_size):
                            stop = min(start + args.batch_size, observation_count)
                            vision_dataset[start:stop] = encode_images(
                                model, processor, observations["agentview_rgb"][start:stop], device
                            )
                        proprio = np.concatenate([observations[field][:] for field in PROPRIO_FIELDS], axis=1).astype(np.float32)
                        actions = source_demo["actions"][:].astype(np.float32)
                        require(proprio.shape == (observation_count, 12), f"{source_path}/demo_{demo_id}: proprio contract")
                        require(actions.shape == (observation_count, 7), f"{source_path}/demo_{demo_id}: action contract")
                        demo_group.create_dataset("proprio", data=proprio, **compression)
                        demo_group.create_dataset("actions", data=actions, **compression)
                        demo_group.create_dataset("timestep", data=np.arange(observation_count, dtype=np.int32), **compression)
                        demo_summaries.append(
                            {
                                "demo_id": demo_id,
                                "split": split,
                                "observation_count": observation_count,
                                "world_model_sample_count": observation_count - HORIZON,
                            }
                        )
                task_summaries.append(
                    {
                        **metadata,
                        "source_hdf5_relative_path": str(source_path.relative_to(PROJECT_ROOT)),
                        "source_hdf5_sha256": expected_hash,
                        "demos": demo_summaries,
                        "observation_count": sum(item["observation_count"] for item in demo_summaries),
                        "train_observation_count": sum(item["observation_count"] for item in demo_summaries if item["split"] == "train"),
                        "validation_observation_count": sum(item["observation_count"] for item in demo_summaries if item["split"] == "validation"),
                        "train_world_model_samples": sum(item["world_model_sample_count"] for item in demo_summaries if item["split"] == "train"),
                        "validation_world_model_samples": sum(item["world_model_sample_count"] for item in demo_summaries if item["split"] == "validation"),
                    }
                )
                print(json.dumps({"task_id": task_id, "observations": task_summaries[-1]["observation_count"], "wm_samples": task_summaries[-1]["train_world_model_samples"] + task_summaries[-1]["validation_world_model_samples"]}), flush=True)
        os.replace(temporary_store, store_path)
    except Exception:
        if temporary_store.exists():
            temporary_store.unlink()
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    store_hash = sha256(store_path)
    manifest = {
        "schema_version": 1,
        "world_model_data_contract": "d13-v1",
        "libero_commit": subprocess.check_output(["git", "-C", str(PROJECT_ROOT / "third_party/LIBERO"), "rev-parse", "HEAD"], text=True).strip(),
        "source_dataset_manifest": "artifacts/stage0/data_manifest.json",
        "split": {"train_demo_ids": list(TRAIN_DEMO_IDS), "validation_demo_ids": list(VALIDATION_DEMO_IDS)},
        "horizon": HORIZON,
        "clip": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(snapshot), "frozen": True, "projection_dim": 512},
        "image_preprocessing": "uint8 RGB; vertical_flip=false; CLIPProcessor from pinned snapshot",
        "proprio_fields": list(PROPRIO_FIELDS),
        "forbidden_fields": ["states", "robot_states", "object_pose", "reward", "done", "success", "future_proprio", "evaluation_episode"],
        "store": {"path": str(store_path.relative_to(PROJECT_ROOT)), "bytes": store_path.stat().st_size, "sha256": store_hash},
        "tasks": task_summaries,
        "totals": {
            "observations": sum(task["observation_count"] for task in task_summaries),
            "train_observations": sum(task["train_observation_count"] for task in task_summaries),
            "validation_observations": sum(task["validation_observation_count"] for task in task_summaries),
            "train_world_model_samples": sum(task["train_world_model_samples"] for task in task_summaries),
            "validation_world_model_samples": sum(task["validation_world_model_samples"] for task in task_summaries),
        },
    }
    manifest["totals"]["world_model_samples"] = manifest["totals"]["train_world_model_samples"] + manifest["totals"]["validation_world_model_samples"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"store": manifest["store"], "totals": manifest["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"world-model feature extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
