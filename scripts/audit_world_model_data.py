#!/usr/bin/env python3
"""Read-only acceptance audit for the D13 observation-aligned store."""

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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark  # noqa: E402
from src.d13 import WorldModelDataset  # noqa: E402


FORBIDDEN_FIELD_NAMES = {"states", "robot_states", "object_pose", "rewards", "reward", "dones", "done", "success", "future_proprio", "evaluation_episode"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_group(store: h5py.File, task_id: int, demo_id: int) -> h5py.Group:
    return store[f"tasks/task_{task_id}/demos/demo_{demo_id}"]


def sample_back_check(
    dataset: WorldModelDataset,
    store: h5py.File,
    source_dir: Path,
    task_id: int,
    demo_id: int,
    start_timestep: int,
) -> dict[str, Any]:
    """Independently read the per-episode store group and official source actions."""

    sample = dataset[dataset.find_index(task_id, demo_id, start_timestep)]
    group = episode_group(store, task_id, demo_id)
    k = start_timestep
    stop = k + 9
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    source_path = source_dir / f"{suite.get_task(task_id).name}_demo.hdf5"
    with h5py.File(source_path, "r") as source:
        source_actions = source[f"data/demo_{demo_id}/actions"][k + 1 : stop].astype(np.float32)
    expected_z0 = group["vision"][k].astype(np.float32)
    expected_actions = group["actions"][k + 1 : stop].astype(np.float32)
    expected_future = group["vision"][k + 1 : stop].astype(np.float32)
    checks = {
        "z0_equals_episode_vision_k": bool(torch.equal(sample["z0"], torch.from_numpy(expected_z0))),
        "actions_0_equals_source_action_k_plus_1": bool(np.array_equal(sample["actions"][0].numpy(), source_actions[0])),
        "actions_7_equals_source_action_k_plus_8": bool(np.array_equal(sample["actions"][7].numpy(), source_actions[7])),
        "actions_slice_equals_episode_actions": bool(torch.equal(sample["actions"], torch.from_numpy(expected_actions))),
        "z_future_0_equals_episode_vision_k_plus_1": bool(torch.equal(sample["z_future"][0], torch.from_numpy(expected_future[0]))),
        "z_future_7_equals_episode_vision_k_plus_8": bool(torch.equal(sample["z_future"][7], torch.from_numpy(expected_future[7]))),
        "z_future_slice_equals_episode_vision": bool(torch.equal(sample["z_future"], torch.from_numpy(expected_future))),
    }
    if not all(checks.values()):
        raise AssertionError(f"D13 index back-check failed for {(task_id, demo_id, k)}: {checks}")
    return {
        "task_id": task_id,
        "demo_id": demo_id,
        "start_timestep": k,
        "sample_shapes": {key: list(value.shape) for key, value in sample.items() if value.ndim > 0},
        "checks": checks,
    }


def oracle_field_audit(store: h5py.File) -> dict[str, Any]:
    found: list[str] = []

    def visit(name: str, item: h5py.Group | h5py.Dataset) -> None:
        leaf = name.rsplit("/", 1)[-1]
        if leaf in FORBIDDEN_FIELD_NAMES:
            found.append(name)

    store.visititems(visit)
    expected_root = {"task_metadata", "tasks"}
    if set(store.keys()) != expected_root:
        raise AssertionError(f"unexpected root groups: {sorted(store.keys())}")
    if found:
        raise AssertionError(f"oracle fields leaked into D13 store: {found}")
    return {
        "forbidden_names_checked": sorted(FORBIDDEN_FIELD_NAMES),
        "forbidden_fields_found": found,
        "passed": True,
    }


def task_counts(store: h5py.File) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for task_id in range(10):
        task_group = store[f"tasks/task_{task_id}"]
        demos: list[dict[str, int | str]] = []
        for demo_id in range(50):
            group = task_group[f"demos/demo_{demo_id}"]
            count = int(group.attrs["observation_count"])
            demos.append(
                {
                    "demo_id": demo_id,
                    "split": str(group.attrs["split"]),
                    "observation_count": count,
                    "world_model_sample_count": int(group.attrs["world_model_sample_count"]),
                }
            )
        result.append(
            {
                "task_id": task_id,
                "task_name": str(task_group.attrs["task_name"]),
                "demos": demos,
                "observations": sum(int(item["observation_count"]) for item in demos),
                "train_world_model_samples": sum(int(item["world_model_sample_count"]) for item in demos if item["split"] == "train"),
                "validation_world_model_samples": sum(int(item["world_model_sample_count"]) for item in demos if item["split"] == "validation"),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/feature_manifest.json")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/data_audit.json")
    args = parser.parse_args()
    train = WorldModelDataset(args.store, split="train")
    validation = WorldModelDataset(args.store, split="validation")
    try:
        train_demo_ids = {(record.task_id, record.demo_id) for record in train.records}
        validation_demo_ids = {(record.task_id, record.demo_id) for record in validation.records}
        if train_demo_ids & validation_demo_ids:
            raise AssertionError("train/validation episode leakage")
        with h5py.File(args.store, "r") as store:
            last_train_k = int(episode_group(store, 8, 44).attrs["observation_count"]) - 9
            last_validation_k = int(episode_group(store, 9, 49).attrs["observation_count"]) - 9
            back_checks = [
                sample_back_check(train, store, args.dataset_dir, 0, 0, 0),
                sample_back_check(train, store, args.dataset_dir, 4, 17, 5),
                sample_back_check(train, store, args.dataset_dir, 8, 44, last_train_k),
                sample_back_check(validation, store, args.dataset_dir, 0, 45, 0),
                sample_back_check(validation, store, args.dataset_dir, 6, 47, 5),
                sample_back_check(validation, store, args.dataset_dir, 9, 49, last_validation_k),
            ]
            fields = oracle_field_audit(store)
            counts = task_counts(store)
        batch = next(iter(DataLoader(train, batch_size=4, shuffle=False)))
        result = {
            "schema_version": 1,
            "world_model_data_contract": "d13-v1",
            "store": {"path": str(args.store.relative_to(PROJECT_ROOT)), "bytes": args.store.stat().st_size, "sha256": sha256(args.store)},
            "manifest": {"path": str(args.manifest.relative_to(PROJECT_ROOT)), "sha256": sha256(args.manifest)},
            "dataset_sizes": {"train": len(train), "validation": len(validation)},
            "split": {
                "train_episode_count": len(train_demo_ids),
                "validation_episode_count": len(validation_demo_ids),
                "intersection": sorted(train_demo_ids & validation_demo_ids),
                "passed": not bool(train_demo_ids & validation_demo_ids),
            },
            "real_sample_trace": back_checks[0],
            "real_batch_trace": {key: list(value.shape) for key, value in batch.items()},
            "index_back_checks": back_checks,
            "oracle_field_audit": fields,
            "task_counts": counts,
            "copy_current_state_shape": list(WorldModelDataset.copy_current_state_prediction(train[0]["z0"]).shape),
        }
    finally:
        train.close()
        validation.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"store": result["store"], "dataset_sizes": result["dataset_sizes"], "split": result["split"], "back_checks": len(back_checks)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"D13 data audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
