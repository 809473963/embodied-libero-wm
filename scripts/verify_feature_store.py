#!/usr/bin/env python3
"""Independently validate a frozen feature store before C0 training consumes it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_DATASETS = {
    "image_features": (512,),
    "proprio_raw": (12,),
    "action_chunks": (8, 7),
    "task_id": (),
    "demo_id": (),
    "timestep": (),
}
FORBIDDEN_NAMES = {
    "states",
    "robot_states",
    "rewards",
    "dones",
    "success",
    "object_pose",
    "eye_in_hand_rgb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_split(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    require(path.is_file(), f"missing feature file: {path}")
    actual_hash = sha256(path)
    require(actual_hash == expected["sha256"], f"{path}: SHA-256 differs from manifest")
    require(path.stat().st_size == expected["bytes"], f"{path}: byte size differs from manifest")
    with h5py.File(path, "r") as feature_file:
        require(feature_file.attrs["schema_version"] == 1, f"{path}: schema version")
        names = set(feature_file.keys())
        require(not (names & FORBIDDEN_NAMES), f"{path}: forbidden top-level fields {names & FORBIDDEN_NAMES}")
        sample_count = int(feature_file["task_id"].shape[0])
        require(sample_count == expected["samples"], f"{path}: sample count")
        for name, tail_shape in REQUIRED_DATASETS.items():
            require(name in feature_file, f"{path}: missing {name}")
            require(feature_file[name].shape == (sample_count, *tail_shape), f"{path}: bad {name} shape")
        image_features = feature_file["image_features"][:]
        proprio = feature_file["proprio_raw"][:]
        actions = feature_file["action_chunks"][:]
        require(np.isfinite(image_features).all(), f"{path}: non-finite image feature")
        require(np.isfinite(proprio).all(), f"{path}: non-finite proprioception")
        require(np.isfinite(actions).all(), f"{path}: non-finite action")
        norms = np.linalg.norm(image_features.astype(np.float32), axis=1)
        require(np.allclose(norms, 1.0, atol=2e-3), f"{path}: CLIP feature norm drift")
        metadata = feature_file["task_metadata"]
        require(metadata["language_features"].shape == (10, 512), f"{path}: language features")
        require(np.isfinite(metadata["language_features"][:]).all(), f"{path}: non-finite language feature")
        return {
            "split": str(feature_file.attrs["split"]),
            "file": str(path),
            "sha256": actual_hash,
            "samples": sample_count,
            "image_feature_norm": {"min": float(norms.min()), "max": float(norms.max())},
            "task_ids": sorted(np.unique(feature_file["task_id"][:]).astype(int).tolist()),
            "demo_ids": sorted(np.unique(feature_file["demo_id"][:]).astype(int).tolist()),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    split_results = []
    seen_demo_pairs: set[tuple[int, int]] = set()
    for expected in manifest["files"]:
        path = root / Path(expected["file"]).name
        result = verify_split(path, expected)
        current_pairs = {(task_id, demo_id) for task_id in result["task_ids"] for demo_id in result["demo_ids"]}
        # Every task has the same predefined demo split, so this cross-product is
        # exact and makes accidental split overlap visible without per-row scans.
        require(not (seen_demo_pairs & current_pairs), "train/validation demo IDs overlap")
        seen_demo_pairs.update(current_pairs)
        split_results.append(result)
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "splits": split_results,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"feature verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
