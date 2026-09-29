from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.probes.visual_target import (
    FEATURE_NAMES,
    FEATURE_SET_GROUPS,
    MASK_NAMES,
    all_patch_coverages,
    build_visual_pair_dataset,
    masked_variants,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/visual_target_suitability_protocol.json"
INVENTORY = ROOT / "results/visual_target_suitability/mask_inventory_v1_2.json"
BLOCKED_INVENTORY = ROOT / "results/visual_target_suitability/mask_inventory.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_protocol_hashes_split_and_forbidden_boundaries() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_BRANCH_MASK_COLLECTION_OR_FEATURE_RESULTS"
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert protocol["partitions"]["historical_reference_only"] == [30, 35]
    assert set(range(40, 50)).issubset(protocol["partitions"]["forbidden_init_state_indices"])
    assert protocol["decision"]["controller_authorization"] is False
    assert subprocess.check_output(
        ["git", "cat-file", "-t", protocol["frozen_upstream"]["primary_tag"]],
        cwd=ROOT,
        text=True,
    ).strip() == "tag"
    assert subprocess.check_output(
        ["git", "rev-parse", f'{protocol["frozen_upstream"]["primary_tag"]}^{{}}'],
        cwd=ROOT,
        text=True,
    ).strip() == protocol["frozen_upstream"]["primary_commit"]
    for item in protocol["frozen_upstream"]["files"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    for item in protocol["implementation"].values():
        assert sha256(ROOT / item["path"]) == item["sha256"]
    assert protocol["mask_inventory"]["sha256"] == sha256(INVENTORY)
    forbidden = " ".join(protocol["forbidden"])
    for phrase in ("40-49", "success", "World Model", "seed23/47", "overwrite"):
        assert phrase in forbidden


def test_corrected_inventory_passes_and_blocked_orientation_is_not_admitted() -> None:
    admitted = json.loads(INVENTORY.read_text())
    blocked = json.loads(BLOCKED_INVENTORY.read_text())
    assert admitted["status"] == "COMPLETE"
    assert all(admitted["aggregate_checks"].values())
    assert len(admitted["tasks"]) == 10
    assert {row["task_id"] for row in admitted["tasks"]} == set(range(10))
    assert all(row["relevant_objects"] for row in admitted["tasks"])
    assert all(set(row["geom_ids"]) == {"robot", "relevant", "other_object"} for row in admitted["tasks"])
    assert blocked["status"] == "BLOCKED_IMAGE_ORIENTATION_PRECHECK"
    assert blocked["admitted_for_protocol"] is False
    assert sha256(BLOCKED_INVENTORY) != sha256(INVENTORY)


def test_mask_variants_and_patch_alignment_preserve_frozen_semantics() -> None:
    masks = np.zeros((4, 128, 128), dtype=bool)
    masks[0, :, :32] = True
    masks[1, :, 32:64] = True
    masks[2, :, 64:96] = True
    masks[3, :, 96:] = True
    image = np.arange(128 * 128 * 3, dtype=np.uint32).reshape(128, 128, 3).astype(np.uint8)
    replacement = np.asarray([17, 23, 29], dtype=np.uint8)
    variants = masked_variants(image, masks, replacement)
    assert set(variants) == {
        "full",
        "robot_only",
        "relevant_object_only",
        "robot_removed",
        "relevant_object_removed",
        "background_only",
    }
    assert np.array_equal(variants["full"], image)
    assert np.array_equal(variants["robot_only"][masks[0]], image[masks[0]])
    assert np.all(variants["robot_only"][~masks[0]] == replacement)
    assert np.all(variants["background_only"][masks[0] | masks[1]] == replacement)
    assert np.array_equal(variants["background_only"][masks[2] | masks[3]], image[masks[2] | masks[3]])

    batched = np.broadcast_to(masks, (2, 3, 8, 4, 128, 128)).copy()
    coverage = all_patch_coverages(batched)
    assert coverage.shape == (2, 3, 8, 4, 49)
    assert np.allclose(coverage.sum(axis=3), 1.0, atol=1e-6)
    assert np.all((coverage >= 0.0) & (coverage <= 1.0))


def test_synthetic_v0_v4_contract_shapes_are_finite() -> None:
    rng = np.random.default_rng(223001)
    branches = 2
    real_rgb = rng.integers(0, 256, size=(branches, 3, 8, 128, 128, 3), dtype=np.uint8)
    real_z = rng.normal(size=(branches, 3, 8, 512)).astype(np.float32)
    real_z /= np.linalg.norm(real_z, axis=-1, keepdims=True)
    candidates = rng.normal(size=(branches, 3, 8, 7)).astype(np.float32)
    masks = np.zeros((branches, 3, 8, 4, 128, 128), dtype=bool)
    masks[..., 0, :, :32] = True
    masks[..., 1, :, 32:64] = True
    masks[..., 2, :, 64:96] = True
    masks[..., 3, :, 96:] = True
    no_robot_relevant = masks[..., 1, :, :].copy()
    patch_tokens = rng.normal(size=(branches, 3, 8, 49, 512)).astype(np.float32)
    patch_tokens /= np.linalg.norm(patch_tokens, axis=-1, keepdims=True)
    coverage = all_patch_coverages(masks)
    dataset = build_visual_pair_dataset(
        {
            "real_rgb": real_rgb,
            "real_z": real_z,
            "candidates": candidates,
            "task_id": np.asarray([0, 1]),
            "init_state_index": np.asarray([31, 32]),
        },
        {"masks": masks, "no_robot_relevant_mask": no_robot_relevant},
        patch_tokens,
        coverage,
    )
    assert list(FEATURE_SET_GROUPS) == ["V0", "V1", "V2", "V3", "V4"]
    expected_dims = {
        "V0": len(FEATURE_NAMES["full_pixel"]),
        "V1": len(FEATURE_NAMES["full_pixel"]) + len(FEATURE_NAMES["robot_pixel"]),
        "V2": 4 + 6 + 6,
        "V3": 4 + 6 + 6 + 4,
        "V4": 4 + 6 + 6 + 4 + 11,
    }
    assert dataset["target_clip"].shape == (branches * 3 * 8,)
    assert dataset["target_delta_z"].shape == (branches * 3 * 8, 512)
    for name, dimensions in expected_dims.items():
        assert dataset["sets"][name].shape == (branches * 3 * 8, dimensions)
        assert np.isfinite(dataset["sets"][name]).all()
    assert tuple(MASK_NAMES) == ("robot", "relevant", "other", "background")
