#!/usr/bin/env python3
"""Calibrate and evaluate the frozen Visual Target Suitability Audit."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_world_model_features import load_clip
from src.probes.missing_state import (
    fit_ridge,
    predict_ridge,
    ridge_from_arrays,
    ridge_to_arrays,
    scalar_metrics,
    select_ridge_alpha,
)
from src.probes.visual_target import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_SET_GROUPS,
    MASK_NAMES,
    PAIR_INDICES,
    PAIR_NAMES,
    VARIANT_NAMES,
    all_patch_coverages,
    build_visual_pair_dataset,
    masked_variants,
)


PROTOCOL_VERSION = "visual-target-suitability-audit-v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_protocol(path: Path, phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected visual-target protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_BRANCH_MASK_COLLECTION_OR_FEATURE_RESULTS":
        raise ValueError("protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise ValueError("protocol must be committed")
    for key in ("analyzer", "feature_module"):
        item = protocol["implementation"][key]
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"implementation changed after preregistration: {item['path']}")
    for item in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen upstream changed: {item['path']}")
    calibration = protocol["partitions"]["calibration"]
    for key in ("source_sidecar", "mask_sidecar", "mask_manifest"):
        if not (ROOT / calibration[key]).exists():
            raise FileNotFoundError(calibration[key])
    if not last_commit(ROOT / calibration["mask_manifest"]):
        raise RuntimeError("calibration mask manifest must be committed")
    if phase == "calibrate":
        forbidden = (
            protocol["partitions"]["evaluation"]["mask_sidecar"],
            protocol["partitions"]["evaluation"]["mask_manifest"],
            protocol["outputs"]["calibration_freeze"],
            protocol["outputs"]["diagnosis"],
        )
        for relative in forbidden:
            if (ROOT / relative).exists():
                raise RuntimeError(f"calibration refuses pre-existing artifact: {relative}")
    else:
        freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("calibration freeze must be committed before held-out analysis")
        freeze = json.loads(freeze_path.read_text())
        if freeze["status"] != "FROZEN_BEFORE_HELD_OUT_MASK_COLLECTION_OR_ANALYSIS":
            raise ValueError("calibration freeze status changed")
        evaluation = protocol["partitions"]["evaluation"]
        for key in ("source_sidecar", "mask_sidecar", "mask_manifest"):
            if not (ROOT / evaluation[key]).exists():
                raise FileNotFoundError(evaluation[key])
    return protocol, commit


def load_partition(protocol: dict[str, Any], partition: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    record = protocol["partitions"][partition]
    source_file = np.load(ROOT / record["source_sidecar"], allow_pickle=False)
    mask_file = np.load(ROOT / record["mask_sidecar"], allow_pickle=False)
    source = {name: np.asarray(source_file[name]) for name in source_file.files}
    masks = {name: np.asarray(mask_file[name]) for name in mask_file.files}
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        if not np.array_equal(source[key], masks[key]):
            raise ValueError(f"partition identity mismatch: {key}")
    if set(np.unique(source["init_state_index"]).tolist()) != set(record["init_state_indices"]):
        raise ValueError("partition init-state boundary changed")
    return source, masks


@torch.inference_mode()
def encode_triplet(
    model: Any,
    processor: Any,
    images: list[np.ndarray],
    device: torch.device,
    include_patch: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(images) != 3:
        raise ValueError("CLIP batch-shape policy requires exactly three candidate frames")
    pixels = processor(
        images=[np.ascontiguousarray(np.asarray(image, dtype=np.uint8)) for image in images],
        return_tensors="pt",
    )["pixel_values"].to(device)
    with autocast(device):
        output = model.vision_model(pixel_values=pixels, return_dict=True)
        global_features = model.visual_projection(output.pooler_output)
        global_features = F.normalize(global_features.float(), dim=-1, eps=1.0e-12)
        patch_features = None
        if include_patch:
            patch_hidden = model.vision_model.post_layernorm(output.last_hidden_state[:, 1:, :])
            patch_features = model.visual_projection(patch_hidden)
            patch_features = F.normalize(patch_features.float(), dim=-1, eps=1.0e-12)
    global_array = global_features.cpu().numpy().astype(np.float32)
    patch_array = None if patch_features is None else patch_features.cpu().numpy().astype(np.float32)
    if not np.isfinite(global_array).all() or (patch_array is not None and not np.isfinite(patch_array).all()):
        raise FloatingPointError("CLIP diagnostic feature contains NaN/Inf")
    return global_array, patch_array


def encode_partition(
    source: dict[str, np.ndarray],
    mask_data: dict[str, np.ndarray],
    replacement_rgb: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    model, processor, snapshot = load_clip(device)
    global_features = np.empty(source["real_rgb"].shape[:3] + (len(VARIANT_NAMES), 512), dtype=np.float16)
    patch_features = np.empty(source["real_rgb"].shape[:3] + (49, 512), dtype=np.float16)
    full_cosine_min = 1.0
    full_max_abs = 0.0
    for branch in range(source["real_rgb"].shape[0]):
        for horizon in range(8):
            variants_by_candidate = [
                masked_variants(
                    source["real_rgb"][branch, candidate, horizon],
                    mask_data["masks"][branch, candidate, horizon],
                    replacement_rgb,
                )
                for candidate in range(3)
            ]
            for variant_index, variant_name in enumerate(VARIANT_NAMES):
                images = [variants_by_candidate[candidate][variant_name] for candidate in range(3)]
                encoded, patch = encode_triplet(
                    model, processor, images, device, include_patch=variant_name == "full"
                )
                global_features[branch, :, horizon, variant_index] = encoded.astype(np.float16)
                if variant_name == "full":
                    if patch is None or patch.shape != (3, 49, 512):
                        raise RuntimeError("pinned CLIP did not expose the preregistered patch tensor")
                    patch_features[branch, :, horizon] = patch.astype(np.float16)
                    reference = source["real_z"][branch, :, horizon].astype(np.float32)
                    cosine = np.sum(encoded * reference, axis=-1) / np.maximum(
                        np.linalg.norm(encoded, axis=-1) * np.linalg.norm(reference, axis=-1), 1.0e-12
                    )
                    full_cosine_min = min(full_cosine_min, float(cosine.min()))
                    full_max_abs = max(full_max_abs, float(np.max(np.abs(encoded - reference))))
        print(json.dumps({"clip_features": branch + 1, "of": int(source["real_rgb"].shape[0])}), flush=True)
    coverage = all_patch_coverages(mask_data["masks"])
    check = {
        "snapshot": str(snapshot),
        "model": "openai/clip-vit-base-patch32",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "batch_shape_policy": "one branch/horizon/variant call with exactly A0,A1,A2 (B=3)",
        "full_reencode_min_cosine_to_frozen_real_z": full_cosine_min,
        "full_reencode_max_abs_difference_to_frozen_real_z": full_max_abs,
        "patch_shape": [49, 512],
        "patch_grid": [7, 7],
        "patch_projection": "vision_model last_hidden_state[:,1:] -> vision_model.post_layernorm -> visual_projection -> L2 normalize",
        "mask_alignment": "128x128 binary -> nearest 224x224 -> mean pool nonoverlapping 32x32 -> 7x7 coverage",
    }
    return global_features, patch_features, coverage, check


def variant_distances(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    rows = []
    for branch in range(value.shape[0]):
        for first, second in PAIR_INDICES:
            for horizon in range(8):
                a = value[branch, first, horizon]
                b = value[branch, second, horizon]
                a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1.0e-12)
                b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1.0e-12)
                rows.append(np.maximum(1.0 - np.sum(a * b, axis=-1), 0.0))
    return np.asarray(rows, dtype=np.float64)


def correlation(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size < 3 or np.std(x) <= 1.0e-20 or np.std(y) <= 1.0e-20:
        return {"n": int(x.size), "pearson": None, "pearson_p": None, "spearman": None, "spearman_p": None}
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = correlation(first, second)["spearman"]
    return 0.0 if value is None else float(value)


def summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_indices(metadata: dict[str, np.ndarray], replicates: int, seed: int) -> list[np.ndarray]:
    branch = metadata["branch"]
    task = metadata["task"]
    rng = np.random.default_rng(seed)
    branches_by_task = {
        task_id: np.unique(branch[task == task_id]) for task_id in sorted(np.unique(task).tolist())
    }
    rows_by_branch = {int(value): np.flatnonzero(branch == value) for value in np.unique(branch)}
    draws = []
    for _ in range(replicates):
        rows = []
        for candidates in branches_by_task.values():
            sampled = rng.choice(candidates, size=candidates.size, replace=True)
            rows.extend(rows_by_branch[int(value)] for value in sampled)
        draws.append(np.concatenate(rows))
    return draws


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def metric_bootstrap(draws: list[np.ndarray], function: Callable[[np.ndarray], float]) -> list[float]:
    return interval([function(index) for index in draws])


def cut_association(
    value: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, np.ndarray],
    draws: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": correlation(value, target), "horizon": [], "task": [], "pair": []}
    if draws is not None:
        result["overall"]["spearman_ci95"] = metric_bootstrap(
            draws, lambda index: safe_spearman(value[index], target[index])
        )
    for name, values in (
        ("horizon", range(1, 9)),
        ("task", range(10)),
        ("pair", range(3)),
    ):
        for item in values:
            mask = metadata[name] == item
            label = PAIR_NAMES[item] if name == "pair" else item
            result[name].append({name: label, **correlation(value[mask], target[mask])})
    return result


def cut_summary(value: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": summary(value), "horizon": [], "task": [], "pair": []}
    for name, values in (("horizon", range(1, 9)), ("task", range(10)), ("pair", range(3))):
        for item in values:
            mask = metadata[name] == item
            label = PAIR_NAMES[item] if name == "pair" else item
            result[name].append({name: label, **summary(value[mask])})
    return result


def deranged_group(group: np.ndarray, metadata: dict[str, np.ndarray], seed: int) -> np.ndarray:
    result = np.asarray(group).copy()
    rng = np.random.default_rng(seed)
    for task in range(10):
        for horizon in range(1, 9):
            for pair in range(3):
                indices = np.flatnonzero(
                    (metadata["task"] == task)
                    & (metadata["horizon"] == horizon)
                    & (metadata["pair"] == pair)
                )
                if indices.size < 2:
                    raise RuntimeError("group shuffle cell has fewer than two branches")
                ordered = indices[np.argsort(metadata["branch"][indices], kind="stable")]
                shift = int(rng.integers(1, ordered.size))
                result[ordered] = group[np.roll(ordered, shift)]
    return result


def save_feature_sidecar(
    path: Path,
    global_features: np.ndarray,
    patch_features: np.ndarray,
    coverage: np.ndarray,
    dataset: dict[str, Any],
    variant: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "global_variant_features": global_features,
        "patch_features": patch_features,
        "patch_coverage": coverage,
        "variant_distances": variant.astype(np.float32),
        "target_clip": dataset["target_clip"].astype(np.float32),
        "target_delta_z": dataset["target_delta_z"].astype(np.float32),
    }
    arrays.update({f"metadata__{name}": value for name, value in dataset["metadata"].items()})
    arrays.update({f"group__{name}": value.astype(np.float32) for name, value in dataset["groups"].items()})
    arrays.update({f"scalar__{name}": value.astype(np.float32) for name, value in dataset["scalars"].items()})
    atomic_npz(path, arrays)


def make_calibration_montage(
    path: Path,
    source: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    dataset: dict[str, Any],
    selected_rows: dict[str, int],
) -> None:
    rows = []
    draw_records = []
    for label in ("low_p10", "median_p50", "high_p90"):
        row = int(selected_rows[label])
        branch = int(dataset["metadata"]["branch"][row])
        pair = int(dataset["metadata"]["pair"][row])
        horizon = int(dataset["metadata"]["horizon"][row]) - 1
        first, second = PAIR_INDICES[pair]
        image_a = source["real_rgb"][branch, first, horizon]
        image_b = source["real_rgb"][branch, second, horizon]
        difference = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16)).mean(axis=-1)
        heat = np.zeros_like(image_a)
        heat[..., 0] = np.clip(difference * 4.0, 0, 255).astype(np.uint8)
        overlay = image_a.copy()
        robot = masks["masks"][branch, first, horizon, 0]
        relevant = masks["masks"][branch, first, horizon, 1]
        overlay[robot] = ((overlay[robot].astype(np.uint16) + np.array([255, 0, 0])) // 2).astype(np.uint8)
        overlay[relevant] = ((overlay[relevant].astype(np.uint16) + np.array([0, 255, 0])) // 2).astype(np.uint8)
        strip = np.concatenate((image_a, image_b, heat, overlay), axis=1)
        canvas = Image.new("RGB", (strip.shape[1], strip.shape[0] + 20), "white")
        canvas.paste(Image.fromarray(strip), (0, 20))
        drawer = ImageDraw.Draw(canvas)
        drawer.text((2, 2), f"{label} task={int(dataset['metadata']['task'][row])} {PAIR_NAMES[pair]} h={horizon+1} D={dataset['target_clip'][row]:.6f}", fill="black")
        rows.append(np.asarray(canvas))
        draw_records.append({"label": label, "row": row, "branch": branch, "pair": PAIR_NAMES[pair], "horizon": horizon + 1})
    output = Image.fromarray(np.concatenate(rows, axis=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def calibrate(protocol: dict[str, Any], commit: str, device: torch.device) -> None:
    source, masks = load_partition(protocol, "calibration")
    replacement = np.rint(source["real_rgb"].astype(np.float64).mean(axis=(0, 1, 2, 3, 4))).astype(np.uint8)
    global_features, patch_features, coverage, clip_check = encode_partition(source, masks, replacement, device)
    if clip_check["full_reencode_min_cosine_to_frozen_real_z"] < float(
        protocol["clip"]["full_reencode_min_cosine"]
    ):
        raise RuntimeError("full-image CLIP re-encoding failed the frozen consistency gate")
    dataset = build_visual_pair_dataset(
        source,
        masks,
        patch_features.astype(np.float32),
        coverage,
        top_k=int(protocol["patch"]["top_k"]),
    )
    variants = variant_distances(global_features.astype(np.float32))
    if float(np.max(np.abs(variants[:, 0] - dataset["target_clip"]))) > float(
        protocol["clip"]["full_distance_max_abs_difference"]
    ):
        raise RuntimeError("recomputed full D_clip differs from frozen target")
    weights: dict[str, np.ndarray] = {}
    model_records: dict[str, Any] = {}
    grid = [float(value) for value in protocol["probe"]["ridge_grid"]]
    for set_name, features in dataset["sets"].items():
        alpha, cv = select_ridge_alpha(
            features, dataset["target_clip"], dataset["metadata"]["init_state"], grid
        )
        fit = fit_ridge(features, dataset["target_clip"], alpha)
        weights.update(ridge_to_arrays(f"{set_name}_scalar", fit))
        model_records[set_name] = {
            "dimensions": int(features.shape[1]),
            "feature_groups": list(FEATURE_SET_GROUPS[set_name]),
            "selected_alpha": alpha,
            "cv": cv,
        }
    weights["constant_clip"] = np.asarray(dataset["target_clip"].mean(), dtype=np.float64)
    weights["replacement_rgb"] = replacement.astype(np.uint8)
    weights_path = ROOT / protocol["outputs"]["calibration_weights"]
    sidecar_path = ROOT / protocol["partitions"]["calibration"]["feature_sidecar"]
    if weights_path.exists() or sidecar_path.exists():
        raise FileExistsError("refusing to overwrite visual-target calibration artifact")
    atomic_npz(weights_path, weights)
    save_feature_sidecar(sidecar_path, global_features, patch_features, coverage, dataset, variants)
    quantiles = {"low_p10": 0.10, "median_p50": 0.50, "high_p90": 0.90}
    selected_rows = {
        label: int(np.argmin(np.abs(dataset["target_clip"] - np.quantile(dataset["target_clip"], quantile))))
        for label, quantile in quantiles.items()
    }
    montage_path = ROOT / protocol["outputs"]["calibration_montage"]
    make_calibration_montage(montage_path, source, masks, dataset, selected_rows)
    freeze = {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_HELD_OUT_MASK_COLLECTION_OR_ANALYSIS",
        "protocol_sha256": sha256(ROOT / "configs/visual_target_suitability_protocol.json"),
        "preregistered_commit": commit,
        "partition": {
            "init_state_indices": sorted(np.unique(dataset["metadata"]["init_state"]).astype(int).tolist()),
            "branch_points": int(np.unique(dataset["metadata"]["branch"]).size),
            "rows": int(dataset["target_clip"].size),
            "evaluation_mask_or_feature_read": False,
        },
        "replacement_background": {
            "rule": "round calibration-partition mean RGB over all A0/A1/A2 future pixels; one global uint8 RGB vector",
            "rgb": replacement.astype(int).tolist(),
        },
        "clip_interface": clip_check,
        "feature_names": {name: list(values) for name, values in FEATURE_NAMES.items()},
        "feature_group_dimensions": {name: int(value.shape[1]) for name, value in dataset["groups"].items()},
        "models": model_records,
        "constant_clip_baseline": float(weights["constant_clip"]),
        "representative_montage": {
            "selection_rule": "nearest calibration D_clip to p10/p50/p90; np.argmin first-row tie break",
            "selected_rows": selected_rows,
            "path": str(montage_path.relative_to(ROOT)),
            "sha256": sha256(montage_path),
        },
        "artifacts": {
            "weights": {"path": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path)},
            "feature_sidecar": {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)},
        },
        "sources": {
            "branch_sidecar_sha256": sha256(ROOT / protocol["partitions"]["calibration"]["source_sidecar"]),
            "mask_sidecar_sha256": sha256(ROOT / protocol["partitions"]["calibration"]["mask_sidecar"]),
        },
        "forbidden_activity_audit": {
            "evaluation_mask_or_feature_read": False,
            "init_40_49_access": False,
            "success_reward_done_read": False,
            "model_or_controller_training": False,
            "evaluation_hyperparameter_selection": False,
        },
    }
    atomic_json(ROOT / protocol["outputs"]["calibration_freeze"], freeze)
    print(json.dumps({"status": "CALIBRATION_FROZEN", "weights_sha256": freeze["artifacts"]["weights"]["sha256"]}))


def model_results(
    protocol: dict[str, Any],
    dataset: dict[str, Any],
    weights: Any,
    draws: list[np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = dataset["target_clip"]
    predictions: dict[str, np.ndarray] = {
        "constant": np.full_like(target, float(np.asarray(weights["constant_clip"])))
    }
    records: dict[str, Any] = {"constant": scalar_metrics(predictions["constant"], target)}
    for set_name, features in dataset["sets"].items():
        fit = ridge_from_arrays(f"{set_name}_scalar", weights)
        predictions[set_name] = predict_ridge(features, fit).reshape(-1)
        records[set_name] = scalar_metrics(predictions[set_name], target)
    increments: dict[str, Any] = {}
    sequence = ("constant", "V0", "V1", "V2", "V3", "V4")
    minimum = float(protocol["gates"]["minimum_incremental_rmse_improvement"])
    for parent, child in zip(sequence[:-1], sequence[1:]):
        parent_rmse = float(records[parent]["rmse"])
        child_rmse = float(records[child]["rmse"])
        improvement = (parent_rmse - child_rmse) / parent_rmse
        ci = metric_bootstrap(
            draws,
            lambda index, p=predictions[parent], c=predictions[child]: (
                np.sqrt(np.mean(np.square(p[index] - target[index])))
                - np.sqrt(np.mean(np.square(c[index] - target[index])))
            )
            / max(np.sqrt(np.mean(np.square(p[index] - target[index]))), 1.0e-12),
        )
        increments[child] = {
            "parent": parent,
            "relative_rmse_improvement": float(improvement),
            "ci95": ci,
            "passes": bool(improvement >= minimum and ci[0] > 0.0),
        }
    records["increments"] = increments
    return records, predictions


def shuffle_results(
    protocol: dict[str, Any],
    dataset: dict[str, Any],
    weights: Any,
    predictions: dict[str, np.ndarray],
    draws: list[np.ndarray],
) -> dict[str, Any]:
    target = dataset["target_clip"]
    groups_for_child = {
        "V1": "robot_pixel",
        "V2": "relevant_pixel",
        "V3": "visibility",
        "V4": "patch",
    }
    result = {}
    for offset, (child, group_name) in enumerate(groups_for_child.items()):
        ordered_groups = list(FEATURE_SET_GROUPS[child])
        components = []
        for name in ordered_groups:
            value = dataset["groups"][name]
            if name == group_name:
                value = deranged_group(
                    value,
                    dataset["metadata"],
                    int(protocol["shuffle"]["seed"]) + offset,
                )
            components.append(value)
        shuffled_features = np.concatenate(components, axis=1)
        fit = ridge_from_arrays(f"{child}_scalar", weights)
        shuffled = predict_ridge(shuffled_features, fit).reshape(-1)
        original_rmse = float(np.sqrt(np.mean(np.square(predictions[child] - target))))
        shuffled_rmse = float(np.sqrt(np.mean(np.square(shuffled - target))))
        degradation = (shuffled_rmse - original_rmse) / original_rmse
        ci = metric_bootstrap(
            draws,
            lambda index, original=predictions[child], alternate=shuffled: (
                np.sqrt(np.mean(np.square(alternate[index] - target[index])))
                - np.sqrt(np.mean(np.square(original[index] - target[index])))
            )
            / max(np.sqrt(np.mean(np.square(original[index] - target[index]))), 1.0e-12),
        )
        result[group_name] = {
            "child": child,
            "original_rmse": original_rmse,
            "shuffled_rmse": shuffled_rmse,
            "relative_degradation": float(degradation),
            "ci95": ci,
            "passes": bool(degradation > 0.0 and ci[0] > 0.0),
        }
    return result


def masked_diagnostics(
    variants: np.ndarray,
    metadata: dict[str, np.ndarray],
    denominator_floor: float,
) -> dict[str, Any]:
    full = variants[:, 0]
    valid = full >= denominator_floor
    result: dict[str, Any] = {
        "denominator_floor": denominator_floor,
        "valid_ratio_rows": int(valid.sum()),
        "invalid_ratio_rows": int((~valid).sum()),
        "variants": {},
    }
    for index, name in enumerate(VARIANT_NAMES):
        distance = variants[:, index]
        retained = np.full_like(distance, np.nan)
        drop = np.full_like(distance, np.nan)
        retained[valid] = distance[valid] / full[valid]
        drop[valid] = (full[valid] - distance[valid]) / full[valid]
        result["variants"][name] = {
            "distance": cut_summary(distance, metadata),
            "retained_fraction": cut_summary(retained[valid], {key: value[valid] for key, value in metadata.items()}),
            "drop_fraction": cut_summary(drop[valid], {key: value[valid] for key, value in metadata.items()}),
        }
    return result


def decision(
    protocol: dict[str, Any],
    models: dict[str, Any],
    shuffle: dict[str, Any],
    associations: dict[str, Any],
    region: dict[str, Any],
    masked: dict[str, Any],
) -> dict[str, Any]:
    gate = protocol["gates"]
    increments = models["increments"]

    def group_pass(child: str, group: str) -> bool:
        return bool(increments[child]["passes"] and shuffle[group]["passes"])

    robot_energy = float(region["robot_energy_contribution"]["overall"]["mean"])
    object_energy = float(region["relevant_energy_contribution"]["overall"]["mean"])
    robot_only = float(masked["variants"]["robot_only"]["retained_fraction"]["overall"]["median"])
    object_only = float(masked["variants"]["relevant_object_only"]["retained_fraction"]["overall"]["median"])
    robot_drop = float(masked["variants"]["robot_removed"]["drop_fraction"]["overall"]["median"])
    object_drop = float(masked["variants"]["relevant_object_removed"]["drop_fraction"]["overall"]["median"])
    energy_ratio = float(gate["dominance_energy_ratio"])
    masked_margin = float(gate["masked_fraction_margin"])
    robot_dominates = bool(
        group_pass("V1", "robot_pixel")
        and robot_energy >= energy_ratio * max(object_energy, 1.0e-12)
        and robot_only >= object_only + masked_margin
        and robot_drop >= object_drop + masked_margin
    )
    object_dominates = bool(
        group_pass("V2", "relevant_pixel")
        and object_energy >= energy_ratio * max(robot_energy, 1.0e-12)
        and object_only >= robot_only + masked_margin
        and object_drop >= robot_drop + masked_margin
    )
    occlusion_ci = associations["object_occlusion_fraction_difference"]["overall"]["spearman_ci95"]
    occlusion_drives = bool(
        group_pass("V3", "visibility")
        and occlusion_ci[0] >= float(gate["minimum_association_ci_lower"])
    )
    background_energy = float(region["background_energy_contribution"]["overall"]["mean"])
    background_retained = float(masked["variants"]["background_only"]["retained_fraction"]["overall"]["median"])
    background_ci = associations["background_pixel_mean"]["overall"]["spearman_ci95"]
    background_sensitivity = bool(
        background_energy >= float(gate["background_energy_fraction"])
        and background_retained >= float(gate["background_retained_fraction"])
        and background_ci[0] >= float(gate["minimum_association_ci_lower"])
    )
    patch_ci = associations["patch_top10_mean"]["overall"]["spearman_ci95"]
    relevant_patch_ci = associations["patch_relevant_mean"]["overall"]["spearman_ci95"]
    relevant_pixel_ci = associations["relevant_pixel_mean"]["overall"]["spearman_ci95"]
    patch_better = bool(
        group_pass("V4", "patch")
        and relevant_patch_ci[0] >= relevant_pixel_ci[1] + float(gate["patch_association_margin"])
    )
    full_median = float(masked["variants"]["full"]["distance"]["overall"]["median"])
    patch_top_median = float(region["patch_top10_mean"]["overall"]["median"])
    compression_ratio = full_median / max(patch_top_median, 1.0e-12)
    global_compression = bool(
        group_pass("V4", "patch")
        and compression_ratio <= float(gate["global_to_top_patch_ratio_max"])
        and patch_ci[0] >= float(gate["minimum_association_ci_lower"])
    )
    task_rows = []
    robot_tasks = 0
    object_tasks = 0
    robot_task_values = region["robot_energy_contribution"]["task"]
    object_task_values = region["relevant_energy_contribution"]["task"]
    for task in range(10):
        r_value = float(robot_task_values[task]["mean"])
        o_value = float(object_task_values[task]["mean"])
        label = "balanced"
        if r_value >= energy_ratio * max(o_value, 1.0e-12):
            label = "robot"
            robot_tasks += 1
        elif o_value >= energy_ratio * max(r_value, 1.0e-12):
            label = "relevant_object"
            object_tasks += 1
        task_rows.append({"task": task, "robot_energy": r_value, "relevant_energy": o_value, "dominance": label})
    task_dependent = bool(
        robot_tasks >= int(gate["task_dependent_min_tasks_each"])
        and object_tasks >= int(gate["task_dependent_min_tasks_each"])
    )
    clip_aligns = bool(object_dominates and not robot_dominates and not occlusion_drives and not background_sensitivity)
    global_unsuitable = bool(
        (robot_dominates or occlusion_drives or background_sensitivity)
        and not object_dominates
        and (patch_better or global_compression)
    )
    labels = []
    if robot_dominates:
        labels.append("ROBOT_APPEARANCE_DOMINATES_CLIP_COUNTERFACTUAL")
    if object_dominates:
        labels.append("RELEVANT_OBJECT_VISUAL_CHANGE_DOMINATES")
    if occlusion_drives:
        labels.append("OCCLUSION_DRIVES_CLIP_COUNTERFACTUAL")
    if background_sensitivity:
        labels.append("BACKGROUND_OR_RENDERING_SENSITIVITY")
    if global_compression:
        labels.append("GLOBAL_CLIP_SPATIAL_COMPRESSION")
    if patch_better:
        labels.append("PATCH_FEATURES_CAPTURE_LOCAL_DYNAMICS_BETTER")
    if task_dependent:
        labels.append("TASK_DEPENDENT_VISUAL_TARGET")
    if clip_aligns:
        labels.append("CLIP_COUNTERFACTUAL_ALIGNS_WITH_TASK_RELEVANT_VISUAL_CHANGE")
    if global_unsuitable:
        labels.append("GLOBAL_CLIP_TARGET_UNSUITABLE_FOR_LOCAL_DYNAMICS")
    if not labels:
        labels.append("CLIP_TARGET_SUITABILITY_UNRESOLVED")
    if object_dominates or patch_better:
        authorization = "GO_SPATIAL_VISUAL_REPRESENTATION_DIAGNOSIS"
    elif robot_dominates or occlusion_drives:
        authorization = "GO_ROBOT_INVARIANT_VISUAL_TARGET_DIAGNOSIS"
    else:
        authorization = "BLOCK_VISUAL_TARGET_REDESIGN"
    return {
        "labels": labels,
        "next_stage_authorization": authorization,
        "controller_authorized": False,
        "checks": {
            "robot_group_pass": group_pass("V1", "robot_pixel"),
            "relevant_group_pass": group_pass("V2", "relevant_pixel"),
            "visibility_group_pass": group_pass("V3", "visibility"),
            "patch_group_pass": group_pass("V4", "patch"),
            "robot_dominates": robot_dominates,
            "object_dominates": object_dominates,
            "occlusion_drives": occlusion_drives,
            "background_sensitivity": background_sensitivity,
            "patch_better": patch_better,
            "global_compression": global_compression,
            "task_dependent": task_dependent,
            "clip_aligns": clip_aligns,
            "global_unsuitable": global_unsuitable,
        },
        "dominance_values": {
            "robot_energy": robot_energy,
            "relevant_object_energy": object_energy,
            "robot_only_retained_median": robot_only,
            "relevant_only_retained_median": object_only,
            "robot_removed_drop_median": robot_drop,
            "relevant_removed_drop_median": object_drop,
            "background_energy": background_energy,
            "background_only_retained_median": background_retained,
            "global_to_top_patch_median_ratio": compression_ratio,
            "robot_dominant_tasks": robot_tasks,
            "relevant_object_dominant_tasks": object_tasks,
            "task_rows": task_rows,
        },
    }


def evaluate(protocol: dict[str, Any], commit: str, device: torch.device) -> None:
    freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    weights_path = ROOT / freeze["artifacts"]["weights"]["path"]
    if sha256(weights_path) != freeze["artifacts"]["weights"]["sha256"]:
        raise ValueError("calibration weights changed")
    weights = np.load(weights_path, allow_pickle=False)
    replacement = np.asarray(freeze["replacement_background"]["rgb"], dtype=np.uint8)
    source, masks = load_partition(protocol, "evaluation")
    global_features, patch_features, coverage, clip_check = encode_partition(source, masks, replacement, device)
    if clip_check["full_reencode_min_cosine_to_frozen_real_z"] < float(protocol["clip"]["full_reencode_min_cosine"]):
        raise RuntimeError("held-out full-image CLIP consistency gate failed")
    dataset = build_visual_pair_dataset(
        source,
        masks,
        patch_features.astype(np.float32),
        coverage,
        top_k=int(protocol["patch"]["top_k"]),
    )
    variants = variant_distances(global_features.astype(np.float32))
    target_difference = float(np.max(np.abs(variants[:, 0] - dataset["target_clip"])))
    if target_difference > float(protocol["clip"]["full_distance_max_abs_difference"]):
        raise RuntimeError("held-out recomputed full D_clip differs from frozen target")
    feature_path = ROOT / protocol["partitions"]["evaluation"]["feature_sidecar"]
    metric_path = ROOT / protocol["outputs"]["evaluation_metrics"]
    diagnosis_path = ROOT / protocol["outputs"]["diagnosis"]
    for path in (feature_path, metric_path, diagnosis_path):
        if path.exists():
            raise FileExistsError(path)
    save_feature_sidecar(feature_path, global_features, patch_features, coverage, dataset, variants)
    draws = bootstrap_indices(
        dataset["metadata"],
        int(protocol["bootstrap"]["replicates"]),
        int(protocol["bootstrap"]["seed"]),
    )
    models, predictions = model_results(protocol, dataset, weights, draws)
    shuffles = shuffle_results(protocol, dataset, weights, predictions, draws)
    association_names = (
        "pixel_mae",
        "pixel_rmse",
        "unequal_pixel_fraction",
        "robot_pixel_mean",
        "relevant_pixel_mean",
        "other_pixel_mean",
        "background_pixel_mean",
        "object_occlusion_fraction_difference",
        "patch_mean",
        "patch_max",
        "patch_top10_mean",
        "patch_robot_mean",
        "patch_relevant_mean",
        "patch_other_mean",
        "patch_background_mean",
    )
    associations = {
        name: cut_association(dataset["scalars"][name], dataset["target_clip"], dataset["metadata"], draws)
        for name in association_names
    }
    action_associations = {}
    for action_name in ("action_translation", "action_rotation"):
        action_associations[action_name] = {
            outcome: cut_association(
                dataset["scalars"][action_name],
                dataset["scalars"][outcome] if outcome != "D_clip" else dataset["target_clip"],
                dataset["metadata"],
                draws,
            )
            for outcome in (
                "robot_pixel_mean",
                "relevant_pixel_mean",
                "patch_mean",
                "patch_robot_mean",
                "patch_relevant_mean",
                "D_clip",
            )
        }
    region_names = (
        "robot_pixel_mean",
        "relevant_pixel_mean",
        "other_pixel_mean",
        "background_pixel_mean",
        "robot_energy_contribution",
        "relevant_energy_contribution",
        "other_energy_contribution",
        "background_energy_contribution",
        "robot_union_area_fraction",
        "relevant_union_area_fraction",
        "robot_visible_area_difference",
        "relevant_visible_area_difference",
        "object_occlusion_fraction_difference",
        "object_occlusion_fraction_mean",
        "patch_mean",
        "patch_max",
        "patch_top10_mean",
        "patch_robot_mean",
        "patch_relevant_mean",
        "patch_other_mean",
        "patch_background_mean",
        "top10_robot_fraction",
        "top10_relevant_fraction",
        "top10_other_fraction",
        "top10_background_fraction",
    )
    region = {name: cut_summary(dataset["scalars"][name], dataset["metadata"]) for name in region_names}
    masked = masked_diagnostics(
        variants,
        dataset["metadata"],
        float(protocol["masked_clip"]["ratio_denominator_floor"]),
    )
    verdict = decision(protocol, models, shuffles, associations, region, masked)
    physical_result = json.loads((ROOT / protocol["comparison"]["structured_state_diagnosis"]).read_text())
    physical_comparison = {
        name: physical_result["raw_associations"][name]["overall"]
        for name in (
            "ee_translation",
            "ee_rotation",
            "object_translation_max",
            "relative_translation_max",
        )
    }
    result = {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "final_mechanism_labels": verdict["labels"],
        "next_stage_authorization": verdict["next_stage_authorization"],
        "controller_authorized": False,
        "protocol": {"path": "configs/visual_target_suitability_protocol.json", "sha256": sha256(ROOT / "configs/visual_target_suitability_protocol.json"), "preregistered_commit": commit},
        "calibration_freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha256(freeze_path)},
        "evaluation_scope": {
            "init_state_indices": sorted(np.unique(dataset["metadata"]["init_state"]).astype(int).tolist()),
            "branch_points": int(np.unique(dataset["metadata"]["branch"]).size),
            "rows": int(dataset["target_clip"].size),
            "tasks": sorted(np.unique(dataset["metadata"]["task"]).astype(int).tolist()),
            "pairs": list(PAIR_NAMES),
            "horizons": list(range(1, 9)),
        },
        "image_preprocessing_verification": {
            **clip_check,
            "full_distance_max_abs_difference": target_difference,
            "replacement_background_rgb": replacement.astype(int).tolist(),
        },
        "patch_feature_availability": "AVAILABLE",
        "clip_target": cut_summary(dataset["target_clip"], dataset["metadata"]),
        "pixel_associations": associations,
        "region_decomposition": region,
        "masked_clip_diagnostics": masked,
        "action_geometry_associations": action_associations,
        "probe_models": models,
        "group_shuffle_controls": shuffles,
        "structured_state_comparison": physical_comparison,
        "mechanical_decision": verdict,
        "interpretation_boundaries": {
            "masking_is_attribution_style_ood_diagnostic": True,
            "patch_distance_is_representation_change_not_attention_attribution": True,
            "simulator_future_masks_are_diagnostic_oracles_not_deployment_inputs": True,
            "success_reward_done_not_used": True,
        },
        "artifacts": {
            "evaluation_feature_sidecar": {"path": str(feature_path.relative_to(ROOT)), "sha256": sha256(feature_path)},
            "evaluation_metrics": {"path": str(metric_path.relative_to(ROOT))},
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "new_world_model_training": False,
            "clip_finetuning": False,
            "controller_or_candidate_change": False,
            "c1_c2_d20_v2_work": False,
            "seed23_47_run": False,
            "init_40_49_access": False,
            "success_reward_done_read": False,
            "simulator_rerun_for_rgb_or_clip": False,
            "simulator_used_only_for_masks": True,
        },
    }
    # Store the dense arrays needed for independent recomputation before the
    # compact result, then write the sidecar hash into the result.
    metric_arrays: dict[str, np.ndarray] = {
        "target_clip": dataset["target_clip"].astype(np.float32),
        "variant_distances": variants.astype(np.float32),
    }
    metric_arrays.update({f"metadata__{name}": value for name, value in dataset["metadata"].items()})
    metric_arrays.update({f"prediction__{name}": value.astype(np.float32) for name, value in predictions.items()})
    metric_arrays.update({f"scalar__{name}": value.astype(np.float32) for name, value in dataset["scalars"].items()})
    atomic_npz(metric_path, metric_arrays)
    result["artifacts"]["evaluation_metrics"]["sha256"] = sha256(metric_path)
    if not finite_tree(result):
        raise FloatingPointError("diagnosis contains NaN/Inf")
    atomic_json(diagnosis_path, result)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "labels": verdict["labels"],
                "authorization": verdict["next_stage_authorization"],
                "diagnosis_sha256": sha256(diagnosis_path),
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/visual_target_suitability_protocol.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol, commit = load_protocol(args.protocol, args.phase)
    if args.phase == "calibrate":
        calibrate(protocol, commit, torch.device(args.device))
    else:
        evaluate(protocol, commit, torch.device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
