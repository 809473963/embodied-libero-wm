#!/usr/bin/env python3
"""Extract frozen CLIP patch grids and run the preregistered Stage R audit."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Required by torch deterministic algorithms for CUDA >=10.2.  This must be
# set before torch initializes cuBLAS; it changes no model or numerical gate.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.allowed_demo_guard import AllowedDemoGuard  # noqa: E402
from src.probes.semantic_value_distillation import (  # noqa: E402
    CounterfactualSet, PairSet, correctness, cross_demo_metrics, fit_temperature,
    make_counterfactual_set, make_cross_demo_controls, make_training_pairs,
    non_tie_metrics, ordinary_pairs, sigmoid,
)
from src.probes.spatial_visual_representation import (  # noqa: E402
    FAMILIES, FeatureStatistics, SpatialRepresentationProbe, attention_arrays,
    attention_diagnostics, fit_feature_statistics, fixed_patch_coordinates,
    parameter_count, predict_logits, train_probe,
)
from src.probes.semantic_value_distillation import load_partition  # noqa: E402


VERSION = "goal-conditioned-spatial-visual-representation-v1"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != VERSION:
        raise ValueError("unexpected Stage-R protocol")
    if protocol.get("status") != "PREREGISTERED_BEFORE_STAGE_R_FEATURE_EXTRACTION_OR_TRAINING":
        raise ValueError("Stage-R protocol status is not frozen")
    if not last_commit(path):
        raise RuntimeError("Stage-R protocol must be committed before execution")
    for row in protocol["frozen_files"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"frozen hash mismatch: {row['path']}")
    architecture_path = ROOT / protocol["representations"]["architectures"]
    architecture = json.loads(architecture_path.read_text())
    if architecture.get("status") != "PREREGISTERED_BEFORE_STAGE_R_FEATURE_EXTRACTION_OR_TRAINING":
        raise ValueError("architecture registry not preregistered")
    return protocol


def clip_snapshot() -> Path:
    path = ROOT / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots" / MODEL_REVISION
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def encode_batch(
    model: CLIPModel, processor: CLIPProcessor, images: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    pixels = processor(
        images=[np.ascontiguousarray(image) for image in images], return_tensors="pt"
    )["pixel_values"].to(device)
    with autocast(device):
        output = model.vision_model(pixel_values=pixels, return_dict=True)
        global_feature = F.normalize(model.visual_projection(output.pooler_output).float(), dim=-1)
        patch_hidden = model.vision_model.post_layernorm(output.last_hidden_state[:, 1:, :])
        patch_feature = F.normalize(model.visual_projection(patch_hidden).float(), dim=-1)
    return (
        global_feature.cpu().numpy().astype(np.float16),
        patch_feature.cpu().numpy().astype(np.float16),
    )


def extract_features(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output_path = ROOT / protocol["outputs"]["patch_features"]
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen patch features: {output_path}")
    semantic, feature_ledger = load_partition(
        ROOT / "results/t1_semantic_teacher/calibration_sidecar.npz",
        ROOT / protocol["frozen_visual_source"]["global_feature_store"],
        "calibration",
    )
    manifest = json.loads((ROOT / protocol["frozen_visual_source"]["feature_manifest"]).read_text())
    task_files = {
        int(row["task_id"]): ROOT / row["source_hdf5_relative_path"] for row in manifest["tasks"]
    }
    guard = AllowedDemoGuard("calibration", task_files, ["obs/agentview_rgb"])
    snapshot = clip_snapshot()
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = sum(int(parameter.requires_grad) for parameter in model.parameters())
    patches: list[np.ndarray] = []
    reconstructed_globals: list[np.ndarray] = []
    repeated_exact = True
    cursor = 0
    minimum_cosine = 1.0
    maximum_absolute = 0.0
    batch_size = int(protocol["patch_extraction"]["batch_size"])
    repeated = False
    for task in range(10):
        for demo in guard.allowed_demo_ids:
            images = guard.read_dataset(task, demo, "obs/agentview_rgb")
            if images.ndim != 4 or images.shape[1:] != (128, 128, 3) or images.dtype != np.uint8:
                raise ValueError(f"unexpected RGB schema for task{task}/demo{demo}: {images.shape} {images.dtype}")
            count = len(images)
            frame_slice = slice(cursor, cursor + count)
            if not (
                np.all(semantic.task_id[frame_slice] == task)
                and np.all(semantic.demo_id[frame_slice] == demo)
                and np.array_equal(semantic.timestep[frame_slice], np.arange(count))
            ):
                raise ValueError("RGB/T1 frame identity mismatch")
            demo_global: list[np.ndarray] = []
            demo_patch: list[np.ndarray] = []
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                global_feature, patch_feature = encode_batch(model, processor, images[start:stop], device)
                if patch_feature.shape != (stop - start, 49, 512):
                    raise ValueError(f"unexpected patch shape: {patch_feature.shape}")
                if not repeated:
                    repeat_global, repeat_patch = encode_batch(model, processor, images[start:stop], device)
                    repeated_exact &= np.array_equal(global_feature, repeat_global)
                    repeated_exact &= np.array_equal(patch_feature, repeat_patch)
                    repeated = True
                reference = semantic.vision[frame_slice][start:stop].astype(np.float32)
                candidate = global_feature.astype(np.float32)
                cosine = np.sum(candidate * reference, axis=-1) / np.maximum(
                    np.linalg.norm(candidate, axis=-1) * np.linalg.norm(reference, axis=-1), 1.0e-12
                )
                minimum_cosine = min(minimum_cosine, float(cosine.min()))
                maximum_absolute = max(maximum_absolute, float(np.max(np.abs(candidate - reference))))
                demo_global.append(global_feature)
                demo_patch.append(patch_feature)
            reconstructed_globals.append(np.concatenate(demo_global))
            patches.append(np.concatenate(demo_patch))
            cursor += count
            print(json.dumps({"patch_extraction": {"task": task, "demo": demo, "frames": count}}), flush=True)
    if cursor != len(semantic.vision):
        raise ValueError("RGB/T1 totals differ")
    patch_array = np.concatenate(patches)
    global_array = np.concatenate(reconstructed_globals)
    all_finite = bool(np.isfinite(patch_array).all() and np.isfinite(global_array).all())
    thresholds = protocol["patch_extraction"]["integrity"]
    integrity_pass = bool(
        all_finite
        and patch_array.shape == (len(semantic.vision), 49, 512)
        and repeated_exact
        and minimum_cosine >= float(thresholds["global_parity_min_cosine"])
        and maximum_absolute <= float(thresholds["global_parity_max_abs"])
        and trainable == 0
        and not any(parameter.grad is not None for parameter in model.parameters())
        and guard.ledger.as_dict()["forbidden_access_count"] == 0
        and feature_ledger["forbidden_access_count"] == 0
    )
    combined_ledger = {
        "stage": "R",
        "partition": "calibration_only",
        "source_rgb": guard.ledger.as_dict(),
        "frozen_feature_store": feature_ledger,
        "stage_r_opened_demo_ids": sorted(set(guard.ledger.opened_demo_ids) | set(feature_ledger["opened_demo_ids"])),
        "stage_r_not_opened_demo_ids": [36, 37, 38, 39],
        "forbidden_demo_ids": list(range(40, 50)),
        "forbidden_numeric_access": 0,
        "forbidden_metadata_access": 0,
    }
    guard.assert_clean()
    patch_digest = arrays_sha256(
        patch_array, semantic.task_id, semantic.demo_id, semantic.timestep, fixed_patch_coordinates()
    )
    atomic_npz(
        output_path,
        {
            "patch": patch_array,
            "global_reencoded": global_array,
            "coordinates": fixed_patch_coordinates(),
            "task_id": semantic.task_id.astype(np.int8),
            "demo_id": semantic.demo_id.astype(np.int8),
            "timestep": semantic.timestep.astype(np.int32),
        },
    )
    atomic_json(ROOT / protocol["outputs"]["calibration_access_ledger"], combined_ledger)
    integrity = {
        "schema_version": 1,
        "stage": "R",
        "mode": "frozen_patch_feature_integrity",
        "checkpoint": str(snapshot),
        "checkpoint_revision": MODEL_REVISION,
        "checkpoint_sha256": protocol["frozen_visual_source"]["checkpoint_sha256"],
        "frames": int(len(patch_array)),
        "patch_shape": list(patch_array.shape),
        "patch_dtype": str(patch_array.dtype),
        "coordinates": fixed_patch_coordinates().astype(float).tolist(),
        "all_finite": all_finite,
        "repeated_float16_exact": bool(repeated_exact),
        "global_parity": {"minimum_cosine": minimum_cosine, "maximum_absolute_error": maximum_absolute},
        "clip_trainable_parameter_tensors": trainable,
        "clip_parameter_gradients_present": bool(any(parameter.grad is not None for parameter in model.parameters())),
        "patch_array_identity_sha256": patch_digest,
        "patch_file_sha256": sha256(output_path),
        "access_ledger_sha256": sha256(ROOT / protocol["outputs"]["calibration_access_ledger"]),
        "complete_gate": integrity_pass,
    }
    atomic_json(ROOT / "results/spatial_visual_representation_patch_integrity.json", integrity)
    if not integrity_pass:
        raise RuntimeError("Stage-R patch integrity gate failed")
    return integrity


def subset_pairset(pairs: PairSet, mask: np.ndarray) -> PairSet:
    return PairSet(*(getattr(pairs, field)[mask] for field in pairs.__dataclass_fields__))


def subset_counterfactual(cf: CounterfactualSet, mask: np.ndarray) -> CounterfactualSet:
    return CounterfactualSet(*(getattr(cf, field)[mask] for field in cf.__dataclass_fields__))


def pairset_for_counterfactual(cf: CounterfactualSet, alternate: bool) -> PairSet:
    return PairSet(
        cf.a, cf.b,
        cf.alternate_language if alternate else cf.original_language,
        cf.alternate_label if alternate else cf.original_label,
        cf.task_id, cf.demo_id, np.zeros(len(cf.a), dtype=np.int64),
        np.zeros(len(cf.a), dtype=np.int8),
        np.full(len(cf.a), "language_counterfactual", dtype="U32"),
    )


def calibration_metrics(
    ordinary_probability: np.ndarray,
    ordinary: PairSet,
    original_probability: np.ndarray,
    alternate_probability: np.ndarray,
    counterfactual: CounterfactualSet,
    matched_probability: np.ndarray,
    matched: PairSet,
    different_probability: np.ndarray,
    different: PairSet,
) -> dict[str, Any]:
    local = np.isin(ordinary.gap_code, (0, 1, 2))
    non_tie = local & (ordinary.label != 0)
    ties = local & (ordinary.label == 0)
    changed = counterfactual.original_label != counterfactual.alternate_label
    direction = counterfactual.alternate_label.astype(np.float64) - counterfactual.original_label.astype(np.float64)
    agreement = (alternate_probability - original_probability) * direction > 0
    reversal = counterfactual.original_label * counterfactual.alternate_label == -1
    reversal_correct = (
        (((original_probability > 0.5) & (counterfactual.original_label == 1)) |
         ((original_probability < 0.5) & (counterfactual.original_label == -1)))
        & (((alternate_probability > 0.5) & (counterfactual.alternate_label == 1)) |
           ((alternate_probability < 0.5) & (counterfactual.alternate_label == -1)))
    )
    same = counterfactual.same_referent
    same_change = np.abs(alternate_probability[same] - original_probability[same])
    same_flip = np.sign(alternate_probability[same] - 0.5) != np.sign(original_probability[same] - 0.5)
    ordinary_result = non_tie_metrics(ordinary_probability[non_tie], ordinary.label[non_tie])
    return {
        "ordinary_local_non_tie": ordinary_result,
        "ordinary_local_tie": {
            "rows": int(ties.sum()),
            "mean_abs_from_half": float(np.abs(ordinary_probability[ties] - 0.5).mean()),
            "median_abs_from_half": float(np.median(np.abs(ordinary_probability[ties] - 0.5))),
        },
        "language_counterfactual": {
            "eligible_rows": int(len(changed)),
            "teacher_changed_rows": int(changed.sum()),
            "teacher_change_agreement": float(agreement[changed].mean()),
            "true_reversal": {
                "rows": int(reversal.sum()),
                "both_correct_rate": float(reversal_correct[reversal].mean()),
            },
            "same_referent": {
                "rows": int(same.sum()),
                "mean_abs_probability_change": float(same_change.mean()),
                "false_preference_flip_rate": float(same_flip.mean()),
            },
        },
        "cross_demo_semantic_match": cross_demo_metrics(matched_probability, matched),
        "cross_demo_semantic_difference": cross_demo_metrics(different_probability, different),
    }


def serialize_statistics(statistics: FeatureStatistics) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(statistics, name)) for name in statistics.__dataclass_fields__}


def calibrate(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    patch_path = ROOT / protocol["outputs"]["patch_features"]
    integrity_path = ROOT / "results/spatial_visual_representation_patch_integrity.json"
    if not patch_path.exists() or not integrity_path.exists():
        raise FileNotFoundError("Stage-R patch extraction must pass before calibration")
    integrity = json.loads(integrity_path.read_text())
    if not integrity.get("complete_gate"):
        raise RuntimeError("patch integrity is not valid")
    data, feature_ledger = load_partition(
        ROOT / "results/t1_semantic_teacher/calibration_sidecar.npz",
        ROOT / protocol["frozen_visual_source"]["global_feature_store"],
        "calibration",
    )
    if feature_ledger["forbidden_access_count"] != 0:
        raise RuntimeError("calibration feature ledger contains forbidden access")
    with np.load(patch_path) as archive:
        patch = archive["patch"]
        patch_task = archive["task_id"]
        patch_demo = archive["demo_id"]
        patch_timestep = archive["timestep"]
        coordinates = archive["coordinates"]
    if not (
        np.array_equal(patch_task, data.task_id)
        and np.array_equal(patch_demo, data.demo_id)
        and np.array_equal(patch_timestep, data.timestep)
        and np.array_equal(coordinates, fixed_patch_coordinates())
    ):
        raise ValueError("patch/T1 identity mismatch")
    train_demos = tuple(protocol["partitions"]["selection_train_demo_ids"])
    validation_demos = tuple(protocol["partitions"]["selection_validation_demo_ids"])
    train_mask = np.isin(data.demo_id, train_demos)
    statistics = fit_feature_statistics(data.vision, patch, data.language, train_mask)
    definition = json.loads((ROOT / "configs/t1_teacher_frozen_definition.json").read_text())
    controls = definition["controls"]
    sampler = json.loads((ROOT / protocol["training"]["pair_sampler"]).read_text())
    training_pairs = make_training_pairs(
        data, train_demos, float(protocol["frozen_teacher"]["epsilon_goal"]), sampler,
        float(controls["robot_match_threshold"]), float(controls["robot_difference_threshold"]),
        np.asarray(controls["proprio_mean"]), np.asarray(controls["proprio_std"]),
    )
    ordinary = ordinary_pairs(data, validation_demos)
    counterfactual_all = make_counterfactual_set(
        data, float(protocol["frozen_teacher"]["epsilon_goal"]), True
    )
    counterfactual = subset_counterfactual(
        counterfactual_all, np.isin(counterfactual_all.demo_id, validation_demos)
    )
    original_pairs = pairset_for_counterfactual(counterfactual, False)
    alternate_pairs = pairset_for_counterfactual(counterfactual, True)
    cross_all = make_cross_demo_controls(
        data, float(protocol["frozen_teacher"]["epsilon_goal"]),
        int(protocol["validation_controls"]["cross_demo_points"]),
        float(protocol["validation_controls"]["cross_demo_clear_margin_min"]),
    )
    matched = subset_pairset(cross_all["matched"], cross_all["matched"].demo_id == 34)
    different = subset_pairset(cross_all["different"], cross_all["different"].demo_id == 34)
    models: dict[str, SpatialRepresentationProbe] = {}
    temperatures: dict[str, float] = {}
    summaries: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}
    training = protocol["training"]
    for family_index, family in enumerate(FAMILIES):
        started = time.perf_counter()
        model = train_probe(
            family, data.vision, patch, data.language, training_pairs, statistics,
            int(training["epochs"]), int(training["batch_size"]),
            int(training["seed"]) + family_index, device,
        )
        if parameter_count(model) > int(protocol["representations"]["parameter_max"]):
            raise RuntimeError(f"{family} exceeds parameter ceiling")
        ordinary_logits = predict_logits(model, data.vision, patch, data.language, ordinary, statistics, device)
        original_logits = predict_logits(model, data.vision, patch, data.language, original_pairs, statistics, device)
        alternate_logits = predict_logits(model, data.vision, patch, data.language, alternate_pairs, statistics, device)
        local = np.isin(ordinary.gap_code, (0, 1, 2))
        temperature = fit_temperature(
            np.concatenate((ordinary_logits[local], original_logits, alternate_logits)),
            np.concatenate((ordinary.label[local], counterfactual.original_label, counterfactual.alternate_label)),
            training["temperature_grid"],
        )
        ordinary_probability = sigmoid(ordinary_logits * temperature)
        original_probability = sigmoid(original_logits * temperature)
        alternate_probability = sigmoid(alternate_logits * temperature)
        matched_probability = sigmoid(
            predict_logits(model, data.vision, patch, data.language, matched, statistics, device) * temperature
        )
        different_probability = sigmoid(
            predict_logits(model, data.vision, patch, data.language, different, statistics, device) * temperature
        )
        summary = calibration_metrics(
            ordinary_probability, ordinary, original_probability, alternate_probability,
            counterfactual, matched_probability, matched, different_probability, different,
        )
        summary.update(
            {
                "family": family,
                "temperature": float(temperature),
                "trainable_parameters": parameter_count(model),
                "training_seconds": float(time.perf_counter() - started),
            }
        )
        models[family] = model
        temperatures[family] = float(temperature)
        summaries[family] = summary
        raw[f"{family}_ordinary_probability"] = ordinary_probability.astype(np.float32)
        raw[f"{family}_cf_original_probability"] = original_probability.astype(np.float32)
        raw[f"{family}_cf_alternate_probability"] = alternate_probability.astype(np.float32)
        raw[f"{family}_cross_match_probability"] = matched_probability.astype(np.float32)
        raw[f"{family}_cross_difference_probability"] = different_probability.astype(np.float32)
        if family != "R0":
            original_attention = attention_arrays(
                model, patch, data.language, counterfactual.a, counterfactual.original_language,
                statistics, device,
            )
            alternate_attention = attention_arrays(
                model, patch, data.language, counterfactual.a, counterfactual.alternate_language,
                statistics, device,
            )
            summary["attention_diagnostics"] = attention_diagnostics(
                original_attention, alternate_attention, coordinates
            )
            raw[f"{family}_cf_original_attention"] = original_attention.astype(np.float16)
            raw[f"{family}_cf_alternate_attention"] = alternate_attention.astype(np.float16)
        print(json.dumps({"stage_r_family_complete": family, "summary": summary}), flush=True)
    priority = sorted(
        ("R1", "R2", "R3"),
        key=lambda family: (
            -summaries[family]["language_counterfactual"]["teacher_change_agreement"],
            summaries[family]["cross_demo_semantic_match"]["mean_abs_from_half"],
            -summaries[family]["cross_demo_semantic_difference"]["accuracy"],
            -summaries[family]["ordinary_local_non_tie"]["accuracy"],
            summaries[family]["ordinary_local_tie"]["mean_abs_from_half"],
            summaries[family]["trainable_parameters"],
            summaries[family]["training_seconds"],
        ),
    )
    baseline = summaries["R0"]
    gates: dict[str, Any] = {}
    for family in ("R1", "R2", "R3"):
        result = summaries[family]
        cf_increment = 100.0 * (
            result["language_counterfactual"]["teacher_change_agreement"]
            - baseline["language_counterfactual"]["teacher_change_agreement"]
        )
        match_improvement = (
            baseline["cross_demo_semantic_match"]["mean_abs_from_half"]
            - result["cross_demo_semantic_match"]["mean_abs_from_half"]
        )
        difference_increment = 100.0 * (
            result["cross_demo_semantic_difference"]["accuracy"]
            - baseline["cross_demo_semantic_difference"]["accuracy"]
        )
        ordinary_increment = 100.0 * (
            result["ordinary_local_non_tie"]["accuracy"]
            - baseline["ordinary_local_non_tie"]["accuracy"]
        )
        complete = bool(
            cf_increment >= 4.0
            and (match_improvement >= 0.02 or difference_increment >= 5.0)
            and ordinary_increment >= -2.0
        )
        gates[family] = {
            "teacher_change_increment_points": cf_increment,
            "cross_demo_match_mean_abs_improvement": match_improvement,
            "cross_demo_difference_increment_points": difference_increment,
            "ordinary_accuracy_increment_points": ordinary_increment,
            "complete": complete,
        }
    eligible = [family for family in priority if gates[family]["complete"]]
    selected = eligible[0] if eligible else None
    passed = selected is not None
    labels = protocol["decision_mapping"]["pass_labels" if passed else "fail_labels"]
    authorization = protocol["decision_mapping"]["pass_authorization" if passed else "fail_authorization"]
    weights_path = ROOT / protocol["outputs"]["weights"]
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "families": {family: model.state_dict() for family, model in models.items()},
            "statistics": serialize_statistics(statistics),
            "temperatures": temperatures,
            "selected_representation": selected,
            "parameter_counts": {family: parameter_count(model) for family, model in models.items()},
        },
        weights_path,
    )
    sidecar = {
        "ordinary_a": ordinary.a, "ordinary_b": ordinary.b,
        "ordinary_label": ordinary.label, "ordinary_task": ordinary.task_id,
        "ordinary_demo": ordinary.demo_id, "ordinary_gap": ordinary.gap,
        "cf_a": counterfactual.a, "cf_b": counterfactual.b,
        "cf_original_language": counterfactual.original_language,
        "cf_alternate_language": counterfactual.alternate_language,
        "cf_original_label": counterfactual.original_label,
        "cf_alternate_label": counterfactual.alternate_label,
        "cf_same_referent": counterfactual.same_referent,
        "cf_task": counterfactual.task_id, "cf_demo": counterfactual.demo_id,
        "cross_match_a": matched.a, "cross_match_b": matched.b,
        "cross_match_label": matched.label,
        "cross_difference_a": different.a, "cross_difference_b": different.b,
        "cross_difference_label": different.label,
        **serialize_statistics(statistics), **raw,
    }
    sidecar_path = ROOT / protocol["outputs"]["calibration_sidecar"]
    atomic_npz(sidecar_path, sidecar)
    result = {
        "schema_version": 1,
        "stage": "R",
        "mode": "calibration_only_representation_selection",
        "protocol_version": VERSION,
        "protocol_commit": last_commit(ROOT / "configs/goal_conditioned_spatial_visual_representation_protocol.json"),
        "partition": {"fit": list(train_demos), "validation": list(validation_demos), "evaluation_opened": False},
        "patch_integrity": integrity,
        "training_pairs": int(len(training_pairs.a)),
        "validation_rows": {
            "ordinary": int(len(ordinary.a)), "counterfactual": int(len(counterfactual.a)),
            "cross_match": int(len(matched.a)), "cross_difference": int(len(different.a)),
        },
        "families": summaries,
        "selection_order": priority,
        "material_gates": gates,
        "selected_representation": selected,
        "complete_gate": passed,
        "labels": labels,
        "authorization": authorization,
        "continuing_blocks": protocol["decision_mapping"]["continuing_blocks"],
        "controller_authorization": False,
        "artifacts": {
            "weights_sha256": sha256(weights_path),
            "sidecar_sha256": sha256(sidecar_path),
            "patch_features_sha256": sha256(patch_path),
            "access_ledger_sha256": sha256(ROOT / protocol["outputs"]["calibration_access_ledger"]),
        },
        "implementation": {
            "probe_sha256": sha256(ROOT / "src/probes/spatial_visual_representation.py"),
            "runner_sha256": sha256(Path(__file__)),
        },
        "finite": True,
        "forbidden_activity": {
            "demo36_39_opened": False,
            "demo40_49_numeric_access": 0,
            "demo40_49_metadata_access": 0,
            "simulator": False,
            "candidate_bridge": False,
            "world_model": False,
            "controller": False,
        },
    }
    if not finite_tree(result) or not all(np.isfinite(value).all() for value in sidecar.values()):
        raise FloatingPointError("Stage-R result contains non-finite values")
    atomic_json(ROOT / protocol["outputs"]["calibration"], result)
    decision = {
        "schema_version": 1,
        "stage": "R",
        "selected_representation": selected,
        "mechanism_labels": labels,
        "next_stage_authorization": authorization,
        "real_candidate_value_bridge_authorization": False,
        "world_model_authorization": False,
        "controller_authorization": False,
        "stop_chain": not passed,
        "source_calibration_sha256": sha256(ROOT / protocol["outputs"]["calibration"]),
    }
    atomic_json(ROOT / protocol["outputs"]["final_decision"], decision)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/goal_conditioned_spatial_visual_representation_protocol.json")
    parser.add_argument("--phase", choices=("extract", "calibrate", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol = validate_protocol(args.protocol.resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase in ("extract", "all"):
        extract_features(protocol, device)
    if args.phase in ("calibrate", "all"):
        result = calibrate(protocol, device)
        print(json.dumps({"stage_r_complete": result["complete_gate"], "selected": result["selected_representation"], "authorization": result["authorization"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
