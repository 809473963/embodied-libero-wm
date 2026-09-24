#!/usr/bin/env python3
"""Final offline validation and matched-counterfactual audit for WM-v2C-EJ."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_wm_action_conditioning import (
    BootstrapPlan,
    CONDITION_NAMES,
    SCALE_CONDITIONS,
    comparison_summary,
    construct_conditions,
    error_summary,
)
from scripts.wm_v2c_ej_common import (
    load_protocol_and_config,
    model_from_config,
    source_commit,
)
from src.world_models import (
    FutureProprioWorldModelDataset,
    apply_world_model_normalization,
    cosine_distance,
    ej_delta_error,
    standardized_proprio_delta_target,
)
from src.world_models.normalization import file_sha256


PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")
ADAPTER_MODES = ("full_ej", "ee_only", "joints_only")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def distribution(value: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if not flat.size or not np.isfinite(flat).all():
        raise ValueError("distribution requires finite non-empty input")
    q = np.quantile(flat, (0.25, 0.5, 0.75, 0.9, 0.95))
    return {
        "n": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "p25": float(q[0]),
        "median": float(q[1]),
        "p75": float(q[2]),
        "p90": float(q[3]),
        "p95": float(q[4]),
        "max": float(flat.max()),
    }


def vector_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def pairwise_cosine(prediction: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.clip(
                1.0
                - np.sum(prediction[:, left] * prediction[:, right], axis=-1)
                / (
                    np.linalg.norm(prediction[:, left], axis=-1)
                    * np.linalg.norm(prediction[:, right], axis=-1)
                ),
                0.0,
                None,
            )
            for left, right in PAIR_INDICES
        ],
        axis=1,
    )


def pairwise_l2(value: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.linalg.norm(value[:, left] - value[:, right], axis=-1) for left, right in PAIR_INDICES],
        axis=1,
    )


def load_validation(
    protocol: dict[str, Any], normalization: dict[str, Any], batch_size: int
) -> dict[str, np.ndarray]:
    dataset = FutureProprioWorldModelDataset(
        ROOT / protocol["frozen_upstream"]["d13_store"], "validation", 8
    )
    if len(dataset) != 5869:
        raise ValueError("validation sample count changed")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    names = (
        "z0",
        "proprio0",
        "proprio_future",
        "language",
        "actions",
        "z_future",
        "task_id",
        "demo_id",
        "start_timestep",
    )
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    target: list[np.ndarray] = []
    try:
        for batch in loader:
            for name in names:
                chunks[name].append(batch[name].numpy())
            target.append(
                standardized_proprio_delta_target(
                    batch["proprio0"], batch["proprio_future"], normalization
                ).numpy()
            )
    finally:
        dataset.close()
    result = {name: np.concatenate(values) for name, values in chunks.items()}
    result["target_p_delta"] = np.concatenate(target)
    identities = np.stack(
        (result["task_id"], result["demo_id"], result["start_timestep"]), axis=1
    )
    if len(np.unique(identities, axis=0)) != 5869:
        raise ValueError("validation identities are not unique")
    if set(result["demo_id"].tolist()) != set(range(45, 50)):
        raise ValueError("validation demo boundary changed")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("validation input contains NaN/Inf")
    return result


def load_candidates(
    protocol: dict[str, Any], data: dict[str, np.ndarray]
) -> np.ndarray:
    path = ROOT / protocol["frozen_upstream"]["candidate_validation_sidecar"]
    with np.load(path) as source:
        for name in ("task_id", "demo_id", "start_timestep"):
            if not np.array_equal(source[name], data[name]):
                raise ValueError(f"candidate identity mismatch: {name}")
        candidates = source["production_candidates"].copy()
        expert = source["expert_target"].copy()
    if candidates.shape != (5869, 3, 8, 7):
        raise ValueError("production candidate shape changed")
    if not np.array_equal(expert, data["actions"]):
        raise ValueError("candidate source action differs from D13")
    return candidates


def load_checkpoint(
    checkpoint_path: Path,
    protocol: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["method"] != "wm_v2c_ej_physical_visual_coupling":
        raise ValueError("checkpoint is not WM-v2C-EJ")
    if checkpoint["version"] != "wm-v2c-ej-seed11-v1":
        raise ValueError("unexpected WM-v2C-EJ checkpoint version")
    if checkpoint["config_sha256"] != protocol["experiment"]["config_sha256"]:
        raise ValueError("checkpoint config differs from preregistration")
    if checkpoint["d13_store_sha256"] != protocol["frozen_upstream"]["d13_store_sha256"]:
        raise ValueError("checkpoint D13 store differs")
    model = model_from_config(config)
    model.load_state_dict(checkpoint["model_state"])
    if sum(parameter.numel() for parameter in model.parameters()) != 901516:
        raise ValueError("WM-v2C-EJ parameter count changed")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("checkpoint contains NaN/Inf")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint["normalization"], checkpoint


@torch.inference_mode()
def validation_inference(
    model: torch.nn.Module,
    normalization: dict[str, Any],
    data: dict[str, np.ndarray],
    conditions: dict[str, np.ndarray],
    candidates: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    count = len(data["task_id"])
    global_error = np.empty((len(CONDITION_NAMES), count, 8), dtype=np.float32)
    global_separation_from_correct = np.empty_like(global_error)
    candidate_error = np.empty((3, 3, count, 8), dtype=np.float32)
    candidate_separation = np.empty((3, 3, count, 8), dtype=np.float32)
    correct_ej_error = np.empty((count, 8), dtype=np.float32)
    base_norm = np.empty((count, 8), dtype=np.float32)
    residual_norm = np.empty((count, 8), dtype=np.float32)
    base_residual_cosine = np.empty((count, 8), dtype=np.float32)
    final_norm_error = np.empty((count, 8), dtype=np.float32)
    residual_pair_l2 = np.empty((3, count, 8), dtype=np.float32)
    base_pair_l2 = np.empty((3, count, 8), dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        z0 = torch.from_numpy(data["z0"][start:stop]).to(device)
        proprio = torch.from_numpy(data["proprio0"][start:stop]).to(device)
        language = torch.from_numpy(data["language"][start:stop]).to(device)
        target = torch.from_numpy(data["z_future"][start:stop]).to(device)
        p_target = torch.from_numpy(data["target_p_delta"][start:stop]).to(device)
        normalized_proprio, _ = apply_world_model_normalization(
            proprio, torch.from_numpy(data["actions"][start:stop]).to(device), normalization
        )
        condition_predictions: list[torch.Tensor] = []
        for condition_index, name in enumerate(CONDITION_NAMES):
            raw = torch.from_numpy(conditions[name][start:stop]).to(device)
            _, normalized_action = apply_world_model_normalization(
                proprio, raw, normalization
            )
            components = model.forward_with_components(
                z0, normalized_proprio, language, normalized_action
            )
            condition_predictions.append(components["pred_z"])
            global_error[condition_index, start:stop] = cosine_distance(
                components["pred_z"], target
            ).cpu().numpy()
            if condition_index == 0:
                correct_ej_error[start:stop] = ej_delta_error(
                    components["pred_proprio_delta"], p_target
                ).cpu().numpy()
                base = components["base_visual_delta"].float()
                residual = components["physical_visual_residual"].float()
                base_norm[start:stop] = torch.linalg.vector_norm(base, dim=-1).cpu().numpy()
                residual_norm[start:stop] = torch.linalg.vector_norm(
                    residual, dim=-1
                ).cpu().numpy()
                base_residual_cosine[start:stop] = torch.nn.functional.cosine_similarity(
                    base, residual, dim=-1
                ).cpu().numpy()
                final_norm_error[start:stop] = (
                    torch.linalg.vector_norm(components["pred_z"].float(), dim=-1) - 1.0
                ).abs().cpu().numpy()

        correct_prediction = condition_predictions[0]
        for condition_index, prediction in enumerate(condition_predictions):
            global_separation_from_correct[condition_index, start:stop] = (
                1.0
                - torch.nn.functional.cosine_similarity(
                    prediction.float(), correct_prediction.float(), dim=-1
                )
            ).clamp_min(0.0).cpu().numpy()

        for mode_index, mode in enumerate(ADAPTER_MODES):
            predictions: list[torch.Tensor] = []
            bases: list[torch.Tensor] = []
            residuals: list[torch.Tensor] = []
            for candidate_index in range(3):
                raw = torch.from_numpy(
                    candidates[start:stop, candidate_index]
                ).to(device)
                _, normalized_action = apply_world_model_normalization(
                    proprio, raw, normalization
                )
                components = model.forward_with_components(
                    z0,
                    normalized_proprio,
                    language,
                    normalized_action,
                    adapter_mode=mode,
                )
                predictions.append(components["pred_z"])
                candidate_error[mode_index, candidate_index, start:stop] = (
                    cosine_distance(components["pred_z"], target).cpu().numpy()
                )
                if mode == "full_ej":
                    bases.append(components["base_visual_delta"])
                    residuals.append(components["physical_visual_residual"])
            stacked = torch.stack(predictions, dim=1)
            for pair_index, (left, right) in enumerate(PAIR_INDICES):
                candidate_separation[mode_index, pair_index, start:stop] = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        stacked[:, left].float(), stacked[:, right].float(), dim=-1
                    )
                ).clamp_min(0.0).cpu().numpy()
                if mode == "full_ej":
                    residual_pair_l2[pair_index, start:stop] = torch.linalg.vector_norm(
                        residuals[left].float() - residuals[right].float(), dim=-1
                    ).cpu().numpy()
                    base_pair_l2[pair_index, start:stop] = torch.linalg.vector_norm(
                        bases[left].float() - bases[right].float(), dim=-1
                    ).cpu().numpy()
    output = {
        "global_error": global_error,
        "global_separation_from_correct": global_separation_from_correct,
        "candidate_error": candidate_error,
        "candidate_separation": candidate_separation,
        "correct_ej_error": correct_ej_error,
        "base_norm": base_norm,
        "residual_norm": residual_norm,
        "base_residual_cosine": base_residual_cosine,
        "final_norm_error": final_norm_error,
        "residual_pair_l2": residual_pair_l2,
        "base_pair_l2": base_pair_l2,
    }
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("validation inference contains NaN/Inf")
    return output


@torch.inference_mode()
def matched_inference(
    model: torch.nn.Module,
    normalization: dict[str, Any],
    source: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    count = len(source["task_id"])
    predictions = np.empty((3, count, 3, 8, 512), dtype=np.float32)
    base = np.empty((count, 3, 8, 512), dtype=np.float32)
    residual = np.empty_like(base)
    predicted_ej = np.empty((count, 3, 8, 10), dtype=np.float32)
    for branch in range(count):
        z0 = torch.from_numpy(source["z0"][branch : branch + 1]).to(device)
        proprio = torch.from_numpy(source["proprio0"][branch : branch + 1]).to(device)
        language = torch.from_numpy(source["language"][branch : branch + 1]).to(device)
        for candidate in range(3):
            raw = torch.from_numpy(
                source["candidates"][branch : branch + 1, candidate]
            ).to(device)
            normalized_proprio, normalized_action = apply_world_model_normalization(
                proprio, raw, normalization
            )
            for mode_index, mode in enumerate(ADAPTER_MODES):
                components = model.forward_with_components(
                    z0,
                    normalized_proprio,
                    language,
                    normalized_action,
                    adapter_mode=mode,
                )
                predictions[mode_index, branch, candidate] = components[
                    "pred_z"
                ][0].cpu().numpy()
                if mode == "full_ej":
                    base[branch, candidate] = components["base_visual_delta"][0].cpu().numpy()
                    residual[branch, candidate] = components[
                        "physical_visual_residual"
                    ][0].cpu().numpy()
                    predicted_ej[branch, candidate] = components["pred_ej_delta"][0].cpu().numpy()
    output = {
        "predictions": predictions,
        "base_visual_delta": base,
        "physical_visual_residual": residual,
        "predicted_ej_delta": predicted_ej,
    }
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("matched inference contains NaN/Inf")
    return output


def direction_bootstrap(
    direction: np.ndarray, replicates: int, seed: int
) -> dict[str, Any]:
    count = direction.shape[0]
    indices = np.random.default_rng(seed).integers(0, count, size=(replicates, count))
    overall = direction[indices].mean(axis=(1, 2, 3))
    per_pair = direction[indices].mean(axis=(1, 3))
    return {
        "replicates": replicates,
        "seed": seed,
        "unit": "matched branch point",
        "overall_ci95": np.quantile(overall, (0.025, 0.975)).tolist(),
        "per_pair_ci95": {
            name: np.quantile(per_pair[:, index], (0.025, 0.975)).tolist()
            for index, name in enumerate(("A0-A1", "A0-A2"))
        },
    }


def matched_summary(
    protocol: dict[str, Any],
    source: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    real = source["real_z"].astype(np.float64)
    task_id = source["task_id"].astype(np.int64)
    real_separation = pairwise_cosine(real)
    modes: dict[str, Any] = {}
    mode_arrays: dict[str, np.ndarray] = {}
    old_error = float(protocol["matched_counterfactual"]["wm_v2b_mean_candidate_specific_error"])
    for mode_index, mode in enumerate(ADAPTER_MODES):
        prediction = arrays["predictions"][mode_index].astype(np.float64)
        separation = pairwise_cosine(prediction)
        error = np.clip(
            1.0
            - np.sum(prediction * real, axis=-1)
            / (np.linalg.norm(prediction, axis=-1) * np.linalg.norm(real, axis=-1)),
            0.0,
            None,
        )
        transfer = float(separation.mean() / real_separation.mean())
        direction = np.stack(
            [
                vector_cosine(
                    prediction[:, right] - prediction[:, left],
                    real[:, right] - real[:, left],
                )
                for left, right in PAIR_INDICES[:2]
            ],
            axis=1,
        )
        bootstrap = direction_bootstrap(
            direction,
            int(protocol["matched_counterfactual"]["bootstrap_replicates"]),
            int(protocol["matched_counterfactual"]["bootstrap_seed"]) + mode_index,
        )
        modes[mode] = {
            "predicted_separation": distribution(separation),
            "real_separation": distribution(real_separation),
            "counterfactual_transfer_ratio": transfer,
            "multiple_vs_wm_v2b": transfer
            / float(protocol["matched_counterfactual"]["wm_v2b_transfer_ratio"]),
            "candidate_specific_error": {
                "overall": distribution(error),
                "relative_change_vs_wm_v2b": float(error.mean() / old_error - 1.0),
                "per_candidate": {
                    f"A{candidate}": distribution(error[:, candidate])
                    for candidate in range(3)
                },
                "per_horizon": error.mean(axis=(0, 1)).tolist(),
                "per_task": {
                    str(task): float(error[task_id == task].mean())
                    for task in range(10)
                },
            },
            "visual_delta_direction": {
                "overall": distribution(direction),
                "per_pair": {
                    name: distribution(direction[:, index])
                    for index, name in enumerate(("A0-A1", "A0-A2"))
                },
                "per_horizon": direction.mean(axis=(0, 1)).tolist(),
                "per_task": {
                    str(task): float(direction[task_id == task].mean())
                    for task in range(10)
                },
                "bootstrap": bootstrap,
            },
            "per_horizon": [
                {
                    "horizon": horizon + 1,
                    "predicted_separation": float(separation[:, :, horizon].mean()),
                    "real_separation": float(real_separation[:, :, horizon].mean()),
                    "transfer_ratio": float(
                        separation[:, :, horizon].mean()
                        / real_separation[:, :, horizon].mean()
                    ),
                    "candidate_specific_error": float(error[:, :, horizon].mean()),
                    "direction": float(direction[:, :, horizon].mean()),
                }
                for horizon in range(8)
            ],
            "per_task": {
                str(task): {
                    "predicted_separation": float(separation[task_id == task].mean()),
                    "real_separation": float(real_separation[task_id == task].mean()),
                    "transfer_ratio": float(
                        separation[task_id == task].mean()
                        / real_separation[task_id == task].mean()
                    ),
                    "candidate_specific_error": float(error[task_id == task].mean()),
                    "direction": float(direction[task_id == task].mean()),
                }
                for task in range(10)
            },
        }
        mode_arrays[f"{mode}_predicted_separation"] = separation.astype(np.float32)
        mode_arrays[f"{mode}_candidate_error"] = error.astype(np.float32)
        mode_arrays[f"{mode}_visual_delta_direction"] = direction.astype(np.float32)
    return modes, {
        "real_separation": real_separation.astype(np.float32),
        **mode_arrays,
    }


def contribution_summary(
    validation: dict[str, np.ndarray], matched: dict[str, np.ndarray]
) -> dict[str, Any]:
    base_norm = validation["base_norm"].astype(np.float64)
    residual_norm = validation["residual_norm"].astype(np.float64)
    ratio = residual_norm / np.maximum(base_norm, 1e-12)
    residual_pair = validation["residual_pair_l2"].astype(np.float64)
    base_pair = validation["base_pair_l2"].astype(np.float64)
    matched_base = matched["base_visual_delta"].astype(np.float64)
    matched_residual = matched["physical_visual_residual"].astype(np.float64)
    matched_final = matched_base + matched_residual
    return {
        "demo_validation_correct_action": {
            "base_visual_delta_l2": distribution(base_norm),
            "physical_visual_residual_l2": distribution(residual_norm),
            "residual_to_base_l2": distribution(ratio),
            "base_residual_direction_cosine": distribution(
                validation["base_residual_cosine"]
            ),
            "final_latent_max_unit_norm_error": float(
                validation["final_norm_error"].max()
            ),
        },
        "production_candidate_sensitivity": {
            "base_pair_l2": distribution(base_pair),
            "adapter_residual_pair_l2": distribution(residual_pair),
            "adapter_pair_l2_to_base_pair_l2": float(
                residual_pair.mean() / base_pair.mean()
            ),
        },
        "matched_branches": {
            "base_visual_delta_l2": distribution(
                np.linalg.norm(matched_base, axis=-1)
            ),
            "physical_visual_residual_l2": distribution(
                np.linalg.norm(matched_residual, axis=-1)
            ),
            "final_visual_delta_l2": distribution(
                np.linalg.norm(matched_final, axis=-1)
            ),
            "base_residual_direction_cosine": distribution(
                vector_cosine(matched_base, matched_residual)
            ),
        },
    }


def make_decision(
    protocol: dict[str, Any],
    ordinary: dict[str, Any],
    global_action: dict[str, Any],
    matched: dict[str, Any],
    contribution: dict[str, Any],
) -> dict[str, Any]:
    gate = protocol["gate"]
    full = matched["full_ej"]
    direction = full["visual_delta_direction"]
    direction_boot = direction["bootstrap"]
    contribution_validation = contribution["demo_validation_correct_action"]
    sensitivity = contribution["production_candidate_sensitivity"]
    checks = {
        "A_ordinary_visual": {
            "d14_relative_degradation": ordinary["relative_change_vs_d14"]
            <= gate["A_ordinary_visual"]["relative_degradation_vs_d14_max"],
            "beats_copy_current": ordinary["overall"] < ordinary["copy_current"],
        },
        "B_global_action": {
            "within_shuffle_relative_delta": global_action["within_task_shuffle"][
                "relative_delta"
            ]
            >= gate["B_global_action"]["within_shuffle_relative_error_increase_min"],
            "within_shuffle_delta_ci": global_action["within_task_shuffle"][
                "delta_bootstrap"
            ]["ci95_lower"]
            > 0,
            "within_shuffle_accuracy": global_action["within_task_shuffle"][
                "true_action_win_accuracy"
            ]
            >= gate["B_global_action"]["correct_action_win_accuracy_min"],
            "within_shuffle_accuracy_ci": global_action["within_task_shuffle"][
                "accuracy_bootstrap"
            ]["ci95_lower"]
            > 0.5,
            "temporal_permutation_delta_ci": global_action["temporal_permutation"][
                "delta_bootstrap"
            ]["ci95_lower"]
            > 0,
            "magnitude_response": global_action["magnitude_distance_spearman"]
            >= gate["B_global_action"]["magnitude_response_spearman_min"],
        },
        "C_local_predicted_real_separation": {
            "absolute_ratio": full["counterfactual_transfer_ratio"]
            >= gate["C_local_predicted_real_separation"]["transfer_ratio_min"],
            "wm_v2b_multiple": full["multiple_vs_wm_v2b"]
            >= gate["C_local_predicted_real_separation"]["multiple_vs_wm_v2b_min"],
        },
        "D_candidate_specific_prediction": {
            "matched_error": full["candidate_specific_error"][
                "relative_change_vs_wm_v2b"
            ]
            <= gate["D_candidate_specific_prediction"][
                "mean_relative_degradation_vs_wm_v2b_max"
            ]
        },
        "E_visual_delta_direction": {
            "aggregate_mean": direction["overall"]["mean"] > 0,
            "aggregate_ci": direction_boot["overall_ci95"][0] > 0,
            "A0_A1_mean": direction["per_pair"]["A0-A1"]["mean"] > 0,
            "A0_A1_ci": direction_boot["per_pair_ci95"]["A0-A1"][0] > 0,
            "A0_A2_mean": direction["per_pair"]["A0-A2"]["mean"] > 0,
            "A0_A2_ci": direction_boot["per_pair_ci95"]["A0-A2"][0] > 0,
        },
        "F_residual_non_degeneracy": {
            "finite": True,
            "candidate_sensitive": sensitivity["adapter_pair_l2_to_base_pair_l2"]
            >= gate["F_residual_non_degeneracy"][
                "adapter_pair_l2_to_base_pair_l2_min"
            ],
            "mean_not_dominant": contribution_validation["residual_to_base_l2"]["mean"]
            <= gate["F_residual_non_degeneracy"][
                "mean_residual_l2_to_base_delta_l2_max"
            ],
            "p95_not_dominant": contribution_validation["residual_to_base_l2"]["p95"]
            <= gate["F_residual_non_degeneracy"][
                "p95_residual_l2_to_base_delta_l2_max"
            ],
            "unit_latent": contribution_validation["final_latent_max_unit_norm_error"]
            <= gate["F_residual_non_degeneracy"][
                "final_latent_unit_norm_tolerance"
            ],
        },
    }
    passed = {name: bool(all(values.values())) for name, values in checks.items()}
    if not passed["A_ordinary_visual"]:
        verdict = "PREDICTION_QUALITY_REGRESSION"
    elif not checks["F_residual_non_degeneracy"]["candidate_sensitive"]:
        verdict = "PHYSICAL_VISUAL_ADAPTER_NOT_USING_LOCAL_SIGNAL"
    elif passed["C_local_predicted_real_separation"] and not passed[
        "D_candidate_specific_prediction"
    ]:
        verdict = "LOCAL_VISUAL_SEPARATION_ARTIFACT"
    elif not all(passed.values()):
        verdict = "PHYSICAL_TO_VISUAL_TRANSFER_STILL_INSUFFICIENT"
    else:
        verdict = "WM_V2C_EJ_PASS"
    return {
        "verdict": verdict,
        "gate_checks": checks,
        "gate_pass": passed,
        "authorization": "GO_RERANKER_REDIAGNOSIS"
        if verdict == "WM_V2C_EJ_PASS"
        else "BLOCK_RERANKER_REDIAGNOSIS",
        "controller_authorized": False,
        "seed23_seed47_authorized": verdict == "WM_V2C_EJ_PASS",
        "legacy_full_12d_mediator_verdict": "BLOCK_WM_V2C_PHYSICAL_MEDIATOR",
        "legacy_reranker_verdict": "RERANKER_SCORE_BLOCK",
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(result: dict[str, Any]) -> str:
    training = result["training"]
    ordinary = result["ordinary_visual_prediction"]
    matched = result["matched_counterfactual"]
    full = matched["modes"]["full_ej"]
    contribution = result["physical_visual_contribution"]
    decision = result["decision"]
    curves = training["curves"]
    curve_rows = [
        [
            str(row["epoch"]),
            f"{row['train_visual_cosine_error']:.6f}",
            f"{row['train_ej_delta_smooth_l1']:.6f}",
            f"{row['train_residual_mean_square']:.3e}",
            f"{row['visual_cosine_error']:.6f}",
            f"{row['ej_delta_smooth_l1']:.6f}",
            f"{row['residual_mean_square']:.3e}",
            f"{row['total_loss']:.6f}",
        ]
        for row in curves
    ]
    horizon_rows = [
        [
            str(row["horizon"]),
            f"{ordinary['per_horizon'][row['horizon']-1]:.6f}",
            f"{row['predicted_separation']:.6e}",
            f"{row['real_separation']:.6e}",
            f"{row['transfer_ratio']:.4f}",
            f"{row['candidate_specific_error']:.6f}",
            f"{row['direction']:.4f}",
        ]
        for row in full["per_horizon"]
    ]
    task_rows = [
        [
            task,
            f"{ordinary['per_task'][task]['error']:.6f}",
            f"{row['predicted_separation']:.6e}",
            f"{row['real_separation']:.6e}",
            f"{row['transfer_ratio']:.4f}",
            f"{row['candidate_specific_error']:.6f}",
            f"{row['direction']:.4f}",
        ]
        for task, row in full["per_task"].items()
    ]
    mode_rows = [
        [
            mode,
            f"{values['predicted_separation']['mean']:.6e}",
            f"{values['counterfactual_transfer_ratio']:.4f}",
            f"{values['multiple_vs_wm_v2b']:.2f}x",
            f"{values['candidate_specific_error']['overall']['mean']:.6f}",
            f"{100*values['candidate_specific_error']['relative_change_vs_wm_v2b']:.2f}%",
            f"{values['visual_delta_direction']['overall']['mean']:.4f}",
        ]
        for mode, values in matched["modes"].items()
    ]
    gate_rows = [
        [name, "pass" if passed else "FAIL"]
        for name, passed in decision["gate_pass"].items()
    ]
    checkpoint = result["model"]
    global_action = result["global_action_conditioning"]
    production = result["production_candidate_validation"]
    residual = contribution["demo_validation_correct_action"]
    sensitivity = contribution["production_candidate_sensitivity"]
    return f"""# Explicit EE+joint coupling {decision['verdict'].replace('_', ' ').lower()}

Date: 2026-09-25

Final verdict: `{decision['verdict']}`

Authorization: `{decision['authorization']}`

Legacy 12-D mediator: `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`

Legacy reranker: `RERANKER_SCORE_BLOCK`

## Executive summary

WM-v2C-EJ initializes every existing module from frozen WM-v2B and adds only
a 75,072-parameter predicted-EE/joint-to-visual residual adapter. Gripper is
excluded by fixed indices and future physical state remains target-only.

The selected seed-11 checkpoint has ordinary validation cosine error
`{ordinary['overall']:.8f}` ({100*ordinary['relative_change_vs_d14']:+.2f}%
versus D14) and matched counterfactual transfer ratio
`{full['counterfactual_transfer_ratio']:.6f}`, or
`{full['multiple_vs_wm_v2b']:.2f}x` frozen WM-v2B. Mean matched candidate
error changes by `{100*full['candidate_specific_error']['relative_change_vs_wm_v2b']:+.2f}%`.

The mechanical verdict is `{decision['verdict']}`. Even a PASS authorizes only
offline reranker rediagnosis; no controller, D20-v2, C2-v2, seed23/47, or
init-state 40--49 work occurs here.

## Frozen provenance and experiment boundary

- Protocol commit: `{result['protocol']['preregistered_commit']}`; protocol SHA-256 `{result['protocol']['sha256']}`.
- WM-v2B initialization checkpoint SHA-256: `{result['frozen_provenance']['wm_v2b_checkpoint_sha256']}`.
- D13 store SHA-256: `{result['frozen_provenance']['d13_store_sha256']}`.
- Matched branch sidecar SHA-256: `{result['frozen_provenance']['matched_sidecar_sha256']}`.
- Train demos 0--44; validation demos 45--49; H=8; 52,381/5,869 samples.
- Matched init 30/35 is final mechanism audit only. Init 40--49 was not accessed.

The prior `PARTIAL_PHYSICAL_MEDIATOR_PASS` and complete 12-D
`BLOCK_WM_V2C_PHYSICAL_MEDIATOR` are unchanged.

## Architecture and parameter difference

```text
shared GRU h_t
  +--> frozen-form base visual delta head
  +--> frozen-form 12-D physical head
          +--> select ee_pos[0:3] + joints[5:12]
                  +--> Linear(10,64)-GELU-Linear(64,128)-GELU-Linear(128,512)
```

`pred_z = normalize(z0 + base_visual_delta + physical_visual_residual)`.
WM-v2B has 826,444 parameters; the adapter adds 75,072; total is 901,516.
The final adapter layer starts at exact zero, so initialization preserves the
WM-v2B visual prediction. No warm-start stage, new GRU, Transformer, camera,
or encoder is used.

## EE/joint target and explicit gripper exclusion

The target is standardized future-minus-current proprio for dimensions
`[0,1,2,5,6,7,8,9,10,11]`: 3-D EE position plus seven joints. Gripper
dimensions `[3,4]` remain in the legacy physical head only and have no edge
into the adapter. Future proprio is training target only.

## Gradient calibration and fixed losses

Train-only shared-GRU gradient norms are `g_z={result['gradient_calibration']['g_z']:.10f}`
and `g_EJ={result['gradient_calibration']['g_ej']:.10f}`. The frozen coefficient
is `lambda_EJ={checkpoint['lambda_ej']:.10f}`; `lambda_res={checkpoint['lambda_res']}`
was fixed before calibration. The objective is `L_z + lambda_EJ*L_EJ +
lambda_res*L_res` with no fake counterfactual target.

## Train and validation curves

{markdown_table(['epoch','train z','train EJ','train res','val z','val EJ','val res','val total'], curve_rows)}

Best checkpoint: epoch {checkpoint['epoch']}, step {checkpoint['global_step']},
SHA-256 `{checkpoint['checkpoint_sha256']}`. GPU peak allocated memory:
{training['gpu_peak_memory_allocated_bytes']} bytes.

## Ordinary visual prediction

Overall correct-action cosine error is `{ordinary['overall']:.8f}` versus D14
`{ordinary['d14_reference']:.8f}`, WM-v2B `{ordinary['wm_v2b_reference']:.8f}`,
and copy-current `{ordinary['copy_current']:.8f}`.

## Global action conditioning

Within-task shuffled actions change error by
`{100*global_action['within_task_shuffle']['relative_delta']:.2f}%` with paired
accuracy `{global_action['within_task_shuffle']['true_action_win_accuracy']:.2%}`.
Fixed temporal permutation delta CI lower is
`{global_action['temporal_permutation']['delta_bootstrap']['ci95_lower']:.8f}`;
magnitude-distance Spearman is `{global_action['magnitude_distance_spearman']:.4f}`.

## Frozen production candidates on demo validation

Mean candidate separation/error ratio is
`{production['full_ej']['separation_to_error_ratio']:.6f}`. References are
WM-v1 `0.004235`, WM-v2A `0.005307`, and WM-v2B `0.005157`.

## Matched real counterfactual visual fidelity

{markdown_table(['mode','pred sep','transfer','vs v2B','candidate error','error delta','direction'], mode_rows)}

The final gate uses only the preregistered `full_ej` row. EE-only and
joints-only are inference ablations and do not redefine the architecture.

## Candidate-specific matched future errors

Full-EJ A0/A1/A2 errors are
`{full['candidate_specific_error']['per_candidate']['A0']['mean']:.6f}` /
`{full['candidate_specific_error']['per_candidate']['A1']['mean']:.6f}` /
`{full['candidate_specific_error']['per_candidate']['A2']['mean']:.6f}`.
Frozen WM-v2B aggregate reference is
`{result['matched_counterfactual']['wm_v2b_mean_candidate_error']:.6f}`.

## Visual-delta direction fidelity

Aggregate A0-relative direction is
`{full['visual_delta_direction']['overall']['mean']:.4f}` with branch-bootstrap
95% CI `[{full['visual_delta_direction']['bootstrap']['overall_ci95'][0]:.4f},
{full['visual_delta_direction']['bootstrap']['overall_ci95'][1]:.4f}]`.

## Base/residual contribution and adapter action sensitivity

Validation residual/base L2 mean is `{residual['residual_to_base_l2']['mean']:.4f}`
and p95 is `{residual['residual_to_base_l2']['p95']:.4f}`. Mean adapter candidate
pair L2 is `{sensitivity['adapter_residual_pair_l2']['mean']:.6f}` versus base
`{sensitivity['base_pair_l2']['mean']:.6f}`, ratio
`{sensitivity['adapter_pair_l2_to_base_pair_l2']:.4f}`. Base/residual direction
cosine is `{residual['base_residual_direction_cosine']['mean']:.4f}`.

## Per-horizon matched analysis

{markdown_table(['h','val error','pred sep','real sep','transfer','matched error','direction'], horizon_rows)}

## Per-task matched analysis

{markdown_table(['task','val error','pred sep','real sep','transfer','matched error','direction'], task_rows)}

## Mechanical gate decision

{markdown_table(['gate','result'], gate_rows)}

Final verdict: `{decision['verdict']}`. Authorization:
`{decision['authorization']}`. Controller authorization is always false.

## Integrity and forbidden-activity audit

- Config SHA-256: `{result['protocol']['config_sha256']}`.
- Result: `{result['artifacts']['result']}`.
- Diagnostic sidecar SHA-256: `{result['artifacts']['sidecar_sha256']}`.
- NaN/Inf audit: `pass`.
- Gripper adapter input: `false`.
- No simulator rerun, training on matched branches, controller success, C1/C2,
  D20-v2, reward/done/success, init 40--49, seed23/47, ensemble, or upstream
  overwrite occurred.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/wm_v2c_ej_protocol.json"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/world_model_v2c_ej_seed11.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/world_model_v2c_ej/seed11/best.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    protocol, config = load_protocol_and_config(args.protocol, args.config)
    output_path = ROOT / protocol["outputs"]["diagnosis"]
    sidecar_path = ROOT / protocol["outputs"]["diagnostic_sidecar"]
    report_path = ROOT / protocol["outputs"]["report"]
    if any(path.exists() for path in (output_path, sidecar_path, report_path)):
        raise FileExistsError("WM-v2C-EJ diagnosis artifact already exists")
    device = torch.device(args.device)
    model, normalization, checkpoint = load_checkpoint(
        args.checkpoint, protocol, config, device
    )
    data = load_validation(protocol, normalization, max(args.batch_size, 256))
    candidate = load_candidates(protocol, data)
    action_protocol = json.loads(
        (ROOT / protocol["frozen_upstream"]["action_conditioning_protocol"]).read_text()
    )
    conditions, within_source, cross_source = construct_conditions(data, action_protocol)
    validation_arrays = validation_inference(
        model,
        normalization,
        data,
        conditions,
        candidate,
        device,
        args.batch_size,
    )
    bootstrap = BootstrapPlan(
        data["task_id"], data["demo_id"], 10000, 220902
    )
    visual_conditions = {
        name: error_summary(validation_arrays["global_error"][index], data["task_id"])
        for index, name in enumerate(CONDITION_NAMES)
    }
    comparisons = {
        name: comparison_summary(
            validation_arrays["global_error"][0],
            validation_arrays["global_error"][CONDITION_NAMES.index(name)],
            data["task_id"],
            bootstrap,
        )
        for name in (
            "zero",
            "shuffle_task",
            "shuffle_cross",
            "reverse",
            "temporal_shuffle",
            "repeat_first",
        )
    }
    scale_distance = []
    scale_separation = []
    for scale, name in SCALE_CONDITIONS:
        scale_distance.append(abs(scale - 1.0))
        scale_separation.append(
            float(
                validation_arrays["global_separation_from_correct"][
                    CONDITION_NAMES.index(name)
                ].mean()
            )
        )
    magnitude = scipy_stats.spearmanr(scale_distance, scale_separation)
    copy_error = np.clip(
        1.0
        - np.sum(data["z0"][:, None] * data["z_future"], axis=-1)
        / (
            np.linalg.norm(data["z0"], axis=-1)[:, None]
            * np.linalg.norm(data["z_future"], axis=-1)
        ),
        0.0,
        None,
    )
    ordinary_error = validation_arrays["global_error"][0]
    ordinary = {
        "overall": float(ordinary_error.mean()),
        "per_horizon": ordinary_error.mean(axis=0).tolist(),
        "per_task": {
            str(task): {
                "samples": int(np.sum(data["task_id"] == task)),
                "error": float(ordinary_error[data["task_id"] == task].mean()),
            }
            for task in range(10)
        },
        "d14_reference": float(protocol["frozen_upstream"]["d14_seed11_validation_error"]),
        "wm_v2b_reference": float(
            json.loads(
                (ROOT / protocol["frozen_upstream"]["wm_v2b_result"]).read_text()
            )["visual_prediction"]["correct_action_error"]
        ),
        "copy_current": float(copy_error.mean()),
        "copy_per_horizon": copy_error.mean(axis=0).tolist(),
        "relative_change_vs_d14": float(
            ordinary_error.mean()
            / protocol["frozen_upstream"]["d14_seed11_validation_error"]
            - 1.0
        ),
    }
    global_action = {
        "condition_error": visual_conditions,
        "within_task_shuffle": comparisons["shuffle_task"],
        "cross_task_shuffle": comparisons["shuffle_cross"],
        "physical_zero": comparisons["zero"],
        "reverse": comparisons["reverse"],
        "temporal_permutation": comparisons["temporal_shuffle"],
        "repeat_first": comparisons["repeat_first"],
        "magnitude_distance_spearman": float(magnitude.statistic),
        "magnitude_distance_spearman_pvalue": float(magnitude.pvalue),
        "magnitude_separation_from_correct": {
            name: float(
                validation_arrays["global_separation_from_correct"][
                    CONDITION_NAMES.index(name)
                ].mean()
            )
            for _, name in SCALE_CONDITIONS
        },
    }
    production = {}
    for mode_index, mode in enumerate(ADAPTER_MODES):
        separation = validation_arrays["candidate_separation"][mode_index]
        error = validation_arrays["candidate_error"][mode_index]
        production[mode] = {
            "mean_pairwise_cosine_separation": float(separation.mean()),
            "per_horizon": separation.mean(axis=(0, 1)).tolist(),
            "separation_to_error_ratio": float(
                separation.mean() / ordinary_error.mean()
            ),
            "candidate_error": float(error.mean()),
            "per_task": {
                str(task): {
                    "separation": float(
                        separation[:, data["task_id"] == task].mean()
                    ),
                    "candidate_error": float(
                        error[:, data["task_id"] == task].mean()
                    ),
                }
                for task in range(10)
            },
        }

    matched_path = ROOT / protocol["frozen_upstream"]["matched_counterfactual_sidecar"]
    with np.load(matched_path) as source_npz:
        matched_source = {name: source_npz[name].copy() for name in source_npz.files}
    if set(matched_source["init_state_index"].tolist()) != {30, 35}:
        raise ValueError("matched branch init-state boundary changed")
    if not np.array_equal(matched_source["candidates"], matched_source["executed_actions"]):
        raise ValueError("matched candidates differ from executed actions")
    matched_arrays = matched_inference(model, normalization, matched_source, device)
    matched_modes, matched_metric_arrays = matched_summary(
        protocol, matched_source, matched_arrays
    )
    contribution = contribution_summary(validation_arrays, matched_arrays)
    decision = make_decision(
        protocol, ordinary, global_action, matched_modes, contribution
    )
    curves = [
        json.loads(line)
        for line in (args.checkpoint.parent / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    calibration = checkpoint["calibration_freeze"]
    gradient_result = json.loads(
        (ROOT / calibration["calibration_result"]).read_text()
    )
    sidecar_arrays = {
        "task_id": data["task_id"].astype(np.int8),
        "demo_id": data["demo_id"].astype(np.int8),
        "start_timestep": data["start_timestep"].astype(np.int32),
        "within_task_source_index": within_source.astype(np.int32),
        "cross_task_source_index": cross_source.astype(np.int32),
        **validation_arrays,
        "matched_task_id": matched_source["task_id"].astype(np.int8),
        "matched_init_state_index": matched_source["init_state_index"].astype(np.int8),
        "matched_planning_index": matched_source["planning_index"].astype(np.int8),
        "matched_predictions": matched_arrays["predictions"],
        "matched_base_visual_delta": matched_arrays["base_visual_delta"],
        "matched_physical_visual_residual": matched_arrays["physical_visual_residual"],
        "matched_predicted_ej_delta": matched_arrays["predicted_ej_delta"],
        **matched_metric_arrays,
    }
    atomic_npz(sidecar_path, sidecar_arrays)
    result = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "COMPLETE",
        "verdict": decision["verdict"],
        "authorization": decision["authorization"],
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": file_sha256(args.protocol),
            "config_sha256": file_sha256(args.config),
            "preregistered_commit": checkpoint["calibration_freeze"][
                "preregistered_commit"
            ],
            "training_source_commit": checkpoint["source_commit"],
            "analysis_source_commit": source_commit(),
        },
        "frozen_provenance": {
            "wm_v2b_1_verdict": "PARTIAL_PHYSICAL_MEDIATOR_PASS",
            "full_12d_mediator": "BLOCK_WM_V2C_PHYSICAL_MEDIATOR",
            "wm_v2b_checkpoint_sha256": protocol["frozen_upstream"][
                "wm_v2b_checkpoint_sha256"
            ],
            "d13_store_sha256": protocol["frozen_upstream"]["d13_store_sha256"],
            "matched_sidecar_sha256": protocol["frozen_upstream"][
                "matched_counterfactual_sidecar_sha256"
            ],
        },
        "scope": {
            "train_demos": "0-44",
            "validation_demos": "45-49",
            "matched_init_states": [30, 35],
            "matched_planning_indices": [0, 8],
            "init_40_49_accessed": False,
            "simulator_rerun": False,
            "controller_success_used": False,
            "future_proprio_is_input": False,
            "gripper_adapter_input": False,
        },
        "model": {
            "parameters": 901516,
            "wm_v2b_parameters": 826444,
            "adapter_parameters": 75072,
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "lambda_ej": float(checkpoint["lambda_ej"]),
            "lambda_res": float(checkpoint["lambda_res"]),
            "ej_indices": config["data"]["ej_target_indices"],
            "gripper_indices_excluded": [3, 4],
        },
        "gradient_calibration": gradient_result["gradient_calibration"],
        "training": {
            "curves": curves,
            "best_validation": json.loads(
                (args.checkpoint.parent / "best_validation.json").read_text()
            ),
            "run_metadata": json.loads(
                (args.checkpoint.parent / "run_metadata.json").read_text()
            ),
            "gpu_peak_memory_allocated_bytes": json.loads(
                (args.checkpoint.parent / "run_metadata.json").read_text()
            )["gpu_peak_memory_allocated_bytes"],
        },
        "ordinary_visual_prediction": ordinary,
        "ej_validation": {
            "smooth_l1": float(validation_arrays["correct_ej_error"].mean()),
            "per_horizon": validation_arrays["correct_ej_error"].mean(axis=0).tolist(),
        },
        "global_action_conditioning": global_action,
        "production_candidate_validation": production,
        "matched_counterfactual": {
            "real_visual_separation": distribution(
                matched_metric_arrays["real_separation"]
            ),
            "wm_v2b_transfer_ratio": float(
                protocol["matched_counterfactual"]["wm_v2b_transfer_ratio"]
            ),
            "wm_v2b_mean_candidate_error": float(
                protocol["matched_counterfactual"][
                    "wm_v2b_mean_candidate_specific_error"
                ]
            ),
            "modes": matched_modes,
        },
        "physical_visual_contribution": contribution,
        "decision": decision,
        "artifacts": {
            "result": str(output_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
            "sidecar": str(sidecar_path.relative_to(ROOT)),
            "sidecar_sha256": sha256(sidecar_path),
            "sidecar_arrays": {
                name: list(value.shape) for name, value in sidecar_arrays.items()
            },
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "upstream_artifact_overwritten": False,
            "d13_contract_split_h_or_normalization_changed": False,
            "future_proprio_used_as_input": False,
            "gripper_entered_adapter": False,
            "fake_counterfactual_training_target": False,
            "matched_branch_used_for_training_or_selection": False,
            "candidate_generator_scale_or_reranker_changed": False,
            "controller_c1_c2_d20v2_or_success_run": False,
            "init_40_49_access": False,
            "seed23_47_or_ensemble_trained": False,
            "legacy_12d_mediator_reinterpreted": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("WM-v2C-EJ result contains NaN/Inf")
    without_hash = copy.deepcopy(result)
    payload = (json.dumps(without_hash, indent=2, ensure_ascii=False) + "\n").encode()
    result["artifacts"]["result_sha256_before_hash_field"] = hashlib.sha256(
        payload
    ).hexdigest()
    atomic_json(output_path, result)
    atomic_text(report_path, render_report(result))
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "verdict": result["verdict"],
                "authorization": result["authorization"],
                "ordinary_visual_error": ordinary["overall"],
                "matched_transfer_ratio": matched_modes["full_ej"][
                    "counterfactual_transfer_ratio"
                ],
                "candidate_error_relative_change": matched_modes["full_ej"][
                    "candidate_specific_error"
                ]["relative_change_vs_wm_v2b"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
