#!/usr/bin/env python3
"""Offline gate for WM-v2A seed 11 using frozen D13/C1 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import stats as scipy_stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_wm_action_conditioning import (
    BootstrapPlan,
    CONDITION_NAMES,
    SCALE_CONDITIONS,
    comparison_summary,
    construct_conditions,
    distribution,
    error_summary,
    load_validation,
)
from scripts.train_world_model import make_model
from scripts.train_world_model_v2a import delta_error
from src.world_models import apply_world_model_normalization, cosine_distance


PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != "wm-v2a-local-dynamics-resolution-v1":
        raise ValueError("unexpected WM-v2A protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_TRAINING":
        raise ValueError("WM-v2A protocol was not preregistered")
    frozen = protocol["frozen_wm_v1"]
    if subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() != frozen["commit"]:
        raise ValueError("WM-v1 freeze tag changed")
    checks = (
        (frozen["c1_v2_2_result"], frozen["c1_v2_2_result_sha256"]),
        (frozen["c1_v2_2_report"], frozen["c1_v2_2_report_sha256"]),
        (frozen["d14_seed11_checkpoint"], frozen["d14_seed11_checkpoint_sha256"]),
        (frozen["d15_transition_result"], frozen["d15_transition_result_sha256"]),
        (protocol["data"]["store"], protocol["data"]["store_sha256"]),
        (protocol["experiment"]["config"], protocol["experiment"]["config_sha256"]),
        (
            protocol["frozen_diagnostics"]["c1_v2_2_protocol"],
            protocol["frozen_diagnostics"]["c1_v2_2_protocol_sha256"],
        ),
        (
            protocol["frozen_diagnostics"]["c1_v2_1_sidecar"],
            protocol["frozen_diagnostics"]["c1_v2_1_sidecar_sha256"],
        ),
        (
            protocol["frozen_diagnostics"]["c1_v2_2_sidecar"],
            protocol["frozen_diagnostics"]["c1_v2_2_sidecar_sha256"],
        ),
    )
    for relative, expected in checks:
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")
    return protocol, sha256(path)


def load_model(
    protocol: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["method"] != "wm_v2a_local_dynamics_resolution":
        raise ValueError("checkpoint is not WM-v2A")
    if state["version"] != "wm-v2a-seed11-v1":
        raise ValueError("unexpected WM-v2A checkpoint version")
    if state["config_sha256"] != protocol["experiment"]["config_sha256"]:
        raise ValueError("checkpoint config changed")
    if state["d13_store_sha256"] != protocol["data"]["store_sha256"]:
        raise ValueError("checkpoint D13 store changed")
    if state["loss_contract"] != state["config"]["loss"]:
        raise ValueError("checkpoint loss contract is inconsistent")
    if float(state["loss_contract"]["lambda_delta"]) != 1.0:
        raise ValueError("checkpoint lambda differs from preregistration")
    model = make_model(state["config"])
    model.load_state_dict(state["model_state"])
    if sum(parameter.numel() for parameter in model.parameters()) != 792000:
        raise ValueError("WM-v2A architecture changed")
    if not all(torch.isfinite(value).all() for value in state["model_state"].values()):
        raise FloatingPointError("checkpoint contains NaN/Inf")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state["normalization"], state


def load_frozen_candidates(
    protocol: dict[str, Any], data: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    path = ROOT / protocol["frozen_diagnostics"]["c1_v2_1_sidecar"]
    with np.load(path) as source:
        required = (
            "task_id",
            "demo_id",
            "start_timestep",
            "expert_target",
            "production_candidates",
            "production_normalized_action_mse",
            "production_chunk_l2_error",
            "1x_raw_action_mse",
        )
        result = {key: source[key].copy() for key in required}
    for key in ("task_id", "demo_id", "start_timestep"):
        if not np.array_equal(result[key], data[key]):
            raise ValueError(f"frozen candidate identity mismatch: {key}")
    if not np.allclose(result["expert_target"], data["actions"], atol=0.0, rtol=0.0):
        raise ValueError("frozen candidate expert target differs from D13 actions")
    if result["production_candidates"].shape != (5869, 3, 8, 7):
        raise ValueError("frozen production candidate shape changed")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("frozen candidate sidecar contains NaN/Inf")
    return result


def _cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(first.float(), second.float(), dim=-1)


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    normalization: dict[str, Any],
    data: dict[str, np.ndarray],
    conditions: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    count = len(data["task_id"])
    global_error = np.empty((len(CONDITION_NAMES), count, 8), dtype=np.float32)
    global_separation = np.empty_like(global_error)
    candidate_error = np.empty((3, count, 8), dtype=np.float32)
    candidate_separation = np.empty((3, count, 8), dtype=np.float32)
    correct_delta = np.empty((count, 8), dtype=np.float32)
    correct_prediction_displacement = np.empty((count, 8), dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        z0 = torch.from_numpy(data["z0"][start:stop]).to(device)
        proprio = torch.from_numpy(data["proprio0"][start:stop]).to(device)
        language = torch.from_numpy(data["language"][start:stop]).to(device)
        target = torch.from_numpy(data["z_future"][start:stop]).to(device)
        normalized_proprio, _ = apply_world_model_normalization(
            proprio,
            torch.from_numpy(data["actions"][start:stop]).to(device),
            normalization,
        )
        predictions = []
        for condition_index, name in enumerate(CONDITION_NAMES):
            raw_action = torch.from_numpy(conditions[name][start:stop]).to(device)
            _, normalized_action = apply_world_model_normalization(
                proprio, raw_action, normalization
            )
            prediction = model(z0, normalized_proprio, language, normalized_action)
            predictions.append(prediction)
            global_error[condition_index, start:stop] = (
                cosine_distance(prediction, target).cpu().numpy()
            )
        correct = predictions[0]
        correct_delta[start:stop] = delta_error(
            correct, z0, target, beta=1.0
        ).cpu().numpy()
        correct_prediction_displacement[start:stop] = (
            _cosine_distance(correct, z0.unsqueeze(1)).clamp_min(0.0).cpu().numpy()
        )
        for condition_index, prediction in enumerate(predictions):
            global_separation[condition_index, start:stop] = (
                _cosine_distance(prediction, correct).clamp_min(0.0).cpu().numpy()
            )

        local_predictions = []
        raw_candidates = candidate["production_candidates"][start:stop]
        for candidate_index in range(3):
            raw_action = torch.from_numpy(raw_candidates[:, candidate_index]).to(device)
            _, normalized_action = apply_world_model_normalization(
                proprio, raw_action, normalization
            )
            prediction = model(z0, normalized_proprio, language, normalized_action)
            local_predictions.append(prediction)
            candidate_error[candidate_index, start:stop] = (
                cosine_distance(prediction, target).cpu().numpy()
            )
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            candidate_separation[pair_index, start:stop] = (
                _cosine_distance(local_predictions[first], local_predictions[second])
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )
    output = {
        "global_error": global_error,
        "global_separation": global_separation,
        "candidate_error": candidate_error,
        "candidate_separation": candidate_separation,
        "correct_delta": correct_delta,
        "correct_prediction_displacement": correct_prediction_displacement,
    }
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("WM-v2A diagnosis produced NaN/Inf")
    return output


def scalar_bootstrap(values: np.ndarray, bootstrap: BootstrapPlan) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return bootstrap.summarize(np.repeat(values[:, None], 8, axis=1))


def directional_summary(
    action_error: np.ndarray,
    prediction_error: np.ndarray,
    task_id: np.ndarray,
    bootstrap: BootstrapPlan,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    action_error = np.asarray(action_error, dtype=np.float64)
    prediction_error = np.asarray(prediction_error, dtype=np.float64)
    prediction_overall = prediction_error.mean(axis=-1).T  # [N,3]
    pair_win = np.empty((len(task_id), 3), dtype=np.float64)
    pair_valid = np.empty_like(pair_win, dtype=bool)
    pair_win_horizon = np.empty((len(task_id), 3, 8), dtype=np.float64)
    for pair_index, (first, second) in enumerate(PAIR_INDICES):
        action_difference = action_error[:, first] - action_error[:, second]
        prediction_difference = (
            prediction_overall[:, first] - prediction_overall[:, second]
        )
        pair_valid[:, pair_index] = action_difference != 0.0
        pair_win[:, pair_index] = (
            action_difference * prediction_difference > 0.0
        ).astype(np.float64)
        for horizon in range(8):
            difference_h = (
                prediction_error[first, :, horizon]
                - prediction_error[second, :, horizon]
            )
            pair_win_horizon[:, pair_index, horizon] = (
                action_difference * difference_h > 0.0
            ).astype(np.float64)
    if not pair_valid.all():
        raise ValueError("unexpected exact action-error tie in production candidates")
    sample_accuracy = pair_win.mean(axis=1)
    horizon_accuracy = pair_win_horizon.mean(axis=1)
    accuracy_bootstrap = scalar_bootstrap(sample_accuracy, bootstrap)
    horizon_bootstrap = bootstrap.summarize(horizon_accuracy)

    within_spearman = np.empty(len(task_id), dtype=np.float64)
    within_kendall = np.empty(len(task_id), dtype=np.float64)
    for index in range(len(task_id)):
        within_spearman[index] = float(
            scipy_stats.spearmanr(action_error[index], prediction_overall[index]).statistic
        )
        within_kendall[index] = float(
            scipy_stats.kendalltau(action_error[index], prediction_overall[index]).statistic
        )
    spearman_bootstrap = scalar_bootstrap(within_spearman, bootstrap)
    per_task = {}
    for task in range(10):
        selected = task_id == task
        per_task[str(task)] = {
            "sample_count": int(selected.sum()),
            "pairwise_ranking_accuracy": float(pair_win[selected].mean()),
            "mean_within_sample_spearman": float(within_spearman[selected].mean()),
            "mean_within_sample_kendall": float(within_kendall[selected].mean()),
            "candidate_prediction_error": prediction_overall[selected]
            .mean(axis=0)
            .tolist(),
            "candidate_action_error": action_error[selected].mean(axis=0).tolist(),
        }
    summary = {
        "definition": "lower expert-action MSE should rank with lower predicted-future cosine error; imitation-oriented only",
        "pairwise_ranking_accuracy": float(pair_win.mean()),
        "pairwise_accuracy_ci95": [
            accuracy_bootstrap["ci95_lower"],
            accuracy_bootstrap["ci95_upper"],
        ],
        "pairwise_accuracy_per_horizon": horizon_accuracy.mean(axis=0).tolist(),
        "pairwise_accuracy_per_horizon_ci95": [
            [record["ci95_lower"], record["ci95_upper"]]
            for record in horizon_bootstrap["per_horizon"]
        ],
        "mean_within_sample_spearman": float(within_spearman.mean()),
        "spearman_ci95": [
            spearman_bootstrap["ci95_lower"],
            spearman_bootstrap["ci95_upper"],
        ],
        "mean_within_sample_kendall": float(within_kendall.mean()),
        "candidate_action_error": action_error.mean(axis=0).tolist(),
        "candidate_prediction_error": prediction_overall.mean(axis=0).tolist(),
        "per_task": per_task,
        "expert_action_caveat": "nearest expert action is not claimed to maximize closed-loop task success",
    }
    arrays = {
        "direction_pair_win": pair_win.astype(np.float32),
        "direction_pair_win_horizon": pair_win_horizon.astype(np.float32),
        "direction_within_spearman": within_spearman.astype(np.float32),
        "direction_within_kendall": within_kendall.astype(np.float32),
    }
    return summary, arrays


def candidate_action_separation(
    candidates: np.ndarray, normalization: dict[str, Any], task_id: np.ndarray
) -> dict[str, Any]:
    mean = np.asarray(normalization["action"]["mean"], dtype=np.float32)
    std = np.asarray(normalization["action"]["std"], dtype=np.float32)
    normalized = (candidates - mean[None, None, None, :]) / std[None, None, None, :]
    raw = []
    norm = []
    pair_records = []
    for name, (first, second) in zip(PAIR_NAMES, PAIR_INDICES):
        raw_values = np.linalg.norm(
            candidates[:, first] - candidates[:, second], axis=(1, 2)
        )
        norm_values = np.linalg.norm(
            normalized[:, first] - normalized[:, second], axis=(1, 2)
        )
        raw.append(raw_values)
        norm.append(norm_values)
        pair_records.append(
            {
                "pair": name,
                "raw_chunk_l2": distribution(raw_values),
                "normalized_chunk_l2": distribution(norm_values),
            }
        )
    raw_array = np.stack(raw, axis=1)
    norm_array = np.stack(norm, axis=1)
    return {
        "pairs": pair_records,
        "mean_pairwise_raw_chunk_l2": float(raw_array.mean()),
        "mean_pairwise_normalized_chunk_l2": float(norm_array.mean()),
        "per_task_mean_normalized_chunk_l2": {
            str(task): float(norm_array[task_id == task].mean()) for task in range(10)
        },
    }


def local_sensitivity_summary(
    separation: np.ndarray,
    ordinary_error: np.ndarray,
    real_displacement: np.ndarray,
    task_id: np.ndarray,
) -> dict[str, Any]:
    separation_h = separation.mean(axis=(0, 1))
    ordinary_h = ordinary_error.mean(axis=0)
    real_h = real_displacement.mean(axis=0)
    return {
        "mean_pairwise_cosine_separation": float(separation.mean()),
        "per_horizon_separation": separation_h.tolist(),
        "ordinary_correct_action_error": float(ordinary_error.mean()),
        "separation_to_ordinary_error_ratio": float(
            separation.mean() / ordinary_error.mean()
        ),
        "per_horizon_separation_to_error_ratio": (separation_h / ordinary_h).tolist(),
        "separation_to_real_motion_ratio": float(
            separation.mean() / real_displacement.mean()
        ),
        "per_horizon_separation_to_real_motion_ratio": (
            separation_h / real_h
        ).tolist(),
        "per_pair": {
            PAIR_NAMES[index]: {
                "overall": float(separation[index].mean()),
                "per_horizon": separation[index].mean(axis=0).tolist(),
            }
            for index in range(3)
        },
        "per_task": {
            str(task): {
                "separation": float(separation[:, task_id == task].mean()),
                "separation_to_error_ratio": float(
                    separation[:, task_id == task].mean()
                    / ordinary_error[task_id == task].mean()
                ),
                "separation_to_real_motion_ratio": float(
                    separation[:, task_id == task].mean()
                    / real_displacement[task_id == task].mean()
                ),
            }
            for task in range(10)
        },
    }


def decide(
    protocol: dict[str, Any],
    prediction: dict[str, Any],
    within: dict[str, Any],
    local: dict[str, Any],
    directional: dict[str, Any],
) -> dict[str, Any]:
    gate = protocol["gate"]
    prediction_checks = {
        "same_seed_relative_error_change": prediction[
            "relative_error_change_vs_d14_seed11"
        ]
        <= gate["prediction_quality"]["same_seed_d14_relative_error_change_max"],
        "beats_copy_current": prediction["correct_action_error"]
        < prediction["copy_current_error"],
    }
    global_checks = {
        "within_relative_error_increase": within["relative_delta"]
        >= gate["global_action_conditioning"][
            "within_shuffle_relative_error_increase_min"
        ],
        "within_delta_ci_lower_positive": within["delta_bootstrap"]["ci95_lower"]
        > 0.0,
        "within_ranking_accuracy": within["true_action_win_accuracy"]
        >= gate["global_action_conditioning"]["within_shuffle_ranking_accuracy_min"],
        "within_ranking_ci_lower_above_random": within["accuracy_bootstrap"][
            "ci95_lower"
        ]
        > 0.5,
    }
    local_checks = {
        "separation_to_error_ratio": local["separation_to_ordinary_error_ratio"]
        >= gate["local_sensitivity"][
            "candidate_separation_to_new_model_error_min"
        ],
        "improvement_multiple_vs_wm_v1": local["improvement_multiple_vs_wm_v1"]
        >= gate["local_sensitivity"]["ratio_to_frozen_wm_v1_reference_min"],
    }
    direction_checks = {
        "pairwise_ranking_accuracy": directional["pairwise_ranking_accuracy"]
        >= gate["local_directional_validity"]["pairwise_ranking_accuracy_min"],
        "pairwise_ci_lower_above_random": directional["pairwise_accuracy_ci95"][0]
        > 0.5,
        "mean_within_sample_spearman": directional["mean_within_sample_spearman"]
        >= gate["local_directional_validity"]["mean_within_sample_spearman_min"],
        "spearman_ci_lower_positive": directional["spearman_ci95"][0] > 0.0,
    }
    passes = {
        "prediction_quality": all(prediction_checks.values()),
        "global_action_conditioning": all(global_checks.values()),
        "local_sensitivity": all(local_checks.values()),
        "local_directional_validity": all(direction_checks.values()),
    }
    if not passes["prediction_quality"]:
        verdict = "PREDICTION_QUALITY_REGRESSION"
    elif not passes["global_action_conditioning"] or not passes["local_sensitivity"]:
        verdict = "DELTA_OBJECTIVE_INSUFFICIENT"
    elif not passes["local_directional_validity"]:
        verdict = "LOCAL_SENSITIVITY_ARTIFACT"
    else:
        verdict = "WM_V2A_LOCAL_SENSITIVITY_PASS"
    return {
        "verdict": verdict,
        "checks": {
            "prediction_quality": prediction_checks,
            "global_action_conditioning": global_checks,
            "local_sensitivity": local_checks,
            "local_directional_validity": direction_checks,
        },
        "gate_pass": passes,
        "seed23_seed47_authorized": verdict == "WM_V2A_LOCAL_SENSITIVITY_PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2a_local_dynamics_protocol.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/world_model_v2a/seed11/best.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    output_path = ROOT / protocol["outputs"]["diagnosis"]
    sidecar_path = ROOT / protocol["outputs"]["sample_sidecar"]
    if output_path.exists() or sidecar_path.exists():
        raise FileExistsError("WM-v2A diagnosis output already exists")
    device = torch.device(args.device)
    model, normalization, checkpoint = load_model(protocol, args.checkpoint, device)
    data = load_validation(protocol, max(args.batch_size, 256))
    frozen_diagnostic_protocol = json.loads(
        (ROOT / protocol["frozen_diagnostics"]["c1_v2_2_protocol"]).read_text()
    )
    conditions, within_source, cross_source = construct_conditions(
        data, frozen_diagnostic_protocol
    )
    candidate = load_frozen_candidates(protocol, data)
    arrays = run_inference(
        model,
        normalization,
        data,
        conditions,
        candidate,
        device,
        args.batch_size,
    )
    with np.load(
        ROOT / protocol["frozen_diagnostics"]["c1_v2_2_sidecar"]
    ) as frozen_v2_2:
        if not all(
            np.array_equal(frozen_v2_2[key], data[key])
            for key in ("task_id", "demo_id", "start_timestep")
        ):
            raise ValueError("C1-v2.2 sidecar identity mismatch")
        real_displacement = frozen_v2_2["real_displacement"].copy()

    condition_error = {
        name: error_summary(arrays["global_error"][index], data["task_id"])
        for index, name in enumerate(CONDITION_NAMES)
    }
    bootstrap = BootstrapPlan(
        data["task_id"],
        data["demo_id"],
        int(protocol["bootstrap"]["replicates"]),
        int(protocol["bootstrap"]["seed"]),
    )
    comparisons = {
        name: comparison_summary(
            arrays["global_error"][0],
            arrays["global_error"][CONDITION_NAMES.index(name)],
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
    copy_current = 1.0 - np.sum(
        data["z0"][:, None, :] * data["z_future"], axis=-1
    ) / (
        np.linalg.norm(data["z0"], axis=-1)[:, None]
        * np.linalg.norm(data["z_future"], axis=-1)
    )
    prediction = {
        "correct_action_error": float(arrays["global_error"][0].mean()),
        "correct_action_error_per_horizon": arrays["global_error"][0]
        .mean(axis=0)
        .tolist(),
        "delta_smooth_l1": float(arrays["correct_delta"].mean()),
        "delta_smooth_l1_per_horizon": arrays["correct_delta"].mean(axis=0).tolist(),
        "copy_current_error": float(copy_current.mean()),
        "copy_current_error_per_horizon": copy_current.mean(axis=0).tolist(),
        "d14_seed11_error": protocol["frozen_wm_v1"]["d14_seed11_validation_error"],
        "d15_ensemble_error": protocol["frozen_wm_v1"][
            "d15_ensemble_validation_error"
        ],
        "relative_error_change_vs_d14_seed11": float(
            arrays["global_error"][0].mean()
            / protocol["frozen_wm_v1"]["d14_seed11_validation_error"]
            - 1.0
        ),
        "relative_error_change_vs_d15_ensemble": float(
            arrays["global_error"][0].mean()
            / protocol["frozen_wm_v1"]["d15_ensemble_validation_error"]
            - 1.0
        ),
        "per_task": {
            str(task): {
                "sample_count": int(np.sum(data["task_id"] == task)),
                "correct_action_error": float(
                    arrays["global_error"][0, data["task_id"] == task].mean()
                ),
                "delta_smooth_l1": float(
                    arrays["correct_delta"][data["task_id"] == task].mean()
                ),
                "copy_current_error": float(
                    copy_current[data["task_id"] == task].mean()
                ),
            }
            for task in range(10)
        },
    }

    action_sep = candidate_action_separation(
        candidate["production_candidates"], normalization, data["task_id"]
    )
    local = local_sensitivity_summary(
        arrays["candidate_separation"],
        arrays["global_error"][0],
        real_displacement,
        data["task_id"],
    )
    local["frozen_wm_v1_ratio"] = protocol["frozen_wm_v1"][
        "production_candidate_local_sensitivity_ratio"
    ]
    local["improvement_multiple_vs_wm_v1"] = (
        local["separation_to_ordinary_error_ratio"] / local["frozen_wm_v1_ratio"]
    )
    local["action_separation"] = action_sep
    directional, direction_arrays = directional_summary(
        candidate["1x_raw_action_mse"],
        arrays["candidate_error"],
        data["task_id"],
        bootstrap,
    )

    magnitude = []
    distance_from_one = []
    separation_from_one = []
    for scale, name in SCALE_CONDITIONS:
        index = CONDITION_NAMES.index(name)
        separation = float(arrays["global_separation"][index].mean())
        distance_from_one.append(abs(scale - 1.0))
        separation_from_one.append(separation)
        magnitude.append(
            {
                "scale": scale,
                "error": condition_error[name]["overall"],
                "separation_from_1x": separation,
                "separation_to_correct_error_ratio": separation
                / prediction["correct_action_error"],
            }
        )
    magnitude_spearman = scipy_stats.spearmanr(
        distance_from_one, separation_from_one
    )
    global_diagnostics = {
        "condition_error": condition_error,
        "comparisons": comparisons,
        "magnitude": magnitude,
        "magnitude_distance_spearman": float(magnitude_spearman.statistic),
        "magnitude_distance_spearman_pvalue": float(magnitude_spearman.pvalue),
    }
    decision = decide(
        protocol, prediction, comparisons["shuffle_task"], local, directional
    )

    output_dir = args.checkpoint.parent
    training_artifacts = {}
    for name in ("config.yaml", "run_metadata.json", "metrics.jsonl", "best_validation.json"):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        training_artifacts[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
    sidecar_arrays = {
        "task_id": data["task_id"].astype(np.int8),
        "demo_id": data["demo_id"].astype(np.int8),
        "start_timestep": data["start_timestep"].astype(np.int32),
        "within_task_source_index": within_source.astype(np.int32),
        "cross_task_source_index": cross_source.astype(np.int32),
        "global_error": arrays["global_error"],
        "global_separation": arrays["global_separation"],
        "candidate_error": arrays["candidate_error"],
        "candidate_separation": arrays["candidate_separation"],
        "correct_delta": arrays["correct_delta"],
        "correct_prediction_displacement": arrays[
            "correct_prediction_displacement"
        ],
        **direction_arrays,
    }
    atomic_npz(sidecar_path, **sidecar_arrays)
    result = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "COMPLETE",
        "verdict": decision["verdict"],
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": protocol_hash,
            "preregistered_commit": checkpoint["source_commit"],
        },
        "scope": {
            "train_demos": "0-44",
            "validation_demos": "45-49",
            "train_samples": protocol["data"]["train_samples"],
            "validation_samples": protocol["data"]["validation_samples"],
            "tasks": 10,
            "simulator_or_controller_rollouts": 0,
            "success_metric_used": False,
        },
        "model": {
            "architecture": "unchanged D14 LatentDynamicsModel",
            "parameters": 792000,
            "seed": 11,
            "lambda_delta": 1.0,
            "loss_contract": checkpoint["loss_contract"],
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_global_step": int(checkpoint["global_step"]),
        },
        "training_artifacts": training_artifacts,
        "prediction_quality": prediction,
        "global_action_diagnostics": global_diagnostics,
        "local_candidate_sensitivity": local,
        "local_directional_validity": directional,
        "decision": decision,
        "sample_artifact": {
            "path": str(sidecar_path.relative_to(ROOT)),
            "sha256": sha256(sidecar_path),
            "arrays": {key: list(value.shape) for key, value in sidecar_arrays.items()},
        },
        "nan_inf_audit": "pass",
        "seed23_seed47_trained": False,
        "wm_v2b_implemented": False,
        "forbidden_activity_audit": {
            "c0_v2_modified": False,
            "candidate_generator_or_scale_modified": False,
            "reranker_modified": False,
            "clip_or_camera_modified": False,
            "world_model_architecture_modified": False,
            "simulator_or_controller_run": False,
            "init_state_outcome_read_or_run": False,
            "d20_v2_or_c2_v2_started": False,
            "d14_or_d15_checkpoint_overwritten": False,
            "success_rate_used_for_selection": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("WM-v2A result contains NaN/Inf")
    atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "correct_error": prediction["correct_action_error"],
                "delta_error": prediction["delta_smooth_l1"],
                "local_ratio": local["separation_to_ordinary_error_ratio"],
                "direction_accuracy": directional["pairwise_ranking_accuracy"],
                "direction_spearman": directional["mean_within_sample_spearman"],
                "seed_expansion_authorized": decision[
                    "seed23_seed47_authorized"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
