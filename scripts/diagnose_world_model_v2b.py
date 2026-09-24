#!/usr/bin/env python3
"""Offline visual/physical gate for the WM-v2B seed-11 checkpoint."""

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
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_world_model_v2b import model_from_config
from scripts.diagnose_wm_action_conditioning import (
    BootstrapPlan,
    CONDITION_NAMES,
    SCALE_CONDITIONS,
    comparison_summary,
    construct_conditions,
    error_summary,
)
from scripts.diagnose_world_model_v2a import (
    PAIR_INDICES,
    PAIR_NAMES,
    candidate_action_separation,
    directional_summary,
)
from src.world_models import (
    FutureProprioWorldModelDataset,
    PROPRIO_GROUPS,
    apply_world_model_normalization,
    cosine_distance,
    standardized_proprio_delta_target,
)
from src.world_models.normalization import file_sha256


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
    if protocol["protocol_version"] != "wm-v2b-future-proprio-auxiliary-v1":
        raise ValueError("unexpected WM-v2B protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_CALIBRATION_AND_TRAINING":
        raise ValueError("WM-v2B protocol was not preregistered")
    frozen = protocol["frozen_upstream"]
    if subprocess.check_output(
        ["git", "rev-parse", f"{frozen['wm_v2a_annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() != frozen["wm_v2a_commit"]:
        raise ValueError("WM-v2A freeze tag changed")
    checks = (
        (frozen["wm_v2a_result"], frozen["wm_v2a_result_sha256"]),
        (frozen["wm_v2a_report"], frozen["wm_v2a_report_sha256"]),
        (frozen["wm_v2a_checkpoint"], frozen["wm_v2a_checkpoint_sha256"]),
        (frozen["d14_seed11_checkpoint"], frozen["d14_seed11_checkpoint_sha256"]),
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
    if state["method"] != "wm_v2b_future_proprio_auxiliary":
        raise ValueError("checkpoint is not WM-v2B")
    if state["version"] != "wm-v2b-seed11-v1":
        raise ValueError("unexpected checkpoint version")
    if state["config_sha256"] != protocol["experiment"]["config_sha256"]:
        raise ValueError("checkpoint config changed")
    if state["d13_store_sha256"] != protocol["data"]["store_sha256"]:
        raise ValueError("checkpoint store changed")
    model = model_from_config(state["config"])
    model.load_state_dict(state["model_state"])
    if sum(parameter.numel() for parameter in model.parameters()) != 826444:
        raise ValueError("WM-v2B parameter count changed")
    if not all(torch.isfinite(value).all() for value in state["model_state"].values()):
        raise FloatingPointError("checkpoint contains NaN/Inf")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state["normalization"], state


def load_validation(
    protocol: dict[str, Any], normalization: dict[str, Any], batch_size: int
) -> dict[str, np.ndarray]:
    dataset = FutureProprioWorldModelDataset(
        ROOT / protocol["data"]["store"], "validation", 8
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
    target_chunks: list[np.ndarray] = []
    try:
        for batch in loader:
            for name in names:
                chunks[name].append(batch[name].numpy())
            target_chunks.append(
                standardized_proprio_delta_target(
                    batch["proprio0"], batch["proprio_future"], normalization
                ).numpy()
            )
    finally:
        dataset.close()
    result = {name: np.concatenate(value, axis=0) for name, value in chunks.items()}
    result["target_p_delta"] = np.concatenate(target_chunks, axis=0)
    if result["proprio_future"].shape != (5869, 8, 12):
        raise ValueError("future proprio shape changed")
    if set(result["demo_id"].tolist()) != set(range(45, 50)):
        raise ValueError("validation demo boundary failed")
    identities = np.stack(
        (result["task_id"], result["demo_id"], result["start_timestep"]), axis=1
    )
    if len(np.unique(identities, axis=0)) != 5869:
        raise ValueError("validation identities are not unique")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("validation data contains NaN/Inf")
    return result


def load_candidates(
    protocol: dict[str, Any], data: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    with np.load(
        ROOT / protocol["frozen_diagnostics"]["c1_v2_1_sidecar"]
    ) as source:
        names = (
            "task_id",
            "demo_id",
            "start_timestep",
            "expert_target",
            "production_candidates",
            "1x_raw_action_mse",
        )
        result = {name: source[name].copy() for name in names}
    for name in ("task_id", "demo_id", "start_timestep"):
        if not np.array_equal(result[name], data[name]):
            raise ValueError(f"candidate identity mismatch: {name}")
    if not np.array_equal(result["expert_target"], data["actions"]):
        raise ValueError("candidate expert target differs from D13")
    if result["production_candidates"].shape != (5869, 3, 8, 7):
        raise ValueError("candidate shape changed")
    return result


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    normalization: dict[str, Any],
    data: dict[str, np.ndarray],
    conditions: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    count = len(data["task_id"])
    condition_count = len(CONDITION_NAMES)
    global_z_error = np.empty((condition_count, count, 8), dtype=np.float32)
    global_z_separation = np.empty_like(global_z_error)
    global_p_error = np.empty_like(global_z_error)
    candidate_z_error = np.empty((3, count, 8), dtype=np.float32)
    candidate_z_separation = np.empty((3, count, 8), dtype=np.float32)
    candidate_p_error = np.empty((3, count, 8), dtype=np.float32)
    candidate_p_separation = np.empty((3, count, 8), dtype=np.float32)
    candidate_p_group_separation = np.empty((3, 3, count, 8), dtype=np.float32)
    correct_p_element_error = np.empty((count, 8, 12), dtype=np.float32)
    correct_p_rmse = np.empty((count, 8), dtype=np.float32)
    group_slices = tuple(PROPRIO_GROUPS.values())
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        z0 = torch.from_numpy(data["z0"][start:stop]).to(device)
        proprio = torch.from_numpy(data["proprio0"][start:stop]).to(device)
        language = torch.from_numpy(data["language"][start:stop]).to(device)
        z_target = torch.from_numpy(data["z_future"][start:stop]).to(device)
        p_target = torch.from_numpy(data["target_p_delta"][start:stop]).to(device)
        normalized_proprio, _ = apply_world_model_normalization(
            proprio, torch.from_numpy(data["actions"][start:stop]).to(device), normalization
        )
        global_z_predictions = []
        for condition_index, name in enumerate(CONDITION_NAMES):
            raw_action = torch.from_numpy(conditions[name][start:stop]).to(device)
            _, normalized_action = apply_world_model_normalization(
                proprio, raw_action, normalization
            )
            pred_z, pred_p = model(z0, normalized_proprio, language, normalized_action)
            global_z_predictions.append(pred_z)
            global_z_error[condition_index, start:stop] = (
                cosine_distance(pred_z, z_target).cpu().numpy()
            )
            global_p_error[condition_index, start:stop] = (
                F.smooth_l1_loss(pred_p.float(), p_target.float(), reduction="none")
                .mean(dim=-1)
                .cpu()
                .numpy()
            )
            if condition_index == 0:
                element = F.smooth_l1_loss(
                    pred_p.float(), p_target.float(), reduction="none"
                )
                correct_p_element_error[start:stop] = element.cpu().numpy()
                correct_p_rmse[start:stop] = (
                    torch.sqrt(torch.mean((pred_p.float() - p_target.float()) ** 2, dim=-1))
                    .cpu()
                    .numpy()
                )
        correct_z = global_z_predictions[0]
        for condition_index, pred_z in enumerate(global_z_predictions):
            global_z_separation[condition_index, start:stop] = (
                (1.0 - F.cosine_similarity(pred_z.float(), correct_z.float(), dim=-1))
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )

        local_z = []
        local_p = []
        raw = candidates["production_candidates"][start:stop]
        for candidate_index in range(3):
            raw_action = torch.from_numpy(raw[:, candidate_index]).to(device)
            _, normalized_action = apply_world_model_normalization(
                proprio, raw_action, normalization
            )
            pred_z, pred_p = model(z0, normalized_proprio, language, normalized_action)
            local_z.append(pred_z)
            local_p.append(pred_p)
            candidate_z_error[candidate_index, start:stop] = (
                cosine_distance(pred_z, z_target).cpu().numpy()
            )
            candidate_p_error[candidate_index, start:stop] = (
                F.smooth_l1_loss(pred_p.float(), p_target.float(), reduction="none")
                .mean(dim=-1)
                .cpu()
                .numpy()
            )
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            candidate_z_separation[pair_index, start:stop] = (
                (1.0 - F.cosine_similarity(local_z[first].float(), local_z[second].float(), dim=-1))
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )
            difference = local_p[first].float() - local_p[second].float()
            candidate_p_separation[pair_index, start:stop] = (
                torch.sqrt(torch.mean(difference**2, dim=-1)).cpu().numpy()
            )
            for group_index, group in enumerate(group_slices):
                candidate_p_group_separation[
                    pair_index, group_index, start:stop
                ] = torch.sqrt(torch.mean(difference[..., group] ** 2, dim=-1)).cpu().numpy()
    output = {
        "global_z_error": global_z_error,
        "global_z_separation": global_z_separation,
        "global_p_error": global_p_error,
        "candidate_z_error": candidate_z_error,
        "candidate_z_separation": candidate_z_separation,
        "candidate_p_error": candidate_p_error,
        "candidate_p_separation": candidate_p_separation,
        "candidate_p_group_separation": candidate_p_group_separation,
        "correct_p_element_error": correct_p_element_error,
        "correct_p_rmse": correct_p_rmse,
    }
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("WM-v2B inference contains NaN/Inf")
    return output


def visual_local_summary(
    separation: np.ndarray,
    error: np.ndarray,
    real_motion: np.ndarray,
    task_id: np.ndarray,
) -> dict[str, Any]:
    separation_h = separation.mean(axis=(0, 1))
    error_h = error.mean(axis=0)
    real_h = real_motion.mean(axis=0)
    return {
        "mean_pairwise_cosine_separation": float(separation.mean()),
        "per_horizon_separation": separation_h.tolist(),
        "separation_to_error_ratio": float(separation.mean() / error.mean()),
        "per_horizon_separation_to_error_ratio": (separation_h / error_h).tolist(),
        "separation_to_real_motion_ratio": float(separation.mean() / real_motion.mean()),
        "per_horizon_separation_to_real_motion_ratio": (separation_h / real_h).tolist(),
        "per_task": {
            str(task): {
                "separation": float(separation[:, task_id == task].mean()),
                "separation_to_error_ratio": float(
                    separation[:, task_id == task].mean() / error[task_id == task].mean()
                ),
            }
            for task in range(10)
        },
    }


def proprio_local_summary(
    separation: np.ndarray,
    group_separation: np.ndarray,
    correct_rmse: np.ndarray,
    task_id: np.ndarray,
) -> dict[str, Any]:
    separation_h = separation.mean(axis=(0, 1))
    error_h = correct_rmse.mean(axis=0)
    group_names = tuple(PROPRIO_GROUPS)
    return {
        "metric": "RMSE in standardized proprio-delta units",
        "mean_pairwise_trajectory_separation": float(separation.mean()),
        "per_horizon_separation": separation_h.tolist(),
        "correct_prediction_rmse": float(correct_rmse.mean()),
        "separation_to_correct_prediction_rmse_ratio": float(
            separation.mean() / correct_rmse.mean()
        ),
        "per_horizon_separation_to_error_ratio": (separation_h / error_h).tolist(),
        "groups": {
            name: {
                "mean_separation": float(group_separation[:, index].mean()),
                "per_horizon": group_separation[:, index].mean(axis=(0, 1)).tolist(),
            }
            for index, name in enumerate(group_names)
        },
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
                "correct_prediction_rmse": float(correct_rmse[task_id == task].mean()),
                "separation_to_error_ratio": float(
                    separation[:, task_id == task].mean()
                    / correct_rmse[task_id == task].mean()
                ),
            }
            for task in range(10)
        },
    }


def direction_checks(
    rule: dict[str, Any], result: dict[str, Any]
) -> dict[str, bool]:
    return {
        "pairwise_accuracy": result["pairwise_ranking_accuracy"]
        >= rule["pairwise_ranking_accuracy_min"],
        "pairwise_ci": result["pairwise_accuracy_ci95"][0] > 0.5,
        "spearman": result["mean_within_sample_spearman"]
        >= rule["mean_within_sample_spearman_min"],
        "spearman_ci": result["spearman_ci95"][0] > 0.0,
    }


def decide(
    protocol: dict[str, Any],
    prediction: dict[str, Any],
    visual_within: dict[str, Any],
    visual_local: dict[str, Any],
    visual_direction: dict[str, Any],
    proprio_prediction: dict[str, Any],
    proprio_within: dict[str, Any],
    proprio_local: dict[str, Any],
    proprio_direction: dict[str, Any],
) -> dict[str, Any]:
    gate = protocol["gate"]
    checks = {
        "prediction_quality": {
            "same_seed_degradation": prediction["relative_error_change_vs_d14_seed11"]
            <= gate["prediction_quality"]["same_seed_d14_relative_error_change_max"],
            "beats_copy": prediction["correct_action_error"] < prediction["copy_current_error"],
        },
        "visual_global": {
            "relative_delta": visual_within["relative_delta"]
            >= gate["visual_global_action_conditioning"]["within_shuffle_relative_error_increase_min"],
            "delta_ci": visual_within["delta_bootstrap"]["ci95_lower"] > 0.0,
            "accuracy": visual_within["true_action_win_accuracy"]
            >= gate["visual_global_action_conditioning"]["within_shuffle_ranking_accuracy_min"],
            "accuracy_ci": visual_within["accuracy_bootstrap"]["ci95_lower"] > 0.5,
        },
        "visual_local": {
            "absolute_ratio": visual_local["separation_to_error_ratio"]
            >= gate["visual_local_sensitivity"]["candidate_separation_to_new_model_error_min"],
            "wm_v1_multiple": visual_local["improvement_multiple_vs_wm_v1"]
            >= gate["visual_local_sensitivity"]["ratio_to_frozen_wm_v1_reference_min"],
        },
        "visual_direction": direction_checks(
            gate["visual_local_directional_validity"], visual_direction
        ),
        "proprio_prediction": {
            "beats_zero_delta": proprio_prediction["correct_action_error"]
            < proprio_prediction["zero_delta_baseline_error"]
        },
        "proprio_action": {
            "relative_delta": proprio_within["relative_delta"]
            >= gate["proprio_action_conditioning"]["within_shuffle_relative_error_increase_min"],
            "delta_ci": proprio_within["delta_bootstrap"]["ci95_lower"] > 0.0,
            "accuracy": proprio_within["true_action_win_accuracy"]
            >= gate["proprio_action_conditioning"]["within_shuffle_ranking_accuracy_min"],
            "accuracy_ci": proprio_within["accuracy_bootstrap"]["ci95_lower"] > 0.5,
        },
        "proprio_local": {
            "separation_ratio": proprio_local[
                "separation_to_correct_prediction_rmse_ratio"
            ]
            >= gate["proprio_local_separation"][
                "candidate_trajectory_rmse_separation_to_correct_prediction_rmse_min"
            ]
        },
        "proprio_direction": direction_checks(
            gate["proprio_local_directional_validity"], proprio_direction
        ),
    }
    passed = {name: all(values.values()) for name, values in checks.items()}
    if not passed["prediction_quality"]:
        verdict = "PREDICTION_QUALITY_REGRESSION"
    elif not passed["proprio_prediction"] or not passed["proprio_action"]:
        verdict = "PROPRIO_AUX_ACTION_SIGNAL_FAIL"
    elif (
        not passed["visual_local"]
        and passed["proprio_local"]
        and passed["proprio_direction"]
    ):
        verdict = "PHYSICAL_SIGNAL_LEARNED_VISUAL_TRANSFER_INSUFFICIENT"
    elif not all(passed.values()):
        verdict = "WM_V2B_AUXILIARY_INSUFFICIENT"
    else:
        verdict = "WM_V2B_LOCAL_SENSITIVITY_PASS"
    return {
        "verdict": verdict,
        "checks": checks,
        "gate_pass": passed,
        "seed23_seed47_authorized": verdict == "WM_V2B_LOCAL_SENSITIVITY_PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2b_future_proprio_protocol.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/world_model_v2b/seed11/best.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    output_path = ROOT / protocol["outputs"]["diagnosis"]
    sidecar_path = ROOT / protocol["outputs"]["sample_sidecar"]
    if output_path.exists() or sidecar_path.exists():
        raise FileExistsError("WM-v2B diagnosis output already exists")
    device = torch.device(args.device)
    model, normalization, checkpoint = load_model(protocol, args.checkpoint, device)
    data = load_validation(protocol, normalization, max(args.batch_size, 256))
    frozen_protocol = json.loads(
        (ROOT / protocol["frozen_diagnostics"]["c1_v2_2_protocol"]).read_text()
    )
    conditions, within_source, cross_source = construct_conditions(data, frozen_protocol)
    candidates = load_candidates(protocol, data)
    arrays = run_inference(
        model, normalization, data, conditions, candidates, device, args.batch_size
    )
    with np.load(
        ROOT / protocol["frozen_diagnostics"]["c1_v2_2_sidecar"]
    ) as frozen_sidecar:
        if not all(
            np.array_equal(frozen_sidecar[name], data[name])
            for name in ("task_id", "demo_id", "start_timestep")
        ):
            raise ValueError("frozen sidecar identity mismatch")
        real_motion = frozen_sidecar["real_displacement"].copy()

    bootstrap = BootstrapPlan(
        data["task_id"],
        data["demo_id"],
        int(protocol["bootstrap"]["replicates"]),
        int(protocol["bootstrap"]["seed"]),
    )
    visual_conditions = {
        name: error_summary(arrays["global_z_error"][index], data["task_id"])
        for index, name in enumerate(CONDITION_NAMES)
    }
    visual_comparisons = {
        name: comparison_summary(
            arrays["global_z_error"][0],
            arrays["global_z_error"][CONDITION_NAMES.index(name)],
            data["task_id"],
            bootstrap,
        )
        for name in ("zero", "shuffle_task", "shuffle_cross", "reverse", "temporal_shuffle", "repeat_first")
    }
    proprio_conditions = {
        name: error_summary(
            arrays["global_p_error"][CONDITION_NAMES.index(name)], data["task_id"]
        )
        for name in ("true", "zero", "shuffle_task", "shuffle_cross")
    }
    proprio_comparisons = {
        name: comparison_summary(
            arrays["global_p_error"][0],
            arrays["global_p_error"][CONDITION_NAMES.index(name)],
            data["task_id"],
            bootstrap,
        )
        for name in ("zero", "shuffle_task", "shuffle_cross")
    }
    copy_error = 1.0 - np.sum(
        data["z0"][:, None, :] * data["z_future"], axis=-1
    )
    zero_p_element = F.smooth_l1_loss(
        torch.zeros_like(torch.from_numpy(data["target_p_delta"])),
        torch.from_numpy(data["target_p_delta"]),
        reduction="none",
    ).numpy()
    prediction = {
        "correct_action_error": float(arrays["global_z_error"][0].mean()),
        "per_horizon": arrays["global_z_error"][0].mean(axis=0).tolist(),
        "copy_current_error": float(copy_error.mean()),
        "copy_per_horizon": copy_error.mean(axis=0).tolist(),
        "d14_seed11_error": protocol["frozen_upstream"]["d14_seed11_validation_error"],
        "relative_error_change_vs_d14_seed11": float(
            arrays["global_z_error"][0].mean()
            / protocol["frozen_upstream"]["d14_seed11_validation_error"]
            - 1.0
        ),
        "per_task": {
            str(task): {
                "sample_count": int(np.sum(data["task_id"] == task)),
                "error": float(arrays["global_z_error"][0, data["task_id"] == task].mean()),
            }
            for task in range(10)
        },
    }
    p_element = arrays["correct_p_element_error"]
    p_dim = p_element.mean(axis=(0, 1))
    proprio_prediction = {
        "correct_action_error": float(arrays["global_p_error"][0].mean()),
        "per_horizon": arrays["global_p_error"][0].mean(axis=0).tolist(),
        "per_dimension": p_dim.tolist(),
        "groups": {
            name: float(p_dim[group].mean()) for name, group in PROPRIO_GROUPS.items()
        },
        "zero_delta_baseline_error": float(zero_p_element.mean()),
        "zero_delta_per_horizon": zero_p_element.mean(axis=(0, 2)).tolist(),
        "per_task": {
            str(task): {
                "error": float(arrays["global_p_error"][0, data["task_id"] == task].mean()),
                "zero_delta_error": float(zero_p_element[data["task_id"] == task].mean()),
            }
            for task in range(10)
        },
    }
    visual_local = visual_local_summary(
        arrays["candidate_z_separation"], arrays["global_z_error"][0], real_motion, data["task_id"]
    )
    visual_local["wm_v1_reference"] = protocol["frozen_upstream"]["wm_v1_local_sensitivity_ratio"]
    visual_local["wm_v2a_reference"] = protocol["frozen_upstream"]["wm_v2a_local_sensitivity_ratio"]
    visual_local["improvement_multiple_vs_wm_v1"] = (
        visual_local["separation_to_error_ratio"] / visual_local["wm_v1_reference"]
    )
    visual_local["improvement_multiple_vs_wm_v2a"] = (
        visual_local["separation_to_error_ratio"] / visual_local["wm_v2a_reference"]
    )
    visual_local["action_separation"] = candidate_action_separation(
        candidates["production_candidates"], normalization, data["task_id"]
    )
    proprio_local = proprio_local_summary(
        arrays["candidate_p_separation"],
        arrays["candidate_p_group_separation"],
        arrays["correct_p_rmse"],
        data["task_id"],
    )
    visual_direction, visual_direction_arrays = directional_summary(
        candidates["1x_raw_action_mse"], arrays["candidate_z_error"], data["task_id"], bootstrap
    )
    proprio_direction, proprio_direction_arrays = directional_summary(
        candidates["1x_raw_action_mse"], arrays["candidate_p_error"], data["task_id"], bootstrap
    )
    magnitude = []
    distances = []
    separations = []
    for scale, name in SCALE_CONDITIONS:
        index = CONDITION_NAMES.index(name)
        separation = float(arrays["global_z_separation"][index].mean())
        distances.append(abs(scale - 1.0))
        separations.append(separation)
        magnitude.append(
            {
                "scale": scale,
                "visual_error": visual_conditions[name]["overall"],
                "visual_separation_from_1x": separation,
            }
        )
    magnitude_spearman = scipy_stats.spearmanr(distances, separations)
    decision = decide(
        protocol,
        prediction,
        visual_comparisons["shuffle_task"],
        visual_local,
        visual_direction,
        proprio_prediction,
        proprio_comparisons["shuffle_task"],
        proprio_local,
        proprio_direction,
    )
    training_dir = args.checkpoint.parent
    training_artifacts = {}
    for name in ("config.yaml", "run_metadata.json", "metrics.jsonl", "best_validation.json"):
        path = training_dir / name
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
        **arrays,
        **{f"visual_{key}": value for key, value in visual_direction_arrays.items()},
        **{f"proprio_{key}": value for key, value in proprio_direction_arrays.items()},
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
            "preregistered_commit": checkpoint["calibration_freeze"]["preregistered_commit"],
            "training_source_commit": checkpoint["source_commit"],
        },
        "scope": {
            "train_demos": "0-44",
            "validation_demos": "45-49",
            "train_samples": 52381,
            "validation_samples": 5869,
            "tasks": 10,
            "simulator_or_controller_rollouts": 0,
            "success_metric_used": False,
            "future_proprio_is_input": False,
        },
        "model": {
            "parameters": 826444,
            "backbone_parameters": 792000,
            "auxiliary_head_parameters": 34444,
            "lambda_p": float(checkpoint["lambda_p"]),
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "calibration_freeze": checkpoint["calibration_freeze"],
            "calibration_freeze_sha256": checkpoint["calibration_freeze_sha256"],
        },
        "training_artifacts": training_artifacts,
        "visual_prediction": prediction,
        "proprio_prediction": proprio_prediction,
        "visual_global_action_diagnostics": {
            "condition_error": visual_conditions,
            "comparisons": visual_comparisons,
            "magnitude": magnitude,
            "magnitude_distance_spearman": float(magnitude_spearman.statistic),
            "magnitude_distance_spearman_pvalue": float(magnitude_spearman.pvalue),
        },
        "proprio_action_diagnostics": {
            "condition_error": proprio_conditions,
            "comparisons": proprio_comparisons,
        },
        "visual_local_candidate_sensitivity": visual_local,
        "proprio_local_candidate_sensitivity": proprio_local,
        "visual_local_directional_validity": visual_direction,
        "proprio_local_directional_validity": proprio_direction,
        "decision": decision,
        "sample_artifact": {
            "path": str(sidecar_path.relative_to(ROOT)),
            "sha256": sha256(sidecar_path),
            "arrays": {name: list(value.shape) for name, value in sidecar_arrays.items()},
        },
        "nan_inf_audit": "pass",
        "seed23_seed47_trained": False,
        "forbidden_activity_audit": {
            "c0_v2_modified": False,
            "candidate_generator_or_scale_modified": False,
            "reranker_modified": False,
            "clip_or_camera_modified": False,
            "d13_contract_or_split_modified": False,
            "future_proprio_used_as_input_or_score": False,
            "simulator_or_controller_run": False,
            "init_state_outcome_read_or_run": False,
            "d20_v2_or_c2_v2_started": False,
            "frozen_checkpoint_overwritten": False,
            "success_rate_used_for_selection": False,
            "fake_counterfactual_supervision_used": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("WM-v2B result contains NaN/Inf")
    atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "visual_error": prediction["correct_action_error"],
                "proprio_error": proprio_prediction["correct_action_error"],
                "visual_local_ratio": visual_local["separation_to_error_ratio"],
                "proprio_local_ratio": proprio_local[
                    "separation_to_correct_prediction_rmse_ratio"
                ],
                "visual_direction_accuracy": visual_direction["pairwise_ranking_accuracy"],
                "proprio_direction_accuracy": proprio_direction["pairwise_ranking_accuracy"],
                "seed_expansion_authorized": decision["seed23_seed47_authorized"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
