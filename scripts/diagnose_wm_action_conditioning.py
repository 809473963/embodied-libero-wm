#!/usr/bin/env python3
"""Frozen D15 action-conditioning diagnosis on D13 validation transitions."""

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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_world_model_ensemble import load_ensemble
from src.d13 import WorldModelDataset


HORIZON = 8
CONDITION_NAMES = (
    "true",
    "zero",
    "shuffle_task",
    "shuffle_cross",
    "reverse",
    "temporal_shuffle",
    "repeat_first",
    "scale_0p25",
    "scale_0p5",
    "scale_1p5",
)
SCALE_CONDITIONS = (
    (0.0, "zero"),
    (0.25, "scale_0p25"),
    (0.5, "scale_0p5"),
    (1.0, "true"),
    (1.5, "scale_1p5"),
)
MEMBER_PAIRS = ((0, 1), (0, 2), (1, 2))


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


def distribution(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("distribution requires non-empty finite values")
    quantiles = np.quantile(flat, (0.25, 0.5, 0.75, 0.9, 0.95))
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "median": float(quantiles[1]),
        "p25": float(quantiles[0]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(flat.max()),
    }


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != "c1-v2.2-wm-action-conditioning-v1":
        raise ValueError("unexpected C1-v2.2 protocol")
    data = protocol["data"]
    if data["demo_ids"] != list(range(45, 50)) or data["task_ids"] != list(
        range(10)
    ):
        raise ValueError("diagnosis must use all tasks and validation demos 45-49")
    if data["simulator_rollout"] or data["init_state_indices_used"]:
        raise ValueError("offline diagnosis may not use simulator/init states")
    frozen = protocol["frozen_c1_v2_1"]
    if ["RERANKER_SCORE_BLOCK", "WM_ACTION_SENSITIVITY_BLOCK"] != frozen[
        "blockers"
    ]:
        raise ValueError("C1-v2.1 negative blockers changed")
    if subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() != frozen["commit"]:
        raise ValueError("C1-v2.1 tag does not resolve to the frozen commit")

    checks = (
        (frozen["result"], frozen["result_sha256"]),
        (frozen["report"], frozen["report_sha256"]),
        (data["store"], data["store_sha256"]),
        (data["feature_manifest"], data["feature_manifest_sha256"]),
        (
            protocol["world_model"]["ensemble_config"],
            protocol["world_model"]["ensemble_config_sha256"],
        ),
        (
            protocol["world_model"]["ordinary_validation_result"],
            protocol["world_model"]["ordinary_validation_result_sha256"],
        ),
    )
    for relative, expected in checks:
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")
    ensemble_config = yaml.safe_load(
        (ROOT / protocol["world_model"]["ensemble_config"]).read_text()
    )
    member_paths = [ROOT / path for path in ensemble_config["member_checkpoints"]]
    if [sha256(path) for path in member_paths] != protocol["world_model"][
        "member_checkpoint_sha256"
    ]:
        raise ValueError("frozen ensemble member changed")
    return protocol, sha256(path)


def load_validation(protocol: dict[str, Any], batch_size: int) -> dict[str, np.ndarray]:
    dataset = WorldModelDataset(ROOT / protocol["data"]["store"], "validation", 8)
    if len(dataset) != protocol["data"]["expected_samples"]:
        raise ValueError("unexpected validation sample count")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    names = (
        "z0",
        "proprio0",
        "language",
        "actions",
        "z_future",
        "task_id",
        "demo_id",
        "start_timestep",
    )
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    try:
        for batch in loader:
            for name in names:
                chunks[name].append(batch[name].numpy())
    finally:
        dataset.close()
    result = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    count = protocol["data"]["expected_samples"]
    expected_shapes = {
        "z0": (count, 512),
        "proprio0": (count, 12),
        "language": (count, 512),
        "actions": (count, 8, 7),
        "z_future": (count, 8, 512),
        "task_id": (count,),
        "demo_id": (count,),
        "start_timestep": (count,),
    }
    if {key: value.shape for key, value in result.items()} != expected_shapes:
        raise ValueError("validation shape audit failed")
    if set(result["task_id"].tolist()) != set(range(10)):
        raise ValueError("validation task coverage failed")
    if set(result["demo_id"].tolist()) != set(range(45, 50)):
        raise ValueError("validation demo boundary failed")
    identities = np.stack(
        (result["task_id"], result["demo_id"], result["start_timestep"]), axis=1
    )
    if len(np.unique(identities, axis=0)) != count:
        raise ValueError("validation identities are not unique")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("validation arrays contain NaN/Inf")
    return result


def shuffled_sources(
    task_id: np.ndarray, within_seed: int, cross_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    task_id = np.asarray(task_id, dtype=np.int64)
    count = task_id.size
    within = np.empty(count, dtype=np.int64)
    rng = np.random.default_rng(within_seed)
    for task in range(10):
        indices = np.flatnonzero(task_id == task)
        order = rng.permutation(indices)
        within[order] = np.roll(order, 1)
    if np.any(within == np.arange(count)) or np.any(task_id[within] != task_id):
        raise RuntimeError("within-task whole-chunk derangement failed")

    cross = np.empty(count, dtype=np.int64)
    rng = np.random.default_rng(cross_seed)
    task_indices = {task: np.flatnonzero(task_id == task) for task in range(10)}
    for destination in range(count):
        source_task = (int(task_id[destination]) + int(rng.integers(1, 10))) % 10
        pool = task_indices[source_task]
        cross[destination] = int(pool[int(rng.integers(0, len(pool)))])
    if np.any(task_id[cross] == task_id):
        raise RuntimeError("cross-task shuffle source has the same task")
    return within, cross


def construct_conditions(
    data: dict[str, np.ndarray], protocol: dict[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    true = data["actions"].astype(np.float32, copy=False)
    condition = protocol["action_conditions"]
    within, cross = shuffled_sources(
        data["task_id"],
        int(condition["within_task_shuffle"]["seed"]),
        int(condition["cross_task_shuffle"]["seed"]),
    )
    temporal_permutation = np.asarray(
        condition["temporal"]["fixed_permutation"], dtype=np.int64
    )
    if sorted(temporal_permutation.tolist()) != list(range(8)):
        raise ValueError("temporal permutation is invalid")
    conditions = {
        "true": true,
        "zero": np.zeros_like(true),
        "shuffle_task": true[within].copy(),
        "shuffle_cross": true[cross].copy(),
        "reverse": true[:, ::-1].copy(),
        "temporal_shuffle": true[:, temporal_permutation].copy(),
        "repeat_first": np.repeat(true[:, :1], HORIZON, axis=1),
        "scale_0p25": true * np.float32(0.25),
        "scale_0p5": true * np.float32(0.5),
        "scale_1p5": true * np.float32(1.5),
    }
    if tuple(conditions) != CONDITION_NAMES:
        raise RuntimeError("condition ordering drifted")
    if not all(value.shape == true.shape for value in conditions.values()):
        raise RuntimeError("action condition shape mismatch")
    if not all(np.isfinite(value).all() for value in conditions.values()):
        raise FloatingPointError("action condition contains NaN/Inf")
    return conditions, within, cross


def _cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(first.float(), second.float(), dim=-1)


@torch.inference_mode()
def run_inference(
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    conditions: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ensemble_config = yaml.safe_load(
        (ROOT / protocol["world_model"]["ensemble_config"]).read_text()
    )
    paths = [ROOT / value for value in ensemble_config["member_checkpoints"]]
    ensemble, contract, members = load_ensemble(paths, device)
    if [member["seed"] for member in members] != protocol["world_model"][
        "member_seeds"
    ]:
        raise ValueError("ensemble member order changed")
    if contract["d13_data_contract_version"] != protocol["data"]["contract_version"]:
        raise ValueError("world-model data contract changed")
    if contract["d13_store_sha256"] != protocol["data"]["store_sha256"]:
        raise ValueError("world-model store changed")
    normalization = contract["normalization"]
    p_mean = torch.tensor(
        normalization["proprio"]["mean"], dtype=torch.float32, device=device
    )
    p_std = torch.tensor(
        normalization["proprio"]["std"], dtype=torch.float32, device=device
    )
    a_mean = torch.tensor(
        normalization["action"]["mean"], dtype=torch.float32, device=device
    ).view(1, 1, 7)
    a_std = torch.tensor(
        normalization["action"]["std"], dtype=torch.float32, device=device
    ).view(1, 1, 7)

    count = len(data["task_id"])
    condition_count = len(CONDITION_NAMES)
    member_count = len(members)
    ensemble_error = np.empty((condition_count, count, 8), dtype=np.float32)
    member_error = np.empty((member_count, condition_count, count, 8), dtype=np.float32)
    predicted_displacement = np.empty_like(ensemble_error)
    member_predicted_displacement = np.empty_like(member_error)
    ensemble_separation_from_true = np.empty_like(ensemble_error)
    member_separation_from_true = np.empty_like(member_error)

    scale_pairs = []
    for first in range(len(SCALE_CONDITIONS)):
        for second in range(first + 1, len(SCALE_CONDITIONS)):
            scale_pairs.append((first, second))
    scale_pair_cosine = np.empty((len(scale_pairs), count, 8), dtype=np.float32)
    scale_pair_l2 = np.empty_like(scale_pair_cosine)
    direction_cosine = np.empty((len(MEMBER_PAIRS), count, 8), dtype=np.float32)

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        z0 = torch.from_numpy(data["z0"][start:stop]).to(device)
        proprio = torch.from_numpy(data["proprio0"][start:stop]).to(device)
        language = torch.from_numpy(data["language"][start:stop]).to(device)
        target = torch.from_numpy(data["z_future"][start:stop]).to(device)
        proprio = (proprio - p_mean) / p_std
        ensemble_predictions: list[torch.Tensor] = []
        member_predictions: list[torch.Tensor] = []
        for condition_index, name in enumerate(CONDITION_NAMES):
            raw_action = torch.from_numpy(conditions[name][start:stop]).to(device)
            normalized_action = (raw_action - a_mean) / a_std
            predictions = ensemble(z0, proprio, language, normalized_action)
            mean_prediction = F.normalize(
                predictions.mean(dim=0), p=2, dim=-1, eps=1.0e-8
            )
            ensemble_predictions.append(mean_prediction)
            member_predictions.append(predictions)
            ensemble_error[condition_index, start:stop] = (
                _cosine_distance(mean_prediction, target).cpu().numpy()
            )
            member_error[:, condition_index, start:stop] = (
                _cosine_distance(predictions, target.unsqueeze(0)).cpu().numpy()
            )
            predicted_displacement[condition_index, start:stop] = (
                _cosine_distance(mean_prediction, z0.unsqueeze(1))
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )
            member_predicted_displacement[:, condition_index, start:stop] = (
                _cosine_distance(predictions, z0.unsqueeze(0).unsqueeze(2))
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )

        true_prediction = ensemble_predictions[0]
        true_member_prediction = member_predictions[0]
        for condition_index in range(condition_count):
            ensemble_separation_from_true[condition_index, start:stop] = (
                _cosine_distance(ensemble_predictions[condition_index], true_prediction)
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )
            member_separation_from_true[:, condition_index, start:stop] = (
                _cosine_distance(
                    member_predictions[condition_index], true_member_prediction
                )
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )

        scale_predictions = [
            ensemble_predictions[CONDITION_NAMES.index(name)]
            for _, name in SCALE_CONDITIONS
        ]
        for pair_index, (first, second) in enumerate(scale_pairs):
            scale_pair_cosine[pair_index, start:stop] = (
                _cosine_distance(scale_predictions[first], scale_predictions[second])
                .clamp_min(0.0)
                .cpu()
                .numpy()
            )
            scale_pair_l2[pair_index, start:stop] = torch.linalg.vector_norm(
                scale_predictions[first] - scale_predictions[second], dim=-1
            ).cpu().numpy()

        action_effect = member_predictions[
            CONDITION_NAMES.index("shuffle_task")
        ] - true_member_prediction
        for pair_index, (first, second) in enumerate(MEMBER_PAIRS):
            direction_cosine[pair_index, start:stop] = F.cosine_similarity(
                action_effect[first].float(), action_effect[second].float(), dim=-1
            ).cpu().numpy()

    arrays = {
        "ensemble_error": ensemble_error,
        "member_error": member_error,
        "predicted_displacement": predicted_displacement,
        "member_predicted_displacement": member_predicted_displacement,
        "ensemble_separation_from_true": ensemble_separation_from_true,
        "member_separation_from_true": member_separation_from_true,
        "scale_pair_cosine": scale_pair_cosine,
        "scale_pair_l2": scale_pair_l2,
        "member_effect_direction_cosine": direction_cosine,
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("world-model inference contains NaN/Inf")
    metadata = {
        "normalization": normalization,
        "member_records": members,
        "scale_pairs": [
            f"{SCALE_CONDITIONS[first][0]:g}x-{SCALE_CONDITIONS[second][0]:g}x"
            for first, second in scale_pairs
        ],
    }
    return arrays, metadata


def error_summary(values: np.ndarray, task_id: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "overall": float(values.mean()),
        "per_horizon": values.mean(axis=0).tolist(),
        "per_task": {
            str(task): {
                "sample_count": int(np.sum(task_id == task)),
                "overall": float(values[task_id == task].mean()),
                "per_horizon": values[task_id == task].mean(axis=0).tolist(),
            }
            for task in range(10)
        },
    }


class BootstrapPlan:
    def __init__(
        self,
        task_id: np.ndarray,
        demo_id: np.ndarray,
        replicates: int,
        seed: int,
    ) -> None:
        self.task_id = np.asarray(task_id)
        self.demo_id = np.asarray(demo_id)
        self.replicates = replicates
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, 5, size=(replicates, 10, 5))

    @staticmethod
    def _interval(draws: np.ndarray) -> tuple[float, float]:
        lower, upper = np.quantile(draws, (0.025, 0.975))
        return float(lower), float(upper)

    def summarize(self, values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError("bootstrap values must be [N,8]")
        sums = np.zeros((10, 5, 8), dtype=np.float64)
        counts = np.zeros((10, 5), dtype=np.float64)
        for task in range(10):
            for demo_offset, demo in enumerate(range(45, 50)):
                selected = (self.task_id == task) & (self.demo_id == demo)
                sums[task, demo_offset] = values[selected].sum(axis=0)
                counts[task, demo_offset] = selected.sum()
                if counts[task, demo_offset] == 0:
                    raise ValueError("empty validation episode in bootstrap")

        total_sum = np.zeros((self.replicates, 8), dtype=np.float64)
        total_count = np.zeros(self.replicates, dtype=np.float64)
        per_task_records: dict[str, Any] = {}
        for task in range(10):
            draw = self.draws[:, task]
            selected_sum = sums[task][draw].sum(axis=1)
            selected_count = counts[task][draw].sum(axis=1)
            total_sum += selected_sum
            total_count += selected_count
            task_horizon = selected_sum / selected_count[:, None]
            task_overall = task_horizon.mean(axis=1)
            lower, upper = self._interval(task_overall)
            per_task_records[str(task)] = {
                "point": float(values[self.task_id == task].mean()),
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        horizon_draws = total_sum / total_count[:, None]
        overall_draws = horizon_draws.mean(axis=1)
        lower, upper = self._interval(overall_draws)
        per_horizon = []
        point_horizon = values.mean(axis=0)
        for horizon in range(8):
            h_lower, h_upper = self._interval(horizon_draws[:, horizon])
            per_horizon.append(
                {
                    "horizon": horizon + 1,
                    "point": float(point_horizon[horizon]),
                    "ci95_lower": h_lower,
                    "ci95_upper": h_upper,
                }
            )
        return {
            "point": float(values.mean()),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "per_horizon": per_horizon,
            "per_task": per_task_records,
            "method": "task-stratified episode-block percentile bootstrap",
            "replicates": self.replicates,
            "seed": self.seed,
        }


def comparison_summary(
    true_error: np.ndarray,
    wrong_error: np.ndarray,
    task_id: np.ndarray,
    bootstrap: BootstrapPlan,
) -> dict[str, Any]:
    delta = np.asarray(wrong_error - true_error, dtype=np.float64)
    wins = np.asarray(true_error < wrong_error, dtype=np.float64)
    ties = np.asarray(true_error == wrong_error, dtype=np.float64)
    delta_bootstrap = bootstrap.summarize(delta)
    accuracy_bootstrap = bootstrap.summarize(wins)
    true_mean = float(np.mean(true_error))
    per_task = {}
    for task in range(10):
        selected = task_id == task
        task_true = float(np.mean(true_error[selected]))
        task_delta = float(np.mean(delta[selected]))
        per_task[str(task)] = {
            "sample_count": int(selected.sum()),
            "true_error": task_true,
            "wrong_error": float(np.mean(wrong_error[selected])),
            "delta": task_delta,
            "relative_delta": task_delta / task_true,
            "true_action_win_accuracy": float(np.mean(wins[selected])),
            "tie_fraction": float(np.mean(ties[selected])),
            "accuracy_ci95": [
                accuracy_bootstrap["per_task"][str(task)]["ci95_lower"],
                accuracy_bootstrap["per_task"][str(task)]["ci95_upper"],
            ],
            "delta_ci95": [
                delta_bootstrap["per_task"][str(task)]["ci95_lower"],
                delta_bootstrap["per_task"][str(task)]["ci95_upper"],
            ],
        }
    return {
        "true_error": true_mean,
        "wrong_error": float(np.mean(wrong_error)),
        "delta": float(delta.mean()),
        "relative_delta": float(delta.mean()) / true_mean,
        "true_action_win_accuracy": float(wins.mean()),
        "random_reference": 0.5,
        "tie_fraction": float(ties.mean()),
        "delta_bootstrap": delta_bootstrap,
        "accuracy_bootstrap": accuracy_bootstrap,
        "per_horizon_delta": delta.mean(axis=0).tolist(),
        "per_horizon_true_action_win_accuracy": wins.mean(axis=0).tolist(),
        "per_task": per_task,
    }


def effect_ratios(
    separation: np.ndarray,
    real_displacement: np.ndarray,
    ordinary_error: np.ndarray,
) -> dict[str, Any]:
    separation_h = separation.mean(axis=0)
    real_h = real_displacement.mean(axis=0)
    return {
        "separation_overall": float(separation_h.mean()),
        "separation_per_horizon": separation_h.tolist(),
        "to_ordinary_wm_error_overall": float(
            separation_h.mean() / ordinary_error.mean()
        ),
        "to_ordinary_wm_error_per_horizon": (separation_h / ordinary_error).tolist(),
        "to_real_latent_displacement_overall": float(
            separation_h.mean() / real_h.mean()
        ),
        "to_real_latent_displacement_per_horizon": (separation_h / real_h).tolist(),
    }


def representation_summary(
    real_displacement: np.ndarray, task_id: np.ndarray, ordinary_error: np.ndarray
) -> dict[str, Any]:
    return {
        "overall": distribution(real_displacement),
        "per_horizon": [
            {"horizon": horizon + 1, **distribution(real_displacement[:, horizon])}
            for horizon in range(8)
        ],
        "per_task": {
            str(task): {
                "sample_count": int(np.sum(task_id == task)),
                "overall": distribution(real_displacement[task_id == task]),
                "per_horizon_mean": real_displacement[task_id == task]
                .mean(axis=0)
                .tolist(),
            }
            for task in range(10)
        },
        "ordinary_wm_error_overall": float(ordinary_error.mean()),
        "ordinary_wm_error_per_horizon": ordinary_error.tolist(),
        "real_displacement_to_ordinary_error_ratio": float(
            real_displacement.mean() / ordinary_error.mean()
        ),
    }


def magnitude_summary(
    protocol: dict[str, Any],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    condition_errors: dict[str, Any],
    real_displacement: np.ndarray,
    ordinary_error: np.ndarray,
    raw_actions: np.ndarray,
) -> dict[str, Any]:
    scale_records = []
    separations = arrays["ensemble_separation_from_true"]
    displacement = arrays["predicted_displacement"]
    scale_values = []
    separation_means = []
    displacement_means = []
    for scale, name in SCALE_CONDITIONS:
        index = CONDITION_NAMES.index(name)
        scale_values.append(scale)
        separation_means.append(float(separations[index].mean()))
        displacement_means.append(float(displacement[index].mean()))
        scaled = raw_actions * np.float32(scale)
        scale_records.append(
            {
                "scale": scale,
                "condition": name,
                "prediction_error": condition_errors[name],
                "predicted_trajectory_displacement": {
                    "overall": float(displacement[index].mean()),
                    "per_horizon": displacement[index].mean(axis=0).tolist(),
                },
                "separation_from_1x": effect_ratios(
                    separations[index], real_displacement, ordinary_error
                ),
                "raw_action_range": [float(scaled.min()), float(scaled.max())],
                "raw_action_abs_gt_1_fraction": float(np.mean(np.abs(scaled) > 1.0)),
            }
        )
    abs_distance = np.abs(np.asarray(scale_values) - 1.0)
    distance_spearman = scipy_stats.spearmanr(abs_distance, separation_means)
    displacement_spearman = scipy_stats.spearmanr(scale_values, displacement_means)
    per_horizon_distance_spearman = []
    per_horizon_displacement_spearman = []
    scale_indices = [CONDITION_NAMES.index(name) for _, name in SCALE_CONDITIONS]
    for horizon in range(8):
        sep_values = [float(separations[index, :, horizon].mean()) for index in scale_indices]
        disp_values = [float(displacement[index, :, horizon].mean()) for index in scale_indices]
        per_horizon_distance_spearman.append(
            float(scipy_stats.spearmanr(abs_distance, sep_values).statistic)
        )
        per_horizon_displacement_spearman.append(
            float(scipy_stats.spearmanr(scale_values, disp_values).statistic)
        )
    pair_records = []
    for pair_index, pair_name in enumerate(metadata["scale_pairs"]):
        pair_records.append(
            {
                "pair": pair_name,
                "cosine_separation_overall": float(
                    arrays["scale_pair_cosine"][pair_index].mean()
                ),
                "cosine_separation_per_horizon": arrays["scale_pair_cosine"][
                    pair_index
                ]
                .mean(axis=0)
                .tolist(),
                "l2_separation_overall": float(
                    arrays["scale_pair_l2"][pair_index].mean()
                ),
                "l2_separation_per_horizon": arrays["scale_pair_l2"][pair_index]
                .mean(axis=0)
                .tolist(),
            }
        )
    return {
        "raw_physical_space": True,
        "clipped": False,
        "scales": scale_records,
        "pairwise_prediction_separation": pair_records,
        "distance_from_1x_monotonicity": {
            "spearman": float(distance_spearman.statistic),
            "pvalue": float(distance_spearman.pvalue),
            "per_horizon_spearman": per_horizon_distance_spearman,
        },
        "predicted_displacement_vs_scale_monotonicity": {
            "spearman": float(displacement_spearman.statistic),
            "pvalue": float(displacement_spearman.pvalue),
            "per_horizon_spearman": per_horizon_displacement_spearman,
        },
    }


def member_summary(
    protocol: dict[str, Any],
    arrays: dict[str, np.ndarray],
    task_id: np.ndarray,
) -> dict[str, Any]:
    true_index = CONDITION_NAMES.index("true")
    shuffle_index = CONDITION_NAMES.index("shuffle_task")
    records = []
    for member_index, seed in enumerate(protocol["world_model"]["member_seeds"]):
        true_error = arrays["member_error"][member_index, true_index]
        shuffle_error = arrays["member_error"][member_index, shuffle_index]
        delta = shuffle_error - true_error
        wins = true_error < shuffle_error
        separation = arrays["member_separation_from_true"][
            member_index, shuffle_index
        ]
        records.append(
            {
                "member_index": member_index,
                "seed": seed,
                "true_error_overall": float(true_error.mean()),
                "shuffle_task_error_overall": float(shuffle_error.mean()),
                "delta_shuffle_overall": float(delta.mean()),
                "delta_shuffle_per_horizon": delta.mean(axis=0).tolist(),
                "true_action_ranking_accuracy_overall": float(wins.mean()),
                "true_action_ranking_accuracy_per_horizon": wins.mean(axis=0).tolist(),
                "action_induced_separation_overall": float(separation.mean()),
                "action_induced_separation_per_horizon": separation.mean(axis=0).tolist(),
                "per_task": {
                    str(task): {
                        "delta_shuffle": float(delta[task_id == task].mean()),
                        "true_action_ranking_accuracy": float(
                            wins[task_id == task].mean()
                        ),
                        "action_induced_separation": float(
                            separation[task_id == task].mean()
                        ),
                    }
                    for task in range(10)
                },
            }
        )
    ensemble_sep = arrays["ensemble_separation_from_true"][shuffle_index]
    mean_member_sep = arrays["member_separation_from_true"][:, shuffle_index].mean(
        axis=0
    )
    directions = []
    for pair_index, pair in enumerate(MEMBER_PAIRS):
        values = arrays["member_effect_direction_cosine"][pair_index]
        directions.append(
            {
                "pair": f"member{pair[0]}-member{pair[1]}",
                "overall_mean_cosine": float(values.mean()),
                "per_horizon_mean_cosine": values.mean(axis=0).tolist(),
                "negative_fraction": float(np.mean(values < 0.0)),
            }
        )
    return {
        "members": records,
        "ensemble_mean_action_separation": float(ensemble_sep.mean()),
        "mean_individual_member_action_separation": float(mean_member_sep.mean()),
        "ensemble_to_mean_member_separation_ratio": float(
            ensemble_sep.mean() / mean_member_sep.mean()
        ),
        "member_action_effect_direction": directions,
    }


def decide(
    protocol: dict[str, Any],
    comparisons: dict[str, Any],
    magnitude: dict[str, Any],
    representation: dict[str, Any],
) -> dict[str, Any]:
    rules = protocol["interpretation_rules"]
    thresholds = rules["action_signal_checks"]
    within = comparisons["shuffle_task"]
    temporal = comparisons["temporal_shuffle"]
    zero_scale = next(item for item in magnitude["scales"] if item["scale"] == 0.0)
    checks = {
        "correct_vs_within_error_advantage": (
            within["relative_delta"]
            >= thresholds[
                "correct_vs_within_shuffle_relative_error_increase_min"
            ]
            and within["delta_bootstrap"]["ci95_lower"] > 0.0
        ),
        "correct_vs_within_ranking_advantage": (
            within["true_action_win_accuracy"]
            >= thresholds["correct_vs_within_shuffle_ranking_accuracy_min"]
            and within["accuracy_bootstrap"]["ci95_lower"] > 0.5
        ),
        "temporal_order_effect": (
            temporal["relative_delta"]
            >= thresholds["temporal_order_relative_error_increase_min"]
            and temporal["delta_bootstrap"]["ci95_lower"] > 0.0
        ),
        "magnitude_response": (
            zero_scale["separation_from_1x"]["to_ordinary_wm_error_overall"]
            >= thresholds["magnitude_separation_to_wm_error_ratio_min"]
            and magnitude["distance_from_1x_monotonicity"]["spearman"]
            >= thresholds["magnitude_distance_response_spearman_min"]
        ),
    }
    action_signal = sum(checks.values()) >= 3 and (
        checks["correct_vs_within_error_advantage"]
        or checks["correct_vs_within_ranking_advantage"]
    )

    task_rule = rules["task_dependent_action_signal"]
    task_accuracy = np.asarray(
        [within["per_task"][str(task)]["true_action_win_accuracy"] for task in range(10)]
    )
    task_relative_delta = np.asarray(
        [within["per_task"][str(task)]["relative_delta"] for task in range(10)]
    )
    spread_rule = (
        float(task_accuracy.max() - task_accuracy.min())
        >= task_rule["within_shuffle_task_accuracy_range_min"]
        and int(np.sum(task_accuracy >= 0.55))
        >= task_rule["tasks_with_accuracy_at_least_0_55_min"]
        and int(np.sum(task_accuracy <= 0.50))
        >= task_rule["tasks_with_accuracy_at_most_0_50_min"]
    )
    sign_rule = int(np.sum(task_relative_delta >= 0.005)) >= 2 and int(
        np.sum(task_relative_delta <= 0.0)
    ) >= 2
    task_dependent = spread_rule or sign_rule

    clip_rule = rules["clip_state_sensitivity_limit"]
    h8_median = representation["per_horizon"][7]["median"]
    clip_limit = (
        representation["real_displacement_to_ordinary_error_ratio"]
        <= clip_rule["real_displacement_to_ordinary_wm_error_ratio_max"]
        and h8_median <= clip_rule["h8_real_displacement_median_max"]
    )
    local_ratio = protocol["frozen_c1_v2_1"][
        "production_candidate_separation_to_wm_error"
    ]
    low_local = action_signal and local_ratio < 0.01

    verdicts = []
    if not action_signal and not task_dependent:
        verdicts.append("WM_ACTION_IGNORING")
    if low_local:
        verdicts.append("WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY")
    if clip_limit:
        verdicts.append("CLIP_STATE_SENSITIVITY_LIMIT")
    if task_dependent:
        verdicts.append("TASK_DEPENDENT_ACTION_SIGNAL")
    no_block = (
        action_signal and local_ratio >= 0.01 and not clip_limit and not task_dependent
    )
    if no_block:
        verdicts.append("NO_ACTION_CONDITIONING_BLOCK")
    if not verdicts:
        raise RuntimeError("interpretation rules produced no verdict")
    priorities = rules["priority_mapping"]
    return {
        "verdicts": verdicts,
        "grouped_action_signal_checks": checks,
        "grouped_action_signal_pass_count": int(sum(checks.values())),
        "action_signal_pass": action_signal,
        "task_dependent_checks": {
            "accuracy_range": float(task_accuracy.max() - task_accuracy.min()),
            "tasks_accuracy_at_least_0_55": int(np.sum(task_accuracy >= 0.55)),
            "tasks_accuracy_at_most_0_50": int(np.sum(task_accuracy <= 0.50)),
            "tasks_relative_delta_at_least_0_005": int(
                np.sum(task_relative_delta >= 0.005)
            ),
            "tasks_nonpositive_relative_delta": int(np.sum(task_relative_delta <= 0.0)),
            "spread_rule_pass": spread_rule,
            "sign_rule_pass": sign_rule,
        },
        "task_dependent_action_signal": task_dependent,
        "clip_state_sensitivity_checks": {
            "real_displacement_to_ordinary_error_ratio": representation[
                "real_displacement_to_ordinary_error_ratio"
            ],
            "h8_real_displacement_median": h8_median,
            "pass": clip_limit,
        },
        "frozen_production_candidate_separation_to_wm_error": local_ratio,
        "low_local_sensitivity": low_local,
        "no_action_conditioning_block": no_block,
        "priority": [priorities[verdict] for verdict in verdicts],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/c1_v2_2_wm_action_conditioning.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    output_path = ROOT / protocol["outputs"]["result"]
    sample_path = ROOT / protocol["outputs"]["sample_sidecar"]
    if output_path.exists() or sample_path.exists():
        raise FileExistsError("C1-v2.2 output already exists; frozen artifacts are immutable")

    data = load_validation(protocol, max(args.batch_size, 256))
    conditions, within_source, cross_source = construct_conditions(data, protocol)
    arrays, metadata = run_inference(
        protocol, data, conditions, torch.device(args.device), args.batch_size
    )
    ordinary_result = json.loads(
        (ROOT / protocol["world_model"]["ordinary_validation_result"]).read_text()
    )
    ordinary_error = np.asarray(
        ordinary_result["learned"]["per_horizon_cosine_error"], dtype=np.float64
    )
    true_index = CONDITION_NAMES.index("true")
    measured_true = arrays["ensemble_error"][true_index].mean(axis=0)
    if not np.allclose(measured_true, ordinary_error, rtol=0.0, atol=2.0e-7):
        raise ValueError("correct-action prediction does not reproduce frozen D16")

    real_displacement = (
        1.0
        - np.sum(data["z0"][:, None, :] * data["z_future"], axis=-1)
        / (
            np.linalg.norm(data["z0"], axis=-1)[:, None]
            * np.linalg.norm(data["z_future"], axis=-1)
        )
    ).astype(np.float32)
    real_displacement = np.maximum(real_displacement, 0.0)
    condition_errors = {
        name: error_summary(arrays["ensemble_error"][index], data["task_id"])
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
            arrays["ensemble_error"][true_index],
            arrays["ensemble_error"][CONDITION_NAMES.index(name)],
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
    representation = representation_summary(
        real_displacement, data["task_id"], ordinary_error
    )
    magnitude = magnitude_summary(
        protocol,
        arrays,
        metadata,
        condition_errors,
        real_displacement,
        ordinary_error,
        data["actions"],
    )
    action_effect = {
        name: effect_ratios(
            arrays["ensemble_separation_from_true"][CONDITION_NAMES.index(name)],
            real_displacement,
            ordinary_error,
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
    frozen_candidate = json.loads(
        (ROOT / protocol["frozen_c1_v2_1"]["result"]).read_text()
    )["production_scale_1x"]["world_model_action_sensitivity"]
    candidate_sep = np.asarray(
        frozen_candidate[
            "ensemble_mean_pairwise_cosine_separation_per_horizon"
        ],
        dtype=np.float64,
    )
    candidate_effect = {
        "separation_per_horizon": candidate_sep.tolist(),
        "to_ordinary_wm_error_per_horizon": (candidate_sep / ordinary_error).tolist(),
        "to_ordinary_wm_error_overall": float(
            candidate_sep.mean() / ordinary_error.mean()
        ),
        "to_real_latent_displacement_per_horizon": (
            candidate_sep / real_displacement.mean(axis=0)
        ).tolist(),
        "to_real_latent_displacement_overall": float(
            candidate_sep.mean() / real_displacement.mean()
        ),
    }
    members = member_summary(protocol, arrays, data["task_id"])
    decision = decide(protocol, comparisons, magnitude, representation)

    task_core = {}
    within = comparisons["shuffle_task"]
    for task in range(10):
        selected = data["task_id"] == task
        task_core[str(task)] = {
            "sample_count": int(selected.sum()),
            "true_error": float(
                arrays["ensemble_error"][true_index, selected].mean()
            ),
            "shuffle_task_error": within["per_task"][str(task)]["wrong_error"],
            "delta_shuffle": within["per_task"][str(task)]["delta"],
            "relative_delta_shuffle": within["per_task"][str(task)][
                "relative_delta"
            ],
            "true_action_ranking_accuracy": within["per_task"][str(task)][
                "true_action_win_accuracy"
            ],
            "ranking_accuracy_ci95": within["per_task"][str(task)][
                "accuracy_ci95"
            ],
            "real_latent_temporal_displacement": float(
                real_displacement[selected].mean()
            ),
            "real_latent_displacement_per_horizon": real_displacement[selected]
            .mean(axis=0)
            .tolist(),
        }

    sidecar_arrays = {
        "condition_names": np.asarray(CONDITION_NAMES),
        "task_id": data["task_id"].astype(np.int8),
        "demo_id": data["demo_id"].astype(np.int8),
        "start_timestep": data["start_timestep"].astype(np.int32),
        "within_task_source_index": within_source.astype(np.int32),
        "cross_task_source_index": cross_source.astype(np.int32),
        "ensemble_error": arrays["ensemble_error"],
        "member_error": arrays["member_error"],
        "real_displacement": real_displacement,
        "predicted_displacement": arrays["predicted_displacement"],
        "ensemble_separation_from_true": arrays[
            "ensemble_separation_from_true"
        ],
        "member_separation_from_true": arrays["member_separation_from_true"],
        "scale_pair_names": np.asarray(metadata["scale_pairs"]),
        "scale_pair_cosine": arrays["scale_pair_cosine"],
        "scale_pair_l2": arrays["scale_pair_l2"],
        "member_effect_direction_cosine": arrays[
            "member_effect_direction_cosine"
        ],
    }
    atomic_npz(sample_path, **sidecar_arrays)
    sample_record = {
        "path": str(sample_path.relative_to(ROOT)),
        "sha256": sha256(sample_path),
        "arrays": {key: list(value.shape) for key, value in sidecar_arrays.items()},
    }
    result = {
        "schema_version": 1,
        "diagnosis_version": protocol["protocol_version"],
        "status": "PASS_OFFLINE_DIAGNOSIS_COMPLETE",
        "verdicts": decision["verdicts"],
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": protocol_hash,
            "preregistered_commit": source_commit(),
        },
        "scope": {
            "split": "D13 validation demos 45-49",
            "sample_count": len(data["task_id"]),
            "episode_count": len(
                np.unique(np.stack((data["task_id"], data["demo_id"]), axis=1), axis=0)
            ),
            "task_count": len(np.unique(data["task_id"])),
            "simulator_rollout_count": 0,
            "init_state_indices_read_or_run": [],
        },
        "action_construction": {
            "conditions": protocol["action_conditions"],
            "within_task_source_identity_all_different": bool(
                np.all(within_source != np.arange(len(within_source)))
            ),
            "within_task_source_task_all_equal": bool(
                np.all(data["task_id"][within_source] == data["task_id"])
            ),
            "cross_task_source_task_all_different": bool(
                np.all(data["task_id"][cross_source] != data["task_id"])
            ),
            "physical_zero_wm_normalized_value": (
                -np.asarray(metadata["normalization"]["action"]["mean"])
                / np.asarray(metadata["normalization"]["action"]["std"])
            ).tolist(),
            "normalized_zero_was_not_used_as_physical_zero": True,
        },
        "prediction_error": condition_errors,
        "true_action_comparisons": comparisons,
        "action_effect_ratios": action_effect,
        "frozen_production_candidate_effect_ratios": candidate_effect,
        "magnitude_ablation": magnitude,
        "representation_sensitivity": representation,
        "ensemble_member_consistency": members,
        "per_task_core": task_core,
        "optional_gradient_analysis": protocol["optional_gradient_analysis"],
        "decision": decision,
        "expert_action_caveat": "expert action is an imitation-oriented diagnostic target; nearest expert action does not imply optimal closed-loop task success",
        "sources": {
            "frozen_c1_v2_1_commit": protocol["frozen_c1_v2_1"]["commit"],
            "frozen_c1_v2_1_tag": protocol["frozen_c1_v2_1"]["annotated_tag"],
            "d13_store_sha256": protocol["data"]["store_sha256"],
            "ensemble_config_sha256": protocol["world_model"][
                "ensemble_config_sha256"
            ],
            "member_checkpoint_sha256": protocol["world_model"][
                "member_checkpoint_sha256"
            ],
            "ordinary_validation_result_sha256": protocol["world_model"][
                "ordinary_validation_result_sha256"
            ],
        },
        "sample_artifact": sample_record,
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "world_model_trained_or_modified": False,
            "world_model_loss_or_architecture_modified": False,
            "candidate_generator_modified": False,
            "reranker_weights_modified": False,
            "controller_or_simulator_rollout_run": False,
            "init_state_0_49_read_or_run": False,
            "d20_v2_started": False,
            "c2_v2_started": False,
            "controller_success_used": False,
            "old_negative_artifact_overwritten": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "verdicts": result["verdicts"],
                "samples": result["scope"]["sample_count"],
                "true_error": condition_errors["true"]["overall"],
                "shuffle_delta": comparisons["shuffle_task"]["delta"],
                "shuffle_ranking_accuracy": comparisons["shuffle_task"][
                    "true_action_win_accuracy"
                ],
                "real_displacement": representation["overall"]["mean"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
