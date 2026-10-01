"""Offline candidate-generation and oracle planning-architecture audit helpers.

This module never evaluates reward, success, or environment outcomes.  The only
objective is frozen expert-action MSE, used as an unavailable-at-deployment
imitation oracle to measure search headroom.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

from src.c0 import C0Actor


HORIZON = 8
ACTION_DIM = 7
TRANSLATION = slice(0, 3)
ROTATION = slice(3, 6)
GRIPPER = slice(6, 7)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OfflineContext:
    image: np.ndarray
    language: np.ndarray
    proprio: np.ndarray
    expert: np.ndarray
    actor_default: np.ndarray
    task_id: np.ndarray
    demo_id: np.ndarray
    timestep: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


def load_offline_context(
    store_path: str | Path,
    checkpoint_path: str | Path,
    demo_ids: Iterable[int],
    device: str = "cpu",
    batch_size: int = 512,
) -> OfflineContext:
    """Load only explicitly admitted demo groups and run the frozen C0 actor."""

    admitted = tuple(int(value) for value in demo_ids)
    if not admitted or any(value < 31 or value > 39 or value == 35 for value in admitted):
        raise ValueError(f"inadmissible demo ids: {admitted}")
    if set(admitted) not in ({31, 32, 33, 34}, {36, 37, 38, 39}):
        raise ValueError("only the frozen calibration or evaluation partition is allowed")

    image, proprio, expert, task_id, demo_id, timestep = [], [], [], [], [], []
    with h5py.File(store_path, "r") as store:
        if str(store.attrs.get("world_model_data_contract")) != "d13-v1":
            raise ValueError("unexpected feature-store contract")
        if int(store.attrs.get("horizon", -1)) != HORIZON:
            raise ValueError("expected horizon 8")
        language_by_task = store["task_metadata/language_features"][:].astype(np.float32)
        if language_by_task.shape != (10, 512):
            raise ValueError("unexpected language feature shape")
        for task in range(10):
            for demo in admitted:
                group = store[f"tasks/task_{task}/demos/demo_{demo}"]
                if int(group.attrs["task_id"]) != task or int(group.attrs["demo_id"]) != demo:
                    raise ValueError("HDF5 identity mismatch")
                count = int(group.attrs["observation_count"])
                vision = group["vision"][:].astype(np.float32)
                prop = group["proprio"][:].astype(np.float32)
                action = group["actions"][:].astype(np.float32)
                if vision.shape != (count, 512) or prop.shape != (count, 12) or action.shape != (count, 7):
                    raise ValueError(f"invalid episode shape task{task}/demo{demo}")
                for start in range(count - HORIZON):
                    image.append(vision[start])
                    proprio.append(prop[start])
                    expert.append(action[start + 1 : start + HORIZON + 1])
                    task_id.append(task)
                    demo_id.append(demo)
                    timestep.append(start)

    arrays = {
        "image": np.asarray(image, dtype=np.float32),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "expert": np.asarray(expert, dtype=np.float32),
        "task_id": np.asarray(task_id, dtype=np.int64),
        "demo_id": np.asarray(demo_id, dtype=np.int64),
        "timestep": np.asarray(timestep, dtype=np.int64),
    }
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_contract = {
        "version": "c0-v2-causal-alignment-v1",
        "pairing": "obs[k] -> actions[k+1:k+9]",
        "action_start_offset": 1,
        "horizon": 8,
    }
    if state.get("supervision_contract") != expected_contract:
        raise ValueError("checkpoint is not frozen C0-v2")
    model = C0Actor(
        feature_dim=512,
        proprio_dim=12,
        fusion_dim=int(state["config"]["model"]["fusion_dim"]),
        action_horizon=HORIZON,
        action_dim=ACTION_DIM,
    )
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    normalization = state["normalization"]
    p_mean = np.asarray(normalization["proprio"]["mean"], dtype=np.float32)
    p_std = np.asarray(normalization["proprio"]["std"], dtype=np.float32)
    a_mean = np.asarray(normalization["action"]["mean"], dtype=np.float32)
    a_std = np.asarray(normalization["action"]["std"], dtype=np.float32)
    language = language_by_task[arrays["task_id"]]
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(arrays["image"]), batch_size):
            stop = min(start + batch_size, len(arrays["image"]))
            predictions.append(
                model(
                    torch.from_numpy(arrays["image"][start:stop]).to(device),
                    torch.from_numpy(language[start:stop]).to(device),
                    torch.from_numpy(
                        (arrays["proprio"][start:stop] - p_mean) / p_std
                    ).to(device),
                ).cpu().numpy()
            )
    normalized = np.concatenate(predictions).astype(np.float32)
    default = np.clip(normalized * a_std[None, None] + a_mean[None, None], -1.0, 1.0)
    if not all(np.isfinite(value).all() for value in (*arrays.values(), language, default)):
        raise FloatingPointError("offline context contains NaN/Inf")
    return OfflineContext(
        arrays["image"], language, arrays["proprio"], arrays["expert"], default,
        arrays["task_id"], arrays["demo_id"], arrays["timestep"], a_mean, a_std,
    )


def generate_unit_noise(sample_count: int, directions: int, seed_base: int) -> np.ndarray:
    """Generate deterministic temporally smoothed unit noise.

    Direction zero exactly reproduces the old per-row C1-v2.1 Torch generator
    rule. Later directions use disjoint fixed seed strides.
    """

    result = np.empty((sample_count, directions, HORIZON, ACTION_DIM), dtype=np.float32)
    for row in range(sample_count):
        for direction in range(directions):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed_base + row + direction * 1_000_003))
            raw = torch.randn((HORIZON, ACTION_DIM), generator=generator)
            padded = torch.cat((raw[:1], raw, raw[-1:]), dim=0)
            result[row, direction] = (
                0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
            ).numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("noise bank contains NaN/Inf")
    return result


def antithetic_noise(unit: np.ndarray, alternatives: int) -> np.ndarray:
    direction_count = (alternatives + 1) // 2
    if direction_count > unit.shape[1]:
        raise ValueError("unit-noise bank too small")
    values = []
    for index in range(direction_count):
        values.extend((unit[:, index], -unit[:, index]))
    return np.stack(values[:alternatives], axis=1)


def make_local_candidates(
    default: np.ndarray,
    unit: np.ndarray,
    count: int,
    std: np.ndarray,
    scale: float = 1.0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("candidate count must include at least one alternative")
    noise = antithetic_noise(unit, count - 1)
    multiplier = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, ACTION_DIM)
    if mask is not None:
        multiplier = multiplier * np.asarray(mask, dtype=np.float32).reshape(1, 1, 1, ACTION_DIM)
    alternatives = default[:, None] + noise * multiplier * np.float32(scale)
    preclip = np.concatenate((default[:, None], alternatives), axis=1)
    return preclip.astype(np.float32), np.clip(preclip, -1.0, 1.0).astype(np.float32)


def make_mixed_candidates(
    default: np.ndarray,
    unit: np.ndarray,
    std: np.ndarray,
    count: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    noise = antithetic_noise(unit, count - 1)
    scales = np.resize(np.asarray((0.5, 1.0, 2.0), dtype=np.float32), count - 1)
    alternatives = default[:, None] + noise * np.asarray(std, dtype=np.float32)[None, None, None] * scales[None, :, None, None]
    preclip = np.concatenate((default[:, None], alternatives), axis=1)
    return preclip.astype(np.float32), np.clip(preclip, -1.0, 1.0).astype(np.float32)


def make_proposal_mixture(
    default: np.ndarray,
    unit: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """P5: A0 + 8 full@1x + 8 full@2x + 8 translation@2x + 7 rotation@2x."""

    base = antithetic_noise(unit, 31)
    multipliers = np.zeros((31, ACTION_DIM), dtype=np.float32)
    multipliers[:8] = np.asarray(std) * 1.0
    multipliers[8:16] = np.asarray(std) * 2.0
    multipliers[16:24, :3] = np.asarray(std)[:3] * 2.0
    multipliers[24:, 3:6] = np.asarray(std)[3:6] * 2.0
    alternatives = default[:, None] + base * multipliers[None, :, None]
    preclip = np.concatenate((default[:, None], alternatives), axis=1)
    return preclip.astype(np.float32), np.clip(preclip, -1.0, 1.0).astype(np.float32)


def make_oracle_cem(
    default: np.ndarray,
    expert: np.ndarray,
    initial_std: np.ndarray,
    seed_base: int,
    iterations: int = 4,
    count: int = 32,
    elite_count: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expert-MSE oracle CEM; returned pools are cumulative and diagnostic only."""

    samples = len(default)
    mean = default.copy()
    initial = np.broadcast_to(np.asarray(initial_std, dtype=np.float32), (samples, HORIZON, ACTION_DIM)).copy()
    spread = initial.copy()
    cumulative_pre, cumulative = [], []
    output = []
    for iteration in range(iterations):
        unit = generate_unit_noise(samples, count - 1, seed_base + 50_000_017 * iteration)
        noise = unit[:, : count - 1]
        alternatives = mean[:, None] + noise * spread[:, None]
        preclip = np.concatenate((mean[:, None], alternatives), axis=1).astype(np.float32)
        candidates = np.clip(preclip, -1.0, 1.0).astype(np.float32)
        errors = np.square(candidates.astype(np.float64) - expert[:, None]).mean(axis=(2, 3))
        elites = np.argsort(errors, axis=1, kind="stable")[:, :elite_count]
        selected = np.take_along_axis(candidates, elites[:, :, None, None], axis=1)
        mean = selected.mean(axis=1).astype(np.float32)
        candidate_std = selected.std(axis=1).astype(np.float32)
        floor = initial * np.float32(0.1)
        ceiling = initial * np.float32(2.0)
        spread = np.minimum(np.maximum(candidate_std, floor), ceiling)
        cumulative_pre.append(preclip)
        cumulative.append(candidates)
        output.append((np.concatenate(cumulative_pre, axis=1), np.concatenate(cumulative, axis=1)))
    return output


def _relative_gain(default: np.ndarray, oracle: np.ndarray) -> float:
    base = float(np.mean(default))
    return float((base - np.mean(oracle)) / base)


def block_bootstrap_draws(replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(replicates, 10, 4), endpoint=False)


def bootstrap_gains(
    default: np.ndarray,
    oracle: np.ndarray,
    task_id: np.ndarray,
    demo_id: np.ndarray,
    demo_values: tuple[int, ...],
    draws: np.ndarray,
) -> np.ndarray:
    sums_default = np.empty((10, 4), dtype=np.float64)
    sums_oracle = np.empty((10, 4), dtype=np.float64)
    counts = np.empty((10, 4), dtype=np.float64)
    for task in range(10):
        for slot, demo in enumerate(demo_values):
            selected = (task_id == task) & (demo_id == demo)
            sums_default[task, slot] = default[selected].sum()
            sums_oracle[task, slot] = oracle[selected].sum()
            counts[task, slot] = selected.sum()
    rows = np.arange(10)[None, :, None]
    selected_default = sums_default[rows, draws].sum(axis=(1, 2))
    selected_oracle = sums_oracle[rows, draws].sum(axis=(1, 2))
    selected_count = counts[rows, draws].sum(axis=(1, 2))
    default_mean = selected_default / selected_count
    oracle_mean = selected_oracle / selected_count
    return (default_mean - oracle_mean) / default_mean


def _effective_rank(candidates: np.ndarray, default: np.ndarray) -> tuple[float, list[float]]:
    delta = (candidates - default[:, None]).reshape(-1, HORIZON * ACTION_DIM).astype(np.float64)
    covariance = np.cov(delta, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0, [0.0] * 10
    probability = eigenvalues / total
    positive = probability[probability > 0]
    effective = float(np.exp(-np.sum(positive * np.log(positive))))
    return effective, [float(value) for value in probability[:10]]


def summarize_family(
    name: str,
    preclip: np.ndarray,
    candidates: np.ndarray,
    context: OfflineContext,
    demo_values: tuple[int, ...],
    draws: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if candidates.shape != preclip.shape or candidates.shape[0] != len(context.expert):
        raise ValueError(f"invalid candidate shape for {name}")
    if not np.isfinite(preclip).all() or not np.isfinite(candidates).all():
        raise FloatingPointError(f"nonfinite candidates in {name}")
    delta = candidates.astype(np.float64) - context.expert[:, None].astype(np.float64)
    errors = np.square(delta).mean(axis=(2, 3))
    default_error = errors[:, 0]
    winner = np.argmin(errors, axis=1)
    oracle_error = errors[np.arange(len(errors)), winner]
    selected = candidates[np.arange(len(candidates)), winner]
    gains = np.divide(default_error - oracle_error, default_error, out=np.zeros_like(default_error), where=default_error > 0)
    if np.any(oracle_error > default_error + 1.0e-15):
        raise AssertionError("oracle worse than default")
    boot = bootstrap_gains(default_error, oracle_error, context.task_id, context.demo_id, demo_values, draws)

    per_task = {}
    for task in range(10):
        mask = context.task_id == task
        per_task[str(task)] = _relative_gain(default_error[mask], oracle_error[mask])
    per_horizon = {}
    squared = np.square(candidates.astype(np.float64) - context.expert[:, None].astype(np.float64))
    for horizon in range(HORIZON):
        step_error = squared[:, :, horizon].mean(axis=2)
        per_horizon[str(horizon + 1)] = _relative_gain(step_error[:, 0], step_error.min(axis=1))
    prefix = {}
    for width in (1, 2, 4, 8):
        value = squared[:, :, :width].mean(axis=(2, 3))
        prefix[str(width)] = _relative_gain(value[:, 0], value.min(axis=1))
    dimensions = {}
    for label, index in (("translation", TRANSLATION), ("rotation", ROTATION), ("gripper", GRIPPER)):
        value = squared[:, :, :, index].mean(axis=(2, 3))
        dimensions[label] = _relative_gain(value[:, 0], value.min(axis=1))

    flat = candidates.astype(np.float64).reshape(len(candidates), candidates.shape[1], -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    n = candidates.shape[1]
    pairwise_rms = float(np.sqrt((2.0 * n / max(n - 1, 1)) * np.mean(np.square(centered))))
    diversity_dimension = {}
    for label, index in (("translation", TRANSLATION), ("rotation", ROTATION), ("gripper", GRIPPER)):
        part = candidates[:, :, :, index].astype(np.float64)
        part = part.reshape(len(part), n, -1)
        part = part - part.mean(axis=1, keepdims=True)
        diversity_dimension[label] = float(np.sqrt((2.0 * n / max(n - 1, 1)) * np.mean(np.square(part))))
    per_step_diversity = []
    for horizon in range(HORIZON):
        part = candidates[:, :, horizon].astype(np.float64)
        part = part - part.mean(axis=1, keepdims=True)
        per_step_diversity.append(float(np.sqrt((2.0 * n / max(n - 1, 1)) * np.mean(np.square(part)))))
    effective_rank, spectrum = _effective_rank(candidates, context.actor_default)
    winner_delta = selected.astype(np.float64) - context.actor_default.astype(np.float64)
    winner_distance = {
        "all_rms": float(np.sqrt(np.mean(np.square(winner_delta)))),
        "translation_rms": float(np.sqrt(np.mean(np.square(winner_delta[:, :, TRANSLATION])))),
        "rotation_rms": float(np.sqrt(np.mean(np.square(winner_delta[:, :, ROTATION])))),
        "gripper_rms": float(np.sqrt(np.mean(np.square(winner_delta[:, :, GRIPPER])))),
    }
    nondefault = winner != 0
    conditional_gain = float(np.mean(gains[nondefault])) if np.any(nondefault) else 0.0
    quantiles = np.quantile(gains, (0.25, 0.5, 0.75, 0.9))
    summary = {
        "name": name,
        "candidate_count": int(n),
        "sample_count": int(len(candidates)),
        "default_raw_action_mse": float(default_error.mean()),
        "oracle_raw_action_mse": float(oracle_error.mean()),
        "absolute_mse_gain": float(default_error.mean() - oracle_error.mean()),
        "relative_mse_gain": _relative_gain(default_error, oracle_error),
        "relative_mse_gain_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "positive_task_count": int(sum(value > 0 for value in per_task.values())),
        "per_task_relative_gain": per_task,
        "per_horizon_oracle_relative_gain": per_horizon,
        "prefix_oracle_relative_gain": prefix,
        "dimension_oracle_relative_gain": dimensions,
        "default_winner_fraction": float(np.mean(winner == 0)),
        "nondefault_winner_fraction": float(np.mean(nondefault)),
        "conditional_nondefault_mean_relative_gain": conditional_gain,
        "sample_relative_gain_distribution": {
            "p25": float(quantiles[0]), "median": float(quantiles[1]),
            "p75": float(quantiles[2]), "p90": float(quantiles[3]),
            "fraction_gt_1pct": float(np.mean(gains > 0.01)),
            "fraction_gt_5pct": float(np.mean(gains > 0.05)),
            "fraction_gt_10pct": float(np.mean(gains > 0.10)),
        },
        "diversity": {
            "pairwise_rms_action": pairwise_rms,
            "by_dimension": diversity_dimension,
            "per_step_rms": per_step_diversity,
            "effective_rank": effective_rank,
            "covariance_spectrum_top10_fraction": spectrum,
        },
        "winner_distance_from_a0": winner_distance,
        "clip_fraction": float(np.mean(np.abs(preclip) > 1.0)),
        "boundary_hit_fraction": float(np.mean(np.abs(candidates) >= 1.0 - 1.0e-7)),
        "invalid_candidate_count": 0,
        "oracle_worse_count": int(np.sum(oracle_error > default_error + 1.0e-15)),
        "compute_multiplier_vs_p0": float(n / 3.0),
    }
    arrays = {
        "default_error": default_error.astype(np.float64),
        "oracle_error": oracle_error.astype(np.float64),
        "winner": winner.astype(np.int16),
        "winner_action": selected.astype(np.float32),
        "bootstrap_gain": boot.astype(np.float64),
    }
    return summary, arrays

