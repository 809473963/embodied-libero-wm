#!/usr/bin/env python3
"""Offline C1-v2.1 candidate, reranker, and frozen-WM mechanism diagnosis."""

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

from src.c0 import C0Actor, CausalActorDataset
from src.world_models import LatentDynamicsEnsemble, LatentDynamicsModel


PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
HORIZON = 8
ACTION_DIM = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
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
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distribution requires non-empty finite values")
    quantiles = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p5": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def correlation(first: np.ndarray, second: np.ndarray, method: str) -> dict[str, Any]:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    selected = np.isfinite(first) & np.isfinite(second)
    first, second = first[selected], second[selected]
    result: dict[str, Any] = {"count": int(first.size), "value": None, "pvalue": None}
    if first.size < 3 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return result
    if method == "pearson":
        measured = scipy_stats.pearsonr(first, second)
    elif method == "spearman":
        measured = scipy_stats.spearmanr(first, second)
    elif method == "kendall":
        measured = scipy_stats.kendalltau(first, second)
    else:
        raise ValueError(method)
    result["value"] = float(measured.statistic)
    result["pvalue"] = float(measured.pvalue)
    return result


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != "c1-v2.1-candidate-discriminability-v1":
        raise ValueError("unexpected C1-v2.1 protocol")
    if protocol["validation"]["demo_ids"] != list(range(45, 50)):
        raise ValueError("C1-v2.1 may use validation demos 45-49 only")
    if protocol["validation"]["init_state_indices_used"]:
        raise ValueError("offline validation protocol may not use simulator init states")
    expected_hashes = (
        (protocol["frozen_c1_v2_negative"]["audit"], "audit_sha256"),
        (protocol["frozen_c1_v2_negative"]["gate"], "gate_sha256"),
        (protocol["validation"]["store"], "store_sha256"),
        (protocol["validation"]["feature_manifest"], "feature_manifest_sha256"),
        (protocol["actor"]["checkpoint"], "checkpoint_sha256"),
        (protocol["candidate_generator"]["config"], "config_sha256"),
        (protocol["world_model"]["ensemble_config"], "ensemble_config_sha256"),
        (
            protocol["world_model"]["ordinary_validation_result"],
            "ordinary_validation_result_sha256",
        ),
    )
    sections = (
        protocol["frozen_c1_v2_negative"],
        protocol["frozen_c1_v2_negative"],
        protocol["validation"],
        protocol["validation"],
        protocol["actor"],
        protocol["candidate_generator"],
        protocol["world_model"],
        protocol["world_model"],
    )
    for (relative, hash_key), section in zip(expected_hashes, sections):
        if sha256(ROOT / relative) != section[hash_key]:
            raise ValueError(f"frozen source changed: {relative}")
    tagged = subprocess.check_output(
        ["git", "rev-parse", f"{protocol['frozen_c1_v2_negative']['annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    if tagged != protocol["frozen_c1_v2_negative"]["commit"]:
        raise ValueError("C1-v2 negative tag does not resolve to the frozen commit")
    gate = json.loads((ROOT / protocol["frozen_c1_v2_negative"]["gate"]).read_text())
    if gate["decision"] != "BLOCK_C1_V2_RERANKER":
        raise ValueError("frozen C1-v2 verdict changed")
    return protocol, sha256(path)


def load_models(
    protocol: dict[str, Any], device: torch.device
) -> tuple[C0Actor, LatentDynamicsEnsemble, dict[str, np.ndarray]]:
    actor_state = torch.load(
        ROOT / protocol["actor"]["checkpoint"], map_location="cpu", weights_only=False
    )
    if actor_state["supervision_contract"] != {
        "version": "c0-v2-causal-alignment-v1",
        "pairing": "obs[k] -> actions[k+1:k+9]",
        "action_start_offset": 1,
        "horizon": 8,
    }:
        raise ValueError("actor is not frozen C0-v2")
    actor = C0Actor(
        feature_dim=512,
        proprio_dim=12,
        fusion_dim=int(actor_state["config"]["model"]["fusion_dim"]),
        action_horizon=8,
        action_dim=7,
    )
    actor.load_state_dict(actor_state["model_state"])
    actor.to(device).eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    ensemble_config = yaml.safe_load(
        (ROOT / protocol["world_model"]["ensemble_config"]).read_text()
    )
    member_paths = [ROOT / value for value in ensemble_config["member_checkpoints"]]
    if [sha256(path) for path in member_paths] != protocol["world_model"][
        "member_checkpoint_sha256"
    ]:
        raise ValueError("frozen WM member checkpoint changed")
    members = []
    reference: dict[str, Any] | None = None
    for path in member_paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        comparable = {
            "contract": state["d13_data_contract_version"],
            "store": state["d13_store_sha256"],
            "normalization": state["normalization"],
            "model": state["config"]["model"],
        }
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError("WM members do not share a frozen contract")
        model = LatentDynamicsModel(
            **{key: int(value) for key, value in state["config"]["model"].items()}
        )
        model.load_state_dict(state["model_state"])
        members.append(model)
    if reference is None:
        raise ValueError("empty WM ensemble")
    ensemble = LatentDynamicsEnsemble(members).to(device).eval()
    for parameter in ensemble.parameters():
        parameter.requires_grad_(False)

    actor_normalization = actor_state["normalization"]
    wm_normalization = reference["normalization"]
    tensors = {
        "actor_proprio_mean": np.asarray(
            actor_normalization["proprio"]["mean"], dtype=np.float32
        ),
        "actor_proprio_std": np.asarray(
            actor_normalization["proprio"]["std"], dtype=np.float32
        ),
        "actor_action_mean": np.asarray(
            actor_normalization["action"]["mean"], dtype=np.float32
        ),
        "actor_action_std": np.asarray(
            actor_normalization["action"]["std"], dtype=np.float32
        ),
        "wm_proprio_mean": np.asarray(
            wm_normalization["proprio"]["mean"], dtype=np.float32
        ),
        "wm_proprio_std": np.asarray(
            wm_normalization["proprio"]["std"], dtype=np.float32
        ),
        "wm_action_mean": np.asarray(
            wm_normalization["action"]["mean"], dtype=np.float32
        ),
        "wm_action_std": np.asarray(
            wm_normalization["action"]["std"], dtype=np.float32
        ),
    }
    if not all(np.isfinite(value).all() for value in tensors.values()):
        raise FloatingPointError("normalization contains NaN/Inf")
    return actor, ensemble, tensors


@torch.inference_mode()
def collect_validation_context(
    protocol: dict[str, Any],
    actor: C0Actor,
    normalization: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dataset = CausalActorDataset(
        ROOT / protocol["validation"]["store"], "validation", in_memory=True
    )
    if len(dataset) != int(protocol["validation"]["expected_samples"]):
        raise ValueError("unexpected C0-v2 validation sample count")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    output: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "z0",
            "language",
            "proprio",
            "target",
            "actor_normalized",
            "task_id",
            "demo_id",
            "start_timestep",
        )
    }
    p_mean = torch.from_numpy(normalization["actor_proprio_mean"]).to(device)
    p_std = torch.from_numpy(normalization["actor_proprio_std"]).to(device)
    for batch in loader:
        z0 = batch["image_features"].to(device)
        language = batch["language_features"].to(device)
        proprio = batch["proprio_raw"].to(device)
        prediction = actor(z0, language, (proprio - p_mean) / p_std)
        output["z0"].append(z0.cpu().numpy())
        output["language"].append(language.cpu().numpy())
        output["proprio"].append(proprio.cpu().numpy())
        output["target"].append(batch["action_chunks"].numpy())
        output["actor_normalized"].append(prediction.float().cpu().numpy())
        output["task_id"].append(batch["task_id"].numpy())
        output["demo_id"].append(batch["demo_id"].numpy())
        output["start_timestep"].append(batch["start_timestep"].numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in output.items()}
    expected = len(dataset)
    if any(len(value) != expected for value in result.values()):
        raise RuntimeError("validation context row accounting mismatch")
    if set(np.unique(result["demo_id"]).tolist()) != set(range(45, 50)):
        raise ValueError("offline analysis escaped validation demos 45-49")
    if set(np.unique(result["task_id"]).tolist()) != set(range(10)):
        raise ValueError("offline analysis does not cover all ten validation tasks")
    dataset.close()
    return result


def generate_base_noise(
    sample_count: int, seed_base: int, noise_std: np.ndarray
) -> np.ndarray:
    output = np.empty((sample_count, HORIZON, ACTION_DIM), dtype=np.float32)
    std = torch.from_numpy(noise_std.astype(np.float32))
    for index in range(sample_count):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_base + index)
        noise = torch.randn((HORIZON, ACTION_DIM), generator=generator) * std
        padded = torch.cat((noise[:1], noise, noise[-1:]), dim=0)
        output[index] = (
            0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        ).numpy()
    if not np.isfinite(output).all() or not np.array_equal(output[:, :, 6], np.zeros((sample_count, HORIZON))):
        raise FloatingPointError("candidate base noise is invalid")
    return output


def make_candidates(
    actor_normalized: np.ndarray,
    base_noise: np.ndarray,
    scale: float,
    normalization: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denormalized = (
        actor_normalized * normalization["actor_action_std"].reshape(1, 1, 7)
        + normalization["actor_action_mean"].reshape(1, 1, 7)
    )
    actor_default = np.clip(denormalized, -1.0, 1.0)
    scaled = base_noise * np.float32(scale)
    preclip = np.stack((actor_default, actor_default + scaled, actor_default - scaled), axis=1)
    candidates = np.clip(preclip, -1.0, 1.0).astype(np.float32)
    wm_normalized = (
        candidates - normalization["wm_action_mean"].reshape(1, 1, 1, 7)
    ) / normalization["wm_action_std"].reshape(1, 1, 1, 7)
    if not all(np.isfinite(value).all() for value in (preclip, candidates, wm_normalized)):
        raise FloatingPointError("candidate construction contains NaN/Inf")
    return preclip, candidates, wm_normalized.astype(np.float32)


def action_quality(
    candidates: np.ndarray,
    target: np.ndarray,
    action_std: np.ndarray,
    task_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    delta = candidates - target[:, None]
    raw_mse = np.square(delta, dtype=np.float64).mean(axis=(2, 3))
    normalized_delta = delta / action_std.reshape(1, 1, 1, 7)
    normalized_mse = np.square(normalized_delta, dtype=np.float64).mean(axis=(2, 3))
    chunk_l2 = np.linalg.norm(delta.astype(np.float64).reshape(len(delta), 3, -1), axis=-1)
    per_dimension_mse = np.square(delta, dtype=np.float64).mean(axis=(0, 2))
    oracle = np.argmin(raw_mse, axis=1)
    default = float(raw_mse[:, 0].mean())
    best = float(raw_mse.min(axis=1).mean())

    def summarize(selected: np.ndarray) -> dict[str, Any]:
        selected_raw = raw_mse[selected]
        selected_normalized = normalized_mse[selected]
        selected_l2 = chunk_l2[selected]
        selected_oracle = oracle[selected]
        default_mean = float(selected_raw[:, 0].mean())
        best_mean = float(selected_raw.min(axis=1).mean())
        return {
            "sample_count": int(selected.sum()),
            "oracle_best_fraction": {
                str(index): float(np.mean(selected_oracle == index)) for index in range(3)
            },
            "alternative_oracle_best_fraction": float(np.mean(selected_oracle != 0)),
            "default_raw_action_mse": default_mean,
            "best_of_3_raw_action_mse": best_mean,
            "best_of_3_raw_mse_relative_gain": (default_mean - best_mean) / default_mean,
            "candidate_raw_action_mse": [
                float(selected_raw[:, index].mean()) for index in range(3)
            ],
            "default_normalized_action_mse": float(selected_normalized[:, 0].mean()),
            "best_of_3_normalized_action_mse": float(
                selected_normalized.min(axis=1).mean()
            ),
            "candidate_normalized_action_mse": [
                float(selected_normalized[:, index].mean()) for index in range(3)
            ],
            "candidate_chunk_l2": [
                float(selected_l2[:, index].mean()) for index in range(3)
            ],
        }

    all_selected = np.ones(len(target), dtype=bool)
    result = summarize(all_selected)
    result["candidate_per_dimension_raw_mse"] = per_dimension_mse.tolist()
    result["per_task"] = {
        str(task_id): summarize(task_ids == task_id)
        for task_id in sorted(np.unique(task_ids).tolist())
    }
    if result["default_raw_action_mse"] != default or result["best_of_3_raw_action_mse"] != best:
        raise RuntimeError("action quality aggregation mismatch")
    return result, {
        "raw_mse": raw_mse.astype(np.float32),
        "normalized_mse": normalized_mse.astype(np.float32),
        "chunk_l2": chunk_l2.astype(np.float32),
        "oracle": oracle.astype(np.int8),
    }


def action_separation(
    candidates: np.ndarray, action_std: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw_l2 = np.empty((len(candidates), 3), dtype=np.float32)
    normalized_l2 = np.empty_like(raw_l2)
    max_absolute = np.empty_like(raw_l2)
    per_dimension: dict[str, list[float]] = {}
    per_horizon: dict[str, list[float]] = {}
    pairs: dict[str, Any] = {}
    for pair_index, (name, (first, second)) in enumerate(zip(PAIR_NAMES, PAIR_INDICES)):
        delta = candidates[:, first] - candidates[:, second]
        raw_l2[:, pair_index] = np.linalg.norm(delta.reshape(len(delta), -1), axis=-1)
        normalized = delta / action_std.reshape(1, 1, 7)
        normalized_l2[:, pair_index] = np.linalg.norm(
            normalized.reshape(len(delta), -1), axis=-1
        )
        max_absolute[:, pair_index] = np.abs(delta).max(axis=(1, 2))
        per_dimension[name] = np.abs(delta).mean(axis=(0, 1)).tolist()
        per_horizon[name] = np.linalg.norm(delta, axis=-1).mean(axis=0).tolist()
        pairs[name] = {
            "raw_chunk_l2": distribution(raw_l2[:, pair_index]),
            "normalized_chunk_l2": distribution(normalized_l2[:, pair_index]),
            "max_absolute_difference": distribution(max_absolute[:, pair_index]),
            "mean_absolute_difference_per_action_dimension": per_dimension[name],
            "mean_raw_l2_per_horizon": per_horizon[name],
        }
    return {
        "pairs": pairs,
        "mean_pairwise_raw_chunk_l2": float(raw_l2.mean()),
        "mean_pairwise_normalized_chunk_l2": float(normalized_l2.mean()),
        "mean_pairwise_max_absolute_difference": float(max_absolute.mean()),
    }, {
        "raw_l2": raw_l2,
        "normalized_l2": normalized_l2,
        "max_absolute": max_absolute,
    }


@torch.inference_mode()
def evaluate_world_model(
    ensemble: LatentDynamicsEnsemble,
    context: dict[str, np.ndarray],
    candidates: np.ndarray,
    wm_normalized: np.ndarray,
    normalization: dict[str, np.ndarray],
    rerank: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    sample_count = len(candidates)
    outputs = {
        "score": np.empty((sample_count, 3), dtype=np.float32),
        "goal_progress": np.empty((sample_count, 3), dtype=np.float32),
        "mean_uncertainty": np.empty((sample_count, 3), dtype=np.float32),
        "base_deviation": np.empty((sample_count, 3), dtype=np.float32),
        "prediction_pair_cosine": np.empty((sample_count, 3, HORIZON), dtype=np.float32),
        "prediction_pair_l2": np.empty((sample_count, 3, HORIZON), dtype=np.float32),
    }
    wm_p_mean = torch.from_numpy(normalization["wm_proprio_mean"]).to(device)
    wm_p_std = torch.from_numpy(normalization["wm_proprio_std"]).to(device)
    wm_a_std = torch.from_numpy(normalization["wm_action_std"]).to(device)
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        count = stop - start
        z0 = torch.from_numpy(context["z0"][start:stop]).to(device)
        language = torch.from_numpy(context["language"][start:stop]).to(device)
        proprio = torch.from_numpy(context["proprio"][start:stop]).to(device)
        actions = torch.from_numpy(wm_normalized[start:stop]).to(device)
        flat_z0 = z0[:, None].expand(-1, 3, -1).reshape(count * 3, 512)
        flat_language = language[:, None].expand(-1, 3, -1).reshape(count * 3, 512)
        normalized_proprio = (proprio - wm_p_mean) / wm_p_std
        flat_proprio = normalized_proprio[:, None].expand(-1, 3, -1).reshape(
            count * 3, 12
        )
        flat_actions = actions.reshape(count * 3, HORIZON, ACTION_DIM)
        mean_prediction, uncertainty, _ = ensemble.predict(
            flat_z0, flat_proprio, flat_language, flat_actions
        )
        prediction = mean_prediction.reshape(count, 3, HORIZON, 512).float()
        uncertainty = uncertainty.reshape(count, 3, HORIZON).float()
        current_similarity = F.cosine_similarity(z0.float(), language.float(), dim=-1)
        final_similarity = F.cosine_similarity(
            prediction[:, :, -1], language[:, None].expand(-1, 3, -1), dim=-1
        )
        goal_progress = final_similarity - current_similarity[:, None]
        mean_uncertainty = uncertainty.mean(dim=-1)
        raw_candidates = torch.from_numpy(candidates[start:stop]).to(device)
        normalized_delta = (raw_candidates - raw_candidates[:, 0:1]) / wm_a_std.reshape(
            1, 1, 1, 7
        )
        base_deviation = normalized_delta.square().mean(dim=(2, 3))
        score = (
            float(rerank["goal_progress_weight"]) * goal_progress
            - float(rerank["uncertainty_penalty_weight"]) * mean_uncertainty
            - float(rerank["base_deviation_penalty_weight"]) * base_deviation
        )
        outputs["score"][start:stop] = score.cpu().numpy()
        outputs["goal_progress"][start:stop] = goal_progress.cpu().numpy()
        outputs["mean_uncertainty"][start:stop] = mean_uncertainty.cpu().numpy()
        outputs["base_deviation"][start:stop] = base_deviation.cpu().numpy()
        for pair_index, (first, second) in enumerate(PAIR_INDICES):
            outputs["prediction_pair_cosine"][start:stop, pair_index] = (
                1.0
                - F.cosine_similarity(
                    prediction[:, first], prediction[:, second], dim=-1
                )
            ).cpu().numpy()
            outputs["prediction_pair_l2"][start:stop, pair_index] = torch.linalg.vector_norm(
                prediction[:, first] - prediction[:, second], dim=-1
            ).cpu().numpy()
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("WM candidate evaluation contains NaN/Inf")
    return outputs


def within_sample_rank_correlations(
    scores: np.ndarray, negative_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    spearman = np.full(len(scores), np.nan, dtype=np.float64)
    kendall = np.full(len(scores), np.nan, dtype=np.float64)
    for index, (score_row, quality_row) in enumerate(zip(scores, negative_error)):
        if np.ptp(score_row) == 0.0 or np.ptp(quality_row) == 0.0:
            continue
        spearman[index] = scipy_stats.spearmanr(score_row, quality_row).statistic
        kendall[index] = scipy_stats.kendalltau(score_row, quality_row).statistic
    return spearman, kendall


def ranking_quality(scores: np.ndarray, raw_mse: np.ndarray) -> dict[str, Any]:
    oracle = np.argmin(raw_mse, axis=1)
    selected = np.argmax(scores, axis=1)
    oracle_ties = np.ptp(raw_mse, axis=1) <= 1.0e-12
    top_ties = np.ptp(scores, axis=1) <= 1.0e-12
    quality_delta = np.concatenate(
        (raw_mse[:, 0] - raw_mse[:, 1], raw_mse[:, 0] - raw_mse[:, 2])
    )
    score_delta = np.concatenate(
        (scores[:, 1] - scores[:, 0], scores[:, 2] - scores[:, 0])
    )
    eligible = np.abs(quality_delta) > 1.0e-12
    pairwise_accuracy = (
        float(np.mean((quality_delta[eligible] * score_delta[eligible]) > 0.0))
        if eligible.any()
        else None
    )
    row_spearman, row_kendall = within_sample_rank_correlations(scores, -raw_mse)
    replacements = selected != 0
    row_index = np.arange(len(scores))
    selected_margin = scores[row_index, selected] - scores[:, 0]
    selected_improvement = raw_mse[:, 0] - raw_mse[row_index, selected]
    margin_spearman = correlation(
        selected_margin[replacements], selected_improvement[replacements], "spearman"
    )
    margin_pearson = correlation(
        selected_margin[replacements], selected_improvement[replacements], "pearson"
    )
    bins = []
    replacement_indices = np.flatnonzero(replacements)
    if replacement_indices.size:
        ordered = replacement_indices[np.argsort(selected_margin[replacement_indices])]
        for bin_index, indices in enumerate(np.array_split(ordered, 5)):
            if not len(indices):
                continue
            bins.append(
                {
                    "bin": bin_index + 1,
                    "count": int(len(indices)),
                    "margin_min": float(selected_margin[indices].min()),
                    "margin_max": float(selected_margin[indices].max()),
                    "mean_score_margin": float(selected_margin[indices].mean()),
                    "mean_oracle_error_improvement": float(
                        selected_improvement[indices].mean()
                    ),
                    "selected_better_fraction": float(
                        np.mean(selected_improvement[indices] > 0.0)
                    ),
                }
            )
    valid_spearman = row_spearman[np.isfinite(row_spearman)]
    valid_kendall = row_kendall[np.isfinite(row_kendall)]
    return {
        "sample_count": int(len(scores)),
        "top_score_equals_oracle_best_fraction": float(np.mean(selected == oracle)),
        "random_top1_reference": 1.0 / 3.0,
        "majority_oracle_class_reference": float(
            np.bincount(oracle, minlength=3).max() / len(oracle)
        ),
        "oracle_error_tie_fraction": float(oracle_ties.mean()),
        "score_tie_fraction": float(top_ties.mean()),
        "default_alternative_pairwise_ranking_accuracy": pairwise_accuracy,
        "default_alternative_pair_count": int(eligible.sum()),
        "pooled_spearman_score_vs_negative_raw_mse": correlation(
            scores, -raw_mse, "spearman"
        ),
        "pooled_kendall_score_vs_negative_raw_mse": correlation(
            scores, -raw_mse, "kendall"
        ),
        "mean_within_sample_spearman": (
            float(valid_spearman.mean()) if valid_spearman.size else None
        ),
        "within_sample_spearman_count": int(valid_spearman.size),
        "mean_within_sample_kendall": (
            float(valid_kendall.mean()) if valid_kendall.size else None
        ),
        "within_sample_kendall_count": int(valid_kendall.size),
        "selected_candidate_fraction": {
            str(index): float(np.mean(selected == index)) for index in range(3)
        },
        "replacement_count": int(replacements.sum()),
        "replacement_rate": float(replacements.mean()),
        "replacement_margin_vs_oracle_improvement": {
            "spearman": margin_spearman,
            "pearson": margin_pearson,
            "margin_bins": bins,
        },
    }


def score_component_summary(
    score: np.ndarray,
    goal_progress: np.ndarray,
    mean_uncertainty: np.ndarray,
    base_deviation: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    components = {
        "goal_progress": (goal_progress, goal_progress),
        "uncertainty": (mean_uncertainty, -0.25 * mean_uncertainty),
        "action_deviation": (base_deviation, -0.001 * base_deviation),
        "total_score": (score, score),
    }
    result: dict[str, Any] = {"components": {}}
    row = np.arange(len(score))
    replacements = selected != 0
    for name, (raw, contribution) in components.items():
        result["components"][name] = {
            "raw": distribution(raw),
            "contribution_after_coefficient": distribution(contribution),
            "by_candidate": {
                str(index): {
                    "raw": distribution(raw[:, index]),
                    "contribution_after_coefficient": distribution(
                        contribution[:, index]
                    ),
                }
                for index in range(3)
            },
            "candidate_1_minus_0_contribution": distribution(
                contribution[:, 1] - contribution[:, 0]
            ),
            "candidate_2_minus_0_contribution": distribution(
                contribution[:, 2] - contribution[:, 0]
            ),
            "selected_minus_default_contribution": distribution(
                contribution[row, selected] - contribution[:, 0]
            ),
            "replacement_selected_minus_default_contribution": (
                distribution(
                    contribution[row[replacements], selected[replacements]]
                    - contribution[replacements, 0]
                )
                if replacements.any()
                else None
            ),
        }
    selected_gain = score[row, selected] - score[:, 0]
    result["selected_candidate_fraction"] = {
        str(index): float(np.mean(selected == index)) for index in range(3)
    }
    result["replacement_rate"] = float(replacements.mean())
    result["selected_score_gain"] = distribution(selected_gain)
    result["replacement_selected_score_gain"] = (
        distribution(selected_gain[replacements]) if replacements.any() else None
    )
    reconstructed = (
        goal_progress - 0.25 * mean_uncertainty - 0.001 * base_deviation
    )
    result["score_formula_max_abs_backcheck"] = float(np.max(np.abs(score - reconstructed)))
    if result["score_formula_max_abs_backcheck"] > 1.0e-6:
        raise ValueError("score-component back-check failed")
    return result


def controller_development_score_components(
    protocol: dict[str, Any]
) -> dict[str, Any]:
    audit_path = ROOT / protocol["frozen_c1_v2_negative"]["audit"]
    audit = json.loads(audit_path.read_text())
    values = {key: [] for key in ("score", "goal", "uncertainty", "deviation")}
    selected_values = []
    event_count = 0
    for source in audit["sources"]["c1_v2"]:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"controller-development evidence changed: {path}")
        result = json.loads(path.read_text())
        for episode in result["episode_results"]:
            if int(episode["init_state_index"]) not in range(30, 40):
                raise ValueError("controller score audit escaped indices 30-39")
            for event in episode["plan_events"]:
                values["score"].append(event["scores"])
                values["goal"].append(event["goal_progress"])
                values["uncertainty"].append(event["mean_uncertainty"])
                values["deviation"].append(event["base_deviation"])
                selected_values.append(int(event["selected_candidate"]))
                event_count += 1
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    selected = np.asarray(selected_values, dtype=np.int64)
    if event_count != 2728 or any(value.shape != (event_count, 3) for value in arrays.values()):
        raise ValueError("unexpected controller-development plan evidence")
    return {
        "partition": "controller-development init-state indices 30-39",
        "plan_count": event_count,
        "summary": score_component_summary(
            arrays["score"],
            arrays["goal"],
            arrays["uncertainty"],
            arrays["deviation"],
            selected,
        ),
    }


def wm_sensitivity(
    prediction_cosine: np.ndarray,
    prediction_l2: np.ndarray,
    ordinary_errors: np.ndarray,
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for pair_index, name in enumerate(PAIR_NAMES):
        pairs[name] = {
            "per_horizon_cosine_distance": [
                distribution(prediction_cosine[:, pair_index, horizon])
                for horizon in range(HORIZON)
            ],
            "per_horizon_l2_distance": [
                distribution(prediction_l2[:, pair_index, horizon])
                for horizon in range(HORIZON)
            ],
            "mean_trajectory_cosine_distance": float(
                prediction_cosine[:, pair_index].mean()
            ),
            "mean_trajectory_l2_distance": float(prediction_l2[:, pair_index].mean()),
        }
    per_horizon_cosine = prediction_cosine.mean(axis=(0, 1))
    per_horizon_l2 = prediction_l2.mean(axis=(0, 1))
    ratios = per_horizon_cosine / ordinary_errors
    return {
        "pairs": pairs,
        "ensemble_mean_pairwise_cosine_separation_per_horizon": per_horizon_cosine.tolist(),
        "ensemble_mean_pairwise_l2_separation_per_horizon": per_horizon_l2.tolist(),
        "ordinary_wm_validation_cosine_error_per_horizon": ordinary_errors.tolist(),
        "cosine_separation_to_ordinary_error_ratio_per_horizon": ratios.tolist(),
        "mean_pairwise_prediction_cosine_separation": float(prediction_cosine.mean()),
        "mean_pairwise_prediction_l2_separation": float(prediction_l2.mean()),
        "mean_pairwise_prediction_cosine_separation_to_ordinary_error_ratio": float(
            prediction_cosine.mean() / ordinary_errors.mean()
        ),
    }


def task_ranking(
    scores: np.ndarray, raw_mse: np.ndarray, task_ids: np.ndarray
) -> dict[str, Any]:
    return {
        str(task_id): ranking_quality(scores[task_ids == task_id], raw_mse[task_ids == task_id])
        for task_id in range(10)
    }


def decide(
    protocol: dict[str, Any],
    quality: dict[str, Any],
    ranking: dict[str, Any],
    separation: dict[str, Any],
    sensitivity: dict[str, Any],
) -> dict[str, Any]:
    rules = protocol["decision_rules"]
    candidate_criteria = rules["candidate_generator_pass_requires_all"]
    candidate_checks = {
        "production_scale_best_of_3_raw_mse_relative_gain_min": quality[
            "best_of_3_raw_mse_relative_gain"
        ]
        >= candidate_criteria["production_scale_best_of_3_raw_mse_relative_gain_min"],
        "production_scale_alternative_oracle_best_fraction_min": quality[
            "alternative_oracle_best_fraction"
        ]
        >= candidate_criteria["production_scale_alternative_oracle_best_fraction_min"],
    }
    reranker_criteria = rules["reranker_score_pass_requires_all"]
    margin_spearman = ranking["replacement_margin_vs_oracle_improvement"]["spearman"][
        "value"
    ]
    reranker_checks = {
        "production_scale_top_score_oracle_accuracy_min": ranking[
            "top_score_equals_oracle_best_fraction"
        ]
        >= reranker_criteria["production_scale_top_score_oracle_accuracy_min"],
        "production_scale_default_alternative_pairwise_accuracy_min": (
            ranking["default_alternative_pairwise_ranking_accuracy"] is not None
            and ranking["default_alternative_pairwise_ranking_accuracy"]
            >= reranker_criteria[
                "production_scale_default_alternative_pairwise_accuracy_min"
            ]
        ),
        "production_scale_mean_within_sample_spearman_min": (
            ranking["mean_within_sample_spearman"] is not None
            and ranking["mean_within_sample_spearman"]
            >= reranker_criteria["production_scale_mean_within_sample_spearman_min"]
        ),
        "production_scale_replacement_margin_improvement_spearman_min": (
            margin_spearman is not None
            and margin_spearman
            >= reranker_criteria[
                "production_scale_replacement_margin_improvement_spearman_min"
            ]
        ),
    }
    wm_criteria = rules["wm_action_sensitivity_pass_requires_all"]
    wm_checks = {
        "production_scale_mean_pairwise_normalized_action_l2_min": separation[
            "mean_pairwise_normalized_chunk_l2"
        ]
        >= wm_criteria["production_scale_mean_pairwise_normalized_action_l2_min"],
        "production_scale_mean_pairwise_prediction_cosine_separation_to_ordinary_error_ratio_min": sensitivity[
            "mean_pairwise_prediction_cosine_separation_to_ordinary_error_ratio"
        ]
        >= wm_criteria[
            "production_scale_mean_pairwise_prediction_cosine_separation_to_ordinary_error_ratio_min"
        ],
    }
    blockers = []
    if not all(candidate_checks.values()):
        blockers.append(rules["candidate_failure_verdict"])
    if not all(reranker_checks.values()):
        blockers.append(rules["reranker_failure_verdict"])
    if not all(wm_checks.values()):
        blockers.append(rules["wm_sensitivity_failure_verdict"])
    if not blockers:
        blockers.append(rules["all_mechanisms_pass_verdict"])
    return {
        "blockers": blockers,
        "candidate_generator": {"checks": candidate_checks, "pass": all(candidate_checks.values())},
        "reranker_score": {"checks": reranker_checks, "pass": all(reranker_checks.values())},
        "wm_action_sensitivity": {"checks": wm_checks, "pass": all(wm_checks.values())},
        "new_c1_controller_design_allowed": blockers == [rules["all_mechanisms_pass_verdict"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/c1_v2_1_candidate_diagnosis.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/c1_v2_1/candidate_diagnosis.json",
    )
    parser.add_argument(
        "--sample-artifact",
        type=Path,
        default=ROOT / "artifacts/stage3/c1_v2_1_candidate_diagnosis/samples.npz",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() or args.sample_artifact.exists():
        raise FileExistsError("refusing to overwrite C1-v2.1 diagnosis evidence")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol, protocol_hash = load_protocol(args.protocol)
    actor, ensemble, normalization = load_models(protocol, device)
    context = collect_validation_context(
        protocol, actor, normalization, device, args.batch_size
    )
    generator_config = protocol["candidate_generator"]
    base_noise = generate_base_noise(
        len(context["z0"]),
        int(generator_config["seed_base"]),
        np.asarray(generator_config["base_noise_std"], dtype=np.float32),
    )
    c1_config = yaml.safe_load((ROOT / generator_config["config"]).read_text())
    rerank = c1_config["rerank"]
    ordinary_result = json.loads(
        (ROOT / protocol["world_model"]["ordinary_validation_result"]).read_text()
    )
    ordinary_errors = np.asarray(
        ordinary_result["learned"]["per_horizon_cosine_error"], dtype=np.float64
    )
    if ordinary_errors.shape != (8,) or not np.isfinite(ordinary_errors).all():
        raise ValueError("invalid ordinary WM validation errors")

    sweep: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    sidecar: dict[str, np.ndarray] = {
        "task_id": context["task_id"].astype(np.int8),
        "demo_id": context["demo_id"].astype(np.int8),
        "start_timestep": context["start_timestep"].astype(np.int32),
        "expert_target": context["target"].astype(np.float32),
        "actor_normalized": context["actor_normalized"].astype(np.float32),
        "base_noise_1x": base_noise.astype(np.float32),
    }
    for scale in generator_config["noise_scales"]:
        scale = float(scale)
        preclip, candidates, wm_normalized = make_candidates(
            context["actor_normalized"], base_noise, scale, normalization
        )
        quality, quality_arrays = action_quality(
            candidates,
            context["target"],
            normalization["actor_action_std"],
            context["task_id"],
        )
        separation, separation_arrays = action_separation(
            candidates, normalization["actor_action_std"]
        )
        model = evaluate_world_model(
            ensemble,
            context,
            candidates,
            wm_normalized,
            normalization,
            rerank,
            device,
            args.batch_size,
        )
        ranking = ranking_quality(model["score"], quality_arrays["raw_mse"])
        sensitivity = wm_sensitivity(
            model["prediction_pair_cosine"],
            model["prediction_pair_l2"],
            ordinary_errors,
        )
        clipping = (preclip < -1.0) | (preclip > 1.0)
        scale_key = f"{scale:g}x"
        sweep[scale_key] = {
            "scale": scale,
            "all_candidate_preclip_clipping_ratio": float(clipping.mean()),
            "action": {
                "mean_pairwise_raw_chunk_l2": separation[
                    "mean_pairwise_raw_chunk_l2"
                ],
                "mean_pairwise_normalized_chunk_l2": separation[
                    "mean_pairwise_normalized_chunk_l2"
                ],
            },
            "oracle": {
                "default_raw_action_mse": quality["default_raw_action_mse"],
                "best_of_3_raw_action_mse": quality["best_of_3_raw_action_mse"],
                "best_of_3_raw_mse_relative_gain": quality[
                    "best_of_3_raw_mse_relative_gain"
                ],
                "alternative_oracle_best_fraction": quality[
                    "alternative_oracle_best_fraction"
                ],
            },
            "world_model": {
                "mean_pairwise_prediction_cosine_separation": sensitivity[
                    "mean_pairwise_prediction_cosine_separation"
                ],
                "mean_pairwise_prediction_l2_separation": sensitivity[
                    "mean_pairwise_prediction_l2_separation"
                ],
                "cosine_separation_to_ordinary_error_ratio": sensitivity[
                    "mean_pairwise_prediction_cosine_separation_to_ordinary_error_ratio"
                ],
            },
            "ranking": {
                "top_score_equals_oracle_best_fraction": ranking[
                    "top_score_equals_oracle_best_fraction"
                ],
                "default_alternative_pairwise_ranking_accuracy": ranking[
                    "default_alternative_pairwise_ranking_accuracy"
                ],
                "mean_within_sample_spearman": ranking[
                    "mean_within_sample_spearman"
                ],
                "replacement_margin_improvement_spearman": ranking[
                    "replacement_margin_vs_oracle_improvement"
                ]["spearman"]["value"],
            },
        }
        prefix = scale_key.replace(".", "p")
        sidecar[f"{prefix}_raw_action_mse"] = quality_arrays["raw_mse"]
        sidecar[f"{prefix}_scores"] = model["score"]
        sidecar[f"{prefix}_prediction_pair_cosine"] = model[
            "prediction_pair_cosine"
        ]
        if scale == float(generator_config["production_scale"]):
            selected = np.argmax(model["score"], axis=1)
            current = {
                "candidate_quality": quality,
                "ranking_quality": {
                    **ranking,
                    "per_task": task_ranking(
                        model["score"], quality_arrays["raw_mse"], context["task_id"]
                    ),
                },
                "score_components": score_component_summary(
                    model["score"],
                    model["goal_progress"],
                    model["mean_uncertainty"],
                    model["base_deviation"],
                    selected,
                ),
                "candidate_action_separation": separation,
                "world_model_action_sensitivity": sensitivity,
            }
            sidecar["production_candidates"] = candidates
            sidecar["production_normalized_action_mse"] = quality_arrays[
                "normalized_mse"
            ]
            sidecar["production_chunk_l2_error"] = quality_arrays["chunk_l2"]
            sidecar["production_oracle_candidate"] = quality_arrays["oracle"]
            sidecar["production_goal_progress"] = model["goal_progress"]
            sidecar["production_mean_uncertainty"] = model["mean_uncertainty"]
            sidecar["production_base_deviation"] = model["base_deviation"]
            sidecar["production_prediction_pair_l2"] = model["prediction_pair_l2"]
    if current is None:
        raise RuntimeError("production scale was not evaluated")

    controller_score = controller_development_score_components(protocol)
    decision = decide(
        protocol,
        current["candidate_quality"],
        current["ranking_quality"],
        current["candidate_action_separation"],
        current["world_model_action_sensitivity"],
    )
    atomic_npz(args.sample_artifact, **sidecar)
    result = {
        "schema_version": 1,
        "diagnosis_version": protocol["protocol_version"],
        "status": "PASS_OFFLINE_DIAGNOSIS_COMPLETE",
        "frozen_c1_v2_verdict": "BLOCK_C1_V2_RERANKER",
        "blockers": decision["blockers"],
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": protocol_hash,
            "preregistered_commit": source_commit(),
        },
        "scope": {
            "split": "demo validation 45-49",
            "sample_count": len(context["z0"]),
            "task_count": 10,
            "demo_count": 50,
            "simulator_rollout_count": 0,
            "init_state_40_49_read_or_run": False,
        },
        "normalization": {
            "actor_action_mean": normalization["actor_action_mean"].tolist(),
            "actor_action_std": normalization["actor_action_std"].tolist(),
            "wm_action_mean": normalization["wm_action_mean"].tolist(),
            "wm_action_std": normalization["wm_action_std"].tolist(),
        },
        "production_scale_1x": current,
        "score_component_scale": {
            "validation_demo_45_49": current["score_components"],
            "controller_development_30_39": controller_score,
            "weights_modified": False,
        },
        "perturbation_scale_sweep": sweep,
        "decision": decision,
        "observed_mismatch_gate_interpretation": {
            "valid_use": "WM candidate-discriminability diagnostic",
            "invalid_use": "counterfactual claim that the default candidate would have controlled better",
            "reason": "selected and default predictions were compared with the same selected-policy observed latent; the default candidate was not executed counterfactually",
            "may_alone_drive_controller_change": False,
        },
        "sample_artifact": {
            "path": str(args.sample_artifact.relative_to(ROOT)),
            "sha256": sha256(args.sample_artifact),
            "arrays": {key: list(value.shape) for key, value in sidecar.items()},
        },
        "sources": {
            "c1_v2_negative_commit": protocol["frozen_c1_v2_negative"]["commit"],
            "c1_v2_negative_tag": protocol["frozen_c1_v2_negative"]["annotated_tag"],
            "validation_store_sha256": protocol["validation"]["store_sha256"],
            "actor_checkpoint_sha256": protocol["actor"]["checkpoint_sha256"],
            "ensemble_member_checkpoint_sha256": protocol["world_model"][
                "member_checkpoint_sha256"
            ],
            "ordinary_wm_validation_result_sha256": protocol["world_model"][
                "ordinary_validation_result_sha256"
            ],
        },
        "forbidden_activity_audit": {
            "score_weights_modified": False,
            "production_noise_scale_selected": False,
            "controller_rollout_run": False,
            "init_state_40_49_read_or_run": False,
            "d20_v2_started": False,
            "c2_v2_started": False,
            "c0_v2_retrained": False,
            "world_model_retrained": False,
            "c1_v2_negative_artifact_overwritten": False,
            "controller_success_used_for_parameter_iteration": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("final diagnosis contains NaN/Inf")
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "blockers": decision["blockers"],
                "sample_count": result["scope"]["sample_count"],
                "candidate_gain": current["candidate_quality"][
                    "best_of_3_raw_mse_relative_gain"
                ],
                "top_score_oracle_accuracy": current["ranking_quality"][
                    "top_score_equals_oracle_best_fraction"
                ],
                "wm_separation_error_ratio": current[
                    "world_model_action_sensitivity"
                ]["mean_pairwise_prediction_cosine_separation_to_ordinary_error_ratio"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
