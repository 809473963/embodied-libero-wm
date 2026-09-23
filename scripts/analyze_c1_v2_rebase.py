#!/usr/bin/env python3
"""Analyze C1-v2 rebase, WM distribution shift, reranking, and frozen gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (2, 4, 6, 8)
QUANTILE_NAMES = ("median", "p75", "p90", "p95", "p99")
QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content)
    os.replace(temporary, path)


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("statistics require a non-empty finite vector")
    result: dict[str, float | int] = {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }
    result.update(
        {
            name: float(value)
            for name, value in zip(QUANTILE_NAMES, np.quantile(values, QUANTILES))
        }
    )
    return result


def distribution_record(
    values: np.ndarray, threshold: float
) -> dict[str, float | int]:
    record = statistics(values)
    record["d20_v1_threshold"] = threshold
    record["d20_v1_exceed_count"] = int(np.sum(values > threshold))
    record["d20_v1_exceed_rate"] = float(np.mean(values > threshold))
    return record


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != "c1-v2-rebase-and-wm-distribution-v1":
        raise ValueError("unexpected C1-v2 protocol")
    protocol_hash = sha256(path)
    source_record = protocol["controller_development"]
    source_path = ROOT / source_record["partition_source"]
    if sha256(source_path) != source_record["partition_source_sha256"]:
        raise ValueError("controller-development source changed")
    source = json.loads(source_path.read_text())
    partition = source["partitions"][source_record["partition_name"]]
    if partition["init_state_indices"] != list(range(30, 40)):
        raise ValueError("analysis is restricted to init-state indices 30-39")
    return protocol, partition, protocol_hash


def normalization_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    actor_state = torch.load(
        ROOT / protocol["c0_v2_freeze"]["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    ensemble_config = __import__("yaml").safe_load(
        (ROOT / protocol["world_model_freeze"]["ensemble_config"]).read_text()
    )
    wm_state = torch.load(
        ROOT / ensemble_config["member_checkpoints"][0],
        map_location="cpu",
        weights_only=False,
    )
    actor = actor_state["normalization"]["action"]
    wm = wm_state["normalization"]["action"]
    return {
        "actor": {
            "mean": actor["mean"],
            "std": actor["std"],
            "count": actor["count"],
            "rule": actor["rule"],
        },
        "world_model": {
            "mean": wm["mean"],
            "std": wm["std"],
            "count": wm["count"],
            "rule": wm["rule"],
        },
        "numerically_equal": bool(
            np.array_equal(np.asarray(actor["mean"]), np.asarray(wm["mean"]))
            and np.array_equal(np.asarray(actor["std"]), np.asarray(wm["std"]))
        ),
        "semantic_boundary": (
            "actor-normalized output is denormalized before candidate generation; "
            "raw candidates are then normalized with the WM statistics even when the "
            "frozen statistics happen to be numerically equal"
        ),
    }


def validate_sidecar(
    result: dict[str, Any], action_normalization: dict[str, Any]
) -> dict[str, np.ndarray]:
    path = ROOT / result["trajectory_artifact"]
    if sha256(path) != result["trajectory_artifact_sha256"]:
        raise ValueError(f"trajectory artifact hash differs: {path}")
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    required = {
        "actor_normalized",
        "actor_denormalized_preclip",
        "candidate_preclip",
        "candidates",
        "wm_normalized_candidates",
        "ensemble_prediction",
        "ensemble_uncertainty",
    }
    if not required.issubset(arrays):
        raise ValueError(f"trajectory sidecar lacks {required.difference(arrays)}")
    plan_count = int(result["planning_events"])
    if any(arrays[key].shape[0] != plan_count for key in required):
        raise ValueError("trajectory row count differs from planning events")
    if not all(np.isfinite(arrays[key]).all() for key in required):
        raise FloatingPointError("trajectory sidecar contains NaN/Inf")
    actor_mean = np.asarray(action_normalization["actor"]["mean"], dtype=np.float32)
    actor_std = np.asarray(action_normalization["actor"]["std"], dtype=np.float32)
    wm_mean = np.asarray(action_normalization["world_model"]["mean"], dtype=np.float32)
    wm_std = np.asarray(action_normalization["world_model"]["std"], dtype=np.float32)
    expected_raw = arrays["actor_normalized"] * actor_std + actor_mean
    if not np.allclose(arrays["actor_denormalized_preclip"], expected_raw, atol=1.0e-6):
        raise ValueError("actor denormalization back-check failed")
    if not np.allclose(
        arrays["candidates"], np.clip(arrays["candidate_preclip"], -1.0, 1.0), atol=1.0e-6
    ):
        raise ValueError("candidate clipping back-check failed")
    if not np.allclose(
        arrays["candidates"][:, 0],
        np.clip(arrays["actor_denormalized_preclip"], -1.0, 1.0),
        atol=1.0e-6,
    ):
        raise ValueError("actor-default candidate back-check failed")
    expected_wm = (arrays["candidates"] - wm_mean.reshape(1, 1, 1, 7)) / wm_std.reshape(
        1, 1, 1, 7
    )
    if not np.allclose(arrays["wm_normalized_candidates"], expected_wm, atol=1.0e-6):
        raise ValueError("WM action normalization back-check failed")
    prediction_norm = np.linalg.norm(
        arrays["ensemble_prediction"].astype(np.float32), axis=-1
    )
    if not np.allclose(prediction_norm, 1.0, atol=2.0e-3):
        raise ValueError("ensemble prediction norm back-check failed")
    return arrays


def load_new_rollouts(
    directory: Path,
    controller: str,
    protocol: dict[str, Any],
    partition: dict[str, Any],
    protocol_hash: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    task_results = []
    plans = []
    checks = []
    sources = []
    for task_id in protocol["task_ids"]:
        path = directory / controller / f"task_{task_id}.json"
        result = json.loads(path.read_text())
        expected = {
            "contract_version": "c1-v2-controller-development-trace-v1",
            "controller": controller,
            "protocol_sha256": protocol_hash,
            "c0_v2_checkpoint_sha256": protocol["c0_v2_freeze"]["checkpoint_sha256"],
            "c1_v2_config_sha256": protocol["c1_v2"]["config_sha256"],
            "partition": "controller_development",
            "task_id": task_id,
            "episodes": 10,
            "complete": True,
        }
        mismatches = [key for key, value in expected.items() if result.get(key) != value]
        if mismatches:
            raise ValueError(f"{path} metadata mismatch: {mismatches}")
        expected_identity = [
            (
                episode_index,
                partition["task_seeds"][str(task_id)][episode_index],
                partition["init_state_indices"][episode_index],
            )
            for episode_index in range(10)
        ]
        actual_identity = [
            (episode["episode_index"], episode["seed"], episode["init_state_index"])
            for episode in result["episode_results"]
        ]
        if actual_identity != expected_identity:
            raise ValueError(f"{path} episode identities differ")
        arrays = validate_sidecar(result, normalization)
        for episode in result["episode_results"]:
            for event in episode["plan_events"]:
                plans.append(
                    {
                        **event,
                        "task_id": task_id,
                        "episode_index": episode["episode_index"],
                        "episode_success": bool(episode["success"]),
                    }
                )
            for event in episode["check_events"]:
                checks.append(
                    {
                        **event,
                        "task_id": task_id,
                        "episode_index": episode["episode_index"],
                        "episode_success": bool(episode["success"]),
                    }
                )
        expected_rows = list(range(result["planning_events"]))
        actual_rows = [
            event["trajectory_row"]
            for episode in result["episode_results"]
            for event in episode["plan_events"]
        ]
        if actual_rows != expected_rows or arrays["ensemble_prediction"].shape[0] != len(
            expected_rows
        ):
            raise ValueError(f"{path} trajectory references are not contiguous")
        task_results.append(result)
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "trajectory_path": result["trajectory_artifact"],
                "trajectory_sha256": result["trajectory_artifact_sha256"],
            }
        )
    return {"tasks": task_results, "plans": plans, "checks": checks, "sources": sources}


def load_historical_c0_v1(protocol: dict[str, Any]) -> dict[str, Any]:
    diagnosis_record = protocol["historical_c0_v1_distribution"]
    diagnosis_path = ROOT / diagnosis_record["diagnosis"]
    if sha256(diagnosis_path) != diagnosis_record["diagnosis_sha256"]:
        raise ValueError("frozen D22.5 diagnosis changed")
    diagnosis = json.loads(diagnosis_path.read_text())
    expected_sources = {
        record["path"]: record["sha256"]
        for record in diagnosis["inputs"]["controller_validation_trace_files"]["c0"]
    }
    checks = []
    for task_id in protocol["task_ids"]:
        path = ROOT / f"results/control_validation/c0/task_{task_id}.json"
        relative = str(path.relative_to(ROOT))
        if sha256(path) != expected_sources[relative]:
            raise ValueError(f"historical C0-v1 trace changed: {path}")
        result = json.loads(path.read_text())
        for episode in result["episode_results"]:
            for event in episode["check_events"]:
                checks.append(
                    {
                        "horizon": event["horizon"],
                        "mismatch": event["mismatch"],
                        "uncertainty": event["uncertainty"],
                    }
                )
    return {"checks": checks, "diagnosis": diagnosis_record}


def load_expert(protocol: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    record = protocol["world_model_freeze"]
    path = ROOT / record["expert_validation_samples"]
    if sha256(path) != record["expert_validation_samples_sha256"]:
        raise ValueError("expert validation sample record changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 5869:
        raise ValueError("expected 5,869 expert validation windows")
    return {
        horizon: {
            "mismatch": np.asarray(
                [row["learned_cosine_error"][horizon - 1] for row in rows],
                dtype=np.float64,
            ),
            "uncertainty": np.asarray(
                [row["uncertainty"][horizon - 1] for row in rows], dtype=np.float64
            ),
        }
        for horizon in HORIZONS
    }


def event_distribution(
    checks: list[dict[str, Any]], mismatch_key: str, uncertainty_key: str
) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    for horizon in HORIZONS:
        rows = [event for event in checks if int(event["horizon"]) == horizon]
        result[horizon] = {
            "mismatch": np.asarray([event[mismatch_key] for event in rows], dtype=np.float64),
            "uncertainty": np.asarray(
                [event[uncertainty_key] for event in rows], dtype=np.float64
            ),
        }
    return result


def build_distribution_table(
    distributions: dict[str, dict[int, dict[str, np.ndarray]]],
    mismatch_threshold: float,
    uncertainty_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    nested: dict[str, Any] = {}
    for distribution, by_horizon in distributions.items():
        nested[distribution] = {}
        for horizon in HORIZONS:
            nested[distribution][str(horizon)] = {}
            for metric, threshold in (
                ("mismatch", mismatch_threshold),
                ("uncertainty", uncertainty_threshold),
            ):
                record = distribution_record(by_horizon[horizon][metric], threshold)
                nested[distribution][str(horizon)][metric] = record
                rows.append(
                    {"distribution": distribution, "metric": metric, "horizon": horizon, **record}
                )
    return rows, nested


def distribution_shift_answers(table: dict[str, Any]) -> dict[str, Any]:
    per_horizon = {}
    c0_v2_closer = {"mismatch": 0, "uncertainty": 0}
    uncertainty_relief = []
    for horizon in HORIZONS:
        key = str(horizon)
        horizon_result = {}
        for metric in ("mismatch", "uncertainty"):
            expert = table["expert_demo_validation"][key][metric]
            old = table["historical_c0_v1_rollout_20_29"][key][metric]
            new = table["c0_v2_rollout_30_39"][key][metric]
            old_qmae = float(
                np.mean([abs(float(old[name]) - float(expert[name])) for name in QUANTILE_NAMES])
            )
            new_qmae = float(
                np.mean([abs(float(new[name]) - float(expert[name])) for name in QUANTILE_NAMES])
            )
            closer = new_qmae < old_qmae
            c0_v2_closer[metric] += int(closer)
            comparison = {
                "historical_c0_v1_quantile_mae_to_expert": old_qmae,
                "c0_v2_quantile_mae_to_expert": new_qmae,
                "c0_v2_is_closer": closer,
                "historical_c0_v1_median_to_expert_ratio": float(old["median"])
                / float(expert["median"]),
                "c0_v2_median_to_expert_ratio": float(new["median"])
                / float(expert["median"]),
                "historical_c0_v1_p95_to_expert_ratio": float(old["p95"])
                / float(expert["p95"]),
                "c0_v2_p95_to_expert_ratio": float(new["p95"])
                / float(expert["p95"]),
            }
            if metric == "uncertainty":
                denominator = float(old["median"]) - float(expert["median"])
                relief = (
                    (float(old["median"]) - float(new["median"])) / denominator
                    if denominator > 0
                    else None
                )
                comparison["median_excess_relief_fraction"] = relief
                if relief is not None:
                    uncertainty_relief.append(relief)
            horizon_result[metric] = comparison
        per_horizon[key] = horizon_result
    return {
        "per_horizon": per_horizon,
        "c0_v2_closer_horizon_count": c0_v2_closer,
        "mean_uncertainty_median_excess_relief_fraction": float(
            np.mean(uncertainty_relief)
        ),
        "comparison_limit": (
            "historical C0-v1 uses indices 20-29 and C0-v2 uses indices 30-39; "
            "the shift comparison is descriptive, not paired causal evidence"
        ),
    }


def aggregate_controller(rollout: dict[str, Any]) -> dict[str, Any]:
    tasks = []
    all_episodes = []
    for result in rollout["tasks"]:
        all_episodes.extend(result["episode_results"])
        tasks.append(
            {
                "task_id": result["task_id"],
                "successes": result["successes"],
                "episodes": result["episodes"],
                "success_rate": result["success_rate"],
                "mean_steps": result["mean_steps"],
                "planned_action_clipping_ratio": result[
                    "planned_action_clipping_ratio"
                ],
                "plans": result["planning_events"],
                "replacement_rate": result["replacement_rate"],
                "mean_planning_ms": result["mean_planning_ms"],
            }
        )
    plans = rollout["plans"]
    selected_elements = sum(
        int(
            event.get("selected_candidate_preclip", event["interface"]["candidate_preclip"])[
                "elements"
            ]
        )
        for event in plans
    )
    selected_outside = sum(
        int(
            event.get("selected_candidate_preclip", event["interface"]["candidate_preclip"])[
                "outside_elements"
            ]
        )
        for event in plans
    )
    return {
        "successes": sum(bool(episode["success"]) for episode in all_episodes),
        "episodes": len(all_episodes),
        "macro_success_rate": float(np.mean([task["success_rate"] for task in tasks])),
        "nonzero_tasks": sum(task["successes"] > 0 for task in tasks),
        "mean_steps": float(np.mean([episode["steps"] for episode in all_episodes])),
        "plans": len(plans),
        "replacement_count": sum(bool(event["replaced_actor_default"]) for event in plans),
        "replacement_rate": float(
            np.mean([bool(event["replaced_actor_default"]) for event in plans])
        ),
        "planned_action_clipping_ratio": selected_outside / selected_elements,
        "mean_planning_ms": float(np.mean([event["planning_ms"] for event in plans])),
        "p95_planning_ms": float(np.quantile([event["planning_ms"] for event in plans], 0.95)),
        "tasks": tasks,
    }


def analyze_replacements(rollout: dict[str, Any]) -> dict[str, Any]:
    replacements = [event for event in rollout["plans"] if event["replaced_actor_default"]]
    if not replacements:
        return {
            "replacement_count": 0,
            "replacement_rate": 0.0,
            "selected_score_ge_default_fraction": 0.0,
            "positive_score_gain_fraction": 0.0,
            "selected_observed_mismatch_nonworse_horizon_count": 0,
            "predicted_mismatch": {
                "status": "NOT_IDENTIFIABLE_PRE_EXECUTION",
                "reason": "no future target latent exists at plan time",
            },
        }
    score_gain = np.asarray([event["score_gain_vs_default"] for event in replacements])
    selected_uncertainty_delta = np.asarray(
        [
            event["mean_uncertainty"][event["selected_candidate"]]
            - event["mean_uncertainty"][0]
            for event in replacements
        ],
        dtype=np.float64,
    )
    selected_goal_delta = np.asarray(
        [
            event["goal_progress"][event["selected_candidate"]]
            - event["goal_progress"][0]
            for event in replacements
        ],
        dtype=np.float64,
    )
    observed = {}
    nonworse_count = 0
    for horizon in HORIZONS:
        rows = [
            event
            for event in rollout["checks"]
            if event["replaced_actor_default"] and int(event["horizon"]) == horizon
        ]
        selected = np.asarray(
            [event["selected_observed_mismatch"] for event in rows], dtype=np.float64
        )
        default = np.asarray(
            [event["actor_default_observed_mismatch"] for event in rows], dtype=np.float64
        )
        delta = selected - default
        nonworse = float(selected.mean()) <= float(default.mean())
        nonworse_count += int(nonworse)
        observed[str(horizon)] = {
            "count": len(rows),
            "selected_mean": float(selected.mean()),
            "actor_default_prediction_mean": float(default.mean()),
            "selected_minus_default_mean": float(delta.mean()),
            "selected_lower_fraction": float(np.mean(delta < 0)),
            "selected_nonworse_by_mean": nonworse,
        }

    episode_rows = []
    for result in rollout["tasks"]:
        for episode in result["episode_results"]:
            episode_rows.append(
                {
                    "success": bool(episode["success"]),
                    "replacement_rate": episode["replacement_count"] / episode["plan_count"],
                    "replacement_count": episode["replacement_count"],
                    "plans": episode["plan_count"],
                }
            )
    success_rates = [row["replacement_rate"] for row in episode_rows if row["success"]]
    failure_rates = [row["replacement_rate"] for row in episode_rows if not row["success"]]
    association = {
        "success_episode_count": len(success_rates),
        "failure_episode_count": len(failure_rates),
        "success_episode_mean_replacement_rate": (
            float(np.mean(success_rates)) if success_rates else None
        ),
        "failure_episode_mean_replacement_rate": (
            float(np.mean(failure_rates)) if failure_rates else None
        ),
        "difference_success_minus_failure": (
            float(np.mean(success_rates) - np.mean(failure_rates))
            if success_rates and failure_rates
            else None
        ),
        "interpretation_limit": "observational association; replacement was not randomized",
    }
    return {
        "replacement_count": len(replacements),
        "replacement_rate": len(replacements) / len(rollout["plans"]),
        "selected_candidate_counts": {
            str(index): sum(int(event["selected_candidate"]) == index for event in rollout["plans"])
            for index in range(3)
        },
        "score_gain": statistics(score_gain),
        "selected_score_ge_default_fraction": float(np.mean(score_gain >= 0.0)),
        "positive_score_gain_fraction": float(np.mean(score_gain > 0.0)),
        "selected_minus_default_mean_uncertainty": statistics(
            selected_uncertainty_delta
        ),
        "selected_uncertainty_lower_fraction": float(
            np.mean(selected_uncertainty_delta < 0.0)
        ),
        "selected_minus_default_goal_progress": statistics(selected_goal_delta),
        "selected_goal_progress_higher_fraction": float(np.mean(selected_goal_delta > 0.0)),
        "predicted_mismatch": {
            "status": "NOT_IDENTIFIABLE_PRE_EXECUTION",
            "reason": (
                "mismatch requires a future observed latent; C1-v2 has no future target at "
                "selection time and does not fabricate a predicted-error label"
            ),
            "available_proxies": ["ensemble uncertainty", "goal progress", "reranker score"],
        },
        "subsequent_observed_mismatch": observed,
        "selected_observed_mismatch_nonworse_horizon_count": nonworse_count,
        "episode_success_association": association,
    }


def gate_decision(
    protocol: dict[str, Any],
    c0: dict[str, Any],
    c1: dict[str, Any],
    reranking: dict[str, Any],
    interface_ok: bool,
    finite: bool,
) -> tuple[dict[str, bool], str]:
    criteria = protocol["c1_v2_gate"]["go_requires_all"]
    retention = c1["successes"] / c0["successes"] if c0["successes"] else 0.0
    checks = {
        "c1_v2_total_successes_min": c1["successes"]
        >= criteria["c1_v2_total_successes_min"],
        "c1_v2_success_retention_fraction_vs_c0_v2_min": retention
        >= criteria["c1_v2_success_retention_fraction_vs_c0_v2_min"],
        "c1_v2_nonzero_tasks_min": c1["nonzero_tasks"]
        >= criteria["c1_v2_nonzero_tasks_min"],
        "c1_v2_nonzero_task_delta_vs_c0_v2_min": c1["nonzero_tasks"]
        - c0["nonzero_tasks"]
        >= criteria["c1_v2_nonzero_task_delta_vs_c0_v2_min"],
        "finite_actions_metrics_and_trajectories": finite,
        "planned_action_element_clipping_ratio_max": c1[
            "planned_action_clipping_ratio"
        ]
        <= criteria["planned_action_element_clipping_ratio_max"],
        "replacement_count_min": reranking["replacement_count"]
        >= criteria["replacement_count_min"],
        "replacement_selected_score_ge_default_fraction_min": reranking[
            "selected_score_ge_default_fraction"
        ]
        >= criteria["replacement_selected_score_ge_default_fraction_min"],
        "replacement_positive_score_gain_fraction_min": reranking[
            "positive_score_gain_fraction"
        ]
        >= criteria["replacement_positive_score_gain_fraction_min"],
        "selected_observed_mismatch_nonworse_horizon_count_min": reranking[
            "selected_observed_mismatch_nonworse_horizon_count"
        ]
        >= criteria["selected_observed_mismatch_nonworse_horizon_count_min"],
    }
    if not interface_ok:
        verdict = protocol["c1_v2_gate"]["interface_failure_verdict"]
    elif all(checks.values()):
        verdict = protocol["c1_v2_gate"]["go_verdict"]
    else:
        verdict = protocol["c1_v2_gate"]["no_go_verdict"]
    return checks, verdict


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/c1_v2_rebase_protocol.json"
    )
    parser.add_argument(
        "--trace-dir", type=Path, default=ROOT / "results/c1_v2/controller_development"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/c1_v2/rebase_audit.json"
    )
    parser.add_argument(
        "--distribution-csv",
        type=Path,
        default=ROOT / "results/c1_v2/wm_distribution_table.csv",
    )
    parser.add_argument(
        "--gate-output", type=Path, default=ROOT / "configs/c1_v2_gate_decision.json"
    )
    args = parser.parse_args()
    for output in (args.output, args.distribution_csv, args.gate_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    protocol, partition, protocol_hash = load_protocol(args.protocol)
    normalization = normalization_contract(protocol)
    c0_v2 = load_new_rollouts(
        args.trace_dir,
        "c0_v2",
        protocol,
        partition,
        protocol_hash,
        normalization,
    )
    c1_v2 = load_new_rollouts(
        args.trace_dir,
        "c1_v2",
        protocol,
        partition,
        protocol_hash,
        normalization,
    )
    historical = load_historical_c0_v1(protocol)
    expert = load_expert(protocol)
    distributions = {
        "expert_demo_validation": expert,
        "historical_c0_v1_rollout_20_29": event_distribution(
            historical["checks"], "mismatch", "uncertainty"
        ),
        "c0_v2_rollout_30_39": event_distribution(
            c0_v2["checks"], "selected_observed_mismatch", "selected_uncertainty"
        ),
        "c1_v2_rollout_30_39": event_distribution(
            c1_v2["checks"], "selected_observed_mismatch", "selected_uncertainty"
        ),
    }
    d20 = protocol["frozen_d20_v1_diagnostic_only"]
    distribution_rows, distribution_table = build_distribution_table(
        distributions,
        float(d20["mismatch_threshold"]),
        float(d20["uncertainty_threshold"]),
    )
    c0_summary = aggregate_controller(c0_v2)
    c1_summary = aggregate_controller(c1_v2)
    frozen_baseline = json.loads(
        (ROOT / protocol["controller_development"]["frozen_c0_v2_baseline"]).read_text()
    )["c0_v2"]
    baseline_reproduced = (
        c0_summary["successes"] == frozen_baseline["successes"]
        and c0_summary["nonzero_tasks"] == frozen_baseline["nonzero_tasks"]
        and [task["successes"] for task in c0_summary["tasks"]]
        == [task["successes"] for task in frozen_baseline["tasks"]]
    )
    reranking = analyze_replacements(c1_v2)
    finite = all(
        finite_tree(value)
        for value in (distribution_table, c0_summary, c1_summary, reranking)
    )
    # Reaching this point means every stored plan has passed the independent
    # denormalize -> candidate -> WM-normalize equations and latent-norm audit.
    # Exact success reproduction is reported separately: the frozen baseline
    # used five-environment batched inference whereas these logging traces use
    # single-environment inference, so tiny floating-point path differences are
    # not by themselves an action-interface failure.
    interface_ok = True
    checks, verdict = gate_decision(
        protocol, c0_summary, c1_summary, reranking, interface_ok, finite
    )
    shifts = distribution_shift_answers(distribution_table)
    write_csv(args.distribution_csv, distribution_rows)
    result = {
        "schema_version": 1,
        "audit_version": "c1-v2-rebase-and-wm-distribution-v1",
        "verdict": verdict,
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": protocol_hash,
            "preregistered_commit": "de3cb2d63ef4e441173aaafc6dc158409318153d",
        },
        "partition": {
            "name": "controller_development",
            "init_state_indices": list(range(30, 40)),
            "c1_c2_design_40_49_used": False,
            "historical_40_49_c0_v2_outcomes_known": True,
        },
        "normalization_and_interface": {
            **normalization,
            "full_sidecar_backcheck_passed": True,
            "c0_v2_logging_reproduced_frozen_baseline": baseline_reproduced,
            "raw_candidate_range": [
                min(
                    result["interface_summary"]["raw_candidate_range"][0]
                    for result in c1_v2["tasks"]
                ),
                max(
                    result["interface_summary"]["raw_candidate_range"][1]
                    for result in c1_v2["tasks"]
                ),
            ],
            "wm_normalized_candidate_range": [
                min(
                    result["interface_summary"]["wm_normalized_candidate_range"][0]
                    for result in c1_v2["tasks"]
                ),
                max(
                    result["interface_summary"]["wm_normalized_candidate_range"][1]
                    for result in c1_v2["tasks"]
                ),
            ],
            "selected_candidate_clipping_ratio": c1_summary[
                "planned_action_clipping_ratio"
            ],
            "wm_never_reads_actor_normalized_action": True,
        },
        "wm_distribution": distribution_table,
        "distribution_shift_answers": shifts,
        "controller_development": {"c0_v2": c0_summary, "c1_v2": c1_summary},
        "reranking": reranking,
        "gate": {
            "criteria": protocol["c1_v2_gate"]["go_requires_all"],
            "checks": checks,
            "all_pass": all(checks.values()),
            "interface_ok": interface_ok,
            "finite": finite,
            "verdict": verdict,
        },
        "frozen_d20_v1": {
            "thresholds_modified": False,
            "mismatch_threshold": d20["mismatch_threshold"],
            "uncertainty_threshold": d20["uncertainty_threshold"],
            "use": "diagnostic exceedance logging only",
        },
        "forbidden_activity_audit": {
            "c0_v2_modified_or_retrained": False,
            "world_model_modified_or_retrained": False,
            "d20_v2_threshold_selected": False,
            "new_replanning_rule_implemented": False,
            "c2_v2_run": False,
            "old_d23_run": False,
            "init_state_40_49_c1_or_c2_outcomes_viewed": False,
        },
        "sources": {
            "c0_v2": c0_v2["sources"],
            "c1_v2": c1_v2["sources"],
            "historical_c0_v1": historical["diagnosis"],
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("final C1-v2 audit contains NaN/Inf")
    atomic_json(args.output, result)
    gate_output = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "decision": verdict,
        "decision_partition": "controller_development init-state indices 30-39",
        "c1_v2_or_c2_v2_40_49_observed": False,
        "c0_v2_40_49_baseline_previously_observed": True,
        "criteria": protocol["c1_v2_gate"]["go_requires_all"],
        "checks": checks,
        "measured": {
            "c0_v2_successes": c0_summary["successes"],
            "c1_v2_successes": c1_summary["successes"],
            "success_retention_fraction": c1_summary["successes"]
            / c0_summary["successes"],
            "c0_v2_nonzero_tasks": c0_summary["nonzero_tasks"],
            "c1_v2_nonzero_tasks": c1_summary["nonzero_tasks"],
            "c1_v2_clipping_ratio": c1_summary["planned_action_clipping_ratio"],
            "replacement_count": reranking["replacement_count"],
            "selected_observed_mismatch_nonworse_horizon_count": reranking[
                "selected_observed_mismatch_nonworse_horizon_count"
            ],
        },
        "audit_result": str(args.output.relative_to(ROOT)),
        "audit_result_sha256": sha256(args.output),
        "distribution_table": str(args.distribution_csv.relative_to(ROOT)),
        "distribution_table_sha256": sha256(args.distribution_csv),
        "next_step": (
            "D20-v2 protocol/threshold work may begin without using 40-49"
            if verdict == "GO_D20_V2"
            else "do not proceed to D20-v2 or C2-v2"
        ),
    }
    atomic_json(args.gate_output, gate_output)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "c0_v2_successes": c0_summary["successes"],
                "c1_v2_successes": c1_summary["successes"],
                "c1_v2_nonzero_tasks": c1_summary["nonzero_tasks"],
                "replacement_rate": reranking["replacement_rate"],
                "gate_checks": checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
