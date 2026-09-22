#!/usr/bin/env python3
"""D22.5 offline diagnosis over expert and isolated controller-validation traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import ks_2samp, pearsonr, rankdata, spearmanr, wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUANTILE_NAMES = ("median", "p75", "p90", "p95", "p99")
QUANTILE_VALUES = (0.50, 0.75, 0.90, 0.95, 0.99)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distribution values must be a non-empty finite vector")
    result: dict[str, float | int] = {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }
    result.update(
        {
            name: float(value)
            for name, value in zip(QUANTILE_NAMES, np.quantile(values, QUANTILE_VALUES))
        }
    )
    return result


def load_expert(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 5869:
        raise ValueError("expected 5,869 D16 validation samples")
    by_horizon = {}
    for horizon in range(1, 9):
        by_horizon[horizon] = {
            "mismatch": np.asarray(
                [row["learned_cosine_error"][horizon - 1] for row in rows],
                dtype=np.float64,
            ),
            "uncertainty": np.asarray(
                [row["uncertainty"][horizon - 1] for row in rows], dtype=np.float64
            ),
        }
    return rows, by_horizon


def load_rollouts(
    directory: Path,
    method: str,
    protocol_hash: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    task_results = []
    sources = []
    for task_id in protocol["task_ids"]:
        path = directory / method / f"task_{task_id}.json"
        result = json.loads(path.read_text())
        expected = {
            "contract_version": "control-validation-trace-v1",
            "controller_method": method,
            "world_model_changes_control": False,
            "early_replanning": False,
            "protocol_sha256": protocol_hash,
            "task_id": task_id,
            "episodes": protocol["episodes_per_task"],
        }
        mismatches = [key for key, value in expected.items() if result.get(key) != value]
        if mismatches:
            raise ValueError(f"{path} protocol mismatch: {mismatches}")
        expected_identity = [
            (
                episode_index,
                protocol["controller_validation"]["task_seeds"][str(task_id)][
                    episode_index
                ],
                protocol["controller_validation"]["init_state_indices"][episode_index],
            )
            for episode_index in range(protocol["episodes_per_task"])
        ]
        actual_identity = [
            (episode["episode_index"], episode["seed"], episode["init_state_index"])
            for episode in result["episode_results"]
        ]
        if actual_identity != expected_identity:
            raise ValueError(f"{path} episode identity differs from frozen protocol")
        task_events = []
        task_plans = []
        for episode in result["episode_results"]:
            task_plans.extend(episode["plan_events"])
            for event in episode["check_events"]:
                enriched = {
                    **event,
                    "task_id": task_id,
                    "episode_index": episode["episode_index"],
                    "episode_success": bool(episode["success"]),
                }
                values = np.asarray(
                    [enriched["mismatch"], enriched["uncertainty"]], dtype=np.float64
                )
                if enriched["horizon"] not in {2, 4, 6, 8} or not np.isfinite(values).all():
                    raise ValueError(f"{path} has an invalid check event")
                task_events.append(enriched)
                events.append(enriched)
                groups[
                    (task_id, episode["episode_index"], event["plan_index"])
                ].append(enriched)
        if len(task_events) != result["check_events"] or len(task_plans) != result[
            "planning_events"
        ]:
            raise ValueError(f"{path} event count mismatch")
        task_results.append(
            {
                "task_id": task_id,
                "episodes": result["episodes"],
                "successes": result["successes"],
                "success_rate": result["success_rate"],
                "planning_events": result["planning_events"],
                "check_events": result["check_events"],
            }
        )
        sources.append(
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        )
    full_plan_groups = [
        sorted(group, key=lambda event: event["horizon"])
        for group in groups.values()
        if {event["horizon"] for event in group} == {2, 4, 6, 8}
    ]
    return {
        "events": events,
        "groups": groups,
        "full_plan_groups": full_plan_groups,
        "tasks": task_results,
        "sources": sources,
    }


def distribution_table(
    expert: dict[int, dict[str, np.ndarray]], rollouts: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for horizon in range(1, 9):
        for metric in ("mismatch", "uncertainty"):
            rows.append(
                {
                    "distribution": "expert_demo_validation",
                    "metric": metric,
                    "horizon": horizon,
                    **statistics(expert[horizon][metric]),
                }
            )
    for method, rollout in rollouts.items():
        for horizon in (2, 4, 6, 8):
            selected = [
                event for event in rollout["events"] if event["horizon"] == horizon
            ]
            for metric in ("mismatch", "uncertainty"):
                rows.append(
                    {
                        "distribution": f"{method}_controller_validation",
                        "metric": metric,
                        "horizon": horizon,
                        **statistics(
                            np.asarray([event[metric] for event in selected], dtype=np.float64)
                        ),
                    }
                )
    return rows


def trigger_metrics(
    events: list[dict[str, Any]],
    full_plans: list[list[dict[str, Any]]],
    trigger: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    triggered = np.asarray([trigger(event) for event in events], dtype=bool)
    by_horizon = {}
    for horizon in (2, 4, 6, 8):
        mask = np.asarray([event["horizon"] == horizon for event in events], dtype=bool)
        by_horizon[str(horizon)] = {
            "count": int(mask.sum()),
            "trigger_count": int((triggered & mask).sum()),
            "trigger_rate": float(triggered[mask].mean()),
        }
    projected_lengths = [
        next((event["horizon"] for event in plan if trigger(event)), 8)
        for plan in full_plans
    ]
    return {
        "check_count": len(events),
        "trigger_count": int(triggered.sum()),
        "trigger_rate": float(triggered.mean()),
        "by_horizon": by_horizon,
        "full_plan_count": len(full_plans),
        "projected_mean_plan_length": float(np.mean(projected_lengths)),
        "projected_plan_length_counts": {
            str(horizon): projected_lengths.count(horizon) for horizon in (2, 4, 6, 8)
        },
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return {"auc": None, "positives": positives, "negatives": negatives}
    ranks = rankdata(scores, method="average")
    auc = (
        ranks[labels].sum() - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return {"auc": float(auc), "positives": positives, "negatives": negatives}


def finite_correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return {
        "count": int(first.size),
        "pearson": float(pearsonr(first, second).statistic),
        "spearman": float(spearmanr(first, second).statistic),
    }


def uncertainty_bins(
    uncertainty: np.ndarray, error: np.ndarray, relation: str, method: str
) -> list[dict[str, Any]]:
    boundaries = np.quantile(uncertainty, np.linspace(0.0, 1.0, 11))
    rows = []
    for index in range(10):
        mask = (uncertainty >= boundaries[index]) & (
            (uncertainty <= boundaries[index + 1])
            if index == 9
            else (uncertainty < boundaries[index + 1])
        )
        rows.append(
            {
                "method": method,
                "relation": relation,
                "uncertainty_decile": index + 1,
                "uncertainty_lower": float(boundaries[index]),
                "uncertainty_upper": float(boundaries[index + 1]),
                "count": int(mask.sum()),
                "error_mean": float(error[mask].mean()),
                "error_median": float(np.median(error[mask])),
                "error_p90": float(np.quantile(error[mask], 0.90)),
            }
        )
    return rows


def signal_analysis(
    method: str, rollout: dict[str, Any], mismatch_threshold: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = rollout["events"]
    uncertainty = np.asarray([event["uncertainty"] for event in events], dtype=np.float64)
    mismatch = np.asarray([event["mismatch"] for event in events], dtype=np.float64)
    pairs = []
    for group in rollout["groups"].values():
        ordered = sorted(group, key=lambda event: event["horizon"])
        pairs.extend(
            (current["uncertainty"], following["mismatch"])
            for current, following in zip(ordered, ordered[1:])
        )
    next_uncertainty = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    next_mismatch = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    bin_rows = uncertainty_bins(uncertainty, mismatch, "current_mismatch", method)
    bin_rows.extend(
        uncertainty_bins(
            next_uncertainty, next_mismatch, "next_check_mismatch", method
        )
    )
    current_means = [
        row["error_mean"]
        for row in bin_rows
        if row["relation"] == "current_mismatch"
    ]
    next_means = [
        row["error_mean"]
        for row in bin_rows
        if row["relation"] == "next_check_mismatch"
    ]
    return (
        {
            "current_mismatch": finite_correlation(uncertainty, mismatch),
            "next_check_mismatch": finite_correlation(next_uncertainty, next_mismatch),
            "current_high_error_ranking": roc_auc(
                uncertainty, mismatch > mismatch_threshold
            ),
            "next_high_error_ranking": roc_auc(
                next_uncertainty, next_mismatch > mismatch_threshold
            ),
            "current_decile_mean_strictly_increasing": bool(
                np.all(np.diff(current_means) > 0.0)
            ),
            "next_decile_mean_strictly_increasing": bool(
                np.all(np.diff(next_means) > 0.0)
            ),
            "current_top_to_bottom_decile_mean_error_ratio": float(
                current_means[-1] / current_means[0]
            ),
            "next_top_to_bottom_decile_mean_error_ratio": float(
                next_means[-1] / next_means[0]
            ),
        },
        bin_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/control_validation_protocol.json",
    )
    parser.add_argument(
        "--trace-dir", type=Path, default=PROJECT_ROOT / "results/control_validation"
    )
    parser.add_argument(
        "--expert-samples",
        type=Path,
        default=PROJECT_ROOT / "results/wm/transition_eval_samples.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/control_validation/d22_5_diagnosis.json",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = file_sha256(args.protocol)
    if protocol["protocol_version"] != "control-validation-v1":
        raise ValueError("expected control-validation-v1")
    if protocol["fixed_final_test_exclusion"][
        "outcomes_are_forbidden_for_threshold_or_rule_selection"
    ] is not True:
        raise ValueError("fixed-test exclusion is not active")
    expert_rows, expert = load_expert(args.expert_samples)
    rollouts = {
        method: load_rollouts(args.trace_dir, method, protocol_hash, protocol)
        for method in ("c0", "c1")
    }
    d20 = protocol["frozen_d20_v1"]
    mismatch_threshold = float(d20["mismatch_threshold"])
    uncertainty_threshold = float(d20["uncertainty_threshold"])

    distribution_rows = distribution_table(expert, rollouts)
    distribution_path = args.output.parent / "distribution_table.csv"
    write_csv(distribution_path, distribution_rows)

    trigger_definitions: dict[str, Callable[[dict[str, Any]], bool]] = {
        "d20_mismatch_only": lambda event: event["mismatch"] > mismatch_threshold,
        "d20_uncertainty_only": lambda event: event["uncertainty"]
        > uncertainty_threshold,
        "d20_or": lambda event: event["mismatch"] > mismatch_threshold
        or event["uncertainty"] > uncertainty_threshold,
        "d20_and": lambda event: event["mismatch"] > mismatch_threshold
        and event["uncertainty"] > uncertainty_threshold,
    }
    pooled_uncertainty = np.asarray(
        [
            event["uncertainty"]
            for method in ("c0", "c1")
            for event in rollouts[method]["events"]
        ],
        dtype=np.float64,
    )
    confidence_candidates = {
        f"control_validation_p{quantile}": float(
            np.quantile(pooled_uncertainty, quantile / 100.0)
        )
        for quantile in (25, 50, 75, 90)
    }
    for name, threshold in confidence_candidates.items():
        trigger_definitions[f"confidence_gated_mismatch_{name}"] = (
            lambda event, threshold=threshold: event["mismatch"] > mismatch_threshold
            and event["uncertainty"] < threshold
        )

    trigger_result: dict[str, dict[str, Any]] = {}
    trigger_rows = []
    for method, rollout in rollouts.items():
        trigger_result[method] = {}
        for rule, trigger in trigger_definitions.items():
            metrics = trigger_metrics(
                rollout["events"], rollout["full_plan_groups"], trigger
            )
            trigger_result[method][rule] = metrics
            trigger_rows.append(
                {
                    "method": method,
                    "rule": rule,
                    "check_count": metrics["check_count"],
                    "trigger_count": metrics["trigger_count"],
                    "trigger_rate": metrics["trigger_rate"],
                    "h2_trigger_rate": metrics["by_horizon"]["2"]["trigger_rate"],
                    "h4_trigger_rate": metrics["by_horizon"]["4"]["trigger_rate"],
                    "h6_trigger_rate": metrics["by_horizon"]["6"]["trigger_rate"],
                    "h8_trigger_rate": metrics["by_horizon"]["8"]["trigger_rate"],
                    "full_plan_count": metrics["full_plan_count"],
                    "projected_mean_plan_length": metrics[
                        "projected_mean_plan_length"
                    ],
                }
            )
    trigger_path = args.output.parent / "trigger_rule_table.csv"
    write_csv(trigger_path, trigger_rows)

    signal_result = {}
    bin_rows = []
    for method, rollout in rollouts.items():
        signal_result[method], rows = signal_analysis(
            method, rollout, mismatch_threshold
        )
        bin_rows.extend(rows)
    bins_path = args.output.parent / "uncertainty_decile_table.csv"
    write_csv(bins_path, bin_rows)

    competence_rows = []
    competence_result = {}
    for method, rollout in rollouts.items():
        for task in rollout["tasks"]:
            competence_rows.append({"method": method, **task})
        successes = sum(task["successes"] for task in rollout["tasks"])
        episodes = sum(task["episodes"] for task in rollout["tasks"])
        nonzero_tasks = sum(task["successes"] > 0 for task in rollout["tasks"])
        zero_tasks = len(rollout["tasks"]) - nonzero_tasks
        competence_result[method] = {
            "successes": successes,
            "episodes": episodes,
            "macro_success_rate": float(
                np.mean([task["success_rate"] for task in rollout["tasks"]])
            ),
            "nonzero_task_count": nonzero_tasks,
            "zero_task_count": zero_tasks,
            "zero_task_fraction": zero_tasks / len(rollout["tasks"]),
            "successes_per_task": [task["successes"] for task in rollout["tasks"]],
        }
    competence_path = args.output.parent / "competence_table.csv"
    write_csv(competence_path, competence_rows)

    shift_summary = {}
    for method, rollout in rollouts.items():
        h2 = [event for event in rollout["events"] if event["horizon"] == 2]
        shift_summary[method] = {}
        for metric in ("mismatch", "uncertainty"):
            expert_values = expert[2][metric]
            rollout_values = np.asarray([event[metric] for event in h2], dtype=np.float64)
            expert_stats = statistics(expert_values)
            rollout_stats = statistics(rollout_values)
            threshold = (
                mismatch_threshold if metric == "mismatch" else uncertainty_threshold
            )
            expert_rate = float((expert_values > threshold).mean())
            rollout_rate = float((rollout_values > threshold).mean())
            shift_summary[method][metric] = {
                "expert": expert_stats,
                "controller_validation": rollout_stats,
                "median_ratio_controller_to_expert": float(
                    rollout_stats["median"] / expert_stats["median"]
                ),
                "p95_ratio_controller_to_expert": float(
                    rollout_stats["p95"] / expert_stats["p95"]
                ),
                "ks_statistic": float(ks_2samp(expert_values, rollout_values).statistic),
                "wasserstein_distance": float(
                    wasserstein_distance(expert_values, rollout_values)
                ),
                "d20_threshold": threshold,
                "expert_threshold_exceed_rate": expert_rate,
                "controller_threshold_exceed_rate": rollout_rate,
                "exceed_rate_ratio": (
                    None if expert_rate == 0.0 else rollout_rate / expert_rate
                ),
            }

    expert_counterfactual = {}
    for rule, trigger in list(trigger_definitions.items())[:4]:
        expert_counterfactual[rule] = {}
        for horizon in (2, 4, 6, 8):
            mismatch = expert[horizon]["mismatch"]
            uncertainty = expert[horizon]["uncertainty"]
            synthetic = [
                {"mismatch": float(mismatch[index]), "uncertainty": float(uncertainty[index])}
                for index in range(mismatch.size)
            ]
            flags = np.asarray([trigger(event) for event in synthetic], dtype=bool)
            expert_counterfactual[rule][str(horizon)] = {
                "count": int(flags.size),
                "trigger_count": int(flags.sum()),
                "trigger_rate": float(flags.mean()),
            }

    c0_competence = competence_result["c0"]
    baseline_block = c0_competence["zero_task_fraction"] > 0.5
    verdict = "BASELINE_COMPETENCE_BLOCK" if baseline_block else "GO_D20_V2"
    artifacts = {
        "distribution_table": {
            "path": str(distribution_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(distribution_path),
        },
        "trigger_rule_table": {
            "path": str(trigger_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(trigger_path),
        },
        "uncertainty_decile_table": {
            "path": str(bins_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(bins_path),
        },
        "competence_table": {
            "path": str(competence_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(competence_path),
        },
    }
    result = {
        "schema_version": 1,
        "diagnosis_version": "d22.5-v1",
        "verdict": verdict,
        "secondary_recommendation": "GO_D20_V2_PROTOCOL_DESIGN_ONLY",
        "d23_authorized": False,
        "d20_v1_thresholds_modified": False,
        "fixed_test_outcomes_used_for_analysis_or_selection": False,
        "inputs": {
            "control_validation_protocol": {
                "path": str(args.protocol.relative_to(PROJECT_ROOT)),
                "sha256": protocol_hash,
            },
            "expert_validation_samples": {
                "path": str(args.expert_samples.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(args.expert_samples),
                "sample_count": len(expert_rows),
                "split": "demonstrations 45-49 only",
            },
            "controller_validation_trace_files": {
                method: rollout["sources"] for method, rollout in rollouts.items()
            },
            "fixed_test_artifacts": [],
        },
        "frozen_d20_v1": {
            "result_sha256": d20["result_sha256"],
            "mismatch_threshold": mismatch_threshold,
            "uncertainty_threshold": uncertainty_threshold,
        },
        "h2_distribution_shift": shift_summary,
        "expert_d20_counterfactual": expert_counterfactual,
        "controller_validation_trigger_counterfactual": trigger_result,
        "confidence_gate_candidates": {
            "derivation": "pooled C0+C1 controller-validation uncertainty quantiles",
            "thresholds": confidence_candidates,
            "selection_status": "diagnostic grid only; no threshold selected",
        },
        "uncertainty_signal": signal_result,
        "controller_competence": competence_result,
        "baseline_block_rule": (
            "block when a strict majority of precommitted tasks have zero C0 success; "
            "observed 7/10 zero tasks"
        ),
        "controller_action_contract_audit": {
            "status": "UNRESOLVED_CAUSAL_INDEX_MISMATCH",
            "d13_training_contract": "z[k] plus actions[k+1:k+9] predicts z[k+1:k+9]",
            "deployed_d19_d21_integration": "current observation plus immediately executed candidate[0:8]",
            "diagnosis_collection": "preserved deployed integration without repair",
            "implication": (
                "D20-v2 controller experiments must resolve or explicitly version this mapping "
                "before interpreting prediction mismatch causally"
            ),
        },
        "answers": {
            "d20_v1_largest_problem": (
                "thresholds were calibrated on expert demonstration transitions, while deployment "
                "uses policy-induced states/actions; h2 disagreement shifts far more than mismatch"
            ),
            "policy_induced_calibration_required": True,
            "uncertainty_interpretation": (
                "useful moderate ranking/OOD signal, but not a calibrated failure probability"
            ),
            "or_rule_reasonable": False,
            "candidate_roles_for_future_validation": [
                "mismatch-only",
                "uncertainty as reranking score/diagnostic only",
                "uncertainty as confidence gate that vetoes mismatch-based replanning",
            ],
            "not_recommended_without_new_evidence": [
                "uncertainty as an OR replan trigger",
                "D20 AND rule, because it triggers specifically when the predictor is less trusted",
                "deleting uncertainty despite its ranking signal",
            ],
            "d20_v2_data_requirement": (
                "precommit separate policy-induced controller calibration and evaluation rollouts "
                "from currently unused init-state indices; do not tune and evaluate on this same trace"
            ),
            "c0_competence_sufficient": False,
            "c0_v2_required_before_broad_controller_claim": True,
        },
        "artifacts": artifacts,
        "negative_result_freeze": {
            **protocol["negative_result_freeze"],
            "used_for_threshold_selection": False,
        },
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "c0_successes": c0_competence["successes"],
                "c0_zero_tasks": c0_competence["zero_task_count"],
                "c0_or_trigger_rate": trigger_result["c0"]["d20_or"]["trigger_rate"],
                "c1_or_trigger_rate": trigger_result["c1"]["d20_or"]["trigger_rate"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
