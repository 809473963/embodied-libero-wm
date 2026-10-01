#!/usr/bin/env python3
"""Run Stage B: frozen-feature goal-conditioned demonstration-progress audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probes.goal_conditioned_progress import (
    GAP_NAMES,
    LOCAL_GAPS,
    AntisymmetricPairMLP,
    PairData,
    ScalarProgressMLP,
    apply_standardizer,
    binary_metrics,
    bootstrap_accuracy_difference,
    evaluate_probabilities,
    fit_standardizer,
    fit_temperature,
    load_trajectories,
    make_cross_demo_controls,
    make_same_demo_pairs,
    parameter_count,
    predict_pairwise,
    predict_scalar,
    sigmoid,
    train_pairwise,
    train_scalar,
)
from src.probes.planning_architecture import block_bootstrap_draws, sha256


VERSION = "goal-conditioned-pairwise-progress-v1"
CONDITIONS = ("visual_language", "proprio_language", "visual_proprio_language")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict): return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)): return bool(np.isfinite(value))
    return True


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != VERSION:
        raise ValueError("unexpected protocol")
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"frozen hash mismatch: {record['path']}")
    if protocol["partitions"]["forbidden_demo_ids"] != list(range(40, 50)):
        raise ValueError("forbidden partition changed")
    return protocol, sha256(path)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def state_arrays(data: Any, statistics: dict[str, dict[str, np.ndarray]] | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    raw = {
        "visual_language": data.vision,
        "proprio_language": data.proprio,
        "visual_proprio_language": np.concatenate((data.vision, data.proprio), axis=1),
    }
    fitted: dict[str, dict[str, np.ndarray]] = {} if statistics is None else statistics
    output = {}
    for name, value in raw.items():
        if statistics is None:
            mean, std = fit_standardizer(value)
            fitted[name] = {"mean": mean, "std": std}
        output[name] = apply_standardizer(value, fitted[name]["mean"], fitted[name]["std"])
    language_rows = data.language_by_task[data.task_id]
    if statistics is None:
        mean, std = fit_standardizer(language_rows)
        fitted["language"] = {"mean": mean, "std": std}
    output["language"] = apply_standardizer(language_rows, fitted["language"]["mean"], fitted["language"]["std"])
    wrong_rows = data.language_by_task[(data.task_id + 1) % 10]
    output["wrong_language"] = apply_standardizer(wrong_rows, fitted["language"]["mean"], fitted["language"]["std"])
    return output, fitted


def terminal_prototypes(data: Any, tail: int) -> np.ndarray:
    values = []
    for task in range(10):
        selected = []
        for trajectory in data.trajectories:
            if trajectory["task"] == task:
                stop = trajectory["start"] + trajectory["length"]
                selected.append(data.vision[max(trajectory["start"], stop - tail):stop])
        prototype = np.concatenate(selected).mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        values.append(prototype)
    return np.asarray(values, dtype=np.float32)


def prototype_scores(data: Any, prototypes: np.ndarray, wrong: bool = False) -> np.ndarray:
    image = data.vision.astype(np.float64)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    task = (data.task_id + (1 if wrong else 0)) % 10
    return np.sum(image * prototypes[task], axis=1)


def state_score_probability(score: np.ndarray, pairs: PairData, temperature: float) -> np.ndarray:
    return sigmoid((score[pairs.b] - score[pairs.a]) * temperature)


def local_mask(pairs: PairData) -> np.ndarray:
    return np.isin(pairs.gap_bin, [GAP_NAMES.index(name) for name in LOCAL_GAPS])


def forward_gap_pairs(data: Any, gap: int) -> PairData:
    a, b, task, demo = [], [], [], []
    for trajectory in data.trajectories:
        for offset in range(trajectory["length"] - gap):
            a.append(trajectory["start"] + offset)
            b.append(trajectory["start"] + offset + gap)
            task.append(trajectory["task"]); demo.append(trajectory["demo"])
    count = len(a)
    return PairData(
        np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), np.ones(count, dtype=np.float32),
        np.asarray(task, dtype=np.int64), np.asarray(demo, dtype=np.int64),
        np.full(count, gap, dtype=np.int64), np.full(count, -1, dtype=np.int8),
    )


def summarize_models(
    data: Any,
    pairs: PairData,
    controls: dict[str, PairData],
    states: dict[str, np.ndarray],
    prototypes: np.ndarray,
    v1: ScalarProgressMLP,
    v2: dict[str, AntisymmetricPairMLP],
    temperatures: dict[str, float],
    demos: tuple[int, ...],
    draws: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    language, wrong_language = states["language"], states["wrong_language"]
    v0_score = prototype_scores(data, prototypes)
    v0_wrong_score = prototype_scores(data, prototypes, wrong=True)
    v1_score = predict_scalar(v1, states["visual_language"], language)
    v1_wrong_score = predict_scalar(v1, states["visual_language"], wrong_language)
    probabilities: dict[str, np.ndarray] = {
        "V0": state_score_probability(v0_score, pairs, temperatures["V0"]),
        "V1": state_score_probability(v1_score, pairs, temperatures["V1"]),
    }
    wrong_probabilities: dict[str, np.ndarray] = {
        "V0": state_score_probability(v0_wrong_score, pairs, temperatures["V0"]),
        "V1": state_score_probability(v1_wrong_score, pairs, temperatures["V1"]),
    }
    for condition, model in v2.items():
        logits = predict_pairwise(model, states[condition], language, pairs)
        wrong_logits = predict_pairwise(model, states[condition], language, pairs, wrong_language)
        probabilities[f"V2_{condition}"] = sigmoid(logits)
        wrong_probabilities[f"V2_{condition}"] = sigmoid(wrong_logits)

    summaries = {}
    local = local_mask(pairs)
    for name, probability in probabilities.items():
        summary = evaluate_probabilities(probability, pairs, demos, draws)
        wrong = wrong_probabilities[name]
        summary["wrong_language"] = evaluate_probabilities(wrong, pairs, demos, draws)
        correct_flag = ((probability >= 0.5) == pairs.label).astype(np.float64)
        wrong_flag = ((wrong >= 0.5) == pairs.label).astype(np.float64)
        drop, interval = bootstrap_accuracy_difference(correct_flag, wrong_flag, pairs, local, demos, draws)
        summary["wrong_language_local_accuracy_drop"] = drop
        summary["wrong_language_drop_ci95"] = interval
        summaries[name] = summary

    primary = "V2_visual_language"
    primary_flag = ((probabilities[primary] >= 0.5) == pairs.label).astype(np.float64)
    for comparison in ("V1", "V2_proprio_language", "V2_visual_proprio_language"):
        other_flag = ((probabilities[comparison] >= 0.5) == pairs.label).astype(np.float64)
        point, interval = bootstrap_accuracy_difference(primary_flag, other_flag, pairs, local, demos, draws)
        summaries[primary].setdefault("paired_local_differences", {})[comparison] = {
            "accuracy_difference": point, "ci95": interval,
        }

    for name in probabilities:
        matched = controls["matched"]
        mismatched = controls["mismatched"]
        if name == "V0":
            matched_probability = state_score_probability(v0_score, matched, temperatures["V0"])
            mismatched_probability = state_score_probability(v0_score, mismatched, temperatures["V0"])
        elif name == "V1":
            matched_probability = state_score_probability(v1_score, matched, temperatures["V1"])
            mismatched_probability = state_score_probability(v1_score, mismatched, temperatures["V1"])
        else:
            condition = name.removeprefix("V2_")
            matched_probability = sigmoid(predict_pairwise(v2[condition], states[condition], language, matched))
            mismatched_probability = sigmoid(predict_pairwise(v2[condition], states[condition], language, mismatched))
        summaries[name]["cross_demo"] = {
            "matched_progress_mean_abs_probability_from_half": float(np.mean(np.abs(matched_probability - 0.5))),
            "matched_progress_p90_abs_probability_from_half": float(np.quantile(np.abs(matched_probability - 0.5), 0.9)),
            "mismatched_progress": binary_metrics(mismatched_probability, mismatched.label, mismatched.signed_gap),
        }

    for name in probabilities:
        summaries[name]["monotonicity"] = {}
        for gap in (1, 4, 8, 16):
            gap_pairs = forward_gap_pairs(data, gap)
            if name == "V0": probability = state_score_probability(v0_score, gap_pairs, temperatures["V0"])
            elif name == "V1": probability = state_score_probability(v1_score, gap_pairs, temperatures["V1"])
            else:
                condition = name.removeprefix("V2_")
                probability = sigmoid(predict_pairwise(v2[condition], states[condition], language, gap_pairs))
            summaries[name]["monotonicity"][str(gap)] = float(np.mean(probability >= 0.5))

    primary_model = v2["visual_language"]
    reverse_pairs = PairData(pairs.b, pairs.a, 1 - pairs.label, pairs.task_id, pairs.demo_id, -pairs.signed_gap, pairs.gap_bin)
    forward_probability = probabilities[primary]
    reverse_probability = sigmoid(predict_pairwise(primary_model, states["visual_language"], language, reverse_pairs))
    summaries[primary]["symmetry"] = {
        "mean_abs_complement_error": float(np.mean(np.abs(forward_probability + reverse_probability - 1))),
        "max_abs_complement_error": float(np.max(np.abs(forward_probability + reverse_probability - 1))),
    }
    arrays = {
        "task_id": pairs.task_id, "demo_id": pairs.demo_id, "a": pairs.a, "b": pairs.b,
        "label": pairs.label, "signed_gap": pairs.signed_gap, "gap_bin": pairs.gap_bin,
    }
    for name, value in probabilities.items():
        arrays[f"{name}__probability"] = value.astype(np.float32)
        arrays[f"{name}__wrong_language_probability"] = wrong_probabilities[name].astype(np.float32)
    return summaries, arrays


def model_state_payload(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def serialize_statistics(statistics: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, torch.Tensor]]:
    return {name: {field: torch.from_numpy(value) for field, value in values.items()} for name, values in statistics.items()}


def deserialize_statistics(value: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, np.ndarray]]:
    return {name: {field: tensor.numpy() for field, tensor in values.items()} for name, values in value.items()}


def envelope(phase: str, protocol_hash: str, data: Any, demos: tuple[int, ...]) -> dict[str, Any]:
    return {
        "schema_version": 1, "audit": VERSION, "phase": phase,
        "protocol": {"path": "configs/goal_conditioned_pairwise_progress_protocol.json", "sha256": protocol_hash},
        "source_commit": git_head(),
        "scope": {
            "demo_ids": list(demos), "state_count": int(len(data.vision)), "trajectory_count": len(data.trajectories),
            "task_count": 10, "simulator_runs": 0, "reward_done_success_reads": 0,
            "expert_action_inputs": 0, "forbidden_demo_access": False,
            "interpretation": "demonstration-progress preference; not task-optimal value",
        },
    }


def run_calibration(protocol: dict[str, Any], protocol_hash: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    store = ROOT / protocol["frozen_inputs"][0]["path"]
    demos = tuple(protocol["partitions"]["calibration_demo_ids"])
    data = load_trajectories(store, demos)
    pairs = make_same_demo_pairs(data, protocol["pair_construction"]["per_demo_per_bin_forward"])
    controls = make_cross_demo_controls(data, protocol["pair_construction"]["cross_demo_points"])
    states, statistics = state_arrays(data)
    epochs, seed = protocol["training"]["epochs"], protocol["training"]["seed"]
    v1 = train_scalar(states["visual_language"], states["language"], data.progress, epochs, seed)
    v2 = {
        condition: train_pairwise(states[condition], states["language"], pairs, epochs, seed + index + 1)
        for index, condition in enumerate(CONDITIONS)
    }
    prototypes = terminal_prototypes(data, protocol["models"]["V0"]["terminal_tail_states"])
    v0_score = prototype_scores(data, prototypes)
    v1_score = predict_scalar(v1, states["visual_language"], states["language"])
    grid = protocol["probability_calibration"]["temperature_grid"]
    temperatures = {
        "V0": fit_temperature(v0_score[pairs.b] - v0_score[pairs.a], pairs.label, grid),
        "V1": fit_temperature(v1_score[pairs.b] - v1_score[pairs.a], pairs.label, grid),
    }
    draws = block_bootstrap_draws(protocol["statistics"]["bootstrap_replicates"], protocol["statistics"]["bootstrap_seed"])
    summaries, arrays = summarize_models(data, pairs, controls, states, prototypes, v1, v2, temperatures, demos, draws)
    weights = {
        "schema_version": 1, "protocol_sha256": protocol_hash,
        "statistics": serialize_statistics(statistics), "prototypes": torch.from_numpy(prototypes),
        "temperatures": temperatures, "v1": model_state_payload(v1),
        "v2": {name: model_state_payload(model) for name, model in v2.items()},
    }
    result = envelope("calibration", protocol_hash, data, demos)
    result.update({
        "pair_count": len(pairs.a), "pair_counts_by_gap": {
            name: int(np.sum(pairs.gap_bin == index)) for index, name in enumerate(GAP_NAMES)
        },
        "model_parameters": {
            "V1": parameter_count(v1), **{f"V2_{name}": parameter_count(model) for name, model in v2.items()}
        },
        "temperatures": temperatures, "metrics": summaries,
        "frozen_primary_model": "V2_visual_language",
        "wrong_language_rule": "task_id -> (task_id + 1) mod 10",
    })
    return result, arrays, weights


def load_models(weights: dict[str, Any]) -> tuple[ScalarProgressMLP, dict[str, AntisymmetricPairMLP]]:
    v1 = ScalarProgressMLP(512); v1.load_state_dict(weights["v1"]); v1.eval()
    dimensions = {"visual_language": 512, "proprio_language": 12, "visual_proprio_language": 524}
    v2 = {}
    for name, dimension in dimensions.items():
        model = AntisymmetricPairMLP(dimension); model.load_state_dict(weights["v2"][name]); model.eval(); v2[name] = model
    return v1, v2


def decide(result: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    gate = protocol["decision_rules"]
    primary = result["metrics"]["V2_visual_language"]
    v1 = result["metrics"]["V1"]
    proprio = result["metrics"]["V2_proprio_language"]
    pair_difference = primary["paired_local_differences"]["V1"]
    robot_difference = primary["paired_local_differences"]["V2_proprio_language"]
    local_pass = (
        primary["local_le8"]["accuracy"] >= gate["local_accuracy_min"]
        and primary["local_accuracy_ci95"][0] > gate["local_ci_lower_min"]
        and primary["positive_task_count"] >= gate["positive_task_min"]
    )
    language_pass = (
        primary["wrong_language_local_accuracy_drop"] >= gate["wrong_language_drop_min"]
        and primary["wrong_language_drop_ci95"][0] > 0
    )
    severe_robot_shortcut = (
        proprio["local_le8"]["accuracy"] >= gate["robot_shortcut_accuracy_floor"]
        and robot_difference["accuracy_difference"] < gate["robot_shortcut_required_visual_margin"]
    )
    noninferior = pair_difference["ci95"][0] >= -gate["v2_noninferiority_margin"]
    absolute_valid = (
        v1["local_le8"]["accuracy"] >= gate["local_accuracy_min"]
        and v1["local_accuracy_ci95"][0] > gate["local_ci_lower_min"]
    )
    outperforms = pair_difference["accuracy_difference"] >= gate["pairwise_outperformance_min"] and pair_difference["ci95"][0] > 0
    terminal_only = primary["per_gap"]["terminal-anchor"]["accuracy"] >= gate["terminal_accuracy_high"] and not local_pass
    passed = local_pass and language_pass and not severe_robot_shortcut and noninferior
    labels = []
    if local_pass: labels.append("PAIRWISE_PROGRESS_SIGNAL_VALID")
    if absolute_valid: labels.append("ABSOLUTE_PROGRESS_BASELINE_VALID")
    if outperforms: labels.append("PAIRWISE_PROGRESS_OUTPERFORMS_ABSOLUTE")
    if language_pass and local_pass: labels.append("GOAL_CONDITIONED_PROGRESS_VALID")
    if not language_pass: labels.append("PROGRESS_MODEL_LANGUAGE_INSENSITIVE")
    if severe_robot_shortcut: labels.append("PROGRESS_MODEL_ROBOT_PHASE_DOMINATED")
    if terminal_only: labels.append("TERMINAL_PROGRESS_ONLY")
    if passed: labels.append("CANDIDATE_VALUE_REPRESENTATION_VALID")
    else: labels.append("CANDIDATE_VALUE_AUDIT_BLOCKED")
    return {
        "stage": "B", "pass": passed, "final_mechanism_labels": labels,
        "world_model_interface_authorization": "GO_WORLD_MODEL_INTERFACE_AUDIT" if passed else "BLOCK_WORLD_MODEL_INTERFACE_AUDIT",
        "spatial_world_model_authorization": "BLOCK_SPATIAL_WM_PROTOTYPE",
        "controller_authorization": False, "selected_value_model": "V2_visual_language" if passed else None,
        "gates": {
            "local_pair": local_pass, "language": language_pass, "severe_robot_shortcut": severe_robot_shortcut,
            "pairwise_noninferior_to_absolute": noninferior, "absolute_baseline_valid": absolute_valid,
            "pairwise_outperforms_absolute": outperforms, "terminal_only": terminal_only,
        },
    }


def run_evaluation(protocol: dict[str, Any], protocol_hash: str, weights: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    store = ROOT / protocol["frozen_inputs"][0]["path"]
    demos = tuple(protocol["partitions"]["evaluation_demo_ids"])
    data = load_trajectories(store, demos)
    pairs = make_same_demo_pairs(data, protocol["pair_construction"]["per_demo_per_bin_forward"])
    controls = make_cross_demo_controls(data, protocol["pair_construction"]["cross_demo_points"])
    statistics = deserialize_statistics(weights["statistics"])
    states, _ = state_arrays(data, statistics)
    v1, v2 = load_models(weights)
    prototypes = weights["prototypes"].numpy()
    draws = block_bootstrap_draws(protocol["statistics"]["bootstrap_replicates"], protocol["statistics"]["bootstrap_seed"])
    summaries, arrays = summarize_models(data, pairs, controls, states, prototypes, v1, v2, weights["temperatures"], demos, draws)
    result = envelope("evaluation", protocol_hash, data, demos)
    result.update({
        "pair_count": len(pairs.a), "pair_counts_by_gap": {
            name: int(np.sum(pairs.gap_bin == index)) for index, name in enumerate(GAP_NAMES)
        },
        "metrics": summaries,
    })
    result["decision"] = decide(result, protocol)
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/goal_conditioned_pairwise_progress_protocol.json")
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    output = ROOT / protocol["outputs"]["directory"]
    if args.phase == "calibration":
        result, arrays, weights = run_calibration(protocol, protocol_hash)
        weights_path = output / "calibration_weights.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights, weights_path)
        result["weights"] = {"path": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path)}
        sidecar_path = output / "calibration_derived.npz"; atomic_npz(sidecar_path, arrays)
        result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
        result_path = output / "calibration_freeze.json"; atomic_json(result_path, result)
        print(json.dumps({"phase": "calibration", "primary": result["metrics"]["V2_visual_language"]}, indent=2))
        return
    calibration_path = output / "calibration_freeze.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["protocol"]["sha256"] != protocol_hash: raise ValueError("calibration protocol mismatch")
    weights_path = ROOT / calibration["weights"]["path"]
    if sha256(weights_path) != calibration["weights"]["sha256"]: raise ValueError("calibration weights changed")
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    result, arrays = run_evaluation(protocol, protocol_hash, weights)
    result["calibration_freeze"] = {"path": str(calibration_path.relative_to(ROOT)), "sha256": sha256(calibration_path)}
    sidecar_path = output / "evaluation_derived.npz"; atomic_npz(sidecar_path, arrays)
    result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
    if not finite_tree(result) or not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite result")
    result_path = output / "diagnosis.json"; atomic_json(result_path, result)
    print(json.dumps({"phase": "evaluation", "decision": result["decision"], "result": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
