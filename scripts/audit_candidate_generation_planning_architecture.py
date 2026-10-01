#!/usr/bin/env python3
"""Run the preregistered offline candidate-generation/planning audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probes.planning_architecture import (
    OfflineContext,
    block_bootstrap_draws,
    generate_unit_noise,
    load_offline_context,
    make_local_candidates,
    make_mixed_candidates,
    make_oracle_cem,
    make_proposal_mixture,
    sha256,
    summarize_family,
)


PROTOCOL_VERSION = "candidate-generation-planning-architecture-audit-v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
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


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected protocol")
    if protocol["partitions"] != {
        "calibration_demo_ids": [31, 32, 33, 34],
        "evaluation_demo_ids": [36, 37, 38, 39],
        "historical_reference_demo_ids": [30, 35],
        "forbidden_demo_ids": list(range(40, 50)),
    }:
        raise ValueError("partition contract changed")
    for record in protocol["frozen_inputs"]:
        target = ROOT / record["path"]
        if sha256(target) != record["sha256"]:
            raise ValueError(f"frozen input changed: {record['path']}")
    for record in protocol["implementation"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"implementation changed: {record['path']}")
    return protocol, sha256(path)


def family_candidates(
    context: OfflineContext, protocol: dict[str, Any]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rule = protocol["candidate_families"]
    std = np.asarray(rule["production_noise_std"], dtype=np.float32)
    unit = generate_unit_noise(len(context.expert), 32, int(rule["seed_base"]))
    output: list[tuple[str, np.ndarray, np.ndarray]] = []

    p0_pre, p0 = make_local_candidates(context.actor_default, unit, 3, std)
    output.append(("P0_N3", p0_pre, p0))
    for count in (8, 16, 32, 64):
        pre, candidate = make_local_candidates(context.actor_default, unit, count, std)
        output.append((f"P1_N{count}", pre, candidate))

    masks = {
        "translation": [1, 1, 1, 0, 0, 0, 0],
        "rotation": [0, 0, 0, 1, 1, 1, 0],
        "translation_rotation": [1, 1, 1, 1, 1, 1, 0],
        "full_non_gripper": [1, 1, 1, 1, 1, 1, 0],
    }
    for label, mask in masks.items():
        pre, candidate = make_local_candidates(
            context.actor_default, unit, 32, std, mask=np.asarray(mask)
        )
        output.append((f"P2_{label}_N32", pre, candidate))
    gripper_std = np.asarray([0, 0, 0, 0, 0, 0, rule["gripper_noise_std"]], dtype=np.float32)
    pre, candidate = make_local_candidates(context.actor_default, unit, 32, gripper_std)
    output.append(("P2_gripper_N32", pre, candidate))

    for label, scale in (("0p5", 0.5), ("1", 1.0), ("2", 2.0)):
        pre, candidate = make_local_candidates(context.actor_default, unit, 32, std, scale=scale)
        output.append((f"P3_scale{label}_N32", pre, candidate))
    pre, candidate = make_mixed_candidates(context.actor_default, unit, std, 32)
    output.append(("P3_mixed_N32", pre, candidate))

    cem = make_oracle_cem(
        context.actor_default,
        context.expert,
        std,
        int(rule["cem_seed_base"]),
        iterations=4,
        count=32,
        elite_count=4,
    )
    for iteration, (pre, candidate) in enumerate(cem, 1):
        output.append((f"P4_iter{iteration}_N{32 * iteration}", pre, candidate))
    pre, candidate = make_proposal_mixture(context.actor_default, unit, std)
    output.append(("P5_mixture_N32", pre, candidate))
    return output


def run_partition(
    phase: str,
    demo_values: tuple[int, ...],
    protocol: dict[str, Any],
    protocol_hash: str,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frozen = {record["role"]: ROOT / record["path"] for record in protocol["frozen_inputs"]}
    context = load_offline_context(
        frozen["d13_store"], frozen["c0_checkpoint"], demo_values, device=device
    )
    expected = protocol["expected_sample_counts"][phase]
    if len(context.expert) != expected:
        raise ValueError(f"{phase} sample count {len(context.expert)} != {expected}")
    draws = block_bootstrap_draws(
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["bootstrap_seed"]),
    )
    summaries: dict[str, Any] = {}
    sidecar: dict[str, np.ndarray] = {
        "task_id": context.task_id,
        "demo_id": context.demo_id,
        "start_timestep": context.timestep,
        "expert": context.expert,
        "actor_default": context.actor_default,
    }
    for name, preclip, candidates in family_candidates(context, protocol):
        summary, arrays = summarize_family(
            name, preclip, candidates, context, demo_values, draws
        )
        summaries[name] = summary
        for key, value in arrays.items():
            sidecar[f"{name}__{key}"] = value
    # Exact aliases are deliberate: 7-D actions have only translation,
    # rotation, and gripper, so translation+rotation == full non-gripper.
    left = sidecar["P2_translation_rotation_N32__oracle_error"]
    right = sidecar["P2_full_non_gripper_N32__oracle_error"]
    if not np.array_equal(left, right):
        raise AssertionError("structured alias failed")
    if not np.array_equal(
        sidecar["P1_N32__oracle_error"], sidecar["P3_scale1_N32__oracle_error"]
    ):
        raise AssertionError("scale-1 alias failed")
    if any(summary["invalid_candidate_count"] for summary in summaries.values()):
        raise FloatingPointError("invalid candidate detected")
    result = {
        "schema_version": 1,
        "audit": PROTOCOL_VERSION,
        "phase": phase,
        "protocol": {"path": protocol["self_path"], "sha256": protocol_hash},
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": {
            "demo_ids": list(demo_values),
            "sample_count": int(len(context.expert)),
            "task_count": 10,
            "demo_block_count": 40,
            "simulator_runs": 0,
            "reward_done_success_reads": 0,
            "forbidden_demo_access": False,
        },
        "families": summaries,
    }
    if not finite_tree(result):
        raise FloatingPointError("nonfinite result tree")
    if not all(np.isfinite(value).all() for value in sidecar.values()):
        raise FloatingPointError("nonfinite sidecar")
    return result, sidecar


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def decision(evaluation: dict[str, Any], sidecar: dict[str, np.ndarray], protocol: dict[str, Any]) -> dict[str, Any]:
    family = evaluation["families"]
    gate = protocol["decision_rules"]

    def complete(name: str, strong: bool = False) -> bool:
        item = family[name]
        threshold = gate["strong_relative_gain"] if strong else gate["meaningful_relative_gain"]
        return (
            item["relative_mse_gain"] >= threshold
            and item["relative_mse_gain_ci95"][0] > 0
            and item["positive_task_count"] >= gate["positive_task_min"]
        )

    def delta(child: str, parent: str) -> dict[str, Any]:
        values = sidecar[f"{child}__bootstrap_gain"] - sidecar[f"{parent}__bootstrap_gain"]
        point = family[child]["relative_mse_gain"] - family[parent]["relative_mse_gain"]
        return {"point": float(point), "ci95": interval(values)}

    one_shot = [
        name for name in family
        if name.startswith(("P1_", "P2_", "P3_", "P5_"))
    ]
    best_one_shot = max(one_shot, key=lambda name: family[name]["relative_mse_gain"])
    expanded = {
        name: {
            "complete": complete(name),
            "delta_vs_p0": delta(name, "P0_N3"),
        }
        for name in one_shot
    }
    qualifying_expanded = [
        name for name, item in expanded.items()
        if item["complete"]
        and item["delta_vs_p0"]["point"] >= gate["expanded_vs_p0_min"]
        and item["delta_vs_p0"]["ci95"][0] > 0
    ]
    iterative_name = "P4_iter4_N128"
    iterative_delta = delta(iterative_name, best_one_shot)
    iterative_significant = (
        complete(iterative_name)
        and iterative_delta["point"] >= gate["iterative_vs_one_shot_min"]
        and iterative_delta["ci95"][0] > 0
    )
    if iterative_significant:
        planner_authorization = "GO_MODEL_BASED_PLANNER_DESIGN"
        case = "C"
    elif qualifying_expanded:
        planner_authorization = "GO_EXPANDED_CANDIDATE_PLANNER_DESIGN"
        case = "B"
    else:
        planner_authorization = "BLOCK_PLANNER_PROTOTYPE"
        case = "A"

    labels: list[str] = []
    if not complete("P0_N3"):
        labels.append("CURRENT_N3_CANDIDATE_SET_UNDERPOWERED")
    n64_delta = delta("P1_N64", "P0_N3")
    if complete("P1_N64") and n64_delta["point"] >= gate["expanded_vs_p0_min"] and n64_delta["ci95"][0] > 0:
        labels.append("MORE_LOCAL_SAMPLES_ADD_HEADROOM")
    else:
        labels.append("LOCAL_CANDIDATE_NEIGHBORHOOD_SATURATED")
    mixed_delta = delta("P3_mixed_N32", "P1_N32")
    if complete("P3_mixed_N32") and mixed_delta["point"] >= gate["family_increment_min"] and mixed_delta["ci95"][0] > 0:
        labels.append("MULTISCALE_SEARCH_ADDS_HEADROOM")
    structured_names = ["P2_translation_N32", "P2_rotation_N32", "P2_full_non_gripper_N32"]
    structured_best = max(structured_names, key=lambda name: family[name]["relative_mse_gain"])
    structured_delta = delta(structured_best, "P1_N32")
    if complete(structured_best) and structured_delta["point"] >= gate["family_increment_min"] and structured_delta["ci95"][0] > 0:
        labels.append("STRUCTURED_ACTION_SEARCH_ADDS_HEADROOM")
    gripper_delta = delta("P2_gripper_N32", "P0_N3")
    if complete("P2_gripper_N32") and gripper_delta["point"] >= gate["family_increment_min"] and gripper_delta["ci95"][0] > 0:
        labels.append("GRIPPER_SEARCH_ADDS_HEADROOM")
    if iterative_significant:
        labels.append("ITERATIVE_ACTION_SEARCH_HAS_ORACLE_VALUE")
        labels.append("C0_SHOULD_BE_PROPOSAL_PRIOR")
        labels.append("ONE_SHOT_RERANKING_ARCHITECTURE_LIMIT")
    elif not any(complete(name) for name in one_shot):
        labels.append("C0_IS_STRONG_LOCAL_OPTIMUM")
    if not any(complete(name, strong=True) for name in family):
        labels.append("CANDIDATE_HEADROOM_REMAINS_LOW")
    if planner_authorization == "BLOCK_PLANNER_PROTOTYPE":
        labels.append("PLANNING_ARCHITECTURE_UNRESOLVED")

    return {
        "case": case,
        "final_mechanism_labels": labels,
        "planner_authorization": planner_authorization,
        "spatial_world_model_authorization": "BLOCK_SPATIAL_WM_PROTOTYPE",
        "controller_authorization": False,
        "meaningful_gate": {
            "relative_gain_min": gate["meaningful_relative_gain"],
            "ci_lower_strictly_positive": True,
            "positive_task_min": gate["positive_task_min"],
        },
        "strong_gate_relative_gain_min": gate["strong_relative_gain"],
        "best_one_shot": best_one_shot,
        "best_one_shot_relative_gain": family[best_one_shot]["relative_mse_gain"],
        "qualifying_expanded_one_shot": qualifying_expanded,
        "iterative_final": iterative_name,
        "iterative_vs_best_one_shot": iterative_delta,
        "iterative_significant": iterative_significant,
        "structured_best": structured_best,
        "structured_vs_p1_n32": structured_delta,
        "mixed_vs_p1_n32": mixed_delta,
        "p1_n64_vs_p0": n64_delta,
        "gripper_vs_p0": gripper_delta,
        "complete_meaningful_families": [name for name in family if complete(name)],
        "complete_strong_families": [name for name in family if complete(name, strong=True)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/candidate_generation_planning_architecture_protocol.json")
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    result_dir = ROOT / protocol["outputs"]["directory"]
    if args.phase == "calibration":
        demos = tuple(protocol["partitions"]["calibration_demo_ids"])
        result, sidecar = run_partition("calibration", demos, protocol, protocol_hash, args.device)
        sidecar_path = result_dir / "calibration_derived.npz"
        atomic_npz(sidecar_path, sidecar)
        result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
        result_path = result_dir / "calibration_freeze.json"
        atomic_json(result_path, result)
        print(json.dumps({"phase": "calibration", "result": str(result_path), "samples": result["scope"]["sample_count"]}, indent=2))
        return

    calibration_path = result_dir / "calibration_freeze.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["protocol"]["sha256"] != protocol_hash or calibration["phase"] != "calibration":
        raise ValueError("calibration freeze does not match protocol")
    if sha256(ROOT / calibration["sidecar"]["path"]) != calibration["sidecar"]["sha256"]:
        raise ValueError("calibration sidecar changed")
    demos = tuple(protocol["partitions"]["evaluation_demo_ids"])
    result, sidecar = run_partition("evaluation", demos, protocol, protocol_hash, args.device)
    result["calibration_freeze"] = {"path": str(calibration_path.relative_to(ROOT)), "sha256": sha256(calibration_path)}
    result["historical_p0_reference"] = {
        "partition": "historical C1-v2.1 demos45-49; not numerically opened by this run",
        "relative_gain": protocol["historical_reference"]["p0_relative_gain"],
        "diagnosis_path": protocol["historical_reference"]["diagnosis_path"],
        "diagnosis_sha256": protocol["historical_reference"]["diagnosis_sha256"],
    }
    result["decision"] = decision(result, sidecar, protocol)
    sidecar_path = result_dir / "evaluation_derived.npz"
    atomic_npz(sidecar_path, sidecar)
    result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
    result_path = result_dir / "diagnosis.json"
    atomic_json(result_path, result)
    print(json.dumps({
        "phase": "evaluation", "result": str(result_path),
        "labels": result["decision"]["final_mechanism_labels"],
        "authorization": result["decision"]["planner_authorization"],
    }, indent=2))


if __name__ == "__main__":
    main()

