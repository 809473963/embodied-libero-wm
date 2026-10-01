#!/usr/bin/env python3
"""Run Stage A: exact N64 expanded-candidate contract correction restart."""

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

from src.probes.expanded_candidate_contract_restart import (
    build_contract_candidates,
    select_gripper_subset,
    summarize_contract,
)
from src.probes.planning_architecture import (
    block_bootstrap_draws,
    generate_unit_noise,
    load_offline_context,
    sha256,
)


VERSION = "expanded-candidate-contract-restart-v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


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
    if protocol["protocol_version"] != VERSION:
        raise ValueError("unexpected protocol")
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"frozen file changed: {record['path']}")
    if protocol["partitions"]["forbidden_demo_ids"] != list(range(40, 50)):
        raise ValueError("forbidden boundary changed")
    return protocol, sha256(path)


def valid(summary: dict[str, Any], protocol: dict[str, Any]) -> bool:
    gate = protocol["validity"]
    return (
        summary["invalid_candidate_count"] == 0
        and summary["clip_fraction"] <= gate["max_component_clip_fraction"]
        and summary["candidate_clip_fraction"] <= gate["max_candidate_clip_fraction"]
        and summary["boundary_hit_fraction"] <= gate["max_boundary_hit_fraction"]
        and summary["exact_contract"]["pass"]
    )


def envelope(phase: str, protocol_hash: str, demos: tuple[int, ...], context: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit": VERSION,
        "phase": phase,
        "protocol": {"path": "configs/expanded_candidate_contract_restart_protocol.json", "sha256": protocol_hash},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": {
            "demo_ids": list(demos), "sample_count": int(len(context.expert)), "task_count": 10,
            "demo_block_count": 40, "simulator_runs": 0, "reward_done_success_reads": 0,
            "forbidden_demo_access": False,
        },
    }


def run_calibration(protocol: dict[str, Any], protocol_hash: str, device: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    frozen = {record["role"]: ROOT / record["path"] for record in protocol["frozen_inputs"]}
    demos = tuple(protocol["partitions"]["calibration_demo_ids"])
    context = load_offline_context(frozen["d13_store"], frozen["c0_checkpoint"], demos, device=device)
    if len(context.expert) != protocol["expected_sample_counts"]["calibration"]:
        raise ValueError("calibration count mismatch")
    if not np.all(np.isin(context.expert[:, :, 6], (-1.0, 1.0))):
        raise ValueError("expert gripper values are not exactly {-1,+1}")
    design = protocol["candidate_design"]
    std = np.asarray(design["production_noise_std"], dtype=np.float32)
    unit = generate_unit_noise(len(context.expert), 26, int(design["seed_base"]))
    draws = block_bootstrap_draws(protocol["statistics"]["bootstrap_replicates"], protocol["statistics"]["bootstrap_seed"])
    summaries: dict[str, Any] = {}
    sidecar: dict[str, np.ndarray] = {
        "task_id": context.task_id, "demo_id": context.demo_id, "start_timestep": context.timestep,
        "expert": context.expert, "actor_default": context.actor_default,
    }
    form_selection: dict[str, Any] = {}
    for allocation in design["allocations"]:
        continuous_count = int(allocation["continuous"])
        gripper_count = int(allocation["gripper"])
        forms, trace = select_gripper_subset(
            context.actor_default, unit, continuous_count, gripper_count, std,
            float(design["scale"]), context.expert,
        )
        name = allocation["name"]
        generated = build_contract_candidates(
            context.actor_default, unit, continuous_count, forms, std, float(design["scale"])
        )
        summary, arrays = summarize_contract(name, generated, context, demos, draws)
        summaries[name] = summary
        form_selection[name] = {"forms": list(forms), "forward_selection_trace": trace}
        for key, value in arrays.items():
            sidecar[f"{name}__{key}"] = value

    eligible = [name for name, item in summaries.items() if valid(item, protocol)]
    if not eligible:
        selected = None
    else:
        selected = max(
            eligible,
            key=lambda name: (
                summaries[name]["relative_mse_gain"],
                0.5 * (summaries[name]["prefix_oracle_relative_gain"]["1"] + summaries[name]["prefix_oracle_relative_gain"]["2"]),
                -summaries[name]["candidate_clip_fraction"],
                summaries[name]["continuous_candidate_count"],
            ),
        )
    contract = {
        "schema_version": 1,
        "name": "Selected Expanded Candidate Interface v1",
        "status": "CALIBRATION_SELECTED" if selected else "NO_VALID_CONTRACT",
        "protocol_sha256": protocol_hash,
        "selected_family": selected,
        "H": 8,
        "N_total": 64,
        "A0": "frozen clipped C0-v2 action chunk; candidate index 0 exactly once",
        "continuous_alternatives": summaries[selected]["continuous_candidate_count"] if selected else None,
        "continuous_antithetic_pairs": summaries[selected]["exact_contract"]["continuous_pair_count"] if selected else None,
        "continuous_scale": float(design["scale"]),
        "continuous_dimensions": ["translation_0_2", "rotation_3_5"],
        "continuous_gripper": "unchanged frozen C0",
        "temporal_smoothing": design["temporal_kernel"],
        "structured_gripper_alternatives": summaries[selected]["structured_gripper_candidate_count"] if selected else None,
        "structured_gripper_forms": form_selection[selected]["forms"] if selected else [],
        "required_G3": "c0_binarized with zero tie -> positive",
        "gripper_semantics": "negative state / positive state; repository does not establish reliable open/close names",
        "bounds": [-1.0, 1.0],
        "seed_base": int(design["seed_base"]),
        "seed_rule": design["seed_rule"],
        "candidate_order": "A0, then +epsilon_i/-epsilon_i pairs by direction index, then gripper forms in frozen selected order",
        "no_cartesian_product": True,
        "calibration_selection": form_selection,
    }
    result = envelope("calibration", protocol_hash, demos, context)
    result["families"] = summaries
    result["selected_family"] = selected
    result["form_selection"] = form_selection
    return result, sidecar, contract


def run_evaluation(protocol: dict[str, Any], protocol_hash: str, calibration: dict[str, Any], contract: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frozen = {record["role"]: ROOT / record["path"] for record in protocol["frozen_inputs"]}
    demos = tuple(protocol["partitions"]["evaluation_demo_ids"])
    context = load_offline_context(frozen["d13_store"], frozen["c0_checkpoint"], demos, device=device)
    if len(context.expert) != protocol["expected_sample_counts"]["evaluation"]:
        raise ValueError("evaluation count mismatch")
    if not np.all(np.isin(context.expert[:, :, 6], (-1.0, 1.0))):
        raise ValueError("expert gripper values are not exactly {-1,+1}")
    design = protocol["candidate_design"]
    std = np.asarray(design["production_noise_std"], dtype=np.float32)
    unit = generate_unit_noise(len(context.expert), 26, int(design["seed_base"]))
    draws = block_bootstrap_draws(protocol["statistics"]["bootstrap_replicates"], protocol["statistics"]["bootstrap_seed"])
    summaries: dict[str, Any] = {}
    sidecar: dict[str, np.ndarray] = {
        "task_id": context.task_id, "demo_id": context.demo_id, "start_timestep": context.timestep,
        "expert": context.expert, "actor_default": context.actor_default,
    }
    for allocation in design["allocations"]:
        name = allocation["name"]
        forms = tuple(calibration["form_selection"][name]["forms"])
        generated = build_contract_candidates(
            context.actor_default, unit, int(allocation["continuous"]), forms, std, float(design["scale"])
        )
        summary, arrays = summarize_contract(name, generated, context, demos, draws)
        summaries[name] = summary
        for key, value in arrays.items():
            sidecar[f"{name}__{key}"] = value
    result = envelope("evaluation", protocol_hash, demos, context)
    result["families"] = summaries
    result["selected_contract"] = contract
    result["decision"] = decide(result, protocol)
    return result, sidecar


def decide(result: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    selected_name = result["selected_contract"]["selected_family"]
    if selected_name is None:
        passed = False
        selected = None
    else:
        selected = result["families"][selected_name]
        gate = protocol["decision_rules"]
        passed = (
            valid(selected, protocol)
            and selected["relative_mse_gain"] >= gate["h8_gain_min"]
            and selected["relative_mse_gain_ci95"][0] > 0
            and selected["positive_task_count"] >= gate["positive_task_min"]
            and selected["prefix_oracle_relative_gain"]["1"] >= gate["h1_gain_min"]
        )
    if passed:
        labels = [
            "EXPANDED_CANDIDATE_INTERFACE_VALID",
            "STRUCTURED_GRIPPER_PROPOSALS_VALID",
            "N64_REQUIRED_FOR_HEADROOM",
            "STRONG_EXPANDED_CANDIDATE_HEADROOM",
        ]
        authorization = "GO_CANDIDATE_VALUE_AUDIT"
    else:
        labels = ["EXPANDED_CANDIDATE_INTERFACE_UNRESOLVED"]
        authorization = "BLOCK_CANDIDATE_VALUE_AUDIT"
    return {
        "stage": "A",
        "pass": passed,
        "final_mechanism_labels": labels,
        "candidate_value_authorization": authorization,
        "spatial_world_model_authorization": "BLOCK_SPATIAL_WM_PROTOTYPE",
        "controller_authorization": False,
        "selected_family": selected_name,
        "selected_metrics": None if selected is None else {
            "h8_gain": selected["relative_mse_gain"],
            "h8_ci95": selected["relative_mse_gain_ci95"],
            "h1_gain": selected["prefix_oracle_relative_gain"]["1"],
            "positive_task_count": selected["positive_task_count"],
            "numeric_validity": valid(selected, protocol),
            "exact_contract": selected["exact_contract"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/expanded_candidate_contract_restart_protocol.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    output = ROOT / protocol["outputs"]["directory"]
    if args.phase == "calibration":
        result, sidecar, contract = run_calibration(protocol, protocol_hash, args.device)
        contract_path = output / "selected_interface.json"
        atomic_json(contract_path, contract)
        result["selected_interface"] = {"path": str(contract_path.relative_to(ROOT)), "sha256": sha256(contract_path)}
        sidecar_path = output / "calibration_derived.npz"
        atomic_npz(sidecar_path, sidecar)
        result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
        result_path = output / "calibration_freeze.json"
        atomic_json(result_path, result)
        print(json.dumps({"phase": "calibration", "selected": result["selected_family"], "contract": contract}, indent=2))
        return

    calibration_path = output / "calibration_freeze.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["protocol"]["sha256"] != protocol_hash:
        raise ValueError("calibration protocol mismatch")
    if sha256(ROOT / calibration["sidecar"]["path"]) != calibration["sidecar"]["sha256"]:
        raise ValueError("calibration sidecar changed")
    contract_path = ROOT / calibration["selected_interface"]["path"]
    if sha256(contract_path) != calibration["selected_interface"]["sha256"]:
        raise ValueError("selected interface changed")
    contract = json.loads(contract_path.read_text())
    result, sidecar = run_evaluation(protocol, protocol_hash, calibration, contract, args.device)
    result["calibration_freeze"] = {"path": str(calibration_path.relative_to(ROOT)), "sha256": sha256(calibration_path)}
    sidecar_path = output / "evaluation_derived.npz"
    atomic_npz(sidecar_path, sidecar)
    result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
    if not finite_tree(result) or not all(np.isfinite(value).all() for value in sidecar.values()):
        raise FloatingPointError("nonfinite result")
    result_path = output / "diagnosis.json"
    atomic_json(result_path, result)
    print(json.dumps({"phase": "evaluation", "decision": result["decision"], "result": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
