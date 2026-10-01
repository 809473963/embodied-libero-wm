#!/usr/bin/env python3
"""Run the preregistered expanded one-shot candidate-interface audit."""

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

from src.probes.expanded_candidate_interface import (
    ALLOCATION_ORDER,
    combined_candidates,
    continuous_candidates,
    paired_delta,
    structured_gripper_candidates,
    summarize_interface,
)
from src.probes.planning_architecture import (
    block_bootstrap_draws,
    generate_unit_noise,
    load_offline_context,
    sha256,
)


PROTOCOL_VERSION = "expanded-candidate-interface-design-audit-v1"


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
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"frozen input changed: {record['path']}")
    for record in protocol["implementation"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"implementation changed: {record['path']}")
    return protocol, sha256(path)


def valid(summary: dict[str, Any], rules: dict[str, Any]) -> bool:
    return (
        summary["invalid_candidate_count"] == 0
        and summary["clip_fraction"] <= rules["max_component_clip_fraction"]
        and summary["candidate_clip_fraction"] <= rules["max_candidate_clip_fraction"]
        and summary["boundary_hit_fraction"] <= rules["max_boundary_hit_fraction"]
    )


def summarize_design(
    name: str,
    generated: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
    context: Any,
    demos: tuple[int, ...],
    draws: np.ndarray,
    summaries: dict[str, Any],
    sidecar: dict[str, np.ndarray],
) -> None:
    preclip, candidates, category, forms = generated
    summary, arrays = summarize_interface(
        name, preclip, candidates, category, forms, context, demos, draws
    )
    summaries[name] = summary
    for key, value in arrays.items():
        sidecar[f"{name}__{key}"] = value


def select_calibration(
    summaries: dict[str, Any], sidecar: dict[str, np.ndarray], protocol: dict[str, Any]
) -> dict[str, Any]:
    rule = protocol["selection_rules"]
    validity = protocol["validity"]

    # Scale: maximize mean H1/H2 among valid meaningful N32 screens.  If a
    # lower scale is within one percentage point, prefer it.
    scale_names = [f"SCALE_{str(scale).replace('.', 'p')}_N32" for scale in protocol["candidate_design"]["scales"]]
    eligible_scales = [
        name for name in scale_names
        if valid(summaries[name], validity)
        and summaries[name]["relative_mse_gain"] >= rule["meaningful_h8_gain"]
    ]
    if not eligible_scales:
        raise RuntimeError("no valid meaningful scale on calibration")
    prefix_score = {
        name: 0.5 * (
            summaries[name]["prefix_oracle_relative_gain"]["1"]
            + summaries[name]["prefix_oracle_relative_gain"]["2"]
        ) for name in eligible_scales
    }
    best_prefix = max(prefix_score.values())
    near_best = [name for name in eligible_scales if best_prefix - prefix_score[name] < rule["scale_tie_pp"]]
    selected_scale_name = min(near_best, key=lambda name: float(name.split("_")[1].replace("p", ".")))
    selected_scale = float(selected_scale_name.split("_")[1].replace("p", "."))

    # Bank style: antithetic is the deterministic preference when both H8 and
    # mean H1/H2 differ by less than one point.
    anti = summaries["STYLE_antithetic_N32"]
    indep = summaries["STYLE_independent_N32"]
    anti_prefix = 0.5 * (anti["prefix_oracle_relative_gain"]["1"] + anti["prefix_oracle_relative_gain"]["2"])
    indep_prefix = 0.5 * (indep["prefix_oracle_relative_gain"]["1"] + indep["prefix_oracle_relative_gain"]["2"])
    if abs(anti["relative_mse_gain"] - indep["relative_mse_gain"]) < rule["bank_style_material_pp"] and abs(anti_prefix - indep_prefix) < rule["bank_style_material_pp"]:
        selected_style = "antithetic"
    else:
        selected_style = max(
            ("antithetic", "independent"),
            key=lambda style: (
                0.5 * (summaries[f"STYLE_{style}_N32"]["prefix_oracle_relative_gain"]["1"] + summaries[f"STYLE_{style}_N32"]["prefix_oracle_relative_gain"]["2"]),
                summaries[f"STYLE_{style}_N32"]["relative_mse_gain"],
            ),
        )

    count_names = {count: f"COUNT_N{count}" for count in protocol["candidate_design"]["budgets"]}
    n64 = summaries[count_names[64]]
    small_eligible = []
    for count in (16, 24, 32):
        item = summaries[count_names[count]]
        if (
            valid(item, validity)
            and item["relative_mse_gain"] >= rule["meaningful_h8_gain"]
            and item["relative_mse_gain"] >= rule["n64_retention"] * n64["relative_mse_gain"]
            and item["positive_task_count"] == 10
        ):
            small_eligible.append(count)
    n32 = summaries[count_names[32]]
    h8_delta_64 = n64["relative_mse_gain"] - n32["relative_mse_gain"]
    prefix_delta_64 = max(
        n64["prefix_oracle_relative_gain"][key] - n32["prefix_oracle_relative_gain"][key]
        for key in ("1", "2")
    )
    dimension_delta_64 = max(
        n64["dimension_oracle_relative_gain"][key] - n32["dimension_oracle_relative_gain"][key]
        for key in ("translation", "rotation")
    )
    upsize_material = (
        h8_delta_64 >= rule["doubling_h8_material_pp"]
        or prefix_delta_64 >= rule["doubling_prefix_or_dimension_material_pp"]
        or dimension_delta_64 >= rule["doubling_prefix_or_dimension_material_pp"]
    )
    if small_eligible:
        selected_count = min(small_eligible)
        count_reason = "smallest_le32_reaching_90pct_n64_and_all_tasks"
    elif upsize_material:
        selected_count = 64
        count_reason = "n64_materially_exceeds_n32"
    else:
        selected_count = 32
        count_reason = "n64_not_material_enough_to_pay_double_compute"

    baseline = summaries["ALLOC_G0"]
    allocation_checks: dict[str, Any] = {}
    qualifying: list[int] = []
    for gripper_count in (4, 8, 12):
        name = f"ALLOC_G{gripper_count}"
        item = summaries[name]
        delta = paired_delta(
            item, baseline,
            sidecar[f"{name}__bootstrap_gain"], sidecar["ALLOC_G0__bootstrap_gain"],
        )
        h1 = item["prefix_oracle_relative_gain"]["1"] - baseline["prefix_oracle_relative_gain"]["1"]
        h2 = item["prefix_oracle_relative_gain"]["2"] - baseline["prefix_oracle_relative_gain"]["2"]
        pass_full = delta["point"] >= rule["gripper_h8_increment"] and delta["ci95"][0] > 0
        pass_prefix = 0.5 * (h1 + h2) >= rule["gripper_prefix_increment"] and h1 > 0 and h2 > 0
        passed = valid(item, validity) and (pass_full or pass_prefix)
        allocation_checks[str(gripper_count)] = {
            "h8_delta": delta, "h1_delta": h1, "h2_delta": h2,
            "full_gate": pass_full, "prefix_gate": pass_prefix, "passed": passed,
        }
        if passed:
            qualifying.append(gripper_count)
    if qualifying:
        best_gain = max(summaries[f"ALLOC_G{count}"]["relative_mse_gain"] for count in qualifying)
        near_best = [count for count in qualifying if best_gain - summaries[f"ALLOC_G{count}"]["relative_mse_gain"] < rule["allocation_tie_pp"]]
        selected_gripper = min(near_best)
    else:
        selected_gripper = 0

    selected_name = f"ALLOC_G{selected_gripper}"
    selected = summaries[selected_name]
    return {
        "selected_scale": selected_scale,
        "selected_scale_screen": selected_scale_name,
        "selected_bank_style": selected_style,
        "selected_total_count": selected_count,
        "selected_gripper_count": selected_gripper,
        "selected_continuous_count": selected_count - 1 - selected_gripper,
        "selected_allocation": selected_name,
        "selected_gripper_forms": list(ALLOCATION_ORDER[:selected_gripper]),
        "candidate_order": ["A0", "continuous proposals in deterministic bank order", "structured gripper proposals in frozen form order"],
        "count_reason": count_reason,
        "n32_vs_n64": {
            "h8_delta": h8_delta_64,
            "max_h1_h2_delta": prefix_delta_64,
            "max_translation_rotation_delta": dimension_delta_64,
            "upsize_material": upsize_material,
        },
        "allocation_checks": allocation_checks,
        "calibration_selected_metrics": {
            "h8_gain": selected["relative_mse_gain"],
            "h1_gain": selected["prefix_oracle_relative_gain"]["1"],
            "positive_tasks": selected["positive_task_count"],
        },
        "unique_selection": True,
    }


def build_designs(context: Any, protocol: dict[str, Any], selection: dict[str, Any] | None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    design = protocol["candidate_design"]
    std = np.asarray(design["production_noise_std"], dtype=np.float32)
    anti_unit = generate_unit_noise(len(context.expert), 32, int(design["antithetic_seed_base"]))
    independent_unit = generate_unit_noise(len(context.expert), 63, int(design["independent_seed_base"]))
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}

    for scale in design["scales"]:
        label = str(scale).replace(".", "p")
        output[f"SCALE_{label}_N32"] = continuous_candidates(context.actor_default, anti_unit, 32, std, scale, "antithetic")

    if selection is None:
        # Provisional selection inputs use the scale rule from the scale screen.
        rule = protocol["selection_rules"]
        # Style and count screens are built for every scale/style, and aliases
        # needed by selection are installed after summaries are first computed.
        for scale in design["scales"]:
            label = str(scale).replace(".", "p")
            output[f"STYLE_antithetic_at_{label}"] = continuous_candidates(context.actor_default, anti_unit, 32, std, scale, "antithetic")
            output[f"STYLE_independent_at_{label}"] = continuous_candidates(context.actor_default, independent_unit, 32, std, scale, "independent")
        return output

    scale = float(selection["selected_scale"])
    style = selection["selected_bank_style"]
    unit = anti_unit if style == "antithetic" else independent_unit
    output["STYLE_antithetic_N32"] = continuous_candidates(context.actor_default, anti_unit, 32, std, scale, "antithetic")
    output["STYLE_independent_N32"] = continuous_candidates(context.actor_default, independent_unit, 32, std, scale, "independent")
    for count in design["budgets"]:
        output[f"COUNT_N{count}"] = continuous_candidates(context.actor_default, unit, count, std, scale, style)

    total = int(selection["selected_total_count"])
    for gripper_count in (0, 4, 8, 12):
        output[f"ALLOC_G{gripper_count}"] = combined_candidates(context.actor_default, unit, total, gripper_count, std, scale, style)

    pure = {
        "GRIP_G0_keep": tuple(),
        "GRIP_G1_constants": ("all_minus", "all_plus"),
        "GRIP_G2_switches": tuple(ALLOCATION_ORDER),
        "GRIP_G3_binarized": ("c0_binarized",),
        "GRIP_Gall": ("c0_binarized", *ALLOCATION_ORDER),
    }
    for name, forms in pure.items():
        output[name] = structured_gripper_candidates(context.actor_default, forms)
    return output


def run_calibration(protocol: dict[str, Any], protocol_hash: str, device: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frozen = {record["role"]: ROOT / record["path"] for record in protocol["frozen_inputs"]}
    demos = tuple(protocol["partitions"]["calibration_demo_ids"])
    context = load_offline_context(frozen["d13_store"], frozen["c0_checkpoint"], demos, device=device)
    if len(context.expert) != protocol["expected_sample_counts"]["calibration"]:
        raise ValueError("calibration sample count mismatch")
    draws = block_bootstrap_draws(protocol["statistics"]["bootstrap_replicates"], protocol["statistics"]["bootstrap_seed"])
    base_sidecar: dict[str, np.ndarray] = {
        "task_id": context.task_id, "demo_id": context.demo_id, "start_timestep": context.timestep,
        "expert": context.expert, "actor_default": context.actor_default,
    }
    summaries: dict[str, Any] = {}

    # First pass: fixed N32 scale screens.
    preliminary = build_designs(context, protocol, None)
    for name, generated in preliminary.items():
        summarize_design(name, generated, context, demos, draws, summaries, base_sidecar)
    scale_names = [f"SCALE_{str(scale).replace('.', 'p')}_N32" for scale in protocol["candidate_design"]["scales"]]
    eligible = [name for name in scale_names if valid(summaries[name], protocol["validity"]) and summaries[name]["relative_mse_gain"] >= protocol["selection_rules"]["meaningful_h8_gain"]]
    if not eligible:
        raise RuntimeError("no scale clears preliminary screen")
    prefix = {name: 0.5 * (summaries[name]["prefix_oracle_relative_gain"]["1"] + summaries[name]["prefix_oracle_relative_gain"]["2"]) for name in eligible}
    best = max(prefix.values())
    near = [name for name in eligible if best - prefix[name] < protocol["selection_rules"]["scale_tie_pp"]]
    scale_screen = min(near, key=lambda name: float(name.split("_")[1].replace("p", ".")))
    provisional_scale = float(scale_screen.split("_")[1].replace("p", "."))

    anti_key = f"STYLE_antithetic_at_{str(provisional_scale).replace('.', 'p')}"
    indep_key = f"STYLE_independent_at_{str(provisional_scale).replace('.', 'p')}"
    summaries["STYLE_antithetic_N32"] = summaries[anti_key]
    summaries["STYLE_antithetic_N32"]["name"] = "STYLE_antithetic_N32"
    summaries["STYLE_independent_N32"] = summaries[indep_key]
    summaries["STYLE_independent_N32"]["name"] = "STYLE_independent_N32"
    for key in ("default_error", "oracle_error", "winner", "winner_action", "bootstrap_gain", "winner_category", "category", "gripper_winner"):
        base_sidecar[f"STYLE_antithetic_N32__{key}"] = base_sidecar[f"{anti_key}__{key}"]
        base_sidecar[f"STYLE_independent_N32__{key}"] = base_sidecar[f"{indep_key}__{key}"]

    # Provisional style selection, then build count/allocation/pure-gripper screens.
    anti = summaries["STYLE_antithetic_N32"]
    indep = summaries["STYLE_independent_N32"]
    anti_prefix = 0.5 * (anti["prefix_oracle_relative_gain"]["1"] + anti["prefix_oracle_relative_gain"]["2"])
    indep_prefix = 0.5 * (indep["prefix_oracle_relative_gain"]["1"] + indep["prefix_oracle_relative_gain"]["2"])
    if abs(anti["relative_mse_gain"] - indep["relative_mse_gain"]) < protocol["selection_rules"]["bank_style_material_pp"] and abs(anti_prefix - indep_prefix) < protocol["selection_rules"]["bank_style_material_pp"]:
        provisional_style = "antithetic"
    else:
        provisional_style = "antithetic" if (anti_prefix, anti["relative_mse_gain"]) >= (indep_prefix, indep["relative_mse_gain"]) else "independent"

    provisional = {"selected_scale": provisional_scale, "selected_bank_style": provisional_style, "selected_total_count": 32}
    stage2 = build_designs(context, protocol, provisional)
    # Count screens first, because allocation total depends on the frozen count rule.
    for name, generated in stage2.items():
        if name.startswith(("COUNT_", "GRIP_")):
            summarize_design(name, generated, context, demos, draws, summaries, base_sidecar)
    # Determine provisional count, rebuild allocation screens at that count.
    n64 = summaries["COUNT_N64"]
    eligible_counts = [count for count in (16, 24, 32) if summaries[f"COUNT_N{count}"]["relative_mse_gain"] >= protocol["selection_rules"]["meaningful_h8_gain"] and summaries[f"COUNT_N{count}"]["relative_mse_gain"] >= protocol["selection_rules"]["n64_retention"] * n64["relative_mse_gain"] and summaries[f"COUNT_N{count}"]["positive_task_count"] == 10 and valid(summaries[f"COUNT_N{count}"], protocol["validity"])]
    n32 = summaries["COUNT_N32"]
    upsize = (n64["relative_mse_gain"] - n32["relative_mse_gain"] >= protocol["selection_rules"]["doubling_h8_material_pp"] or max(n64["prefix_oracle_relative_gain"][key] - n32["prefix_oracle_relative_gain"][key] for key in ("1", "2")) >= protocol["selection_rules"]["doubling_prefix_or_dimension_material_pp"] or max(n64["dimension_oracle_relative_gain"][key] - n32["dimension_oracle_relative_gain"][key] for key in ("translation", "rotation")) >= protocol["selection_rules"]["doubling_prefix_or_dimension_material_pp"])
    provisional_count = min(eligible_counts) if eligible_counts else (64 if upsize else 32)
    provisional["selected_total_count"] = provisional_count
    stage3 = build_designs(context, protocol, provisional)
    for name, generated in stage3.items():
        if name.startswith("ALLOC_"):
            summarize_design(name, generated, context, demos, draws, summaries, base_sidecar)

    selection = select_calibration(summaries, base_sidecar, protocol)
    if selection["selected_total_count"] != provisional_count or selection["selected_scale"] != provisional_scale or selection["selected_bank_style"] != provisional_style:
        raise AssertionError("staged selection drift")
    result = result_envelope("calibration", protocol_hash, demos, context, summaries)
    result["selection"] = selection
    return result, base_sidecar


def result_envelope(phase: str, protocol_hash: str, demos: tuple[int, ...], context: Any, summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit": PROTOCOL_VERSION,
        "phase": phase,
        "protocol": {"path": "configs/expanded_candidate_interface_design_protocol.json", "sha256": protocol_hash},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": {
            "demo_ids": list(demos), "sample_count": int(len(context.expert)), "task_count": 10,
            "demo_block_count": 40, "simulator_runs": 0, "reward_done_success_reads": 0,
            "forbidden_demo_access": False,
        },
        "families": summaries,
    }


def run_evaluation(protocol: dict[str, Any], protocol_hash: str, calibration: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frozen = {record["role"]: ROOT / record["path"] for record in protocol["frozen_inputs"]}
    demos = tuple(protocol["partitions"]["evaluation_demo_ids"])
    context = load_offline_context(frozen["d13_store"], frozen["c0_checkpoint"], demos, device=device)
    if len(context.expert) != protocol["expected_sample_counts"]["evaluation"]:
        raise ValueError("evaluation sample count mismatch")
    if not np.all(np.isin(context.expert[:, :, 6], (-1.0, 1.0))):
        raise ValueError("expert gripper convention changed")
    draws = block_bootstrap_draws(protocol["statistics"]["bootstrap_replicates"], protocol["statistics"]["bootstrap_seed"])
    sidecar: dict[str, np.ndarray] = {
        "task_id": context.task_id, "demo_id": context.demo_id, "start_timestep": context.timestep,
        "expert": context.expert, "actor_default": context.actor_default,
    }
    summaries: dict[str, Any] = {}
    designs = build_designs(context, protocol, calibration["selection"])
    for name, generated in designs.items():
        summarize_design(name, generated, context, demos, draws, summaries, sidecar)
    result = result_envelope("evaluation", protocol_hash, demos, context, summaries)
    result["frozen_selection"] = calibration["selection"]
    result["decision"] = make_decision(summaries, sidecar, calibration["selection"], protocol)
    return result, sidecar


def make_decision(summaries: dict[str, Any], sidecar: dict[str, np.ndarray], selection: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    rule = protocol["selection_rules"]
    validity = protocol["validity"]
    final_name = selection["selected_allocation"]
    final = summaries[final_name]
    base = summaries["ALLOC_G0"]
    final_vs_base = paired_delta(final, base, sidecar[f"{final_name}__bootstrap_gain"], sidecar["ALLOC_G0__bootstrap_gain"])
    h1_delta = final["prefix_oracle_relative_gain"]["1"] - base["prefix_oracle_relative_gain"]["1"]
    h2_delta = final["prefix_oracle_relative_gain"]["2"] - base["prefix_oracle_relative_gain"]["2"]
    gripper_pass = selection["selected_gripper_count"] > 0 and (
        (final_vs_base["point"] >= rule["gripper_h8_increment"] and final_vs_base["ci95"][0] > 0)
        or (0.5 * (h1_delta + h2_delta) >= rule["gripper_prefix_increment"] and h1_delta > 0 and h2_delta > 0)
    )
    full_gate = (
        valid(final, validity)
        and final["relative_mse_gain"] >= rule["meaningful_h8_gain"]
        and final["relative_mse_gain_ci95"][0] > 0
        and final["positive_task_count"] >= rule["positive_task_min"]
        and final["prefix_oracle_relative_gain"]["1"] >= rule["h1_gain_min"]
    )
    competing = {
        name: item for name, item in summaries.items()
        if name.startswith(("COUNT_", "ALLOC_")) and item["candidate_count"] <= final["candidate_count"]
    }
    dominators = [
        name for name, item in competing.items()
        if name != final_name
        and item["relative_mse_gain"] >= final["relative_mse_gain"]
        and item["prefix_oracle_relative_gain"]["1"] >= final["prefix_oracle_relative_gain"]["1"]
        and (item["relative_mse_gain"] > final["relative_mse_gain"] or item["prefix_oracle_relative_gain"]["1"] > final["prefix_oracle_relative_gain"]["1"])
    ]
    pareto = not dominators
    interface_valid = full_gate and pareto and selection["unique_selection"]

    labels: list[str] = []
    if gripper_pass:
        labels.append("STRUCTURED_GRIPPER_PROPOSALS_VALID")
    elif selection["selected_gripper_count"] > 0 or summaries["GRIP_Gall"]["relative_mse_gain"] >= rule["meaningful_h8_gain"]:
        labels.append("GRIPPER_PROPOSALS_REMAIN_DIAGNOSTIC")

    scale_name = f"SCALE_{str(selection['selected_scale']).replace('.', 'p')}_N32"
    scale_delta = paired_delta(summaries[scale_name], summaries["SCALE_1p0_N32"], sidecar[f"{scale_name}__bootstrap_gain"], sidecar["SCALE_1p0_N32__bootstrap_gain"])
    if selection["selected_scale"] > 1.0 and scale_delta["point"] >= rule["scale_label_increment"] and scale_delta["ci95"][0] > 0:
        labels.append("PRODUCTION_SCALE_TOO_CONSERVATIVE")

    count32 = summaries["COUNT_N32"]
    count64 = summaries["COUNT_N64"]
    n64_h8_delta = count64["relative_mse_gain"] - count32["relative_mse_gain"]
    n64_prefix_delta = max(count64["prefix_oracle_relative_gain"][key] - count32["prefix_oracle_relative_gain"][key] for key in ("1", "2"))
    n64_dimension_delta = max(count64["dimension_oracle_relative_gain"][key] - count32["dimension_oracle_relative_gain"][key] for key in ("translation", "rotation"))
    n64_material = n64_h8_delta >= rule["doubling_h8_material_pp"] or n64_prefix_delta >= rule["doubling_prefix_or_dimension_material_pp"] or n64_dimension_delta >= rule["doubling_prefix_or_dimension_material_pp"]
    if selection["selected_total_count"] <= 32:
        labels.append("N32_SUFFICIENT")
        if not n64_material:
            labels.append("CANDIDATE_COMPUTE_TRADEOFF_LIMIT")
    else:
        labels.append("N64_REQUIRED_FOR_HEADROOM")

    if full_gate:
        labels.append("STRONG_EXPANDED_CANDIDATE_HEADROOM" if final["relative_mse_gain"] >= rule["strong_h8_gain"] else "MEANINGFUL_EXPANDED_CANDIDATE_HEADROOM")
    if interface_valid:
        labels.insert(0, "EXPANDED_CANDIDATE_INTERFACE_VALID")
        authorization = "GO_CANDIDATE_VALUE_AUDIT"
    else:
        labels.append("EXPANDED_CANDIDATE_INTERFACE_UNRESOLVED")
        authorization = "BLOCK_CANDIDATE_VALUE_AUDIT"

    return {
        "final_mechanism_labels": labels,
        "candidate_value_authorization": authorization,
        "spatial_world_model_authorization": "BLOCK_SPATIAL_WM_PROTOTYPE",
        "controller_authorization": False,
        "selected_interface": selection,
        "selected_family": final_name,
        "selected_metrics": {
            "h8_gain": final["relative_mse_gain"], "h8_ci95": final["relative_mse_gain_ci95"],
            "h1_gain": final["prefix_oracle_relative_gain"]["1"],
            "positive_task_count": final["positive_task_count"],
            "valid": valid(final, validity), "pareto_frontier": pareto,
            "pareto_dominators": dominators,
        },
        "structured_gripper_vs_continuous": {
            "h8_delta": final_vs_base, "h1_delta": h1_delta, "h2_delta": h2_delta,
            "passes": gripper_pass,
        },
        "selected_scale_vs_1x": scale_delta,
        "n64_vs_n32": {
            "h8_delta": n64_h8_delta, "max_h1_h2_delta": n64_prefix_delta,
            "max_translation_rotation_delta": n64_dimension_delta, "material": n64_material,
        },
        "complete_gate": interface_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/expanded_candidate_interface_design_protocol.json")
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    result_dir = ROOT / protocol["outputs"]["directory"]

    if args.phase == "calibration":
        result, sidecar = run_calibration(protocol, protocol_hash, args.device)
        sidecar_path = result_dir / "calibration_derived.npz"
        atomic_npz(sidecar_path, sidecar)
        result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
        result_path = result_dir / "calibration_freeze.json"
        atomic_json(result_path, result)
        print(json.dumps({"phase": "calibration", "selection": result["selection"], "result": str(result_path)}, indent=2))
        return

    calibration_path = result_dir / "calibration_freeze.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["phase"] != "calibration" or calibration["protocol"]["sha256"] != protocol_hash:
        raise ValueError("calibration freeze mismatch")
    if sha256(ROOT / calibration["sidecar"]["path"]) != calibration["sidecar"]["sha256"]:
        raise ValueError("calibration sidecar changed")
    result, sidecar = run_evaluation(protocol, protocol_hash, calibration, args.device)
    result["calibration_freeze"] = {"path": str(calibration_path.relative_to(ROOT)), "sha256": sha256(calibration_path)}
    sidecar_path = result_dir / "evaluation_derived.npz"
    atomic_npz(sidecar_path, sidecar)
    result["sidecar"] = {"path": str(sidecar_path.relative_to(ROOT)), "sha256": sha256(sidecar_path)}
    if not finite_tree(result) or not all(np.isfinite(value).all() for value in sidecar.values()):
        raise FloatingPointError("nonfinite result")
    result_path = result_dir / "diagnosis.json"
    atomic_json(result_path, result)
    print(json.dumps({"phase": "evaluation", "labels": result["decision"]["final_mechanism_labels"], "authorization": result["decision"]["candidate_value_authorization"], "result": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
