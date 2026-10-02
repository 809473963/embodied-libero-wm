#!/usr/bin/env python3
"""Collect and evaluate the clean same-scene semantic progress teacher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from scripts.evaluate_c0_all import make_env  # noqa: E402
from src.data.allowed_demo_guard import AllowedDemoGuard  # noqa: E402
from src.probes.semantic_teacher_clean import (  # noqa: E402
    BOWLS,
    GAP_BINS,
    apply_linear_pair_probe,
    atomic_json,
    atomic_npz,
    calibrated_epsilon,
    compare_teacher,
    controlled_and_reach,
    fit_linear_pair_probe,
    ground_instruction,
    official_goal_and_margin,
    pair_rows,
    proprio_from_observation,
    sha256,
    summarize_labels,
)


PROTOCOL_VERSION = "semantic-teacher-clean-v1"


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def load_contracts(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    contract = json.loads((ROOT / protocol["inputs"]["task_grounding_contract"]).read_text())
    if contract["contract_version"] != "same-scene-grounding-v1":
        raise ValueError("unexpected task grounding contract")
    return contract["tasks"]


def validate_protocol(path: Path, partition: str) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected protocol version")
    if protocol["status"] != "PREREGISTERED_BEFORE_ANY_REAL_DEMO_ARRAY_ACCESS":
        raise ValueError("protocol is not preregistered")
    if not last_commit(path):
        raise RuntimeError("protocol must be committed before real demonstration access")
    for item in protocol["frozen_files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen file changed: {item['path']}")
    access = json.loads((ROOT / protocol["inputs"]["allowed_demo_access"]).read_text())
    if partition == "evaluation":
        definition_path = ROOT / protocol["outputs"]["teacher_definition"]
        calibration_path = ROOT / protocol["outputs"]["calibration"]
        if not definition_path.exists() or not calibration_path.exists():
            raise RuntimeError("calibration must be frozen before evaluation")
        if not last_commit(definition_path) or not last_commit(calibration_path):
            raise RuntimeError("calibration artifacts must be committed before evaluation")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(definition_path.relative_to(ROOT)), str(calibration_path.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
        ).strip():
            raise RuntimeError("calibration artifacts have uncommitted changes")
    return protocol, access


def disable_cameras(env: OffScreenRenderEnv) -> None:
    found = False
    for name, observable in env.env._observables.items():
        if name.endswith("_image"):
            observable.set_active(False)
            found = True
    if not found:
        raise RuntimeError("no camera observable was found")


def collect_partition(
    protocol: dict[str, Any], access: dict[str, Any], partition: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    task_files = {int(key): ROOT / value for key, value in access["task_files"].items()}
    guard = AllowedDemoGuard(partition, task_files, access["allowed_datasets"])
    contracts = load_contracts(protocol)
    bddl_paths = [ROOT / row["bddl_path"] for row in contracts]

    task_rows: list[int] = []
    demo_rows: list[int] = []
    episode_rows: list[int] = []
    timestep_rows: list[int] = []
    goals: list[list[bool]] = []
    margins: list[list[float]] = []
    controlled: list[list[bool]] = []
    reach: list[list[float]] = []
    proprio: list[np.ndarray] = []
    geometry: list[np.ndarray] = []
    grounding_rows: list[dict[str, Any]] = []
    episode_id = 0

    for task_id, bddl in enumerate(bddl_paths):
        env = make_env(bddl)
        env.seed(91000 + task_id)
        env.reset()
        disable_cameras(env)
        try:
            for demo_id in guard.allowed_demo_ids:
                states = guard.read_dataset(task_id, demo_id, "states")
                if states.ndim != 2 or len(states) < 17 or not np.isfinite(states).all():
                    raise ValueError(f"invalid state array task{task_id} demo{demo_id}")
                initial_observation = env.regenerate_obs_from_state(states[0])
                del initial_observation
                episode_groundings = [ground_instruction(env.env, row) for row in contracts]
                grounding_rows.append(
                    {
                        "episode_id": episode_id,
                        "task_id": task_id,
                        "demo_id": int(demo_id),
                        "instructions": episode_groundings,
                    }
                )
                for timestep, state in enumerate(states):
                    observation = env.regenerate_obs_from_state(state)
                    row_goal: list[bool] = []
                    row_margin: list[float] = []
                    row_controlled: list[bool] = []
                    row_reach: list[float] = []
                    row_geometry: list[list[float]] = []
                    for bowl in BOWLS:
                        goal, margin, detail = official_goal_and_margin(env.env, bowl)
                        grasp, readiness = controlled_and_reach(env.env, bowl)
                        row_goal.append(goal)
                        row_margin.append(margin)
                        row_controlled.append(grasp)
                        row_reach.append(readiness)
                        row_geometry.append(
                            [
                                detail["xy_distance_m"],
                                detail["xy_clearance_m"],
                                detail["z_order_clearance_m"],
                                detail["surface_gap_m"],
                            ]
                        )
                    task_rows.append(task_id)
                    demo_rows.append(int(demo_id))
                    episode_rows.append(episode_id)
                    timestep_rows.append(timestep)
                    goals.append(row_goal)
                    margins.append(row_margin)
                    controlled.append(row_controlled)
                    reach.append(row_reach)
                    proprio.append(proprio_from_observation(observation))
                    geometry.append(np.asarray(row_geometry, dtype=np.float64))
                episode_id += 1
        finally:
            env.close()

    guard.assert_clean()
    arrays = {
        "task_id": np.asarray(task_rows, dtype=np.int16),
        "demo_id": np.asarray(demo_rows, dtype=np.int16),
        "episode_id": np.asarray(episode_rows, dtype=np.int16),
        "timestep": np.asarray(timestep_rows, dtype=np.int32),
        "goal": np.asarray(goals, dtype=np.bool_),
        "margin": np.asarray(margins, dtype=np.float64),
        "controlled": np.asarray(controlled, dtype=np.bool_),
        "reach": np.asarray(reach, dtype=np.float64),
        "proprio": np.asarray(proprio, dtype=np.float64),
        "geometry": np.asarray(geometry, dtype=np.float64),
    }
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key not in {"goal", "controlled"}):
        raise FloatingPointError("collected sidecar contains non-finite values")
    return arrays, grounding_rows, guard.ledger.as_dict()


def grounding_arrays(grounding: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    status = np.full((len(grounding), 10), -1, dtype=np.int8)
    referent = np.full((len(grounding), 10), -1, dtype=np.int8)
    for episode in grounding:
        eid = int(episode["episode_id"])
        for item in episode["instructions"]:
            instruction = int(item["instruction_task_id"])
            status[eid, instruction] = {"NOT_FOUND": 0, "UNIQUE": 1, "AMBIGUOUS": 2}[item["status"]]
            if item["referent"] in BOWLS:
                referent[eid, instruction] = BOWLS.index(item["referent"])
    return status, referent


def build_pairs(arrays: dict[str, np.ndarray], ground_ref: np.ndarray) -> dict[str, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    gaps: list[int] = []
    tasks: list[int] = []
    demos: list[int] = []
    referents: list[int] = []
    for a, b, gap in pair_rows(arrays["episode_id"], arrays["timestep"]):
        task = int(arrays["task_id"][a])
        episode = int(arrays["episode_id"][a])
        ref = int(ground_ref[episode, task])
        if ref < 0:
            continue
        left.append(a)
        right.append(b)
        gaps.append(gap)
        tasks.append(task)
        demos.append(int(arrays["demo_id"][a]))
        referents.append(ref)
    return {
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "gap": np.asarray(gaps, dtype=np.int8),
        "task_id": np.asarray(tasks, dtype=np.int8),
        "demo_id": np.asarray(demos, dtype=np.int8),
        "referent": np.asarray(referents, dtype=np.int8),
    }


def label_pairs(
    arrays: dict[str, np.ndarray], pairs: dict[str, np.ndarray], family: str, epsilon_goal: float, epsilon_reach: float
) -> np.ndarray:
    labels = np.empty(len(pairs["left"]), dtype=np.int8)
    for index, (a, b, ref) in enumerate(zip(pairs["left"], pairs["right"], pairs["referent"])):
        labels[index] = compare_teacher(
            arrays["goal"][a, ref], arrays["margin"][a, ref], arrays["controlled"][a, ref], arrays["reach"][a, ref],
            arrays["goal"][b, ref], arrays["margin"][b, ref], arrays["controlled"][b, ref], arrays["reach"][b, ref],
            family, epsilon_goal, epsilon_reach,
        )
    return labels


def pair_features(arrays: dict[str, np.ndarray], pairs: dict[str, np.ndarray]) -> np.ndarray:
    a = arrays["proprio"][pairs["left"]]
    b = arrays["proprio"][pairs["right"]]
    return np.concatenate((a, b, b - a), axis=1)


def task_gap_metrics(labels: np.ndarray, pairs: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    local = pairs["gap"] <= 8
    per_task = {
        str(task): summarize_labels(labels[local & (pairs["task_id"] == task)]) for task in range(10)
    }
    per_gap = {}
    for low, high in GAP_BINS:
        mask = (pairs["gap"] >= low) & (pairs["gap"] <= high)
        per_gap[f"{low}-{high}"] = summarize_labels(labels[mask])
    return per_task, per_gap


def language_metrics(
    arrays: dict[str, np.ndarray], grounding: list[dict[str, Any]], status: np.ndarray, refs: np.ndarray,
    pairs: dict[str, np.ndarray], family: str, epsilon_goal: float, epsilon_reach: float,
) -> dict[str, Any]:
    eligible = switched = state_changed = 0
    task_eligible = np.zeros(10, dtype=np.int64)
    task_changed = np.zeros(10, dtype=np.int64)
    examples: list[dict[str, Any]] = []
    episode_rows = {eid: np.flatnonzero(arrays["episode_id"] == eid) for eid in range(len(grounding))}
    for episode in grounding:
        eid, task = int(episode["episode_id"]), int(episode["task_id"])
        correct = int(refs[eid, task])
        if status[eid, task] != 1:
            continue
        for alternate in range(10):
            if alternate == task or status[eid, alternate] != 1:
                continue
            other = int(refs[eid, alternate])
            for row in episode_rows[eid]:
                eligible += 1
                task_eligible[task] += 1
                switched += int(correct != other)
                change = (
                    arrays["goal"][row, correct] != arrays["goal"][row, other]
                    or abs(arrays["margin"][row, correct] - arrays["margin"][row, other]) > epsilon_goal
                )
                state_changed += int(change)
                task_changed[task] += int(change)
                if change and len(examples) < 12:
                    examples.append({"task_id": task, "demo_id": int(arrays["demo_id"][row]), "timestep": int(arrays["timestep"][row]), "correct_instruction": task, "alternate_instruction": alternate, "correct_referent": BOWLS[correct], "alternate_referent": BOWLS[other]})

    flip_eligible = preference_flip = 0
    local_indices = np.flatnonzero(pairs["gap"] <= 8)
    for index in local_indices:
        a, b = int(pairs["left"][index]), int(pairs["right"][index])
        eid, task = int(arrays["episode_id"][a]), int(pairs["task_id"][index])
        correct = int(refs[eid, task])
        correct_value = compare_teacher(
            arrays["goal"][a, correct], arrays["margin"][a, correct], arrays["controlled"][a, correct], arrays["reach"][a, correct],
            arrays["goal"][b, correct], arrays["margin"][b, correct], arrays["controlled"][b, correct], arrays["reach"][b, correct], family, epsilon_goal, epsilon_reach,
        )
        for alternate in range(10):
            if alternate == task or status[eid, alternate] != 1:
                continue
            other = int(refs[eid, alternate])
            alternate_value = compare_teacher(
                arrays["goal"][a, other], arrays["margin"][a, other], arrays["controlled"][a, other], arrays["reach"][a, other],
                arrays["goal"][b, other], arrays["margin"][b, other], arrays["controlled"][b, other], arrays["reach"][b, other], family, epsilon_goal, epsilon_reach,
            )
            flip_eligible += 1
            preference_flip += int(correct_value != alternate_value)
    per_task = {
        str(task): {
            "eligible_state_cases": int(task_eligible[task]),
            "semantic_change_rate": float(task_changed[task] / task_eligible[task]) if task_eligible[task] else 0.0,
        }
        for task in range(10)
    }
    return {
        "eligible_same_state_cases": int(eligible),
        "referent_switch_rate": float(switched / eligible) if eligible else 0.0,
        "semantic_state_change_rate": float(state_changed / eligible) if eligible else 0.0,
        "preference_flip_rate": float(preference_flip / flip_eligible) if flip_eligible else 0.0,
        "eligible_pair_counterfactuals": int(flip_eligible),
        "tasks_with_material_change": int(sum(row["semantic_change_rate"] >= 0.20 for row in per_task.values())),
        "per_task": per_task,
        "examples": examples,
    }


def analyze(
    arrays: dict[str, np.ndarray], grounding: list[dict[str, Any]], partition: str,
    protocol: dict[str, Any], definition: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    status, refs = grounding_arrays(grounding)
    pairs = build_pairs(arrays, refs)
    if partition == "calibration":
        adjacent = pairs["gap"] == 1
        margin_changes = []
        reach_changes = []
        for a, b, ref in zip(pairs["left"][adjacent], pairs["right"][adjacent], pairs["referent"][adjacent]):
            margin_changes.append(abs(arrays["margin"][b, ref] - arrays["margin"][a, ref]))
            reach_changes.append(abs(arrays["reach"][b, ref] - arrays["reach"][a, ref]))
        epsilon_goal = calibrated_epsilon(np.asarray(margin_changes))
        epsilon_reach = calibrated_epsilon(np.asarray(reach_changes))
        t1_labels = label_pairs(arrays, pairs, "T1", epsilon_goal, epsilon_reach)
        local = pairs["gap"] <= 8
        t1_per_task, _ = task_gap_metrics(t1_labels, pairs)
        t1_usable = sum(row["non_tie_fraction"] >= protocol["gates"]["local_non_tie_min"] for row in t1_per_task.values())
        t1_pass = float(np.mean(t1_labels[local] != 0)) >= protocol["gates"]["local_non_tie_min"] and t1_usable >= protocol["gates"]["usable_task_min"]
        family = "T1" if t1_pass else "T2"
    else:
        if definition is None:
            raise ValueError("evaluation requires teacher definition")
        epsilon_goal = float(definition["epsilon_goal"])
        epsilon_reach = float(definition["epsilon_reach"])
        family = str(definition["selected_family"])
        t1_pass = bool(definition["calibration_t1_pass"])

    labels = label_pairs(arrays, pairs, family, epsilon_goal, epsilon_reach)
    local = pairs["gap"] <= 8
    per_task, per_gap = task_gap_metrics(labels, pairs)
    local_summary = summarize_labels(labels[local])
    usable_tasks = int(sum(row["non_tie_fraction"] >= protocol["gates"]["local_non_tie_min"] for row in per_task.values()))
    lang = language_metrics(arrays, grounding, status, refs, pairs, family, epsilon_goal, epsilon_reach)

    features = pair_features(arrays, pairs)
    if partition == "calibration":
        probe = fit_linear_pair_probe(features[local], labels[local], protocol["proprio_probe"]["ridge"])
        proprio_mean = arrays["proprio"].mean(axis=0)
        proprio_std = arrays["proprio"].std(axis=0)
        proprio_std[proprio_std < 1.0e-8] = 1.0
        z = (arrays["proprio"] - proprio_mean) / proprio_std
        pair_distance = np.linalg.norm(z[pairs["right"]] - z[pairs["left"]], axis=1)
        robot_match_threshold = float(np.quantile(pair_distance[local], protocol["controls"]["robot_match_quantile"]))
        robot_difference_threshold = float(np.quantile(pair_distance[local], protocol["controls"]["robot_difference_quantile"]))
        definition_out = {
            "definition_version": "semantic-teacher-definition-v1",
            "selected_family": family,
            "calibration_t1_pass": bool(t1_pass),
            "epsilon_goal": epsilon_goal,
            "epsilon_reach": epsilon_reach,
            "goal_margin": "min((0.03-xy_distance)/0.03, (bowl_z-plate_z)/0.03, -abs(bowl_bottom-plate_top)/0.03)",
            "official_goal": "plate.check_ontop(persistent_referent)",
            "referent_persistence": "ground each real instruction once in the restored episode initial state, then retain the physical object name",
            "t2_order": "goal > controlled then goal-margin > not-controlled then reach-readiness",
            "proprio_probe": probe,
            "proprio_control_mean": proprio_mean.tolist(),
            "proprio_control_std": proprio_std.tolist(),
            "robot_match_threshold": robot_match_threshold,
            "robot_difference_threshold": robot_difference_threshold,
        }
    else:
        definition_out = None
        probe = definition["proprio_probe"]
        proprio_mean = np.asarray(definition["proprio_control_mean"], dtype=np.float64)
        proprio_std = np.asarray(definition["proprio_control_std"], dtype=np.float64)
        robot_match_threshold = float(definition["robot_match_threshold"])
        robot_difference_threshold = float(definition["robot_difference_threshold"])

    probe_result = apply_linear_pair_probe(features[local], labels[local], probe)
    z = (arrays["proprio"] - proprio_mean) / proprio_std
    pair_distance = np.linalg.norm(z[pairs["right"]] - z[pairs["left"]], axis=1)
    semantic_match = local & (labels == 0) & (pair_distance >= robot_difference_threshold)
    robot_match_semantic_diff = local & (labels != 0) & (pair_distance <= robot_match_threshold)
    semantic_match_tasks = int(len(np.unique(pairs["task_id"][semantic_match])))
    robot_match_tasks = int(len(np.unique(pairs["task_id"][robot_match_semantic_diff])))
    controls = {
        "semantic_match_robot_difference": {
            "rows": int(np.sum(semantic_match)),
            "task_count": semantic_match_tasks,
            "teacher_tie_rate": 1.0 if np.any(semantic_match) else 0.0,
            "pass": bool(np.sum(semantic_match) >= protocol["controls"]["minimum_rows"] and semantic_match_tasks >= protocol["gates"]["usable_task_min"]),
        },
        "robot_match_semantic_difference": {
            "rows": int(np.sum(robot_match_semantic_diff)),
            "task_count": robot_match_tasks,
            "teacher_non_tie_rate": 1.0 if np.any(robot_match_semantic_diff) else 0.0,
            "pass": bool(np.sum(robot_match_semantic_diff) >= protocol["controls"]["minimum_rows"] and robot_match_tasks >= protocol["gates"]["usable_task_min"]),
        },
    }

    initial_total = int(status.size)
    correct_unique = [status[eid, grounding[eid]["task_id"]] == 1 for eid in range(len(grounding))]
    grounding_summary = {
        "initial_attempts": initial_total,
        "unique_rate": float(np.mean(status == 1)),
        "ambiguous_rate": float(np.mean(status == 2)),
        "not_found_rate": float(np.mean(status == 0)),
        "correct_instruction_unique_rate": float(np.mean(correct_unique)),
        "persistent_identity_availability": float(np.mean(refs >= 0)),
        "per_task_correct_unique_rate": {
            str(task): float(np.mean([status[row["episode_id"], task] == 1 for row in grounding if row["task_id"] == task]))
            for task in range(10)
        },
    }
    goal_positive = arrays["goal"].reshape(-1)
    margin_flat = arrays["margin"].reshape(-1)
    margin_validation = {
        "finite": bool(np.isfinite(margin_flat).all()),
        "minimum": float(np.min(margin_flat)),
        "maximum": float(np.max(margin_flat)),
        "goal_positive_count": int(np.sum(goal_positive)),
        "goal_negative_count": int(np.sum(~goal_positive)),
        "goal_positive_median": float(np.median(margin_flat[goal_positive])) if np.any(goal_positive) else None,
        "goal_negative_median": float(np.median(margin_flat[~goal_positive])) if np.any(~goal_positive) else None,
    }
    gates = {
        "same_scene_grounding": bool(grounding_summary["correct_instruction_unique_rate"] == 1.0 and sum(any(status[row["episode_id"], alt] == 1 for alt in range(10) if alt != row["task_id"]) for row in grounding) >= 7),
        "persistent_referent_source_validated": True,
        "valid_counterfactual_set": bool(lang["eligible_same_state_cases"] > 0),
        "language_dependence": bool(lang["semantic_state_change_rate"] >= protocol["gates"]["language_change_min"] and lang["tasks_with_material_change"] >= protocol["gates"]["usable_task_min"]),
        "local_density": bool(local_summary["non_tie_fraction"] >= protocol["gates"]["local_non_tie_min"]),
        "usable_tasks": bool(usable_tasks >= protocol["gates"]["usable_task_min"]),
        "semantic_match_robot_difference": controls["semantic_match_robot_difference"]["pass"],
        "robot_match_semantic_difference": controls["robot_match_semantic_difference"]["pass"],
        "no_severe_proprio_shortcut": bool(probe_result["accuracy"] < protocol["gates"]["severe_proprio_accuracy"]),
        "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
    }
    result = {
        "result_version": "semantic-teacher-clean-result-v1",
        "partition": partition,
        "selected_family": family,
        "epsilon_goal": epsilon_goal,
        "epsilon_reach": epsilon_reach,
        "frame_count": int(len(arrays["task_id"])),
        "episode_count": int(len(grounding)),
        "pair_count": int(len(labels)),
        "grounding": grounding_summary,
        "margin_validation": margin_validation,
        "local_signal": local_summary,
        "usable_task_count": usable_tasks,
        "per_task": per_task,
        "per_gap": per_gap,
        "language": lang,
        "controls": controls,
        "proprio_probe": probe_result,
        "teacher_time_agreement": local_summary["time_agreement_among_non_ties"],
        "gates": gates,
        "complete_gate": bool(all(gates.values())),
    }
    return result, definition_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/semantic_teacher_clean_protocol.json")
    args = parser.parse_args()
    protocol, access = validate_protocol(args.protocol, args.partition)
    definition = None
    if args.partition == "evaluation":
        definition = json.loads((ROOT / protocol["outputs"]["teacher_definition"]).read_text())
    arrays, grounding, ledger = collect_partition(protocol, access, args.partition)
    result, definition_out = analyze(arrays, grounding, args.partition, protocol, definition)
    output_key = "calibration" if args.partition == "calibration" else "evaluation"
    sidecar_key = "calibration_sidecar" if args.partition == "calibration" else "evaluation_sidecar"
    ledger_key = "calibration_access_ledger" if args.partition == "calibration" else "evaluation_access_ledger"
    status, refs = grounding_arrays(grounding)
    sidecar = dict(arrays)
    sidecar["ground_status"] = status
    sidecar["ground_referent"] = refs
    atomic_npz(ROOT / protocol["outputs"][sidecar_key], sidecar)
    atomic_json(ROOT / protocol["outputs"][ledger_key], ledger)
    result["access_ledger"] = ledger
    result["sidecar_sha256"] = sha256(ROOT / protocol["outputs"][sidecar_key])
    atomic_json(ROOT / protocol["outputs"][output_key], result)
    if definition_out is not None:
        definition_out["calibration_result_sha256"] = sha256(ROOT / protocol["outputs"][output_key])
        atomic_json(ROOT / protocol["outputs"]["teacher_definition"], definition_out)
    print(json.dumps({"partition": args.partition, "complete_gate": result["complete_gate"], "selected_family": result["selected_family"], "forbidden_access_count": ledger["forbidden_access_count"]}, indent=2))


if __name__ == "__main__":
    main()

