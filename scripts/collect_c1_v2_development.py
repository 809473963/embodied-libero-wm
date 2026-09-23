#!/usr/bin/env python3
"""Collect C0-v2 logging-only and C1-v2 traces on init states 30--39."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scripts.evaluate_c0_all import (
    make_env,
    policy_observation,
    refresh_policy_observation,
    trusted_init_states,
)
from src.policies.c1_v2_rerank_policy import C1V2RerankPolicy, file_sha256


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **arrays)
    os.replace(temporary, path)


def disable_all_camera_observables(env: OffScreenRenderEnv) -> None:
    disabled = []
    for name, observable in env.env._observables.items():
        if name.endswith("_image"):
            observable.set_active(False)
            disabled.append(name)
    if "agentview_image" not in disabled:
        raise KeyError("controller-development environment has no agentview_image")
    if any(name not in {"agentview_image", "robot0_eye_in_hand_image"} for name in disabled):
        raise ValueError(f"unexpected camera observables: {disabled}")


def validate_protocol(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != "c1-v2-rebase-and-wm-distribution-v1":
        raise ValueError("expected c1-v2 rebase protocol")
    source_record = protocol["controller_development"]
    source_path = ROOT / source_record["partition_source"]
    if file_sha256(source_path) != source_record["partition_source_sha256"]:
        raise ValueError("controller-development partition source changed")
    source = json.loads(source_path.read_text())
    partition = source["partitions"][source_record["partition_name"]]
    if partition["init_state_indices"] != list(range(30, 40)):
        raise ValueError("C1-v2 collector is restricted to init-state indices 30-39")
    if file_sha256(ROOT / protocol["c0_v2_freeze"]["checkpoint"]) != protocol[
        "c0_v2_freeze"
    ]["checkpoint_sha256"]:
        raise ValueError("frozen C0-v2 checkpoint changed")
    if file_sha256(ROOT / protocol["c1_v2"]["config"]) != protocol["c1_v2"][
        "config_sha256"
    ]:
        raise ValueError("frozen C1-v2 configuration changed")
    wm = protocol["world_model_freeze"]
    if file_sha256(ROOT / wm["ensemble_config"]) != wm["ensemble_config_sha256"]:
        raise ValueError("frozen ensemble configuration changed")
    d20 = protocol["frozen_d20_v1_diagnostic_only"]
    if file_sha256(ROOT / d20["result"]) != d20["result_sha256"]:
        raise ValueError("frozen D20-v1 result changed")
    return protocol, partition


def append_plan_tensors(
    destination: dict[str, list[np.ndarray]], trace: dict[str, np.ndarray]
) -> None:
    for key in destination:
        value = np.asarray(trace[key])
        if not np.isfinite(value).all():
            raise FloatingPointError(f"plan trace {key} contains NaN/Inf")
        destination[key].append(value)


def summarize_interface(plan_events: list[dict[str, Any]]) -> dict[str, Any]:
    if not plan_events:
        raise ValueError("cannot summarize an empty plan set")

    def aggregate_outside(key_path: tuple[str, ...]) -> dict[str, float | int]:
        records = []
        for event in plan_events:
            value: Any = event
            for key in key_path:
                value = value[key]
            records.append(value)
        elements = sum(int(record["elements"]) for record in records)
        outside = sum(int(record["outside_elements"]) for record in records)
        return {
            "elements": elements,
            "outside_elements": outside,
            "clipped_element_ratio": outside / elements,
            "min": min(float(record["min"]) for record in records),
            "max": max(float(record["max"]) for record in records),
        }

    selected_records = [
        event.get("selected_candidate_preclip", event["interface"]["candidate_preclip"])
        for event in plan_events
    ]
    selected_elements = sum(int(record["elements"]) for record in selected_records)
    selected_outside = sum(int(record["outside_elements"]) for record in selected_records)
    return {
        "actor_normalized_range": [
            min(float(event["interface"]["actor_normalized_range"][0]) for event in plan_events),
            max(float(event["interface"]["actor_normalized_range"][1]) for event in plan_events),
        ],
        "actor_denormalized_preclip": aggregate_outside(
            ("interface", "actor_denormalized_preclip")
        ),
        "all_candidate_preclip": aggregate_outside(("interface", "candidate_preclip")),
        "selected_candidate_preclip": {
            "elements": selected_elements,
            "outside_elements": selected_outside,
            "clipped_element_ratio": selected_outside / selected_elements,
            "min": min(float(record["min"]) for record in selected_records),
            "max": max(float(record["max"]) for record in selected_records),
        },
        "raw_candidate_range": [
            min(float(event["interface"]["raw_candidate_range"][0]) for event in plan_events),
            max(float(event["interface"]["raw_candidate_range"][1]) for event in plan_events),
        ],
        "wm_normalized_candidate_range": [
            min(
                float(event["interface"]["wm_normalized_candidate_range"][0])
                for event in plan_events
            ),
            max(
                float(event["interface"]["wm_normalized_candidate_range"][1])
                for event in plan_events
            ),
        ],
    }


def evaluate_task(
    protocol_path: Path,
    protocol: dict[str, Any],
    partition: dict[str, Any],
    policy: C1V2RerankPolicy,
    controller: str,
    task_id: int,
    artifact_path: Path,
) -> dict[str, Any]:
    suite = benchmark.get_benchmark_dict()[protocol["suite"]]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = trusted_init_states(suite, task_id)
    indices = partition["init_state_indices"]
    seeds = partition["task_seeds"][str(task_id)]
    d20 = protocol["frozen_d20_v1_diagnostic_only"]
    tensor_rows: dict[str, list[np.ndarray]] = {
        "actor_normalized": [],
        "actor_denormalized_preclip": [],
        "candidate_preclip": [],
        "candidates": [],
        "wm_normalized_candidates": [],
        "ensemble_prediction": [],
        "ensemble_uncertainty": [],
    }
    episodes = []
    trajectory_row = 0
    for episode_index, (seed, init_state_index) in enumerate(zip(seeds, indices)):
        env: OffScreenRenderEnv = make_env(bddl)
        try:
            env.seed(int(seed))
            env.reset()
            observation = env.set_init_state(init_states[int(init_state_index)])
            disable_all_camera_observables(env)
            policy.reset(seed=int(seed))
            chunk = np.empty((0, 7), dtype=np.float32)
            chunk_index = 8
            current_prediction: np.ndarray | None = None
            current_uncertainty: np.ndarray | None = None
            selected_candidate = 0
            plan_events: list[dict[str, Any]] = []
            check_events: list[dict[str, Any]] = []
            success = False
            for step in range(1, int(protocol["max_steps"]) + 1):
                if chunk_index >= 8:
                    observation = refresh_policy_observation(env)
                    live = policy_observation(observation)
                    if controller == "c0_v2":
                        chunk, diagnostics = policy.plan_actor_default(live, task.language)
                    else:
                        chunk, diagnostics = policy.plan_chunk(live, task.language)
                    trace = policy._last_plan_tensors
                    if trace is None:
                        raise RuntimeError("policy did not retain plan tensors")
                    append_plan_tensors(tensor_rows, trace)
                    current_prediction = trace["ensemble_prediction"]
                    current_uncertainty = trace["ensemble_uncertainty"]
                    selected_candidate = int(diagnostics["selected_candidate"])
                    event = {
                        **diagnostics,
                        "episode_step_before_chunk": step - 1,
                        "trajectory_row": trajectory_row,
                        "trajectory_artifact": str(artifact_path.relative_to(ROOT)),
                        "executed_actions": 0,
                    }
                    plan_events.append(event)
                    trajectory_row += 1
                    chunk_index = 0
                action = np.asarray(chunk[chunk_index], dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("controller produced an invalid raw physical action")
                observation, _, _, _ = env.step(action)
                chunk_index += 1
                plan_events[-1]["executed_actions"] = chunk_index
                success_now = bool(env.check_success())
                if chunk_index in protocol["check_horizons"]:
                    observation = refresh_policy_observation(env)
                    actual = policy.encode_observation(policy_observation(observation))
                    if current_prediction is None or current_uncertainty is None:
                        raise RuntimeError("missing current WM prediction")
                    horizon_index = chunk_index - 1
                    selected_prediction = torch.from_numpy(
                        current_prediction[selected_candidate, horizon_index]
                    ).view(1, 512).to(policy.device)
                    default_prediction = torch.from_numpy(
                        current_prediction[0, horizon_index]
                    ).view(1, 512).to(policy.device)
                    selected_mismatch = float(
                        1.0
                        - F.cosine_similarity(
                            selected_prediction.float(), actual.float(), dim=-1
                        ).item()
                    )
                    default_mismatch = float(
                        1.0
                        - F.cosine_similarity(
                            default_prediction.float(), actual.float(), dim=-1
                        ).item()
                    )
                    selected_uncertainty = float(
                        current_uncertainty[selected_candidate, horizon_index]
                    )
                    default_uncertainty = float(current_uncertainty[0, horizon_index])
                    values = np.asarray(
                        [
                            selected_mismatch,
                            default_mismatch,
                            selected_uncertainty,
                            default_uncertainty,
                        ],
                        dtype=np.float64,
                    )
                    if not np.isfinite(values).all():
                        raise FloatingPointError("development check contains NaN/Inf")
                    check_events.append(
                        {
                            "plan_index": int(plan_events[-1]["plan_index"]),
                            "trajectory_row": int(plan_events[-1]["trajectory_row"]),
                            "horizon": chunk_index,
                            "episode_step_after_check": step,
                            "selected_candidate": selected_candidate,
                            "replaced_actor_default": selected_candidate != 0,
                            "selected_observed_mismatch": selected_mismatch,
                            "actor_default_observed_mismatch": default_mismatch,
                            "selected_uncertainty": selected_uncertainty,
                            "actor_default_uncertainty": default_uncertainty,
                            "selected_mismatch_exceeded_d20_v1": selected_mismatch
                            > float(d20["mismatch_threshold"]),
                            "selected_uncertainty_exceeded_d20_v1": selected_uncertainty
                            > float(d20["uncertainty_threshold"]),
                        }
                    )
                if success_now:
                    success = True
                    break
        finally:
            env.close()
        episodes.append(
            {
                "episode_index": episode_index,
                "seed": int(seed),
                "init_state_index": int(init_state_index),
                "success": success,
                "steps": step,
                "plan_count": len(plan_events),
                "replacement_count": sum(
                    bool(event["replaced_actor_default"]) for event in plan_events
                ),
                "plan_events": plan_events,
                "check_events": check_events,
            }
        )
        print(
            json.dumps(
                {
                    "controller": controller,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "success": success,
                    "steps": step,
                    "plans": len(plan_events),
                }
            ),
            flush=True,
        )

    arrays = {
        key: np.stack(values).astype(np.float16 if key == "ensemble_prediction" else np.float32)
        for key, values in tensor_rows.items()
    }
    arrays["task_id"] = np.asarray(task_id, dtype=np.int64)
    arrays["controller_code"] = np.asarray(0 if controller == "c0_v2" else 1, dtype=np.int64)
    atomic_npz(artifact_path, arrays)
    artifact_sha = file_sha256(artifact_path)
    plan_events = [event for episode in episodes for event in episode["plan_events"]]
    check_events = [event for episode in episodes for event in episode["check_events"]]
    selected_counts = Counter(int(event["selected_candidate"]) for event in plan_events)
    interface = summarize_interface(plan_events)
    return {
        "schema_version": 1,
        "contract_version": "c1-v2-controller-development-trace-v1",
        "controller": controller,
        "world_model_changes_actions": controller == "c1_v2",
        "world_model_replans": False,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "c0_v2_checkpoint_sha256": protocol["c0_v2_freeze"]["checkpoint_sha256"],
        "c1_v2_config_sha256": protocol["c1_v2"]["config_sha256"],
        "ensemble_config_sha256": protocol["world_model_freeze"]["ensemble_config_sha256"],
        "partition": "controller_development",
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "episodes": len(episodes),
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "success_rate": sum(bool(episode["success"]) for episode in episodes) / len(episodes),
        "mean_steps": float(np.mean([episode["steps"] for episode in episodes])),
        "planning_events": len(plan_events),
        "check_events": len(check_events),
        "selected_candidate_counts": {
            str(index): selected_counts[index] for index in range(3)
        },
        "replacement_count": sum(bool(event["replaced_actor_default"]) for event in plan_events),
        "replacement_rate": float(
            np.mean([bool(event["replaced_actor_default"]) for event in plan_events])
        ),
        "mean_planning_ms": float(np.mean([event["planning_ms"] for event in plan_events])),
        "interface_summary": interface,
        "planned_action_clipping_ratio": interface["selected_candidate_preclip"][
            "clipped_element_ratio"
        ],
        "trajectory_artifact": str(artifact_path.relative_to(ROOT)),
        "trajectory_artifact_sha256": artifact_sha,
        "trajectory_rows": len(plan_events),
        "trajectory_arrays": {key: list(value.shape) for key, value in arrays.items()},
        "complete": True,
        "episode_results": episodes,
    }


def validate_existing(
    path: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    controller: str,
    task_id: int,
) -> None:
    result = json.loads(path.read_text())
    expected = {
        "contract_version": "c1-v2-controller-development-trace-v1",
        "controller": controller,
        "protocol_sha256": file_sha256(protocol_path),
        "c0_v2_checkpoint_sha256": protocol["c0_v2_freeze"]["checkpoint_sha256"],
        "c1_v2_config_sha256": protocol["c1_v2"]["config_sha256"],
        "partition": "controller_development",
        "task_id": task_id,
        "episodes": 10,
        "complete": True,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    artifact = ROOT / result["trajectory_artifact"]
    if not artifact.exists() or file_sha256(artifact) != result["trajectory_artifact_sha256"]:
        mismatches.append("trajectory_artifact_sha256")
    if mismatches:
        raise ValueError(f"incompatible existing trace {path}: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True, choices=("c0_v2", "c1_v2"))
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/c1_v2_rebase_protocol.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/c1_v2/controller_development"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts/stage3/c1_v2_controller_development",
    )
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(10))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol, partition = validate_protocol(args.protocol)
    task_ids = protocol["task_ids"] if args.task_ids is None else args.task_ids
    checkpoint = ROOT / protocol["c0_v2_freeze"]["checkpoint"]
    config = ROOT / protocol["c1_v2"]["config"]
    policy = C1V2RerankPolicy(checkpoint=checkpoint, config=config, device=args.device)
    for task_id in task_ids:
        output_path = args.output_dir / args.controller / f"task_{task_id}.json"
        artifact_path = args.artifact_dir / args.controller / f"task_{task_id}.npz"
        if output_path.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {output_path}; use --resume")
            validate_existing(output_path, args.protocol, protocol, args.controller, task_id)
            print(json.dumps({"resume_skip": str(output_path)}), flush=True)
            continue
        result = evaluate_task(
            args.protocol,
            protocol,
            partition,
            policy,
            args.controller,
            task_id,
            artifact_path,
        )
        atomic_json(output_path, result)
        print(
            json.dumps(
                {
                    "completed": str(output_path),
                    "successes": result["successes"],
                    "plans": result["planning_events"],
                    "replacements": result["replacement_count"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
