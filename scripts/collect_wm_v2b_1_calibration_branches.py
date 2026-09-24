#!/usr/bin/env python3
"""Collect preregistered matched branches for WM-v2B.1 calibration.

Calibration and evaluation partitions are intentionally separate invocations.
The evaluation invocation refuses to run until the calibration coefficients
have been committed to git.
"""

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

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path

from scripts import audit_local_candidate_counterfactual as base
from scripts.evaluate_c0_all import policy_observation, refresh_policy_observation, trusted_init_states


PROTOCOL_VERSION = "wm-v2b-1-differential-physical-calibration-v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def path_last_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT)
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(relative)],
        cwd=ROOT,
        text=True,
    ).strip()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def validate_protocol(path: Path, partition_name: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected calibration protocol version")
    if protocol["status"] != "PREREGISTERED_BEFORE_INIT_31_39_BRANCH_EXECUTION":
        raise ValueError("calibration protocol is not preregistered")
    if partition_name not in {"calibration", "evaluation"}:
        raise ValueError("partition must be calibration or evaluation")
    preregistered_commit = path_last_commit(path)
    if not preregistered_commit:
        raise ValueError("protocol must be committed before branch execution")

    frozen = protocol["frozen_upstream"]
    if subprocess.check_output(
        ["git", "cat-file", "-t", frozen["mediator_annotated_tag"]],
        cwd=ROOT,
        text=True,
    ).strip() != "tag":
        raise ValueError("mediator freeze tag is not annotated")
    if subprocess.check_output(
        ["git", "rev-parse", f"{frozen['mediator_annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() != frozen["mediator_commit"]:
        raise ValueError("mediator freeze tag changed")

    for path_key, hash_key in (
        ("mediator_result", "mediator_result_sha256"),
        ("mediator_report", "mediator_report_sha256"),
        ("historical_branch_result", "historical_branch_result_sha256"),
        ("historical_branch_sidecar", "historical_branch_sidecar_sha256"),
        ("historical_branch_protocol", "historical_branch_protocol_sha256"),
        ("historical_branch_amendment", "historical_branch_amendment_sha256"),
        ("historical_collector", "historical_collector_sha256"),
        ("validation_sidecar", "validation_sidecar_sha256"),
        ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
        ("controller_partition_protocol", "controller_partition_protocol_sha256"),
    ):
        if sha256(ROOT / frozen[path_key]) != frozen[hash_key]:
            raise ValueError(f"frozen source changed: {frozen[path_key]}")

    partition = protocol["partitions"][partition_name]
    expected = {
        "calibration": ([31, 32, 33, 34], [1, 2, 3, 4]),
        "evaluation": ([36, 37, 38, 39], [6, 7, 8, 9]),
    }[partition_name]
    if partition["init_state_indices"] != expected[0] or partition["episode_positions"] != expected[1]:
        raise ValueError("partition identities changed")
    if partition["planning_indices"] != [8] or partition["prefix_steps"] != [64]:
        raise ValueError("planning-point scope changed")
    if partition["branch_points"] != 40 or partition["candidate_branches"] != 120:
        raise ValueError("branch count changed")
    if set(protocol["partitions"]["forbidden_init_state_indices"]) != (
        set(range(30)) | set(range(40, 50))
    ):
        raise ValueError("forbidden init-state boundary changed")

    controller = json.loads((ROOT / frozen["controller_partition_protocol"]).read_text())
    development = controller["partitions"]["controller_development"]
    for task_id in protocol["branch_construction"]["task_ids"]:
        indices = [development["init_state_indices"][position] for position in partition["episode_positions"]]
        seeds = [development["task_seeds"][str(task_id)][position] for position in partition["episode_positions"]]
        expected_seeds = [30000 + 100 * task_id + position for position in partition["episode_positions"]]
        if indices != partition["init_state_indices"] or seeds != expected_seeds:
            raise ValueError(f"task {task_id}: partition seed/index mapping changed")

    base_protocol = json.loads((ROOT / frozen["historical_branch_protocol"]).read_text())
    if base_protocol["scope"]["planning_indices"] != [0, 8]:
        raise ValueError("historical branch construction changed")
    for key in (
        "c0_v2_checkpoint",
        "c0_v2_checkpoint_sha256",
        "c1_v2_config",
        "c1_v2_config_sha256",
        "d15_checkpoints",
        "wm_v2a_checkpoint",
        "wm_v2a_checkpoint_sha256",
        "wm_v2b_checkpoint",
        "wm_v2b_checkpoint_sha256",
    ):
        if key not in base_protocol["frozen_upstream"]:
            raise KeyError(f"historical protocol missing frozen field: {key}")

    artifacts = protocol["artifacts"]
    calibration_freeze = ROOT / artifacts["calibration_freeze"]
    evaluation_sidecar = ROOT / artifacts["evaluation_sidecar"]
    evaluation_manifest = ROOT / artifacts["evaluation_manifest"]
    if partition_name == "calibration":
        if calibration_freeze.exists() or evaluation_sidecar.exists() or evaluation_manifest.exists():
            raise RuntimeError("calibration must run before alpha freeze and evaluation artifacts")
    else:
        if not calibration_freeze.exists():
            raise RuntimeError("evaluation is blocked until calibration alpha freeze exists")
        freeze_commit = path_last_commit(calibration_freeze)
        if not freeze_commit:
            raise RuntimeError("calibration alpha freeze must be committed before evaluation")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(calibration_freeze.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
        ).strip():
            raise RuntimeError("calibration alpha freeze has uncommitted changes")
        freeze = json.loads(calibration_freeze.read_text())
        if freeze["protocol_sha256"] != sha256(path):
            raise RuntimeError("calibration alpha freeze references another protocol")
        calibration_manifest = ROOT / artifacts["calibration_manifest"]
        calibration_sidecar = ROOT / artifacts["calibration_sidecar"]
        if sha256(calibration_manifest) != freeze["calibration_manifest_sha256"]:
            raise RuntimeError("calibration manifest changed after alpha freeze")
        if sha256(calibration_sidecar) != freeze["calibration_sidecar_sha256"]:
            raise RuntimeError("calibration sidecar changed after alpha freeze")
    return protocol, base_protocol, preregistered_commit


def stack_arrays(records: list[dict[str, Any]], collected: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    arrays = {
        key: np.stack(values)
        for key, values in collected.items()
        if key != "prefix_actions"
    }
    maximum_prefix = max(value.shape[0] for value in collected["prefix_actions"])
    prefix = np.zeros((len(records), maximum_prefix, 7), dtype=np.float32)
    prefix_lengths = np.empty(len(records), dtype=np.int64)
    for index, value in enumerate(collected["prefix_actions"]):
        prefix[index, : value.shape[0]] = value
        prefix_lengths[index] = value.shape[0]
    arrays["prefix_actions"] = prefix
    arrays["prefix_lengths"] = prefix_lengths
    for name in ("task_id", "episode_position", "init_state_index", "seed", "planning_index"):
        arrays[name] = np.asarray([record[name] for record in records], dtype=np.int64)
    return arrays


def collect_partition(
    protocol: dict[str, Any],
    base_protocol: dict[str, Any],
    partition_name: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    frozen = base_protocol["frozen_upstream"]
    policy = base.C1V2RerankPolicy(
        checkpoint=ROOT / frozen["c0_v2_checkpoint"],
        device=str(device),
        config=ROOT / frozen["c1_v2_config"],
    )
    v2a_model, v2a_normalization, _ = base.load_single_visual_model(
        ROOT / frozen["wm_v2a_checkpoint"],
        "wm_v2a_local_dynamics_resolution",
        frozen["wm_v2a_checkpoint_sha256"],
        device,
    )
    v2b_model, v2b_normalization, _ = base.load_v2b_model(
        ROOT / frozen["wm_v2b_checkpoint"],
        frozen["wm_v2b_checkpoint_sha256"],
        device,
    )
    d15_state = torch.load(
        ROOT / frozen["d15_checkpoints"][0]["path"],
        map_location="cpu",
        weights_only=False,
    )
    d15_normalization = d15_state["normalization"]
    base.normalization_identity([d15_normalization, v2a_normalization, v2b_normalization])

    partition = protocol["partitions"][partition_name]
    controller = json.loads(
        (ROOT / protocol["frozen_upstream"]["controller_partition_protocol"]).read_text()
    )["partitions"]["controller_development"]
    suite = benchmark.get_benchmark_dict()[protocol["branch_construction"]["suite"]]()
    records: list[dict[str, Any]] = []
    collected: dict[str, list[np.ndarray]] = {}
    planning_index = 8
    for task_id in protocol["branch_construction"]["task_ids"]:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_states = trusted_init_states(suite, task_id)
        for episode_position, init_state_index in zip(
            partition["episode_positions"], partition["init_state_indices"]
        ):
            seed = int(controller["task_seeds"][str(task_id)][episode_position])
            init_state = init_states[init_state_index]
            reference_env, _ = base.create_episode_env(bddl, seed, init_state)
            prefix_rows: list[np.ndarray] = []
            try:
                for plan in range(planning_index + 1):
                    observation = refresh_policy_observation(reference_env)
                    context = policy._actor_context(policy_observation(observation), task.language)
                    plan_seed = policy.candidate_seed + seed * 1_000_003 + plan
                    interface = policy._interface(context["actor_normalized"], plan_seed)
                    candidates = interface["candidates"].detach().float().cpu().numpy().copy()
                    if plan == planning_index:
                        prefix = np.stack(prefix_rows).astype(np.float32)
                        record, arrays = base.collect_branch_point(
                            protocol,
                            policy,
                            v2a_model,
                            v2b_model,
                            d15_normalization,
                            bddl,
                            seed,
                            init_state,
                            task_id,
                            init_state_index,
                            episode_position,
                            planning_index,
                            prefix,
                            task.language,
                            plan_seed,
                        )
                        records.append(record)
                        for key, value in arrays.items():
                            collected.setdefault(key, []).append(value)
                        print(
                            json.dumps(
                                {
                                    "partition": partition_name,
                                    "task_id": task_id,
                                    "init_state_index": init_state_index,
                                    "branch_points": len(records),
                                    "prefix_pass": True,
                                }
                            ),
                            flush=True,
                        )
                    else:
                        for action in candidates[0]:
                            reference_env.step(np.asarray(action, dtype=np.float32))
                            prefix_rows.append(np.asarray(action, dtype=np.float32).copy())
            finally:
                reference_env.close()
            if len(prefix_rows) != 64:
                raise RuntimeError("reference prefix length differs from preregistration")
    if len(records) != partition["branch_points"]:
        raise RuntimeError("partition branch-point count differs from preregistration")
    arrays = stack_arrays(records, collected)
    if not all(
        np.isfinite(value).all()
        for name, value in arrays.items()
        if name not in {"real_rgb", "planning_rgb"}
    ):
        raise FloatingPointError("collected partition contains NaN/Inf")
    return records, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("calibration", "evaluation"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2b_1_differential_calibration_protocol.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol, base_protocol, preregistered_commit = validate_protocol(
        args.protocol, args.partition
    )
    sidecar = ROOT / protocol["artifacts"][f"{args.partition}_sidecar"]
    manifest = ROOT / protocol["artifacts"][f"{args.partition}_manifest"]
    for path in (sidecar, manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        records, arrays = collect_partition(protocol, base_protocol, args.partition, device)
    except base.CounterfactualReplayError as failure:
        blocked = {
            "schema_version": 1,
            "experiment_version": PROTOCOL_VERSION,
            "status": "BLOCKED",
            "partition": args.partition,
            "verdict": "BLOCK_COUNTERFACTUAL_REPLAY",
            "protocol_sha256": sha256(args.protocol),
            "preregistered_commit": preregistered_commit,
            "failure": {"message": str(failure), "prefix_audit": failure.audit},
        }
        atomic_json(manifest, blocked)
        print(json.dumps({"status": "BLOCKED", "failure": str(failure)}), flush=True)
        return 2

    atomic_npz(sidecar, arrays)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "partition": args.partition,
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": sha256(args.protocol),
            "preregistered_commit": preregistered_commit,
            "execution_source_commit": source_commit(),
        },
        "scope": protocol["partitions"][args.partition],
        "branch_identities": records,
        "artifacts": {
            "sidecar": str(sidecar.relative_to(ROOT)),
            "sidecar_sha256": sha256(sidecar),
            "sidecar_arrays": {name: list(value.shape) for name, value in arrays.items()},
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "init_state_outside_partition": False,
            "mid_episode_state_restore": False,
            "success_reward_done_used": False,
            "model_training_or_modification": False,
            "candidate_or_controller_modified": False,
            "evaluation_before_committed_alpha_freeze": False,
            "init_state_40_49_access": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("manifest contains NaN/Inf")
    atomic_json(manifest, result)
    result["artifacts"]["manifest_sha256_before_hash_field"] = sha256(manifest)
    atomic_json(manifest, result)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "partition": args.partition,
                "branch_points": len(records),
                "sidecar_sha256": result["artifacts"]["sidecar_sha256"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
