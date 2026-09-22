#!/usr/bin/env python3
"""Collect isolated C0/C1 rollout traces with a logging-only WM observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scripts.evaluate_c0_all import (
    disable_camera_observable,
    make_env,
    policy_observation,
    refresh_policy_observation,
    trusted_init_states,
)
from src.policies.c1_rerank_policy import C1RerankPolicy


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def encode_image(
    policy: C1RerankPolicy, observation: dict[str, np.ndarray]
) -> torch.Tensor:
    image = np.asarray(observation["agentview_image"], dtype=np.uint8)
    if image.shape != (128, 128, 3):
        raise ValueError(f"expected agentview_image [128,128,3], got {image.shape}")
    pixels = policy.processor(
        images=[np.ascontiguousarray(image)], return_tensors="pt"
    )["pixel_values"].to(policy.device)
    with torch.inference_mode(), policy._autocast():
        latent = policy.clip.get_image_features(pixel_values=pixels)
        latent = latent / latent.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return latent.float()


def predict_chunk(
    policy: C1RerankPolicy,
    z0: torch.Tensor,
    observation: dict[str, np.ndarray],
    language: str,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.shape != (8, 7):
        raise ValueError(f"expected actions [8,7], got {tuple(actions.shape)}")
    proprio = torch.from_numpy(policy._proprio(observation)).view(1, 12).to(policy.device)
    language_feature = policy._language_features(language)
    normalized_proprio = (proprio - policy.wm_proprio_mean) / policy.wm_proprio_std
    normalized_actions = (
        actions.view(1, 8, 7) - policy.wm_action_mean
    ) / policy.wm_action_std
    with torch.inference_mode():
        mean_prediction, uncertainty, _ = policy.ensemble.predict(
            z0.float(), normalized_proprio, language_feature, normalized_actions
        )
    return mean_prediction[0].float(), uncertainty[0].float()


def plan_c0_observed(
    policy: C1RerankPolicy,
    observation: dict[str, np.ndarray],
    language: str,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Reproduce the frozen C0 forward pass and attach logging-only WM output."""

    required = (
        "agentview_image",
        "robot0_eef_pos",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
    )
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"C0 observer input is missing {missing}")
    started = time.perf_counter()
    image = np.asarray(observation["agentview_image"], dtype=np.uint8)
    pixels = policy.processor(
        images=[np.ascontiguousarray(image)], return_tensors="pt"
    )["pixel_values"].to(policy.device)
    proprio = torch.from_numpy(policy._proprio(observation)).view(1, 12).to(policy.device)
    language_feature = policy._language_features(language)
    with torch.inference_mode(), policy._autocast():
        z0 = policy.clip.get_image_features(pixel_values=pixels)
        z0 = z0 / z0.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        actor_output = policy.model(
            z0.float(),
            language_feature,
            (proprio - policy.proprio_mean) / policy.proprio_std,
        )
        raw = actor_output * policy.action_std + policy.action_mean
    actions = raw[0].float().clamp(-1.0, 1.0)
    prediction, uncertainty = predict_chunk(policy, z0, observation, language, actions)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    diagnostics = {
        "selected_candidate": 0,
        "planning_ms": (time.perf_counter() - started) * 1000.0,
        "raw_base_clipping": policy._clipping_statistics(raw.float().cpu().numpy())[0],
    }
    return actions.cpu().numpy(), prediction, uncertainty, diagnostics


def plan_c1_observed(
    policy: C1RerankPolicy,
    observation: dict[str, np.ndarray],
    language: str,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict[str, Any]]:
    actions, diagnostics = policy.plan_chunk(observation, language)
    prediction = policy._last_selected_prediction
    uncertainty = policy._last_selected_uncertainty
    if prediction is None or uncertainty is None:
        raise RuntimeError("C1 did not retain its selected prediction")
    return actions, prediction, uncertainty, diagnostics


def evaluate_episode(
    policy: C1RerankPolicy,
    env: OffScreenRenderEnv,
    initial_state: np.ndarray,
    task: Any,
    task_id: int,
    method: str,
    episode_index: int,
    seed: int,
    init_state_index: int,
    max_steps: int,
    mismatch_threshold: float,
    uncertainty_threshold: float,
) -> dict[str, Any]:
    env.seed(seed)
    env.reset()
    observation = env.set_init_state(initial_state)
    disable_camera_observable(env)
    policy.reset(seed=seed)
    plan_function = plan_c0_observed if method == "c0" else plan_c1_observed
    chunk = np.empty((0, 7), dtype=np.float32)
    chunk_index = 8
    plan_index = -1
    prediction: torch.Tensor | None = None
    uncertainty: torch.Tensor | None = None
    plan_events: list[dict[str, Any]] = []
    check_events: list[dict[str, Any]] = []
    success = False
    for step in range(1, max_steps + 1):
        if chunk_index >= 8:
            observation = refresh_policy_observation(env)
            chunk, prediction, uncertainty, diagnostics = plan_function(
                policy, policy_observation(observation), task.language
            )
            if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
                raise ValueError("observer controller produced an invalid action chunk")
            plan_index += 1
            plan_events.append(
                {
                    "plan_index": plan_index,
                    "episode_step_before_chunk": step - 1,
                    "selected_candidate": int(diagnostics["selected_candidate"]),
                    "planning_ms": float(diagnostics["planning_ms"]),
                    "executed_actions": 0,
                }
            )
            chunk_index = 0
        action = np.asarray(chunk[chunk_index], dtype=np.float32)
        observation, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
        chunk_index += 1
        plan_events[-1]["executed_actions"] = chunk_index
        success_now = bool(env.check_success())
        if chunk_index % 2 == 0:
            observation = refresh_policy_observation(env)
            actual = encode_image(policy, policy_observation(observation))
            assert prediction is not None and uncertainty is not None
            mismatch = float(
                1.0
                - F.cosine_similarity(
                    prediction[chunk_index - 1 : chunk_index].float(), actual, dim=-1
                ).item()
            )
            disagreement = float(uncertainty[chunk_index - 1].item())
            values = np.asarray([mismatch, disagreement], dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError("control-validation trace contains NaN/Inf")
            mismatch_exceeded = mismatch > mismatch_threshold
            uncertainty_exceeded = disagreement > uncertainty_threshold
            check_events.append(
                {
                    "plan_index": plan_index,
                    "horizon": chunk_index,
                    "episode_step_after_check": step,
                    "mismatch": mismatch,
                    "uncertainty": disagreement,
                    "mismatch_exceeded_d20_v1": mismatch_exceeded,
                    "uncertainty_exceeded_d20_v1": uncertainty_exceeded,
                    "or_trigger_d20_v1": mismatch_exceeded or uncertainty_exceeded,
                    "and_trigger_d20_v1": mismatch_exceeded and uncertainty_exceeded,
                }
            )
        if success_now:
            success = True
            break
    return {
        "task_id": task_id,
        "episode_index": episode_index,
        "seed": seed,
        "init_state_index": init_state_index,
        "success": success,
        "steps": step,
        "plan_count": len(plan_events),
        "check_count": len(check_events),
        "plan_events": plan_events,
        "check_events": check_events,
    }


def task_result(
    protocol_path: Path,
    protocol: dict[str, Any],
    policy: C1RerankPolicy,
    suite: Any,
    method: str,
    task_id: int,
) -> dict[str, Any]:
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    states = trusted_init_states(suite, task_id)
    indices = [int(value) for value in protocol["controller_validation"]["init_state_indices"]]
    seeds = [
        int(value)
        for value in protocol["controller_validation"]["task_seeds"][str(task_id)]
    ]
    d20 = protocol["frozen_d20_v1"]
    episodes = []
    for episode_index, (seed, state_index) in enumerate(zip(seeds, indices)):
        env: OffScreenRenderEnv = make_env(bddl)
        try:
            episode = evaluate_episode(
                policy,
                env,
                states[state_index],
                task,
                task_id,
                method,
                episode_index,
                seed,
                state_index,
                int(protocol["max_steps"]),
                float(d20["mismatch_threshold"]),
                float(d20["uncertainty_threshold"]),
            )
        finally:
            env.close()
        episodes.append(episode)
        print(
            json.dumps(
                {
                    "method": method,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "success": episode["success"],
                    "steps": episode["steps"],
                    "checks": episode["check_count"],
                }
            ),
            flush=True,
        )
    checks = [event for episode in episodes for event in episode["check_events"]]
    plans = [event for episode in episodes for event in episode["plan_events"]]
    selected = Counter(int(event["selected_candidate"]) for event in plans)
    return {
        "schema_version": 1,
        "contract_version": "control-validation-trace-v1",
        "method": f"{method}_logging_only_world_model_observer",
        "controller_method": method,
        "world_model_changes_control": False,
        "early_replanning": False,
        "wm_action_sequence": (
            "executed_chunk[0:8], preserving the deployed D19/D21 controller integration "
            "for diagnosis without repair"
        ),
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "episodes": len(episodes),
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "success_rate": sum(bool(episode["success"]) for episode in episodes) / len(episodes),
        "max_steps": int(protocol["max_steps"]),
        "planning_events": len(plans),
        "check_events": len(checks),
        "selected_candidate_counts": {str(index): selected[index] for index in range(3)},
        "d20_v1_counterfactual": {
            "mismatch_threshold": float(d20["mismatch_threshold"]),
            "uncertainty_threshold": float(d20["uncertainty_threshold"]),
            "mismatch_exceeded": sum(event["mismatch_exceeded_d20_v1"] for event in checks),
            "uncertainty_exceeded": sum(
                event["uncertainty_exceeded_d20_v1"] for event in checks
            ),
            "or_trigger": sum(event["or_trigger_d20_v1"] for event in checks),
            "and_trigger": sum(event["and_trigger_d20_v1"] for event in checks),
        },
        "episode_results": episodes,
    }


def validate_existing(
    path: Path, protocol_path: Path, method: str, task_id: int
) -> None:
    existing = json.loads(path.read_text())
    expected = {
        "contract_version": "control-validation-trace-v1",
        "controller_method": method,
        "world_model_changes_control": False,
        "early_replanning": False,
        "protocol_sha256": file_sha256(protocol_path),
        "task_id": task_id,
        "episodes": 10,
    }
    mismatches = [key for key, value in expected.items() if existing.get(key) != value]
    if mismatches:
        raise ValueError(f"incompatible existing task result {path}: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("c0", "c1"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/control_validation_protocol.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/control_validation"
    )
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(10))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["protocol_version"] != "control-validation-v1":
        raise ValueError("expected frozen control-validation-v1")
    if file_sha256(PROJECT_ROOT / protocol["fixed_final_test_exclusion"]["seed_file"]) != protocol[
        "fixed_final_test_exclusion"
    ]["seed_file_sha256"]:
        raise ValueError("fixed-test exclusion record changed")
    if file_sha256(PROJECT_ROOT / protocol["frozen_d20_v1"]["result"]) != protocol[
        "frozen_d20_v1"
    ]["result_sha256"]:
        raise ValueError("D20-v1 threshold record changed")
    c0_record = protocol["collection_methods"]["c0_rollout"]
    if file_sha256(PROJECT_ROOT / c0_record["checkpoint"]) != c0_record[
        "checkpoint_sha256"
    ]:
        raise ValueError("frozen C0 checkpoint changed")
    c1_record = protocol["collection_methods"]["c1_selected_rollout"]
    if file_sha256(PROJECT_ROOT / c1_record["config"]) != c1_record["config_sha256"]:
        raise ValueError("frozen C1 configuration changed")
    ensemble_record = protocol["world_model"]
    ensemble_path = PROJECT_ROOT / ensemble_record["ensemble_config"]
    if file_sha256(ensemble_path) != ensemble_record["ensemble_config_sha256"]:
        raise ValueError("frozen ensemble configuration changed")
    import yaml

    ensemble_config = yaml.safe_load(ensemble_path.read_text())
    member_hashes = [
        file_sha256(PROJECT_ROOT / relative)
        for relative in ensemble_config["member_checkpoints"]
    ]
    if member_hashes != ensemble_record["member_checkpoint_sha256"]:
        raise ValueError("frozen ensemble checkpoint changed")
    task_ids = args.task_ids if args.task_ids is not None else protocol["task_ids"]
    checkpoint = PROJECT_ROOT / c0_record["checkpoint"]
    policy = C1RerankPolicy(checkpoint=checkpoint, device=args.device)
    suite = benchmark.get_benchmark_dict()[protocol["suite"]]()
    for task_id in task_ids:
        path = args.output_dir / args.method / f"task_{task_id}.json"
        if path.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {path}; use --resume")
            validate_existing(path, args.protocol, args.method, task_id)
            print(json.dumps({"resume_skip": str(path)}), flush=True)
            continue
        result = task_result(args.protocol, protocol, policy, suite, args.method, task_id)
        atomic_json(path, result)
        print(
            json.dumps(
                {
                    "completed": str(path),
                    "successes": result["successes"],
                    "episodes": result["episodes"],
                    "checks": result["check_events"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
