#!/usr/bin/env python3
"""Matched simulator branches for frozen C1-v2 production candidates.

Every candidate branch starts from the official episode initial state and
replays the exact stored prefix actions.  Mid-episode simulator-state restore
is deliberately absent from this module.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scripts.calibrate_world_model_v2b import model_from_config as v2b_model_from_config
from scripts.evaluate_c0_all import (
    make_env,
    policy_observation,
    refresh_policy_observation,
    trusted_init_states,
)
from src.policies.c1_v2_rerank_policy import C1V2RerankPolicy
from src.world_models import LatentDynamicsModel, apply_world_model_normalization


PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")
MODEL_NAMES = ("wm_v1_d15_ensemble", "wm_v2a_seed11", "wm_v2b_seed11")


class CounterfactualReplayError(RuntimeError):
    """A deterministic-prefix identity gate failed."""

    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content)
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


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def path_last_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT)
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(relative)],
        cwd=ROOT,
        text=True,
    ).strip()


def validate_protocol(
    path: Path, amendment_path: Path | None = None
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != "local-candidate-counterfactual-dynamics-v1":
        raise ValueError("unexpected local-counterfactual protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_SIMULATOR_BRANCHES":
        raise ValueError("protocol is not preregistered")
    scope = protocol["scope"]
    expected_scope = {
        "task_ids": list(range(10)),
        "partition": "controller_development",
        "init_state_indices_per_task": [30, 35],
        "episode_positions_in_partition": [0, 5],
        "planning_indices": [0, 8],
        "prefix_steps": [0, 64],
        "candidate_count": 3,
        "horizon": 8,
        "total_branch_points": 40,
        "total_independent_candidate_branches": 120,
        "success_reward_done_recorded": False,
    }
    mismatches = [key for key, expected in expected_scope.items() if scope.get(key) != expected]
    if mismatches:
        raise ValueError(f"protocol scope changed: {mismatches}")
    branch = protocol["branch_construction"]
    if branch["mid_episode_flattened_state_restore"]:
        raise ValueError("mid-episode state restore is forbidden")
    if branch["independent_noise_per_branch"] or branch["replanning_inside_branch"]:
        raise ValueError("branch construction changed")

    frozen = protocol["frozen_upstream"]
    tag_type = subprocess.check_output(
        ["git", "cat-file", "-t", frozen["wm_v2b_annotated_tag"]],
        cwd=ROOT,
        text=True,
    ).strip()
    tag_commit = subprocess.check_output(
        ["git", "rev-parse", f"{frozen['wm_v2b_annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    if tag_type != "tag" or tag_commit != frozen["wm_v2b_commit"]:
        raise ValueError("WM-v2B annotated freeze tag changed")
    checks: list[tuple[str, str]] = [
        (frozen["wm_v2b_diagnosis"], frozen["wm_v2b_diagnosis_sha256"]),
        (frozen["c0_v2_checkpoint"], frozen["c0_v2_checkpoint_sha256"]),
        (frozen["c1_v2_config"], frozen["c1_v2_config_sha256"]),
        (
            frozen["controller_partition_protocol"],
            frozen["controller_partition_protocol_sha256"],
        ),
        (frozen["d15_ensemble_config"], frozen["d15_ensemble_config_sha256"]),
        (frozen["wm_v2a_checkpoint"], frozen["wm_v2a_checkpoint_sha256"]),
        (frozen["wm_v2a_config"], frozen["wm_v2a_config_sha256"]),
        (frozen["wm_v2b_checkpoint"], frozen["wm_v2b_checkpoint_sha256"]),
        (frozen["wm_v2b_config"], frozen["wm_v2b_config_sha256"]),
    ]
    checks.extend((item["path"], item["sha256"]) for item in frozen["d15_checkpoints"])
    for relative, expected in checks:
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")

    partition_source = json.loads((ROOT / frozen["controller_partition_protocol"]).read_text())
    partition = partition_source["partitions"][scope["partition"]]
    for task_id in scope["task_ids"]:
        positions = scope["episode_positions_in_partition"]
        indices = [partition["init_state_indices"][position] for position in positions]
        seeds = [partition["task_seeds"][str(task_id)][position] for position in positions]
        if indices != scope["init_state_indices_per_task"]:
            raise ValueError(f"task {task_id}: init-state preregistration mismatch")
        expected_seeds = [30000 + 100 * task_id + position for position in positions]
        if seeds != expected_seeds:
            raise ValueError(f"task {task_id}: seed preregistration mismatch")
    preregistered_commit = path_last_commit(path)
    if not preregistered_commit:
        raise ValueError("protocol must be committed before simulator execution")
    amendment = None
    if amendment_path is not None:
        amendment = json.loads(amendment_path.read_text())
        if amendment["amendment_version"] != "local-candidate-counterfactual-dynamics-v1.1":
            raise ValueError("unexpected protocol amendment")
        if amendment["status"] != (
            "PREREGISTERED_AFTER_V1_PREFLIGHT_BLOCK_AND_BEFORE_ANY_MECHANISM_BRANCH_RESULT"
        ):
            raise ValueError("protocol amendment is not preregistered")
        if amendment["base_protocol"] != str(path.resolve().relative_to(ROOT)):
            raise ValueError("amendment base protocol path changed")
        if amendment["base_protocol_sha256"] != sha256(path):
            raise ValueError("amendment base protocol hash changed")
        if amendment["base_preregistered_commit"] != preregistered_commit:
            raise ValueError("amendment base protocol commit changed")
        blocker = amendment["v1_preflight_blocker"]
        if sha256(ROOT / blocker["result"]) != blocker["result_sha256"]:
            raise ValueError("v1 preflight blocker result changed")
        if sha256(ROOT / blocker["report"]) != blocker["report_sha256"]:
            raise ValueError("v1 preflight blocker report changed")
        if float(amendment["single_authorized_measurement_fix"]["clip_identity_threshold_unchanged"]) != float(
            protocol["prefix_identity_gate"]["clip_latent_min_cosine"]
        ):
            raise ValueError("v1.1 changed the CLIP identity threshold")
        preregistered_commit = path_last_commit(amendment_path)
        if not preregistered_commit:
            raise ValueError("amendment must be committed before simulator execution")
    return protocol, preregistered_commit, amendment


def disable_all_cameras(env: OffScreenRenderEnv) -> None:
    camera_names = []
    for name, observable in env.env._observables.items():
        if name.endswith("_image"):
            observable.set_active(False)
            camera_names.append(name)
    if "agentview_image" not in camera_names:
        raise KeyError("agentview camera observable is missing")


def proprio_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
            np.asarray(observation["robot0_joint_pos"], dtype=np.float32).reshape(7),
        )
    )


def simulator_state(env: OffScreenRenderEnv) -> np.ndarray:
    state = env.env.sim.get_state()
    flattened = state.flatten() if hasattr(state, "flatten") else np.asarray(state)
    result = np.asarray(flattened, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all():
        raise FloatingPointError("simulator state contains NaN/Inf")
    return result


def observation_identity(
    reference: dict[str, np.ndarray], candidates: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    keys = sorted(reference)
    details: dict[str, Any] = {}
    exact = True
    max_abs = 0.0
    for branch_index, observation in enumerate(candidates):
        branch_keys = sorted(observation)
        if branch_keys != keys:
            exact = False
            details[str(branch_index)] = {"same_keys": False}
            continue
        branch_exact = True
        branch_max = 0.0
        for key in keys:
            first = np.asarray(reference[key])
            second = np.asarray(observation[key])
            same_meta = first.shape == second.shape and first.dtype == second.dtype
            same = same_meta and np.array_equal(first, second)
            branch_exact = branch_exact and same
            if same_meta and np.issubdtype(first.dtype, np.number) and first.size:
                difference = float(
                    np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)))
                )
                branch_max = max(branch_max, difference)
        exact = exact and branch_exact
        max_abs = max(max_abs, branch_max)
        details[str(branch_index)] = {
            "same_keys": True,
            "exact": branch_exact,
            "max_abs_difference": branch_max,
        }
    return {
        "keys": keys,
        "exact": exact,
        "max_abs_difference": max_abs,
        "branches": details,
    }


@torch.inference_mode()
def encode_images(policy: C1V2RerankPolicy, images: Iterable[np.ndarray]) -> np.ndarray:
    rows = [np.ascontiguousarray(np.asarray(image, dtype=np.uint8)) for image in images]
    pixels = policy.processor(images=rows, return_tensors="pt")["pixel_values"].to(
        policy.device
    )
    with policy._autocast():
        latent = policy.clip.get_image_features(pixel_values=pixels)
        latent = F.normalize(latent.float(), p=2, dim=-1, eps=1.0e-12)
    result = latent.cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("CLIP encoding contains NaN/Inf")
    return result


def load_single_visual_model(
    checkpoint_path: Path, expected_method: str, expected_hash: str, device: torch.device
) -> tuple[LatentDynamicsModel, dict[str, Any], dict[str, Any]]:
    if sha256(checkpoint_path) != expected_hash:
        raise ValueError(f"checkpoint hash changed: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["method"] != expected_method:
        raise ValueError(f"unexpected checkpoint method: {checkpoint_path}")
    model = LatentDynamicsModel(
        **{key: int(value) for key, value in state["config"]["model"].items()}
    )
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state["normalization"], state


def load_v2b_model(
    checkpoint_path: Path, expected_hash: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if sha256(checkpoint_path) != expected_hash:
        raise ValueError("WM-v2B checkpoint hash changed")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["method"] != "wm_v2b_future_proprio_auxiliary":
        raise ValueError("unexpected WM-v2B checkpoint method")
    model = v2b_model_from_config(state["config"])
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state["normalization"], state


def normalization_identity(values: list[dict[str, Any]]) -> None:
    first = json.dumps(values[0], sort_keys=True, separators=(",", ":"))
    if any(json.dumps(value, sort_keys=True, separators=(",", ":")) != first for value in values[1:]):
        raise ValueError("world models do not share the frozen normalization")


@torch.inference_mode()
def predict_candidates(
    policy: C1V2RerankPolicy,
    v2a_model: LatentDynamicsModel,
    v2b_model: torch.nn.Module,
    normalization: dict[str, Any],
    z0: np.ndarray,
    proprio0: np.ndarray,
    language: np.ndarray,
    candidates: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    device = policy.device
    z_tensor = torch.from_numpy(z0).view(1, 512).to(device).expand(3, -1)
    p_raw = torch.from_numpy(proprio0).view(1, 12).to(device)
    language_tensor = torch.from_numpy(language).view(1, 512).to(device).expand(3, -1)
    raw_actions = torch.from_numpy(candidates).to(device)
    p_normalized, action_normalized = apply_world_model_normalization(
        p_raw, raw_actions, normalization
    )
    p_normalized = p_normalized.expand(3, -1)
    v1, _, _ = policy.ensemble.predict(
        z_tensor, p_normalized, language_tensor, action_normalized
    )
    v2a = v2a_model(z_tensor, p_normalized, language_tensor, action_normalized)
    v2b, p_delta = v2b_model(
        z_tensor, p_normalized, language_tensor, action_normalized
    )
    predicted_p_future = p_normalized.unsqueeze(1) + p_delta
    predictions = {
        "wm_v1_d15_ensemble": v1.float().cpu().numpy(),
        "wm_v2a_seed11": v2a.float().cpu().numpy(),
        "wm_v2b_seed11": v2b.float().cpu().numpy(),
    }
    p_result = predicted_p_future.float().cpu().numpy()
    if not all(np.isfinite(value).all() for value in (*predictions.values(), p_result)):
        raise FloatingPointError("world-model prediction contains NaN/Inf")
    return predictions, p_result


def create_episode_env(
    bddl: Path, seed: int, init_state: Any
) -> tuple[OffScreenRenderEnv, dict[str, np.ndarray]]:
    env = make_env(bddl)
    env.seed(int(seed))
    env.reset()
    observation = env.set_init_state(init_state)
    disable_all_cameras(env)
    return env, observation


def replay_prefix(
    bddl: Path,
    seed: int,
    init_state: Any,
    prefix: np.ndarray,
) -> tuple[OffScreenRenderEnv, dict[str, np.ndarray], np.ndarray]:
    env, _ = create_episode_env(bddl, seed, init_state)
    try:
        for action_index, action in enumerate(prefix, start=1):
            env.step(np.asarray(action, dtype=np.float32))
            # Match the reference C0-v2 planning cadence. Rendering does not
            # advance physics, but it does update robosuite observable buffers.
            if action_index % 8 == 0 and action_index < len(prefix):
                refresh_policy_observation(env)
        observation = refresh_policy_observation(env)
        return env, observation, simulator_state(env)
    except Exception:
        env.close()
        raise


def prefix_audit(
    protocol: dict[str, Any],
    policy: C1V2RerankPolicy,
    prefix: np.ndarray,
    reference_observation: dict[str, np.ndarray],
    reference_state: np.ndarray,
    branch_observations: list[dict[str, np.ndarray]],
    branch_states: list[np.ndarray],
    identity_latents: np.ndarray,
) -> dict[str, Any]:
    gate = protocol["prefix_identity_gate"]
    prefix_hash = array_sha256(prefix.astype(np.float32))
    branch_prefix_hashes = [array_sha256(prefix.astype(np.float32)) for _ in range(3)]
    rgb_reference = np.asarray(reference_observation["agentview_image"], dtype=np.uint8)
    rgb_hash = array_sha256(rgb_reference)
    branch_rgb_hashes = [
        array_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
        for observation in branch_observations
    ]
    rgb_equal = all(
        np.array_equal(rgb_reference, observation["agentview_image"])
        for observation in branch_observations
    )
    reference_p = proprio_from_observation(reference_observation).astype(np.float64)
    proprio_max = max(
        float(np.max(np.abs(reference_p - proprio_from_observation(obs).astype(np.float64))))
        for obs in branch_observations
    )
    state_max = 0.0
    states_same_shape = all(state.shape == reference_state.shape for state in branch_states)
    if states_same_shape:
        state_max = max(
            float(np.max(np.abs(reference_state - state))) if state.size else 0.0
            for state in branch_states
        )
    else:
        state_max = float("inf")
    if identity_latents.shape != (4, 512):
        raise ValueError("identity CLIP batch must contain reference plus three branches")
    reference_latent = identity_latents[0]
    clip_cosines = np.sum(identity_latents[1:] * reference_latent[None, :], axis=-1)
    observation = observation_identity(reference_observation, branch_observations)
    passed = bool(
        prefix_hash == branch_prefix_hashes[0] == branch_prefix_hashes[1] == branch_prefix_hashes[2]
        and rgb_equal
        and all(value == rgb_hash for value in branch_rgb_hashes)
        and observation["exact"]
        and proprio_max <= float(gate["proprio_max_abs_tolerance"])
        and states_same_shape
        and state_max <= float(gate["simulator_flattened_state_max_abs_tolerance"])
        and float(clip_cosines.min()) >= float(gate["clip_latent_min_cosine"])
    )
    return {
        "pass": passed,
        "prefix_action_sha256": prefix_hash,
        "branch_prefix_action_sha256": branch_prefix_hashes,
        "reference_rgb_sha256": rgb_hash,
        "branch_rgb_sha256": branch_rgb_hashes,
        "rgb_exact": rgb_equal,
        "planning_observation": observation,
        "proprio_max_abs_difference": proprio_max,
        "simulator_state_same_shape": states_same_shape,
        "simulator_state_max_abs_difference": state_max,
        "clip_cosine_to_reference": clip_cosines.astype(float).tolist(),
    }


def collect_branch_point(
    protocol: dict[str, Any],
    policy: C1V2RerankPolicy,
    v2a_model: LatentDynamicsModel,
    v2b_model: torch.nn.Module,
    normalization: dict[str, Any],
    bddl: Path,
    seed: int,
    init_state: Any,
    task_id: int,
    init_state_index: int,
    episode_position: int,
    planning_index: int,
    prefix: np.ndarray,
    reference_observation: dict[str, np.ndarray],
    reference_state: np.ndarray,
    candidates: np.ndarray,
    context: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    envs: list[OffScreenRenderEnv] = []
    try:
        branch_observations: list[dict[str, np.ndarray]] = []
        branch_states: list[np.ndarray] = []
        for _ in range(3):
            env, observation, state = replay_prefix(bddl, seed, init_state, prefix)
            envs.append(env)
            branch_observations.append(observation)
            branch_states.append(state)
        identity_latents = encode_images(
            policy,
            [reference_observation["agentview_image"]]
            + [observation["agentview_image"] for observation in branch_observations],
        )
        audit = prefix_audit(
            protocol,
            policy,
            prefix,
            reference_observation,
            reference_state,
            branch_observations,
            branch_states,
            identity_latents,
        )
        identity = {
            "task_id": task_id,
            "episode_position": episode_position,
            "init_state_index": init_state_index,
            "seed": seed,
            "planning_index": planning_index,
            "prefix_steps": int(prefix.shape[0]),
        }
        if not audit["pass"]:
            raise CounterfactualReplayError(
                f"deterministic prefix identity failed: {identity}",
                {"identity": identity, **audit},
            )

        real_rgb = np.empty((3, 8, 128, 128, 3), dtype=np.uint8)
        real_z = np.empty((3, 8, 512), dtype=np.float32)
        real_p = np.empty((3, 8, 12), dtype=np.float32)
        executed = np.empty((3, 8, 7), dtype=np.float32)
        for horizon in range(8):
            images = []
            for candidate_index, env in enumerate(envs):
                action = np.asarray(candidates[candidate_index, horizon], dtype=np.float32)
                env.step(action)
                observation = refresh_policy_observation(env)
                image = np.asarray(observation["agentview_image"], dtype=np.uint8)
                real_rgb[candidate_index, horizon] = image
                real_p[candidate_index, horizon] = proprio_from_observation(observation)
                executed[candidate_index, horizon] = action
                images.append(image)
            real_z[:, horizon] = encode_images(policy, images)

        z0 = context["z0"][0].detach().float().cpu().numpy()
        p0 = context["proprio"][0].detach().float().cpu().numpy()
        language = context["language"][0].detach().float().cpu().numpy()
        predictions, pred_p = predict_candidates(
            policy,
            v2a_model,
            v2b_model,
            normalization,
            z0,
            p0,
            language,
            candidates,
        )
        if not np.array_equal(executed, candidates):
            raise RuntimeError("executed branch action differs from frozen candidate tensor")
        candidate_hash = array_sha256(candidates.astype(np.float32))
        record = {
            **identity,
            "candidate_sha256": candidate_hash,
            "prefix_audit": audit,
        }
        arrays = {
            "z0": z0,
            "proprio0": p0,
            "language": language,
            "candidates": candidates.astype(np.float32),
            "executed_actions": executed,
            "real_rgb": real_rgb,
            "real_z": real_z,
            "real_proprio": real_p,
            "pred_z_wm_v1_d15_ensemble": predictions["wm_v1_d15_ensemble"],
            "pred_z_wm_v2a_seed11": predictions["wm_v2a_seed11"],
            "pred_z_wm_v2b_seed11": predictions["wm_v2b_seed11"],
            "pred_proprio_wm_v2b_standardized": pred_p,
            "prefix_actions": prefix.astype(np.float32),
            "reference_rgb": np.asarray(
                reference_observation["agentview_image"], dtype=np.uint8
            ),
        }
        if not all(
            np.isfinite(value).all()
            for key, value in arrays.items()
            if key not in {"real_rgb", "reference_rgb"}
        ):
            raise FloatingPointError("branch point contains NaN/Inf")
        return record, arrays
    finally:
        for env in envs:
            env.close()


def collect(
    protocol: dict[str, Any], device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    frozen = protocol["frozen_upstream"]
    policy = C1V2RerankPolicy(
        checkpoint=ROOT / frozen["c0_v2_checkpoint"],
        device=str(device),
        config=ROOT / frozen["c1_v2_config"],
    )
    v2a_model, v2a_normalization, _ = load_single_visual_model(
        ROOT / frozen["wm_v2a_checkpoint"],
        "wm_v2a_local_dynamics_resolution",
        frozen["wm_v2a_checkpoint_sha256"],
        device,
    )
    v2b_model, v2b_normalization, _ = load_v2b_model(
        ROOT / frozen["wm_v2b_checkpoint"],
        frozen["wm_v2b_checkpoint_sha256"],
        device,
    )
    d15_state = torch.load(
        ROOT / frozen["d15_checkpoints"][0]["path"], map_location="cpu", weights_only=False
    )
    d15_normalization = d15_state["normalization"]
    normalization_identity([d15_normalization, v2a_normalization, v2b_normalization])

    partition = json.loads(
        (ROOT / frozen["controller_partition_protocol"]).read_text()
    )["partitions"][protocol["scope"]["partition"]]
    suite = benchmark.get_benchmark_dict()[protocol["scope"]["suite"]]()
    records: list[dict[str, Any]] = []
    collected: dict[str, list[np.ndarray]] = {}
    max_prefix = max(protocol["scope"]["prefix_steps"])
    max_plan = max(protocol["scope"]["planning_indices"])
    for task_id in protocol["scope"]["task_ids"]:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_states = trusted_init_states(suite, task_id)
        for episode_position, init_state_index in zip(
            protocol["scope"]["episode_positions_in_partition"],
            protocol["scope"]["init_state_indices_per_task"],
        ):
            seed = int(partition["task_seeds"][str(task_id)][episode_position])
            init_state = init_states[init_state_index]
            reference_env, _ = create_episode_env(bddl, seed, init_state)
            prefix_rows: list[np.ndarray] = []
            try:
                for planning_index in range(max_plan + 1):
                    reference_observation = refresh_policy_observation(reference_env)
                    live = policy_observation(reference_observation)
                    context = policy._actor_context(live, task.language)
                    plan_seed = (
                        policy.candidate_seed + seed * 1_000_003 + planning_index
                    )
                    interface = policy._interface(context["actor_normalized"], plan_seed)
                    candidates = (
                        interface["candidates"].detach().float().cpu().numpy().copy()
                    )
                    if planning_index in protocol["scope"]["planning_indices"]:
                        prefix = (
                            np.stack(prefix_rows).astype(np.float32)
                            if prefix_rows
                            else np.empty((0, 7), dtype=np.float32)
                        )
                        record, arrays = collect_branch_point(
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
                            reference_observation,
                            simulator_state(reference_env),
                            candidates,
                            context,
                        )
                        records.append(record)
                        for key, value in arrays.items():
                            collected.setdefault(key, []).append(value)
                        print(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "init_state_index": init_state_index,
                                    "planning_index": planning_index,
                                    "branch_points": len(records),
                                    "prefix_pass": True,
                                }
                            ),
                            flush=True,
                        )
                    if planning_index < max_plan:
                        for action in candidates[0]:
                            reference_env.step(np.asarray(action, dtype=np.float32))
                            prefix_rows.append(np.asarray(action, dtype=np.float32).copy())
            finally:
                reference_env.close()
            if len(prefix_rows) != max_prefix:
                raise RuntimeError("reference prefix length differs from preregistration")
    if len(records) != int(protocol["scope"]["total_branch_points"]):
        raise RuntimeError("branch-point count differs from preregistration")

    arrays = {key: np.stack(values) for key, values in collected.items() if key != "prefix_actions"}
    maximum_prefix = max(value.shape[0] for value in collected["prefix_actions"])
    prefix_padded = np.zeros((len(records), maximum_prefix, 7), dtype=np.float32)
    prefix_lengths = np.empty(len(records), dtype=np.int64)
    for index, value in enumerate(collected["prefix_actions"]):
        prefix_padded[index, : value.shape[0]] = value
        prefix_lengths[index] = value.shape[0]
    arrays["prefix_actions"] = prefix_padded
    arrays["prefix_lengths"] = prefix_lengths
    for name in ("task_id", "episode_position", "init_state_index", "seed", "planning_index"):
        arrays[name] = np.asarray([record[name] for record in records], dtype=np.int64)
    return records, arrays


def cosine_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first.astype(np.float64) * second.astype(np.float64), axis=-1)
    denominator = np.linalg.norm(first.astype(np.float64), axis=-1) * np.linalg.norm(
        second.astype(np.float64), axis=-1
    )
    return np.maximum(1.0 - numerator / np.maximum(denominator, 1.0e-12), 0.0)


def pairwise(values: np.ndarray, metric: str) -> np.ndarray:
    rows = []
    for first, second in PAIR_INDICES:
        if metric == "cosine":
            rows.append(cosine_distance(values[:, first], values[:, second]))
        elif metric == "l2":
            rows.append(
                np.linalg.norm(
                    values[:, first].astype(np.float64)
                    - values[:, second].astype(np.float64),
                    axis=-1,
                )
            )
        elif metric == "rmse":
            rows.append(
                np.sqrt(
                    np.mean(
                        np.square(
                            values[:, first].astype(np.float64)
                            - values[:, second].astype(np.float64)
                        ),
                        axis=-1,
                    )
                )
            )
        else:
            raise ValueError(metric)
    return np.stack(rows, axis=1)


def summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("cannot summarize empty finite values")
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray, minimum: float) -> dict[str, Any]:
    pred = float(np.mean(numerator))
    real = float(np.mean(denominator))
    return {
        "predicted_mean": pred,
        "real_mean": real,
        "predicted_minus_real": pred - real,
        "minimum_real_mean": minimum,
        "eligible": real >= minimum,
        "ratio_of_means": None if real < minimum else pred / real,
    }


def correlation(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(x.size), "pearson": None, "spearman": None}
    pearson = scipy_stats.pearsonr(x, y)
    spearman = scipy_stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def analyze(
    protocol: dict[str, Any], records: list[dict[str, Any]], arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    real_z = arrays["real_z"].astype(np.float64)
    real_p = arrays["real_proprio"].astype(np.float64)
    real_rgb = arrays["real_rgb"].astype(np.float64)
    normalization = torch.load(
        ROOT / protocol["frozen_upstream"]["d15_checkpoints"][0]["path"],
        map_location="cpu",
        weights_only=False,
    )["normalization"]
    p_mean = np.asarray(normalization["proprio"]["mean"], dtype=np.float64).reshape(1, 1, 1, 12)
    p_std = np.asarray(normalization["proprio"]["std"], dtype=np.float64).reshape(1, 1, 1, 12)
    real_p_standardized = (real_p - p_mean) / p_std

    real_visual_cos = pairwise(real_z, "cosine")
    real_visual_l2 = pairwise(real_z, "l2")
    real_p_rmse = pairwise(real_p_standardized, "rmse")
    real_p_raw_l2 = pairwise(real_p, "l2")
    pixel_mae_rows, pixel_rmse_rows, pixel_fraction_rows = [], [], []
    for first, second in PAIR_INDICES:
        difference = real_rgb[:, first] - real_rgb[:, second]
        pixel_mae_rows.append(np.mean(np.abs(difference), axis=(-1, -2, -3)))
        pixel_rmse_rows.append(np.sqrt(np.mean(np.square(difference), axis=(-1, -2, -3))))
        pixel_fraction_rows.append(np.mean(difference != 0, axis=(-1, -2, -3)))
    pixel_mae = np.stack(pixel_mae_rows, axis=1)
    pixel_rmse = np.stack(pixel_rmse_rows, axis=1)
    pixel_fraction = np.stack(pixel_fraction_rows, axis=1)

    predicted_separation: dict[str, np.ndarray] = {}
    candidate_error: dict[str, np.ndarray] = {}
    model_array_names = {
        "wm_v1_d15_ensemble": "pred_z_wm_v1_d15_ensemble",
        "wm_v2a_seed11": "pred_z_wm_v2a_seed11",
        "wm_v2b_seed11": "pred_z_wm_v2b_seed11",
    }
    for model, array_name in model_array_names.items():
        prediction = arrays[array_name].astype(np.float64)
        predicted_separation[model] = pairwise(prediction, "cosine")
        candidate_error[model] = cosine_distance(prediction, real_z)
    pred_p = arrays["pred_proprio_wm_v2b_standardized"].astype(np.float64)
    v2b_p_error = np.sqrt(np.mean(np.square(pred_p - real_p_standardized), axis=-1))
    v2b_p_separation = pairwise(pred_p, "rmse")

    minimum = float(protocol["metric_contract"]["ratio_minimum_real_cosine"])
    per_horizon = []
    for horizon in range(8):
        row: dict[str, Any] = {
            "horizon": horizon + 1,
            "real_visual_cosine": summary(real_visual_cos[:, :, horizon]),
            "real_visual_l2": summary(real_visual_l2[:, :, horizon]),
            "real_proprio_standardized_rmse": summary(real_p_rmse[:, :, horizon]),
            "real_proprio_raw_l2": summary(real_p_raw_l2[:, :, horizon]),
            "pixel_mae": summary(pixel_mae[:, :, horizon]),
            "pixel_rmse": summary(pixel_rmse[:, :, horizon]),
            "unequal_pixel_fraction": summary(pixel_fraction[:, :, horizon]),
            "proprio_visual_correlation": correlation(
                real_p_rmse[:, :, horizon], real_visual_cos[:, :, horizon]
            ),
            "models": {},
        }
        for model in MODEL_NAMES:
            row["models"][model] = {
                "predicted_visual_cosine": summary(
                    predicted_separation[model][:, :, horizon]
                ),
                "transfer": safe_ratio(
                    predicted_separation[model][:, :, horizon],
                    real_visual_cos[:, :, horizon],
                    minimum,
                ),
                "candidate_error": {
                    f"A{candidate}": summary(candidate_error[model][:, candidate, horizon])
                    for candidate in range(3)
                },
            }
        row["wm_v2b_proprio"] = {
            "candidate_error": {
                f"A{candidate}": summary(v2b_p_error[:, candidate, horizon])
                for candidate in range(3)
            },
            "predicted_candidate_separation": summary(v2b_p_separation[:, :, horizon]),
        }
        per_horizon.append(row)

    per_task: dict[str, Any] = {}
    task_ids = arrays["task_id"].astype(np.int64)
    for task_id in range(10):
        mask = task_ids == task_id
        task_row: dict[str, Any] = {
            "branch_points": int(mask.sum()),
            "real_visual_cosine": summary(real_visual_cos[mask]),
            "real_visual_l2": summary(real_visual_l2[mask]),
            "real_proprio_standardized_rmse": summary(real_p_rmse[mask]),
            "pixel_mae": summary(pixel_mae[mask]),
            "proprio_visual_correlation": correlation(real_p_rmse[mask], real_visual_cos[mask]),
            "models": {},
        }
        for model in MODEL_NAMES:
            task_row["models"][model] = {
                "predicted_visual_cosine": summary(predicted_separation[model][mask]),
                "transfer": safe_ratio(
                    predicted_separation[model][mask], real_visual_cos[mask], minimum
                ),
                "A0_error": summary(candidate_error[model][mask, 0]),
                "A1_error": summary(candidate_error[model][mask, 1]),
                "A2_error": summary(candidate_error[model][mask, 2]),
            }
        task_row["wm_v2b_proprio_error"] = {
            f"A{candidate}": summary(v2b_p_error[mask, candidate])
            for candidate in range(3)
        }
        per_task[str(task_id)] = task_row

    per_pair: dict[str, Any] = {}
    for pair_index, pair_name in enumerate(PAIR_NAMES):
        pair_row: dict[str, Any] = {
            "real_visual_cosine": summary(real_visual_cos[:, pair_index]),
            "real_proprio_standardized_rmse": summary(real_p_rmse[:, pair_index]),
            "models": {},
        }
        for model in MODEL_NAMES:
            pair_row["models"][model] = {
                "predicted_visual_cosine": summary(predicted_separation[model][:, pair_index]),
                "transfer": safe_ratio(
                    predicted_separation[model][:, pair_index],
                    real_visual_cos[:, pair_index],
                    minimum,
                ),
            }
        per_pair[pair_name] = pair_row

    reference = protocol["decision_rules"]["reference_values"]
    visual_reference = float(reference["d15_ordinary_validation_visual_error"])
    physical_reference = float(reference["wm_v2b_validation_proprio_rmse"])
    real_visual_mean = float(real_visual_cos.mean())
    real_p_mean = float(real_p_rmse.mean())
    horizon_real_means = real_visual_cos.mean(axis=(0, 1))
    visual_meaningful = bool(
        real_visual_mean / visual_reference >= 0.01
        and int(np.sum(horizon_real_means >= minimum)) >= 6
    )
    physical_meaningful = bool(
        real_p_mean / physical_reference >= 0.05
        and np.all(real_p_rmse.mean(axis=(0, 1)) > 0)
    )
    v2b_transfer = safe_ratio(
        predicted_separation["wm_v2b_seed11"], real_visual_cos, minimum
    )
    under_horizons = int(
        np.sum(
            predicted_separation["wm_v2b_seed11"].mean(axis=(0, 1))
            < horizon_real_means
        )
    )
    transfer_failure = bool(
        visual_meaningful
        and physical_meaningful
        and v2b_transfer["eligible"]
        and float(v2b_transfer["ratio_of_means"]) <= 0.50
        and under_horizons >= 6
    )
    clip_limit = bool(
        physical_meaningful
        and (
            real_visual_mean / visual_reference < 0.01
            or int(np.sum(horizon_real_means >= minimum)) < 6
        )
    )

    prediction_failures: dict[str, Any] = {}
    for model in MODEL_NAMES:
        error = candidate_error[model]
        a0 = error[:, 0]
        alternatives = error[:, 1:]
        ordinary = float(reference[
            {
                "wm_v1_d15_ensemble": "d15_ordinary_validation_visual_error",
                "wm_v2a_seed11": "wm_v2a_ordinary_validation_visual_error",
                "wm_v2b_seed11": "wm_v2b_ordinary_validation_visual_error",
            }[model]
        ])
        matched_a0 = np.repeat(a0[:, None, :], 2, axis=1)
        worse_fraction = float(np.mean(alternatives > matched_a0))
        criteria = {
            "alternative_to_validation_error": float(alternatives.mean() / ordinary),
            "alternative_to_matched_A0_error": float(alternatives.mean() / a0.mean()),
            "alternative_worse_than_A0_fraction": worse_fraction,
        }
        failed = bool(
            criteria["alternative_to_validation_error"] >= 1.50
            and criteria["alternative_to_matched_A0_error"] >= 1.10
            and worse_fraction > 0.55
        )
        prediction_failures[model] = {**criteria, "fail": failed}

    verdicts = []
    if transfer_failure:
        verdicts.append("VISUAL_TRANSFER_FAILURE_CONFIRMED")
    if clip_limit:
        verdicts.append("LOCAL_CLIP_COUNTERFACTUAL_LIMIT")
    if any(value["fail"] for value in prediction_failures.values()):
        verdicts.append("WM_LOCAL_COUNTERFACTUAL_PREDICTION_FAIL")
    if not verdicts:
        verdicts.append("LOCAL_COUNTERFACTUAL_AUDIT_INCONCLUSIVE")

    prefix_records = [record["prefix_audit"] for record in records]
    candidate_hashes = [record["candidate_sha256"] for record in records]
    analysis = {
        "real_counterfactual_dynamics": {
            "visual_cosine": summary(real_visual_cos),
            "visual_l2": summary(real_visual_l2),
            "proprio_standardized_rmse": summary(real_p_rmse),
            "proprio_raw_l2": summary(real_p_raw_l2),
            "pixel_mae": summary(pixel_mae),
            "pixel_rmse": summary(pixel_rmse),
            "unequal_pixel_fraction": summary(pixel_fraction),
            "real_visual_to_d15_validation_error": real_visual_mean / visual_reference,
            "real_proprio_to_wm_v2b_validation_rmse": real_p_mean / physical_reference,
        },
        "model_transfer": {
            model: {
                "predicted_visual_cosine": summary(predicted_separation[model]),
                "transfer": safe_ratio(predicted_separation[model], real_visual_cos, minimum),
                "candidate_error": {
                    f"A{candidate}": summary(candidate_error[model][:, candidate])
                    for candidate in range(3)
                },
            }
            for model in MODEL_NAMES
        },
        "wm_v2b_proprio": {
            "candidate_error": {
                f"A{candidate}": summary(v2b_p_error[:, candidate])
                for candidate in range(3)
            },
            "predicted_candidate_separation": summary(v2b_p_separation),
        },
        "proprio_visual_correlation": {
            "overall": correlation(real_p_rmse, real_visual_cos),
            "per_horizon": [
                {"horizon": horizon + 1, **correlation(real_p_rmse[:, :, horizon], real_visual_cos[:, :, horizon])}
                for horizon in range(8)
            ],
        },
        "per_horizon": per_horizon,
        "per_task": per_task,
        "per_candidate_pair": per_pair,
        "prefix_determinism": {
            "branch_points": len(prefix_records),
            "all_pass": all(record["pass"] for record in prefix_records),
            "rgb_exact_all": all(record["rgb_exact"] for record in prefix_records),
            "planning_observation_exact_all": all(
                record["planning_observation"]["exact"] for record in prefix_records
            ),
            "max_proprio_abs_difference": max(
                record["proprio_max_abs_difference"] for record in prefix_records
            ),
            "max_simulator_state_abs_difference": max(
                record["simulator_state_max_abs_difference"] for record in prefix_records
            ),
            "minimum_clip_cosine": min(
                min(record["clip_cosine_to_reference"]) for record in prefix_records
            ),
        },
        "candidate_hashes": {
            "count": len(candidate_hashes),
            "ordered_manifest_sha256": stable_json_sha256(candidate_hashes),
            "values": candidate_hashes,
        },
        "decision": {
            "real_visual_meaningful": visual_meaningful,
            "real_physical_meaningful": physical_meaningful,
            "wm_v2b_predicted_below_real_horizons": under_horizons,
            "visual_transfer_failure_confirmed": transfer_failure,
            "local_clip_counterfactual_limit": clip_limit,
            "prediction_failure_by_model": prediction_failures,
            "verdicts": verdicts,
            "final_verdict": (
                verdicts[0]
                if len(verdicts) == 1
                else "MIXED: " + " + ".join(verdicts)
            ),
        },
    }
    metric_arrays = {
        "real_visual_cosine": real_visual_cos.astype(np.float32),
        "real_visual_l2": real_visual_l2.astype(np.float32),
        "real_proprio_standardized_rmse": real_p_rmse.astype(np.float32),
        "real_proprio_raw_l2": real_p_raw_l2.astype(np.float32),
        "pixel_mae": pixel_mae.astype(np.float32),
        "pixel_rmse": pixel_rmse.astype(np.float32),
        "unequal_pixel_fraction": pixel_fraction.astype(np.float32),
        "wm_v2b_proprio_error": v2b_p_error.astype(np.float32),
        "wm_v2b_predicted_proprio_separation": v2b_p_separation.astype(np.float32),
    }
    for model in MODEL_NAMES:
        metric_arrays[f"{model}_predicted_visual_separation"] = predicted_separation[model].astype(np.float32)
        metric_arrays[f"{model}_candidate_visual_error"] = candidate_error[model].astype(np.float32)
    return analysis, metric_arrays


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    decision = analysis["decision"]
    real = analysis["real_counterfactual_dynamics"]
    transfer_rows = []
    for model in MODEL_NAMES:
        item = analysis["model_transfer"][model]
        transfer = item["transfer"]
        transfer_rows.append(
            [
                model,
                f"{item['predicted_visual_cosine']['mean']:.8g}",
                f"{transfer['real_mean']:.8g}",
                "n/a" if transfer["ratio_of_means"] is None else f"{transfer['ratio_of_means']:.4f}",
                f"{item['candidate_error']['A0']['mean']:.6f}",
                f"{item['candidate_error']['A1']['mean']:.6f}",
                f"{item['candidate_error']['A2']['mean']:.6f}",
            ]
        )
    horizon_rows = []
    for row in analysis["per_horizon"]:
        horizon_rows.append(
            [
                str(row["horizon"]),
                f"{row['real_visual_cosine']['mean']:.8g}",
                f"{row['real_proprio_standardized_rmse']['mean']:.6f}",
                f"{row['pixel_mae']['mean']:.4f}",
                *[
                    "n/a"
                    if row["models"][model]["transfer"]["ratio_of_means"] is None
                    else f"{row['models'][model]['transfer']['ratio_of_means']:.3f}"
                    for model in MODEL_NAMES
                ],
            ]
        )
    task_rows = []
    for task_id, row in analysis["per_task"].items():
        task_rows.append(
            [
                task_id,
                f"{row['real_visual_cosine']['mean']:.8g}",
                f"{row['real_proprio_standardized_rmse']['mean']:.6f}",
                *[
                    "n/a"
                    if row["models"][model]["transfer"]["ratio_of_means"] is None
                    else f"{row['models'][model]['transfer']['ratio_of_means']:.3f}"
                    for model in MODEL_NAMES
                ],
            ]
        )
    correlation_rows = []
    for row in analysis["proprio_visual_correlation"]["per_horizon"]:
        correlation_rows.append(
            [
                str(row["horizon"]),
                "n/a" if row["pearson"] is None else f"{row['pearson']:.4f}",
                "n/a" if row["spearman"] is None else f"{row['spearman']:.4f}",
                str(row["n"]),
            ]
        )
    prefix = analysis["prefix_determinism"]
    identity_rows = [
        [
            str(item["task_id"]),
            str(item["init_state_index"]),
            str(item["seed"]),
            str(item["planning_index"]),
            str(item["prefix_steps"]),
            item["candidate_sha256"][:16] + "…",
        ]
        for item in result["branch_identities"]
    ]
    prediction_failure_rows = [
        [
            model,
            f"{values['alternative_to_validation_error']:.3f}x",
            f"{values['alternative_to_matched_A0_error']:.3f}x",
            f"{100*values['alternative_worse_than_A0_fraction']:.2f}%",
            "yes" if values["fail"] else "no",
        ]
        for model, values in decision["prediction_failure_by_model"].items()
    ]
    return f"""# Matched simulator branches reveal the local candidate dynamics mechanism

Date: 2026-09-24

Final verdict: `{decision['final_verdict']}`

## Executive summary

This audit uses {result['scope']['branch_points']} preregistered branch points across all
ten LIBERO Spatial tasks. Each A0/A1/A2 future comes from a fresh environment
that starts at the same official init state and replays the same stored raw
prefix actions. No mid-episode simulator-state restore, reward, success, or
task-completion signal is used.

Real candidate futures have mean visual cosine separation
`{real['visual_cosine']['mean']:.8g}` and standardized-proprio RMSE separation
`{real['proprio_standardized_rmse']['mean']:.6f}`. Relative to the frozen
D15 ordinary validation error, real visual separation is
`{100*real['real_visual_to_d15_validation_error']:.3f}%`; relative to the
frozen WM-v2B proprio prediction RMSE, real physical separation is
`{100*real['real_proprio_to_wm_v2b_validation_rmse']:.2f}%`.

The preregistered rule assigns `{decision['final_verdict']}`. This is a
mechanism conclusion only. It does not rank candidates by task success and
does not authorize a controller, threshold, candidate-scale, or model change.

## Frozen provenance and data boundary

- WM-v2B freeze commit: `{result['protocol']['wm_v2b_commit']}`.
- Annotated tag: `{result['protocol']['wm_v2b_tag']}`.
- Counterfactual protocol commit: `{result['protocol']['preregistered_commit']}`.
- Protocol SHA-256: `{result['protocol']['sha256']}`.
- Frozen C0-v2, C1-v2 generator, D15 ensemble, WM-v2A, and WM-v2B hashes all passed before execution.
- Only controller-development init states 30 and 35 were used; 0--29 and 40--49 were not run.
- No reward, done, success, or final task completion value is present in the result or sidecar.

## Preregistered branch identities and candidate hashes

{markdown_table(['Task', 'Init', 'Seed', 'Plan', 'Prefix steps', 'Candidate SHA-256'], identity_rows)}

The ordered 40-candidate hash manifest is
`{analysis['candidate_hashes']['ordered_manifest_sha256']}`. A candidate tensor
was generated once at its reference planning observation and copied unchanged
to all three independent branches.

## Prefix determinism passed before every branch

All {prefix['branch_points']} branch points pass. RGB and every planning
observation array are bitwise identical. Maximum proprio difference is
`{prefix['max_proprio_abs_difference']:.3g}`, maximum flattened simulator-state
difference is `{prefix['max_simulator_state_abs_difference']:.3g}`, and minimum
CLIP cosine to the reference planning observation is
`{prefix['minimum_clip_cosine']:.9f}`. Flattened state is used only for this
identity audit and never for restore.

## Real and predicted local visual separation

{markdown_table(['Model', 'Predicted cosine', 'Real cosine', 'Pred/real', 'A0 error', 'A1 error', 'A2 error'], transfer_rows)}

Ratios use ratio-of-means and are withheld when the matching real group mean
is below the preregistered `1e-5` denominator floor. Absolute predicted, real,
and signed differences remain in the JSON for every horizon, task, and pair.

## Per-horizon dynamics

{markdown_table(['h', 'Real visual cosine', 'Real proprio RMSE', 'Pixel MAE', 'WM-v1 ratio', 'WM-v2A ratio', 'WM-v2B ratio'], horizon_rows)}

## Per-task dynamics

{markdown_table(['Task', 'Real visual cosine', 'Real proprio RMSE', 'WM-v1 ratio', 'WM-v2A ratio', 'WM-v2B ratio'], task_rows)}

Each task contributes four branch points: init states 30/35 crossed with
planning indices 0/8. Per-pair mean/median/p75/p90/p95 and model ratios are in
the compact diagnosis JSON.

## Candidate-specific future prediction fidelity

{markdown_table(['Model', 'A1/A2 vs validation', 'A1/A2 vs matched A0', 'Alt worse fraction', 'Fail'], prediction_failure_rows)}

The prediction-failure gate requires all three preregistered conditions. It
therefore distinguishes general policy-rollout error from an additional error
specific to nearby A1/A2 counterfactual actions.

WM-v2B standardized-proprio RMSE is
`{analysis['wm_v2b_proprio']['candidate_error']['A0']['mean']:.6f}` for A0,
`{analysis['wm_v2b_proprio']['candidate_error']['A1']['mean']:.6f}` for A1,
and `{analysis['wm_v2b_proprio']['candidate_error']['A2']['mean']:.6f}` for A2.

## Does local physical motion map into CLIP motion?

Overall Pearson correlation is
`{analysis['proprio_visual_correlation']['overall']['pearson']}` and Spearman
correlation is `{analysis['proprio_visual_correlation']['overall']['spearman']}`.

{markdown_table(['h', 'Pearson', 'Spearman', 'n'], correlation_rows)}

This correlation is descriptive across candidate pairs and branch points; it
does not make future proprio a task-success objective.

## Mechanical decision

- Real visual meaningful: `{decision['real_visual_meaningful']}`.
- Real physical meaningful: `{decision['real_physical_meaningful']}`.
- WM-v2B predicted separation is below real separation on `{decision['wm_v2b_predicted_below_real_horizons']}/8` horizons.
- `VISUAL_TRANSFER_FAILURE_CONFIRMED`: `{decision['visual_transfer_failure_confirmed']}`.
- `LOCAL_CLIP_COUNTERFACTUAL_LIMIT`: `{decision['local_clip_counterfactual_limit']}`.
- Any model meets `WM_LOCAL_COUNTERFACTUAL_PREDICTION_FAIL`: `{any(value['fail'] for value in decision['prediction_failure_by_model'].values())}`.

Final verdict: `{decision['final_verdict']}`.

## Integrity and forbidden-activity audit

- Compact result: `{result['artifacts']['result']}`, SHA-256 recorded after write.
- Ignored sidecar: `{result['artifacts']['sidecar']}`, SHA-256 `{result['artifacts']['sidecar_sha256']}`.
- Sidecar arrays: all {len(result['artifacts']['sidecar_arrays'])} named arrays are finite where numeric; RGB is uint8.
- NaN/Inf audit: `{result['nan_inf_audit']}`.
- Simulator/controller success comparison: not performed.
- Model training, candidate modification, reranker modification, D20-v2, C2-v2, and init-state 40--49 access: not performed.
"""


def blocker_result(
    protocol_path: Path,
    protocol: dict[str, Any],
    preregistered_commit: str,
    amendment_path: Path | None,
    amendment: dict[str, Any] | None,
    failure: CounterfactualReplayError,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_version": (
            protocol["protocol_version"]
            if amendment is None
            else amendment["amendment_version"]
        ),
        "status": "BLOCKED",
        "verdict": "BLOCK_COUNTERFACTUAL_REPLAY",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256(protocol_path),
            "preregistered_commit": preregistered_commit,
            "amendment": None if amendment_path is None else str(amendment_path.relative_to(ROOT)),
            "amendment_sha256": None if amendment_path is None else sha256(amendment_path),
            "wm_v2b_commit": protocol["frozen_upstream"]["wm_v2b_commit"],
            "wm_v2b_tag": protocol["frozen_upstream"]["wm_v2b_annotated_tag"],
        },
        "failure": {"message": str(failure), "prefix_audit": failure.audit},
        "forbidden_activity_audit": {
            "mid_episode_state_restore": False,
            "success_reward_done_used": False,
            "init_state_outside_30_39": False,
            "model_training": False,
            "controller_or_candidate_modified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/local_candidate_counterfactual_protocol.json",
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=ROOT / "configs/local_candidate_counterfactual_protocol_v1_1_amendment.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/local_candidate_counterfactual/diagnosis.json",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "artifacts/stage3/local_candidate_counterfactual/branches.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/local_candidate_counterfactual_dynamics_audit.md",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for path in (args.output, args.sidecar, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    protocol, preregistered_commit, amendment = validate_protocol(
        args.protocol, args.amendment
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        records, arrays = collect(protocol, device)
    except CounterfactualReplayError as failure:
        result = blocker_result(
            args.protocol,
            protocol,
            preregistered_commit,
            args.amendment,
            amendment,
            failure,
        )
        atomic_json(args.output, result)
        atomic_text(
            args.report,
            "# Deterministic prefix replay failed\n\n"
            "Final verdict: `BLOCK_COUNTERFACTUAL_REPLAY`\n\n"
            f"Failure: {failure}\n",
        )
        print(json.dumps({"verdict": result["verdict"], "failure": str(failure)}))
        return 2

    analysis, metric_arrays = analyze(protocol, records, arrays)
    arrays.update(metric_arrays)
    atomic_npz(args.sidecar, arrays)
    sidecar_hash = sha256(args.sidecar)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_version": amendment["amendment_version"],
        "status": "COMPLETE",
        "verdict": analysis["decision"]["final_verdict"],
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": sha256(args.protocol),
            "preregistered_commit": preregistered_commit,
            "base_preregistered_commit": amendment["base_preregistered_commit"],
            "amendment": str(args.amendment.relative_to(ROOT)),
            "amendment_sha256": sha256(args.amendment),
            "execution_source_commit": source_commit(),
            "wm_v2b_commit": protocol["frozen_upstream"]["wm_v2b_commit"],
            "wm_v2b_tag": protocol["frozen_upstream"]["wm_v2b_annotated_tag"],
        },
        "scope": {
            "suite": protocol["scope"]["suite"],
            "tasks": 10,
            "init_state_indices": protocol["scope"]["init_state_indices_per_task"],
            "planning_indices": protocol["scope"]["planning_indices"],
            "branch_points": len(records),
            "candidate_branches": len(records) * 3,
            "branch_horizon": 8,
            "rewards_recorded": False,
            "done_recorded": False,
            "success_recorded": False,
            "init_state_indices_read_or_run": protocol["scope"]["init_state_indices_per_task"],
        },
        "branch_identities": records,
        "analysis": analysis,
        "artifacts": {
            "result": str(args.output.relative_to(ROOT)),
            "sidecar": str(args.sidecar.relative_to(ROOT)),
            "sidecar_sha256": sidecar_hash,
            "sidecar_arrays": {key: list(value.shape) for key, value in arrays.items()},
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "mid_episode_state_restore": False,
            "success_reward_done_used": False,
            "init_state_outside_30_39": False,
            "model_training": False,
            "wm_v2b_modified": False,
            "candidate_scale_or_generator_modified": False,
            "reranker_modified": False,
            "d20_v2_or_c2_v2_started": False,
            "controller_tuned_from_results": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    atomic_json(args.output, result)
    result["artifacts"]["result_sha256_before_hash_field"] = sha256(args.output)
    # The compact file intentionally does not self-hash recursively. The report
    # and final handoff calculate its final SHA-256 after this last write.
    atomic_json(args.output, result)
    atomic_text(args.report, render_report(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "branch_points": len(records),
                "result": str(args.output),
                "sidecar_sha256": sidecar_hash,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
