#!/usr/bin/env python3
"""Read-only audit of LIBERO-Spatial post-conversion temporal semantics.

This script never modifies source HDF5 trajectories, feature stores, or model
checkpoints.  It writes only its JSON evidence file.  The audit distinguishes
the HDF5's official imitation-learning array convention from the physical
action that advances one recorded observation to the next.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
# This is the literal value in LIBERO's pinned scripts/create_dataset.py:123.
CAP_INDEX = 5


def comma_separated_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def image_metrics(live: np.ndarray, recorded: np.ndarray) -> dict[str, float]:
    delta = np.asarray(live, dtype=np.float64) - np.asarray(recorded, dtype=np.float64)
    mse = float(np.mean(np.square(delta)))
    return {
        "pixel_mae": float(np.mean(np.abs(delta))),
        "pixel_mse": mse,
        "psnr_db": 100.0 if mse == 0.0 else float(10.0 * np.log10((255.0**2) / mse)),
    }


def proprio_from_live(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
            np.asarray(observation["robot0_joint_pos"], dtype=np.float32).reshape(7),
        )
    )


def proprio_from_recorded(demo: h5py.Group, index: int) -> np.ndarray:
    obs = demo["obs"]
    return np.concatenate(
        (
            obs["ee_pos"][index].astype(np.float32),
            obs["gripper_states"][index].astype(np.float32),
            obs["joint_states"][index].astype(np.float32),
        )
    )


def restore(env: OffScreenRenderEnv, state: np.ndarray) -> dict[str, np.ndarray]:
    """Reset controller state before restoring a MuJoCo state for one replay trial."""

    env.reset()
    return env.regenerate_obs_from_state(state)


def replay_metrics(
    env: OffScreenRenderEnv,
    start_state: np.ndarray,
    action: np.ndarray,
    target_state: np.ndarray,
    target_proprio: np.ndarray,
    target_rgb: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    observation = restore(env, start_state)
    post_observation, _, _, _ = env.step(action)
    result: dict[str, Any] = {
        "state_l2": l2(env.get_sim_state(), target_state),
        "proprio_l2": l2(proprio_from_live(post_observation), target_proprio),
        **image_metrics(post_observation["agentview_image"], target_rgb),
    }
    return result, post_observation["agentview_image"].copy(), target_rgb.copy()


def autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def attach_clip_cosines(
    image_pairs: list[tuple[dict[str, Any], np.ndarray, np.ndarray]], device: torch.device
) -> dict[str, Any]:
    """Attach frozen-CLIP cosine values to replay records in bounded batches."""

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=PROJECT_ROOT / ".cache/huggingface/hub",
            allow_patterns=["*.json", "vocab.json", "merges.txt", "pytorch_model.bin"],
            local_files_only=True,
        )
    )
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    values: list[float] = []
    with torch.inference_mode(), autocast(device):
        for start in range(0, len(image_pairs), 32):
            batch = image_pairs[start : start + 32]
            live_pixels = processor(images=[item[1] for item in batch], return_tensors="pt")["pixel_values"].to(device)
            target_pixels = processor(images=[item[2] for item in batch], return_tensors="pt")["pixel_values"].to(device)
            live_features = model.get_image_features(pixel_values=live_pixels)
            target_features = model.get_image_features(pixel_values=target_pixels)
            live_features = live_features / live_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cosines = (live_features * target_features).sum(dim=-1).float().cpu().numpy()
            for (record, _, _), cosine in zip(batch, cosines):
                value = float(cosine)
                record["clip_cosine_to_recorded_rgb"] = value
                values.append(value)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "local_snapshot": str(snapshot),
        "samples": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def median(records: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(record[key]) for record in records]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial")
    parser.add_argument("--task-ids", type=comma_separated_ints, default=[0, 4, 8])
    parser.add_argument("--demo-ids", type=comma_separated_ints, default=[0, 17, 44])
    parser.add_argument("--timesteps", type=comma_separated_ints, default=[0, 5, 10])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-clip", action="store_true", help="diagnostic only; normal audit must retain CLIP")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/stage1/temporal_alignment.json")
    args = parser.parse_args()

    if len(args.task_ids) < 3 or len(args.demo_ids) < 3 or len(args.timesteps) < 3:
        raise ValueError("audit requires at least 3 tasks, 3 demos per task, and 3 timesteps per demo")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    replay_rows: list[dict[str, Any]] = []
    same_index_records: list[dict[str, Any]] = []
    shifted_records: list[dict[str, Any]] = []
    pre_state_records: list[dict[str, Any]] = []
    clip_pairs: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []

    for task_id in args.task_ids:
        if task_id < 0 or task_id >= suite.n_tasks:
            raise ValueError(f"invalid task_id {task_id}")
        task = suite.get_task(task_id)
        dataset_path = args.dataset_dir / f"{task.name}_demo.hdf5"
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=128,
            camera_widths=128,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
        )
        try:
            with h5py.File(dataset_path, "r") as source:
                for demo_id in args.demo_ids:
                    demo_name = f"demo_{demo_id}"
                    if demo_name not in source["data"]:
                        raise KeyError(f"{dataset_path}: missing {demo_name}")
                    demo = source["data"][demo_name]
                    states = demo["states"][:]
                    actions = demo["actions"][:]
                    rgb = demo["obs/agentview_rgb"][:]
                    if not (len(states) == len(actions) == len(rgb)):
                        raise ValueError(f"{dataset_path}/{demo_name}: processed arrays have unequal lengths")
                    for k in args.timesteps:
                        if k < 0 or k + 2 >= len(actions):
                            raise ValueError(f"{dataset_path}/{demo_name}: timestep {k} lacks k+2 target")
                        current = restore(env, states[k + 1])
                        current_metrics = {
                            "restored_state_l2": l2(env.get_sim_state(), states[k + 1]),
                            "recorded_obs_k_proprio_l2": l2(proprio_from_live(current), proprio_from_recorded(demo, k)),
                            **{
                                f"recorded_obs_k_{name}": value
                                for name, value in image_metrics(current["agentview_image"], rgb[k]).items()
                            },
                        }
                        target_proprio = proprio_from_recorded(demo, k + 1)
                        same, same_live, same_target = replay_metrics(
                            env, states[k + 1], actions[k], states[k + 2], target_proprio, rgb[k + 1]
                        )
                        shifted, shifted_live, shifted_target = replay_metrics(
                            env, states[k + 1], actions[k + 1], states[k + 2], target_proprio, rgb[k + 1]
                        )
                        pre_state, pre_live, pre_target = replay_metrics(
                            env, states[k], actions[k], states[k + 1], proprio_from_recorded(demo, k), rgb[k]
                        )
                        row = {
                            "task_id": task_id,
                            "task_name": task.name,
                            "demo_id": demo_id,
                            "processed_timestep_k": k,
                            "raw_index_j": k + CAP_INDEX,
                            "current_observation_state_index": k + 1,
                            "next_observation_state_index": k + 2,
                            "restore_check": current_metrics,
                            "candidate_1_obs_k_plus_action_k_to_obs_k_plus_1": same,
                            "candidate_2_obs_k_plus_action_k_plus_1_to_obs_k_plus_1": shifted,
                            "supporting_pre_state_transition_state_k_plus_action_k_to_obs_k": pre_state,
                        }
                        replay_rows.append(row)
                        same_index_records.append(same)
                        shifted_records.append(shifted)
                        pre_state_records.append(pre_state)
                        clip_pairs.extend(
                            [(same, same_live, same_target), (shifted, shifted_live, shifted_target), (pre_state, pre_live, pre_target)]
                        )
        finally:
            env.close()

    if args.skip_clip:
        clip_summary: dict[str, Any] = {"skipped": True, "reason": "--skip-clip was explicitly supplied"}
    else:
        clip_summary = attach_clip_cosines(clip_pairs, device)

    shifted_state_wins = sum(
        float(shifted["state_l2"]) < float(same["state_l2"])
        for same, shifted in zip(same_index_records, shifted_records)
    )
    result = {
        "schema_version": 1,
        "mode": "read_only_source_and_feature_audit",
        "input_mutation": "none",
        "libero_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT / "third_party/LIBERO"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "converter_source": "third_party/LIBERO/scripts/create_dataset.py",
        "converter_source_lines": {
            "cap_index": 123,
            "step_before_recording_obs": [175, 177, 190, 195, 220],
            "post_loop_subset": [226, 227],
            "processed_write": [248, 260, 261],
        },
        "official_loader_source": "third_party/LIBERO/libero/lifelong/datasets.py",
        "official_loader_convention": "SequenceDataset stores obs and actions under the same sequence index (dataset_keys=['actions']; load_next_obs=False).",
        "source_semantic_trace": {
            "raw_trajectory_availability": "The pre-conversion raw input HDF5 is not retained locally; this mapping follows the pinned converter source and is independently checked by state/action replay of the published processed HDF5.",
            "cap_index": CAP_INDEX,
            "valid_index_definition": "valid_index[k] = k + cap_index for k >= 0",
            "processed_actions_k": "raw_actions[k + cap_index]",
            "processed_states_k": "raw_states[k + cap_index] (the pre-step simulator state selected after the replay loop)",
            "processed_obs_k": "observation returned after env.step(raw_actions[k + cap_index]), which the converter checks against raw_states[k + cap_index + 1]",
            "derived_current_state_for_processed_obs_k": "processed_states[k + 1]",
            "derived_observed_transition": "processed_obs[k] + processed_actions[k + 1] -> processed_obs[k + 1]",
        },
        "replay_protocol": {
            "tasks": args.task_ids,
            "demos_per_task": args.demo_ids,
            "timesteps_per_demo": args.timesteps,
            "trials": len(replay_rows),
            "candidate_1": "restore processed_states[k+1] (state underlying obs[k]); step processed_actions[k]; compare to processed_states[k+2] and obs[k+1]",
            "candidate_2": "restore processed_states[k+1] (state underlying obs[k]); step processed_actions[k+1]; compare to processed_states[k+2] and obs[k+1]",
            "controller_caveat": "A MuJoCo state vector does not serialize robosuite controller goal internals. Each independent restore resets that controller, so absolute replay errors are expected; candidate comparison and the pinned converter's own state check establish index causality.",
        },
        "summary": {
            "candidate_1_median": {key: median(same_index_records, key) for key in ("state_l2", "proprio_l2", "pixel_mae", "psnr_db")},
            "candidate_2_median": {key: median(shifted_records, key) for key in ("state_l2", "proprio_l2", "pixel_mae", "psnr_db")},
            "supporting_pre_state_median": {key: median(pre_state_records, key) for key in ("state_l2", "proprio_l2", "pixel_mae", "psnr_db")},
            "candidate_2_state_l2_wins": shifted_state_wins,
            "candidate_comparisons": len(replay_rows),
        },
        "clip": clip_summary,
        "interpretation": {
            "official_il_pairing": "obs[k] -> actions[k] (same HDF5 sequence index)",
            "physical_observed_transition_pairing": "obs[k] + actions[k+1] -> obs[k+1]",
            "c0_pairing": "Keep obs[k] -> action_chunks[k:k+H] to preserve the official IL convention; this audit does not authorize shifting the existing C0 labels.",
            "d13_world_model_pairing": "If a future D13 model predicts the next observed latent, use z_obs[k], action[k+1], z_obs[k+1], with boundary handling that excludes the final unavailable next observation.",
        },
        "replay_rows": replay_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": result["summary"], "clip": clip_summary}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"temporal alignment audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
