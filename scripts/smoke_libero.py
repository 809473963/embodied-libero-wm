#!/usr/bin/env python3
"""Run the stage-0 LIBERO environment smoke test and save verifiable evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def get_trusted_task_init_states(task_suite: object, task_id: int) -> object:
    """Load LIBERO's Git-pinned task states under PyTorch 2.6+.

    LIBERO 0.1.0 calls ``torch.load`` without ``weights_only``. PyTorch 2.6
    changed that default to True, while LIBERO's checked-in numpy state files
    require regular pickle deserialization. Scope the compatibility behavior to
    this known local benchmark asset rather than changing global loading policy.
    """

    original_load = torch.load

    def load_with_trusted_default(*load_args: object, **load_kwargs: object) -> object:
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = load_with_trusted_default  # type: ignore[assignment]
    try:
        return task_suite.get_task_init_states(task_id)  # type: ignore[attr-defined]
    finally:
        torch.load = original_load  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    libero_root = project_root / "third_party" / "LIBERO"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "smoke_run.json"
    screenshot_path = args.output_dir / "smoke_agentview.png"
    env = None

    result: dict[str, object] = {
        "libero_commit": git_commit(libero_root),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "reset_ok": False,
        "step_ok": False,
        "success_api_checked": False,
    }

    try:
        task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
        task = task_suite.get_task(args.task_id)
        task_bddl_file = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        if not task_bddl_file.is_file():
            raise FileNotFoundError(task_bddl_file)

        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=128,
            camera_widths=128,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
        )
        env.seed(args.seed)
        env.reset()
        result["reset_ok"] = True

        init_states = get_trusted_task_init_states(task_suite, args.task_id)
        observation = env.set_init_state(init_states[0])
        action = np.zeros(7, dtype=np.float32)
        observation, reward, done, info = env.step(action)
        result["step_ok"] = True

        rgb = observation["agentview_image"]
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"unexpected agentview image shape: {rgb.shape}")
        if not cv2.imwrite(str(screenshot_path), rgb[::-1, :, ::-1]):
            raise RuntimeError(f"could not save screenshot to {screenshot_path}")

        result.update(
            {
                "task_name": task.name,
                "task_language": task.language,
                "rgb_shape": list(rgb.shape),
                "action_shape": list(action.shape),
                "reward_after_one_zero_action": float(reward),
                "done_after_one_zero_action": bool(done),
                "success_after_one_zero_action": bool(env.check_success()),
                "success_api_checked": True,
                "screenshot": str(screenshot_path.relative_to(project_root)),
                "observation_keys": sorted(observation.keys()),
                "info_keys": sorted(info.keys()),
            }
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        raise
    else:
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    finally:
        if env is not None:
            env.close()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
