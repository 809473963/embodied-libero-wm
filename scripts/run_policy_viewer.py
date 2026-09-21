#!/usr/bin/env python3
"""Render a deployed LIBERO policy live in a GUI window.

The policy adapter receives only ``agentview_image`` and robot proprioception;
object poses, simulator state, rewards, and success flags are deliberately not
exposed to it. An adapter factory uses the form ``module.submodule:function``
and must accept keyword arguments ``checkpoint`` and ``device``. It returns an
object exposing ``reset()`` and ``act(observation, language) -> action[7]``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pynput.keyboard import Key, Listener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv


ROBOT_OBSERVATION_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
)


class ViewerControls:
    """Global controls while a policy rollout is displayed."""

    def __init__(self) -> None:
        self.quit_requested = False
        self.reset_requested = False
        self.listener = Listener(on_release=self.on_release)
        self.listener.start()

    def on_release(self, key: object) -> None:
        if key == Key.esc:
            self.quit_requested = True
        else:
            try:
                if key.char == "q":
                    self.reset_requested = True
            except AttributeError:
                pass

    def close(self) -> None:
        self.listener.stop()


def get_trusted_task_init_states(task_suite: object, task_id: int) -> object:
    """Load LIBERO's fixed task-init asset under PyTorch 2.6+."""

    original_load = torch.load

    def load_with_trusted_default(*load_args: object, **load_kwargs: object) -> object:
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = load_with_trusted_default  # type: ignore[assignment]
    try:
        return task_suite.get_task_init_states(task_id)  # type: ignore[attr-defined]
    finally:
        torch.load = original_load  # type: ignore[assignment]


def load_policy(spec: str, checkpoint: str | None, device: str) -> Any:
    try:
        module_name, factory_name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError("--adapter must have the form 'module.submodule:factory'") from exc
    factory = getattr(importlib.import_module(module_name), factory_name)
    policy = factory(checkpoint=checkpoint, device=device)
    if not callable(getattr(policy, "act", None)):
        raise TypeError("adapter factory must return an object with act(observation, language)")
    return policy


def make_environment(
    task_suite_name: str, task_id: int, seed: int
) -> tuple[ControlEnv, str, dict[str, np.ndarray]]:
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_file.is_file():
        raise FileNotFoundError(bddl_file)
    env = ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=True,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_gpu_device_id=-1,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
        ignore_done=True,
        hard_reset=False,
        horizon=2000,
    )
    env.seed(seed)
    env.reset()
    observation = env.set_init_state(get_trusted_task_init_states(task_suite, task_id)[0])
    return env, task.language, observation


def policy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Filter raw simulator observations to the policy's declared input contract."""

    filtered = {"agentview_image": np.ascontiguousarray(observation["agentview_image"])}
    for key in ROBOT_OBSERVATION_KEYS:
        if key in observation:
            filtered[key] = np.ascontiguousarray(observation[key])
    return filtered


def checked_action(action: Any) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().float().cpu().numpy()
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape != (7,):
        raise ValueError(f"policy action must have shape (7,), received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("policy action contains non-finite values")
    return np.clip(array, -1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Python adapter factory: module:factory")
    parser.add_argument("--checkpoint", help="Optional checkpoint path passed to the adapter factory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--control-hz", type=float, default=20.0)
    args = parser.parse_args()

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("No GUI display detected. Run inside a WSLg/X11 desktop session.")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive")

    policy = load_policy(args.adapter, args.checkpoint, args.device)
    controls = ViewerControls()
    env = None
    try:
        env, language, observation = make_environment(args.task_suite, args.task_id, args.seed)
        print(f"Policy adapter: {args.adapter}")
        print(f"Task: {language}")
        print("The policy is acting. q resets the rollout; ESC exits.")
        env.env.render()
        if callable(getattr(policy, "reset", None)):
            policy.reset()

        step = 0
        while not controls.quit_requested and step < args.max_steps:
            if controls.reset_requested:
                controls.reset_requested = False
                env.close()
                env, language, observation = make_environment(args.task_suite, args.task_id, args.seed)
                env.env.render()
                if callable(getattr(policy, "reset", None)):
                    policy.reset()
                step = 0
                print(f"Reset task: {language}")

            step_started = time.perf_counter()
            action = checked_action(policy.act(policy_observation(observation), language))
            observation, _, _, _ = env.step(action)
            env.env.render()
            step += 1
            remaining = 1.0 / args.control_hz - (time.perf_counter() - step_started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        controls.close()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
