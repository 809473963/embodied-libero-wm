#!/usr/bin/env python3
"""Open a keyboard-controllable LIBERO task in a MuJoCo GUI window."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from pynput.keyboard import Key
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv


class TeleopKeyboard(Keyboard):
    """Use robosuite's keyboard mappings and add an explicit ESC exit."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.quit_requested = False
        super().__init__(*args, **kwargs)

    def on_release(self, key: object) -> None:
        super().on_release(key)
        if key == Key.esc:
            self.quit_requested = True


def get_trusted_task_init_states(task_suite: object, task_id: int) -> object:
    """Read LIBERO's Git-pinned task states under PyTorch 2.6+."""

    original_load = torch.load

    def load_with_trusted_default(*load_args: object, **load_kwargs: object) -> object:
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = load_with_trusted_default  # type: ignore[assignment]
    try:
        return task_suite.get_task_init_states(task_id)  # type: ignore[attr-defined]
    finally:
        torch.load = original_load  # type: ignore[assignment]


def make_environment(task_suite_name: str, task_id: int, seed: int) -> tuple[ControlEnv, str]:
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_file.is_file():
        raise FileNotFoundError(bddl_file)

    env = ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        render_gpu_device_id=-1,
        ignore_done=True,
        hard_reset=False,
        horizon=2000,
    )
    env.seed(seed)
    env.reset()
    init_states = get_trusted_task_init_states(task_suite, task_id)
    env.set_init_state(init_states[0])
    return env, task.language


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pos-sensitivity", type=float, default=0.35)
    parser.add_argument("--rot-sensitivity", type=float, default=0.5)
    args = parser.parse_args()

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("No GUI display detected. Run this inside a WSLg/X11 desktop session.")

    env = None
    device = None
    try:
        env, language = make_environment(args.task_suite, args.task_id, args.seed)
        print(f"Task: {language}")
        print("Focus the MuJoCo window before pressing keys. ESC exits; q resets the task.")

        device = TeleopKeyboard(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.env.viewer.add_keypress_callback(device.on_press)
        env.env.render()
        device.start_control()

        while not device.quit_requested:
            action, _ = input2action(device=device, robot=env.robots[0])
            if action is None:
                env.close()
                env, language = make_environment(args.task_suite, args.task_id, args.seed)
                env.env.viewer.add_keypress_callback(device.on_press)
                env.env.render()
                device.start_control()
                print(f"Reset task: {language}")
                continue
            env.step(action)
            env.env.render()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        if device is not None:
            device.listener.stop()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
