# Embodied LIBERO World Model

This repository is prepared for the `LIBERO-Spatial` stage of the prediction-verified embodied-agent project.

## Environment

Run the verified smoke test from the repository root:

```bash
./scripts/run_smoke.sh
```

It uses the project-local Python environment, clears the inherited ROS `PYTHONPATH`, sets the project-local LIBERO configuration, and enables MuJoCo EGL rendering on GPU 0. Evidence is written to `artifacts/stage0/`.

## Interactive simulation

WSLg is available on this machine, so open a real-time MuJoCo window with:

```bash
./scripts/run_teleop.sh
```

Focus that window before using the controls: `w/a/s/d` moves horizontally; `r/f` moves vertically; `z/x`, `t/g`, and `c/v` rotate around the x, y, and z axes; `Space` toggles the gripper; `q` resets the fixed task; `Esc` exits. Choose another Spatial task with, for example, `./scripts/run_teleop.sh --task-id 3`.

## Live policy viewer

Once a C0/C1/C2 checkpoint exists, run it in the real-time GUI with an adapter factory:

```bash
./scripts/run_policy_viewer.sh \
  --adapter your_package.your_policy:create_policy \
  --checkpoint results/checkpoints/your_checkpoint.pt
```

The factory receives keyword arguments `checkpoint` and `device`; it must return an object with `reset()` and `act(observation, language)`. `act` must return a finite normalized NumPy or Torch action with shape `(7,)`. Its observation contains only `agentview_image` plus robot proprioception, never object poses, rewards, simulator state, or success labels.

The current `artifacts/stage1/c0_backward_probe` checkpoint is only a two-step optimizer / recovery probe, not a trained policy and must not be evaluated as one. Once a real C0 checkpoint is trained, it can run in the window through the included adapter:

```bash
./scripts/run_policy_viewer.sh \
  --adapter src.policies.c0_actor_policy:create_policy \
  --checkpoint results/c0/<run-id>/best.pt
```

The adapter executes an `H=8` action chunk one 7-D action at a time and uses only the C0 contract's RGB, language, end-effector position, gripper state, and joint position. Verify the GUI-to-policy wiring without claiming learned behavior by running the explicit zero-action fixture:

```bash
./scripts/run_policy_viewer.sh --adapter src.policies.zero_policy:create_policy
```

## C0 single-task reproduction

The first completed C0 run is deliberately limited to Spatial task 0 and training seed 11. It is a stage-1 baseline check, not the 10-task/three-seed main result.

```bash
env -u PYTHONPATH .venv/bin/python scripts/train_c0.py \
  --task-id 0 --in-memory --seed 11 \
  --output-dir results/c0/task0_seed11

env -u PYTHONPATH LIBERO_CONFIG_PATH="$PWD/.libero" MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl .venv/bin/python scripts/evaluate_c0.py \
  --checkpoint results/c0/task0_seed11/best.pt --task-id 0 \
  --output results/c0/task0_seed11/eval_task0_seed11.json
```

The earlier flipped-image run reported 12/20 success (0.60); the corrected upright-RGB run reports 11/20 (0.55) on the same episodes. Both are preserved as an honest preprocessing ablation in `results/c0/`; the corrected RGB convention is retained, and neither run is the final benchmark score.

The upstream LIBERO source and official Spatial demonstrations are versioned in `configs/source_versions.yaml`; data file checksums are recorded in `artifacts/stage0/data_manifest.json`.

The project environment reuses the already-verified local `torch==2.6.0+cu124` runtime to avoid a duplicate multi-GB CUDA installation; its version and location are recorded in the freeze artifacts. Do not delete `~/.local/lib/python3.10/site-packages/torch` or its `nvidia` runtime packages while using this environment. All LIBERO, MuJoCo, robosuite, and dataset dependencies are project-local.

## Scope

Only the official 10-task `libero_spatial` demonstrations are present. The other LIBERO suites have deliberately not been downloaded.
