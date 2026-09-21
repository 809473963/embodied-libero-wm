#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID
export LIBERO_CONFIG_PATH="${project_root}/.libero"
export MUJOCO_GL=glfw

exec "${project_root}/.venv/bin/python" "${project_root}/scripts/teleop_libero.py" "$@"
