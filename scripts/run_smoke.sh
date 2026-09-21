#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${SMOKE_OUTPUT_DIR:-${project_root}/artifacts/stage0}"
unset PYTHONPATH
export LIBERO_CONFIG_PATH="${project_root}/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

exec "${project_root}/.venv/bin/python" "${project_root}/scripts/smoke_libero.py" \
  --output-dir "${output_dir}" "$@"
