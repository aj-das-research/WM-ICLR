#!/usr/bin/env bash
# Isolated SOFA/Python launch; does not alter the active project environment.
set -euo pipefail
surgery_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$surgery_dir/../.." && pwd)"
export SOFA_ROOT="$surgery_dir/SOFA_v24.06.00_Linux"
surgery_python_lib="$($surgery_dir/.venv/bin/python -c 'import sys; print(sys.base_prefix + "/lib")')"
export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$project_dir/external/sofa_env:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib:$surgery_python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SOFA_PLUGIN_PATH="$SOFA_ROOT/plugins:$SOFA_ROOT/lib"
export PYOPENGL_PLATFORM=egl
# The registered dataset was rendered by NVIDIA EGL device 0. Pyglet's
# EGL_PLATFORM_DEVICE_EXT selection ignores LIBGL_ALWAYS_SOFTWARE for NVIDIA.
# Pin the vendor explicitly; Mesa/llvmpipe is physically consistent but changes
# raster pixels and is not a drop-in renderer for these visual references.
surgery_egl_vendor=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [[ ! -r "$surgery_egl_vendor" ]]; then
  echo "Surgery requires the registered NVIDIA EGL renderer (RTX 5000 Ada, driver 570.195.03). Run inside an allocated GPU job; vendor file missing: $surgery_egl_vendor" >&2
  exit 1
fi
export __EGL_VENDOR_LIBRARY_FILENAMES="$surgery_egl_vendor"
export LIBGL_ALWAYS_SOFTWARE=0
export LP_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
exec "$surgery_dir/.venv/bin/python" "$@"
