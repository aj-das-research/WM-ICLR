#!/usr/bin/env bash
set -euo pipefail
shiftwm_project_root="${SHIFTWM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
cd "$shiftwm_project_root"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/external/le-wm:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="$PWD/data/hf"
export STABLEWM_HOME="$PWD/data/upstream"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/shiftwm-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
python scripts/run_record.py -- "$@"
