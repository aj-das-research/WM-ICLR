#!/usr/bin/env bash
# Full official-config reproduction on the isolated compatible historical backend.
# This script does not install dependencies or modify live training imports.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
bundle="$project_root/artifacts/upstream_planning_reproduction"
upstream_python="$bundle/.venv/bin/python"
environment="${1:-}"
case "$environment" in pusht|reacher) ;; *) echo "Usage: bash scripts/run_upstream_planning_reproduction.sh {pusht|reacher} [--config-only]" >&2; exit 2;; esac
hydra_inspection=()
if [[ "${2:-}" == "--config-only" && $# -eq 2 ]]; then
  hydra_inspection=(--cfg job)
  export CUDA_VISIBLE_DEVICES=""
elif [[ $# -ne 1 ]]; then
  echo "Only an environment and optional --config-only are accepted." >&2
  exit 2
fi
if [[ ! -x "$upstream_python" ]]; then
  echo "Missing isolated reproduction environment: $upstream_python (see reports/upstream_planning_reproduction.md)" >&2
  exit 3
fi
export PYTHONPATH="$bundle/stable-worldmodel:$bundle/le-wm"
export STABLEWM_HOME="$bundle/cache"
export MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
"$upstream_python" -c 'import stable_pretraining, sklearn, stable_worldmodel, os; from pathlib import Path; assert Path(stable_worldmodel.__file__).resolve().is_relative_to(Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()); from stable_worldmodel.solver import CEMSolver'
cd "$bundle/le-wm"
exec "$upstream_python" eval.py --config-name "$environment" \
  "policy=$environment/lewm" "cache_dir=$bundle/cache" \
  "output.filename=${environment}_results.txt" \
  "hydra.run.dir=$bundle/logs/$environment" "${hydra_inspection[@]}"
