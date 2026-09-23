#!/bin/bash
# Shared setup for the DINO-WM (+ShiftWM head) external-benchmark tasks. Sourced by dinowm_<env>_{base,plugin,plan}.sh.
# Uses the separate venv environments/dinowm/.venv (overrides the .venv activated by job.sbatch).
set -eo pipefail
export PROJ=/home/Test/abhijit.das/projects/WM-ICLR
cd "$PROJ"
deactivate 2>/dev/null || true
source environments/dinowm/.venv/bin/activate
export PYTHONPATH="$PROJ/src:$PROJ/external/dino_wm"
export DATASET_DIR="$PROJ/data/dinowm"
export TORCH_HOME="$PROJ/.cache/torch"          # pre-populated: DINOv2 hub repo+weights, VGG16 (LPIPS)
export WANDB_MODE=offline WANDB_DIR="$PROJ/runs/dinowm_plugin/wandb" WANDB_SILENT=true
export HYDRA_FULL_ERROR=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export DINOWM_FAST_SLICES=${DINOWM_FAST_SLICES:-1}  # output-identical PushT slice decoding (see shims.py/tests)
mkdir -p "$WANDB_DIR"
RUNS=$PROJ/runs/dinowm_plugin                      # checkpoints (gitignored)
RES=$PROJ/results/v2/external/dinowm_plugin
SCR=$PROJ/scripts/external_dinowm_plugin
PLUGIN_TARGET=shiftwm.v2.dinowm_plugin.ShiftViTPredictor

# Per-environment official settings (conf/env/*.yaml + README: frameskip 5, num_hist 3; all else from train.yaml)
# EPOCHS is the reduced, arm-identical budget (official default training.epochs=100).
env_settings() {
  case "$1" in
    pusht) EPOCHS=${EPOCHS:-8};  PLAN_CFG=plan_pusht.yaml ;;
    wall)  EPOCHS=${EPOCHS:-20}; PLAN_CFG=plan_wall.yaml ;;
    point_maze) EPOCHS=${EPOCHS:-10}; PLAN_CFG=plan_point_maze.yaml; export DINOWM_ENV=point_maze ;;
    *) echo "unknown env $1"; exit 2 ;;
  esac
}

# train_arm ENV ARM(dinowm|dinowm_shiftwm) : resumable official training + open-loop eval + copy results
train_arm() {
  local env=$1 arm=$2; env_settings "$env"
  local name=${env}_${arm} extra=()
  [ "$arm" = dinowm_shiftwm ] && extra+=("predictor._target_=$PLUGIN_TARGET")
  local rd=$RUNS/outputs/$name out=$RES/$env/$arm
  mkdir -p "$rd" "$out"
  if [ ! -f "$rd/checkpoints/model_${EPOCHS}.pth" ]; then
    python "$SCR/run_train.py" --config-name train.yaml env="$env" frameskip=5 num_hist=3 \
      training.epochs="$EPOCHS" training.seed=0 env.num_workers=${NUM_WORKERS:-24} \
      ckpt_base_path="$RUNS" hydra.run.dir="$rd" "${extra[@]}" ${TRAIN_EXTRA:-}
  fi
  cp "$rd/hydra.yaml" "$rd/metrics.jsonl" "$rd/wrapper_info.json" "$out/" 2>/dev/null || true
  if [ ! -f "$out/openloop.json" ]; then
    python "$SCR/eval_openloop.py" --run_dir "$rd" --epoch "$EPOCHS" --out "$out/openloop.json" ${EVAL_EXTRA:-}
  fi
}

# plan_arm ENV ARM PLANNER SEED : official plan.py, official plan_<env>.yaml (MPC-CEM) or planner=cem (open-loop CEM)
plan_arm() {
  local env=$1 arm=$2 planner=$3 seed=$4; env_settings "$env"
  local out=$RES/$env/$arm/plan_${planner}_seed${seed} rd=$RUNS/plan/${env}_${arm}_${planner}_seed${seed}
  [ -f "$out/final.json" ] && { echo "skip $out"; return; }
  local extra=()
  case "$planner" in
    mpc_cem) extra+=("planner.max_iter=${MPC_MAX_ITER:-10}") ;;  # official cfg has max_iter=null (unbounded)
    cem) extra+=("planner=cem") ;;
  esac
  rm -rf "$rd"; mkdir -p "$rd" "$out"
  python "$SCR/run_plan.py" --config-name "$PLAN_CFG" ckpt_base_path="$RUNS" model_name="${env}_${arm}" \
    model_epoch="$EPOCHS" seed="$seed" n_evals=${N_EVALS:-50} hydra.run.dir="$rd" "${extra[@]}" ${PLAN_EXTRA:-}
  cp "$rd/logs.json" "$out/logs.json"
  tail -n 1 "$rd/logs.json" > "$out/final.json"
}
