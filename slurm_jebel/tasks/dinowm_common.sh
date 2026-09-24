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
# Official code pin: git clone https://github.com/gaoyuezhou/dino_wm external/dino_wm && git -C external/dino_wm checkout 0a9492fa12044b852ae9e001cc74604b79c8bb0c
DWM_PIN=0a9492fa12044b852ae9e001cc74604b79c8bb0c
[ "$(git -C external/dino_wm rev-parse HEAD)" = "$DWM_PIN" ] || { echo "external/dino_wm is not at $DWM_PIN"; exit 3; }
[ -z "$(git -C external/dino_wm status --porcelain --untracked-files=no)" ] || { echo "external/dino_wm has local edits"; exit 3; }
RUNS=$PROJ/runs/dinowm_plugin                      # checkpoints (gitignored)
RES=$PROJ/results/v2/external/dinowm_plugin
SCR=$PROJ/scripts/external_dinowm_plugin
PLUGIN_TARGET=shiftwm.v2.dinowm_plugin.ShiftViTPredictor

# Per-environment settings = the hydra.yaml shipped with the OFFICIAL released checkpoints (OSF outputs.zip):
#   pusht: frameskip 5, num_hist 3, batch 32, emb_dropout 0      (released cfg: epochs 100)
#   wall : frameskip 5, num_hist 1, batch 128, emb_dropout 0.1   (released cfg: epochs 1000, save every 5)
# everything else = conf/train.yaml (AdamW predictor/action lr 5e-4, Adam decoder 3e-4, frozen DINOv2 ViT-S/14, 224px).
# EPOCHS is the reduced, arm-identical budget.
env_settings() {
  case "$1" in
    pusht) EPOCHS=${EPOCHS:-2};  PLAN_CFG=plan_pusht.yaml; TRAIN_ENV=(env=pusht frameskip=5 num_hist=3 training.batch_size=32) ;;
    wall)  EPOCHS=${EPOCHS:-40}; PLAN_CFG=plan_wall.yaml;  TRAIN_ENV=(env=wall frameskip=5 num_hist=1 training.batch_size=128 predictor.emb_dropout=0.1) ;;
    point_maze) EPOCHS=${EPOCHS:-10}; PLAN_CFG=plan_point_maze.yaml; export DINOWM_ENV=point_maze
                TRAIN_ENV=(env=point_maze frameskip=5 num_hist=3 training.batch_size=32) ;;
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
    python "$SCR/run_train.py" --config-name train.yaml "${TRAIN_ENV[@]}" \
      training.epochs="$EPOCHS" training.seed=0 env.num_workers=${NUM_WORKERS:-24} \
      ckpt_base_path="$RUNS" hydra.run.dir="$rd" "${extra[@]}" ${TRAIN_EXTRA:-}
  fi
  for f in hydra.yaml metrics.jsonl wrapper_info.json; do [ -f "$rd/$f" ] && cat "$rd/$f" > "$out/$f"; done
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
    # open-loop CEM = upstream conf/planner/cem.yaml values, overriding the inline MPC planner of plan_<env>.yaml
    cem) extra+=(planner._target_=planning.cem.CEMPlanner planner.name=cem +planner.horizon=5 +planner.topk=30
                 +planner.num_samples=300 +planner.var_scale=1 +planner.opt_steps=30 +planner.eval_every=1) ;;
  esac
  rm -rf "$rd"; mkdir -p "$rd" "$out"
  python "$SCR/run_plan.py" --config-name "$PLAN_CFG" ckpt_base_path="$RUNS" model_name="${env}_${arm}" \
    model_epoch="$EPOCHS" seed="$seed" n_evals=${N_EVALS:-50} hydra.run.dir="$rd" "${extra[@]}" ${PLAN_EXTRA:-}
  cat "$rd/logs.json" > "$out/logs.json"
  tail -n 1 "$rd/logs.json" > "$out/final.json"
}
