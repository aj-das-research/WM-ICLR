#!/bin/bash
# Official-protocol planning with trained v2 predictors (shiftwm.v2.planning.v2wm).
#   ENVS="pusht tworoom reacher"  ARMS="shiftwm ar direct ar_tf"  TRAIN_SEED=0  SEEDS="42 43 44"  EXTRA=""
# Skips missing checkpoints and finished results. Output results/v2/planning/<env>/v2_<arm>_s<TRAIN_SEED>/<seed>.json
set -uo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8
ENVS=${ENVS:-"pusht tworoom reacher"}; ARMS=${ARMS:-"shiftwm ar direct ar_tf"}
TRAIN_SEED=${TRAIN_SEED:-0}; SEEDS=${SEEDS:-"42 43 44"}; EXTRA=${EXTRA:-}
declare -A DS=([pusht]=data/upstream/pusht/pusht_expert_train.h5 [reacher]=data/upstream/reacher/reacher.h5 [tworoom]=data/upstream/tworoom/tworoom.h5)
rc=0
for E in $ENVS; do for A in $ARMS; do for S in $SEEDS; do
  ck=results/v2/plan/$E/$A/s$TRAIN_SEED/best.pt; M=v2_${A}_s$TRAIN_SEED
  [ -f $ck ] || { echo "skip $ck (missing)"; continue; }
  [ -f results/v2/planning/$E/$M/$S.json ] && { echo "skip $E/$M/$S (done)"; continue; }
  echo "== $(date -Is) $E $M seed $S"
  python scripts/v2/planning_eval.py --env $E --seed $S --model $M --predictor shiftwm.v2.planning.v2wm:build \
    --ckpt $ck --dataset ${DS[$E]} $EXTRA || rc=1
done; done; done
echo "== done rc=$rc"; exit $rc
