#!/bin/bash
# Official-protocol planning for any pluggable predictor (E4 step 2+). Configure via env:
#   PREDICTOR=module:factory  CKPT=path  MODEL=name  ENVS="pusht tworoom reacher"  SEEDS="42 43 44"  EXTRA=""
# Run: sbatch -J v2-plan-X --export=ALL,PREDICTOR=...,CKPT=...,MODEL=... slurm_jebel/job.sbatch slurm_jebel/tasks/planning_eval_model.sh
set -uo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8
PREDICTOR=${PREDICTOR:-shiftwm.v2.planning.lewm:build}; MODEL=${MODEL:-lewm}; CKPT=${CKPT:-}
ENVS=${ENVS:-"pusht tworoom reacher"}; SEEDS=${SEEDS:-"42 43 44"}; EXTRA=${EXTRA:-}
declare -A DS=([pusht]=data/upstream/pusht/pusht_expert_train.h5 [reacher]=data/upstream/reacher/reacher.h5 [tworoom]=data/upstream/tworoom/tworoom.h5)
rc=0
for E in $ENVS; do for S in $SEEDS; do
  A=(--env $E --seed $S --model $MODEL --predictor $PREDICTOR --dataset ${DS[$E]})
  [ -n "$CKPT" ] && A+=(--ckpt $CKPT)
  echo "== $(date -Is) planning_eval.py ${A[*]} $EXTRA"
  python scripts/v2/planning_eval.py "${A[@]}" $EXTRA || rc=1
done; done
echo "== done rc=$rc"; exit $rc
