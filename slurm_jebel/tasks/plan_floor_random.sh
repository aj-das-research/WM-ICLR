#!/bin/bash
# Old-draft reuse (audit items 16 + 7): official LeWM planning protocol,
#  (a) uniform random-action floor (same tasks/seeds/budget) -> results/v2/planning/<env>/random/<seed>.json
#  (b) released LeWM re-run with the physical-goal-error recorder -> results/v2/planning/<env>/lewm/<seed>_phys.json
set -uo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8
declare -A DS=([pusht]=data/upstream/pusht/pusht_expert_train.h5 [reacher]=data/upstream/reacher/reacher.h5 [tworoom]=data/upstream/tworoom/tworoom.h5)
rc=0
for E in pusht tworoom reacher; do for S in 42 43 44; do
  [ -f results/v2/planning/$E/random/$S.json ] || python scripts/v2/planning_eval.py --env $E --seed $S --policy random --model random --dataset ${DS[$E]} || rc=1
done; done
for E in pusht tworoom reacher; do for S in 42 43 44; do
  [ -f results/v2/planning/$E/lewm/${S}_phys.json ] || python scripts/v2/planning_eval.py --env $E --seed $S --model lewm --tag phys --dataset ${DS[$E]} || rc=1
done; done
echo "== done rc=$rc"; exit $rc
