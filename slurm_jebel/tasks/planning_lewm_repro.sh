#!/bin/bash
# E4 step 1: reproduce released LeWM checkpoints at the official LeWM planning protocol.
# Run: sbatch -J v2-plan-lewm slurm_jebel/job.sbatch slurm_jebel/tasks/planning_lewm_repro.sh
# (job.sbatch activates .venv, sets PYTHONPATH=src HF_HUB_OFFLINE=1). Expected wall ~15-25 min.
# PushT (seeds 42/43/44 + native/hist3 checks) is already done in results/v2/planning/pusht/lewm/.
set -uo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8
U=data/upstream
for E in reacher tworoom; do
  [ -f $U/$E/.extracted ] || { echo "dataset $U/$E not extracted (run: tar --use-compress-program=zstd -xf $U/$E.tar.zst -C $U/$E && touch $U/$E/.extracted)"; exit 2; }
done
rc=0
run() { echo "== $(date -Is) planning_eval.py $*"; python scripts/v2/planning_eval.py "$@" || rc=1; }
# official released protocol (le-wm config/eval/*.yaml), 3 eval seeds (42 = official selection)
for S in 42 43 44; do
  run --env reacher --seed $S --dataset $U/reacher/reacher.h5 $([ $S = 42 ] && echo --check-adapter)
  run --env tworoom --seed $S --dataset $U/tworoom/tworoom.h5 $([ $S = 42 ] && echo --check-adapter)
done
# paper App. D/F.1 variant (TwoRoom goal +100 / budget 150; CEM 10 iters outside PushT)
run --env reacher --seed 42 --variant paper --dataset $U/reacher/reacher.h5 --tag paper
run --env tworoom --seed 42 --variant paper --dataset $U/tworoom/tworoom.h5 --tag paper
echo "== done rc=$rc"; exit $rc
