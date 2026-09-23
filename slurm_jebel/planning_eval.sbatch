#!/bin/bash
#SBATCH --job-name=v2-planning-eval
#SBATCH --partition=h200
#SBATCH --comment="h200test2609"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=07:50:00
#SBATCH --chdir=/home/Test/abhijit.das/projects/WM-ICLR
#SBATCH --output=logs/slurm-%x-%j.out
#
# Official-protocol LeWM planning evaluation (E4). Configure via env vars:
#   ENVS="pusht tworoom reacher"   environments, run sequentially
#   SEEDS="42"                     eval seeds (42 = official task selection / CEM seed)
#   PREDICTOR=shiftwm.v2.planning.lewm:build   CKPT=<path or empty>   MODEL=lewm
#   NUM_EVAL=<n> (smoke)  TAG=<suffix>  EXTRA="--check-adapter ..."  (extra planning_eval.py args)
# e.g.  sbatch --export=ALL,ENVS=pusht,NUM_EVAL=5,TAG=smoke,EXTRA=--check-adapter slurm_jebel/planning_eval.sbatch
set -uo pipefail
ENVS=${ENVS:-"pusht tworoom reacher"}
SEEDS=${SEEDS:-42}
PREDICTOR=${PREDICTOR:-shiftwm.v2.planning.lewm:build}
MODEL=${MODEL:-lewm}
CKPT=${CKPT:-}
TAG=${TAG:-}
EXTRA=${EXTRA:-}

source .venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
echo "job=$SLURM_JOB_ID host=$(hostname) envs=[$ENVS] seeds=[$SEEDS] model=$MODEL predictor=$PREDICTOR"

U=data/upstream
# Datasets are needed for the protocol itself: start states, goal images (+25 steps in the same
# expert/random trajectory), goal states for the success check and the action StandardScaler.
ensure_dataset() {  # $1=env -> echo path of the .h5
  case "$1" in
    pusht)   out=$U/pusht/pusht_expert_train.h5
             if [ ! -s "$out" ]; then mkdir -p $U/pusht
               zstd -d -f --no-progress $U/pusht_expert_train.h5.zst -o $out.tmp && mv $out.tmp $out; fi ;;
    reacher|tworoom)
             arc=$U/$1.tar.zst; d=$U/$1
             if [ ! -f "$d/.extracted" ]; then mkdir -p $d
               tar --use-compress-program=zstd -xf $arc -C $d && touch $d/.extracted; fi
             out=$(find $d -name '*.h5' -size +1M | head -1) ;;
  esac
  echo "$out"
}

rc=0
for E in $ENVS; do
  t0=$(date +%s)
  DS=$(ensure_dataset $E)
  echo "[$E] dataset=$DS ($(( $(date +%s)-t0 ))s to prepare)"
  if [ -z "$DS" ] || [ ! -s "$DS" ]; then echo "[$E] dataset missing, skipping"; rc=1; continue; fi
  for S in $SEEDS; do
    ARGS=(--env $E --seed $S --model $MODEL --predictor $PREDICTOR --dataset $DS)
    [ -n "$CKPT" ] && ARGS+=(--ckpt $CKPT)
    [ -n "${NUM_EVAL:-}" ] && ARGS+=(--num-eval $NUM_EVAL)
    [ -n "$TAG" ] && ARGS+=(--tag $TAG)
    echo "== $(date -Is) python scripts/v2/planning_eval.py ${ARGS[*]} $EXTRA"
    python scripts/v2/planning_eval.py "${ARGS[@]}" $EXTRA || rc=1
  done
done
echo "== done rc=$rc $(date -Is)"
exit $rc
