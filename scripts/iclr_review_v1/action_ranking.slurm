#!/bin/bash
#SBATCH --job-name=spatial-ranking-diagnostic
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iclr-ranking-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) task=pusht ;;
  1) task=reacher ;;
  *) exit 1 ;;
esac
artifacts/upstream_planning_reproduction/.venv/bin/python scripts/iclr_review_v1/action_ranking.py --task "$task"
