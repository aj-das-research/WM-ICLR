#!/bin/bash
#SBATCH --job-name=iclr-rgb-endpoint-v2
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iclr-rgb-endpoint-v2-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) task=pusht ;;
  1) task=bimanual_box ;;
  2) task=bimanual_rope ;;
  *) exit 1 ;;
esac
.venv/bin/python scripts/iclr_review_v1/rgb_endpoint_v2.py --task "$task"
