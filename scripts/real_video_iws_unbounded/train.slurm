#!/bin/bash
#SBATCH --job-name=iws-unbounded
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-unbounded-%A_%a.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_iws_unbounded/campaign.py run-index --config "${1:-configs/real_video_iws_unbounded/training_v1.json}" --index "${SLURM_ARRAY_TASK_ID:?array index required}"
