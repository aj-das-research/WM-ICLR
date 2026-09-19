#!/bin/bash
#SBATCH --job-name=iws-single-observation
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=23:50:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-single-%A_%a.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_iws/campaign.py run-index --config "${1:?config required}" --index "${SLURM_ARRAY_TASK_ID:?array index required}"
