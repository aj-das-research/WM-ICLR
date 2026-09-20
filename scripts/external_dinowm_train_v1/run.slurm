#!/bin/bash
#SBATCH --job-name=external-dinowm-train
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/external-dinowm-train-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/external_dinowm_train_v1/campaign.py run --index "${SLURM_ARRAY_TASK_ID:-${1:?run index required}}"
