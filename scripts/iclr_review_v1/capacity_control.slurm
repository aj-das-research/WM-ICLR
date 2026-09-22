#!/bin/bash
#SBATCH --job-name=iclr-capacity-control
#SBATCH --partition=gpu
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iclr-capacity-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/iclr_review_v1/capacity_control.py --seed "${SLURM_ARRAY_TASK_ID:?}"
