#!/bin/bash
#SBATCH --job-name=droid-metrics-r2-v1
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=/home/abhijit.das/projects/TTA/reports/metrics_completion_v1/droid/recovery_v2/slurm-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/metrics_completion_v1/droid_recovery_v2_evaluate.py --index "${SLURM_ARRAY_TASK_ID:-${1:?registered index required}}"
