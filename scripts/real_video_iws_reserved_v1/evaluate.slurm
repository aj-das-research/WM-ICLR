#!/bin/bash
#SBATCH --job-name=iws-reserved-score
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-35%2
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-reserved-score-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
srun --cpu-bind=cores .venv/bin/python -I scripts/real_video_iws_reserved_v1/run_registered.py --index "$SLURM_ARRAY_TASK_ID"
