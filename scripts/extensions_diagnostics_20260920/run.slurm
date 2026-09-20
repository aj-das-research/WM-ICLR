#!/bin/bash
#SBATCH --job-name=extension-geometry-diagnostic
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/extension-geometry-diagnostic-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
srun --cpu-bind=cores .venv/bin/python -I scripts/extensions_diagnostics_20260920/diagnose.py run
