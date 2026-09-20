#!/bin/bash
#SBATCH --job-name=iws-bound-diagnostic-v1
#SBATCH --partition=ws-ia
#SBATCH --account=ia-std
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0
#SBATCH --output=reports/iws_bound_diagnostic_v1/slurm-%j.log
set -euo pipefail
cd /home/abhijit.das/projects/TTA
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
.venv/bin/python -B scripts/iws_bound_diagnostic_v1/diagnose.py run
