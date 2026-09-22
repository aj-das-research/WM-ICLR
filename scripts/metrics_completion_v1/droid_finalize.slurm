#!/bin/bash
#SBATCH --job-name=droid-metrics-finalize-v1
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=/home/abhijit.das/projects/TTA/reports/metrics_completion_v1/droid/finalizer-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python scripts/metrics_completion_v1/droid_finalize.py
