#!/bin/bash
#SBATCH --job-name=spatial-sim-finalize
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/spatial-sim-finalize-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/metrics_completion_v1/planning_finalize.py
