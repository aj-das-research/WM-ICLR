#!/bin/bash
#SBATCH --job-name=qualitative-closest-v1
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/qualitative-closest-%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B scripts/qualitative_closest_v1/replay.py replay
