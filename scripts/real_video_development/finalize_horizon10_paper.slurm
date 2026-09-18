#!/bin/bash
#SBATCH --job-name=droid-h10-paper
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/droid-horizon10-paper-%j.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_development/finalize_horizon10_paper.py --build-paper
