#!/bin/bash
#SBATCH --job-name=droid-horizon10
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/droid-horizon10-%j.log
set -euo pipefail
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_development/horizon10_campaign.py --group "${1:?registered seed group required}"
