#!/bin/bash
#SBATCH --job-name=droid-components
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=07:50:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/droid-components-%j.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_spatial_components/campaign.py run --name "${1:?registered model name required}"
