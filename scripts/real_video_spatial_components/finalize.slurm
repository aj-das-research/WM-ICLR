#!/bin/bash
#SBATCH --job-name=droid-components-finalize
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/droid-components-finalize-%j.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_spatial_components/campaign.py finalize
