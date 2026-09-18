#!/bin/bash
#SBATCH --job-name=real-droid-finalize
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/real-droid-finalize-%j.log
set -euo pipefail
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video/finalize_campaign.py --campaign configs/real_video/campaign.json
.venv/bin/python scripts/real_video/progress.py
bash paper/build.sh
