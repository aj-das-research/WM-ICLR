#!/bin/bash
#SBATCH --job-name=real-droid-features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/real-droid-features-%j.log
set -euo pipefail
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python -m shiftwm.real_video.features --data data/real_video/droid_selected/processed --output data/features/droid_selected_v1
