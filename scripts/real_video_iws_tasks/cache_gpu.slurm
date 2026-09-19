#!/bin/bash
#SBATCH --job-name=iws-task-cache
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-task-cache-%j.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_iws_tasks/prepare_cache.py build --device cuda --task "$1"

.venv/bin/python scripts/real_video_iws_tasks/prepare_cache.py validate --task "$1" --output "data/features/iws_${1}_spatial_v1"
