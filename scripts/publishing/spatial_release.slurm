#!/bin/bash
#SBATCH --job-name=spatial-model-release
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/spatial-model-release-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python artifacts/publishing/spatial_v1/frozen_exporter.py \
  --root /home/abhijit.das/projects/TTA \
  --directory artifacts/publishing/spatial_v1 \
  --publish \
  --credential-file /home/abhijit.das/.config/shiftwm/github-git.credentials
