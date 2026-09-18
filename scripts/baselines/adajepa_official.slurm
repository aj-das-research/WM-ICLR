#!/usr/bin/env bash
#SBATCH --job-name=adajepa-official
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/adajepa-official-%j.log

set -euo pipefail
cd /home/abhijit.das/projects/TTA
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TORCH_HOME=/home/abhijit.das/projects/TTA/environments/adajepa/torch_cache
environments/adajepa/.venv/bin/python scripts/baselines/adajepa_pin_dinov2.py
environments/adajepa/.venv/bin/python scripts/baselines/adajepa_campaign.py
