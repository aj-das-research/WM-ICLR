#!/bin/bash
#SBATCH --job-name=spatial-sim-eval
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/spatial-sim-eval-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
artifacts/upstream_planning_reproduction/.venv/bin/python scripts/metrics_completion_v1/planning_campaign.py evaluate --index "${SLURM_ARRAY_TASK_ID:?array index required}" --split test
