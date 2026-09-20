#!/bin/bash
#SBATCH --job-name=iws-inference-resources
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-inference-resources-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
srun --cpu-bind=cores .venv/bin/python -I scripts/real_video_iws_resources_v1/profile.py run
