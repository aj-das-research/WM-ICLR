#!/bin/bash
#SBATCH --job-name=external-dinowm-profile
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/external-dinowm-profile-%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/external_dinowm_profile_v1/profile_cuda.py
