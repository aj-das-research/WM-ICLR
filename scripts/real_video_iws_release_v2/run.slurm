#!/bin/bash
#SBATCH --job-name=iws-rowwise-release
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-rowwise-release-%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B scripts/real_video_iws_release_v2/build.py build
