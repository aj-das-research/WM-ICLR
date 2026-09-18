#!/usr/bin/env bash
#SBATCH --job-name=droid-fresh-decode
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/real-droid-fresh-decode-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export TF_CPP_MIN_LOG_LEVEL=3
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1
environments/real_video/.venv/bin/python scripts/real_video_development/fresh_evaluation_prepare.py
