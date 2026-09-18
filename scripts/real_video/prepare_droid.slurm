#!/usr/bin/env bash
#SBATCH --job-name=real-droid-extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/real-droid-extract-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1
environments/real_video/.venv/bin/python scripts/real_video/prepare_droid.py \
  --raw data/real_video/droid_selected/raw \
  --output data/real_video/droid_selected/processed \
  --inventory reports/evidence/real_video/droid_selected_inventory.json \
  --expected-episodes 1126 --workers 4
