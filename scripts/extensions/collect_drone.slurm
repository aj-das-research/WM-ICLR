#!/usr/bin/env bash
#SBATCH --job-name=shiftwm-drone-data
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/drone-collection-%j.log

set -euo pipefail
cd /home/abhijit.das/projects/TTA
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
environments/drone/.venv/bin/python scripts/extensions/collect_drone.py \
  --output data/extensions/drone_v1 \
  --train-per-gain 64 --val-per-gain 16 \
  --development-per-gain 16 --test-per-gain 32 \
  --native-steps 200 --action-block 5 --image-size 128 --workers 8
