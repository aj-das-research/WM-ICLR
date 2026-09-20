#!/bin/bash
#SBATCH --job-name=external-dinowm-raw-finalize
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/external-dinowm-raw-finalize-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B scripts/external_dinowm_raw_reporting_v2/finalize.py
