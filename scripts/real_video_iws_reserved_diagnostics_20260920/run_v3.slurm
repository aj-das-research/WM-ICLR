#!/bin/bash
#SBATCH --job-name=iws-prefix-synthetic-scorer
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-reserved-prefix-probe-v3-%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
srun --cpu-bind=cores .venv/bin/python -I scripts/real_video_iws_reserved_diagnostics_20260920/probe_v3.py --output reports/real_video_iws_reserved_diagnostics_20260920/probe_cpu8_v3.json
