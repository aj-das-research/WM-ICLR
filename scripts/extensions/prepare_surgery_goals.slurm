#!/usr/bin/env bash
#SBATCH --job-name=shiftwm-surgery-goals
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/surgery-goals-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
environments/surgery/run_python.sh scripts/extensions/validate_surgery_data.py
environments/surgery/run_python.sh scripts/extensions/prepare_surgery_goals.py --workers 8
