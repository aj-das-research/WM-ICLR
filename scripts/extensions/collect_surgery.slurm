#!/usr/bin/env bash
#SBATCH --job-name=shiftwm-surgery-data
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/surgery-data-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p reports/surgery/slurm-profile
# First warm SOFA's precomputed compliance state; its initial generation may
# produce micrometer differences. Require a full fresh replay after warmup.
environments/surgery/run_python.sh scripts/extensions/profile_surgery.py --calls 30 --output "reports/surgery/slurm-profile/warmup-${SLURM_JOB_ID}" || true
environments/surgery/run_python.sh scripts/extensions/profile_surgery.py --calls 30 --output "reports/surgery/slurm-profile/verified-${SLURM_JOB_ID}"
environments/surgery/run_python.sh scripts/extensions/collect_surgery.py --workers 8
