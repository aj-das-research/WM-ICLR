#!/bin/bash
#SBATCH --job-name=iws-rgb-shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-rgb-shared-%A_%a.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
tasks=(pusht bimanual_box bimanual_rope)
index="${SLURM_ARRAY_TASK_ID:?array index required}"
if [[ "$index" -lt 0 || "$index" -gt 2 ]]; then exit 2; fi
set +e
.venv/bin/python scripts/metrics_completion_v1/rgb_train.py --task "${tasks[$index]}" --stage pipeline
status=$?
set -e
if [[ "$status" -eq 75 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 8 ]]; then exit 75; fi
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
.venv/bin/python scripts/metrics_completion_v1/rgb_finalize.py --if-ready
