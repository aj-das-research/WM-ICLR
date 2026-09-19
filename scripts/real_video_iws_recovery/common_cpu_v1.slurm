#!/bin/bash
#SBATCH --job-name=iws-common-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --open-mode=append
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-common-cpu-%A_%a.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
if [[ "${1:-run}" == "finalize" ]]; then
 .venv/bin/python scripts/real_video_iws_recovery/common_cpu_v1.py finalize
else
 .venv/bin/python scripts/real_video_iws_recovery/common_cpu_v1.py run --index "${SLURM_ARRAY_TASK_ID:?array index required}"
fi
