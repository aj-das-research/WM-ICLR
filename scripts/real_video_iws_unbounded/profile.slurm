#!/bin/bash
#SBATCH --job-name=iws-unbounded-profile
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=ws-ia
#SBATCH --account=students
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/iws-unbounded-profile-%j.log
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for task in pusht bimanual_box bimanual_rope; do
 .venv/bin/python scripts/real_video_iws_unbounded/train.py profile --config configs/real_video_iws_unbounded/training_v1.json --task "$task" --mode all --microbatch 64 --updates 2 --output "reports/real_video_iws_unbounded/profile_${task}.json"
done
