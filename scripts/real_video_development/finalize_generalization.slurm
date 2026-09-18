#!/bin/bash
#SBATCH --job-name=droid-generalization-finalize
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abhijit.das/projects/TTA
#SBATCH --output=logs/droid-generalization-finalize-%j.log
set -euo pipefail
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/real_video_development/finalize_generalization.py --build-paper
# The existing sync timer publishes only after the workspace has settled. No
# credentials or separate remote pushes are needed in this compute-node job.
