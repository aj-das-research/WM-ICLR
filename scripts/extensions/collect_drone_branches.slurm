#!/bin/bash
#SBATCH --job-name=shiftwm-drone-branches
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/drone-branches-%j.log
set -euo pipefail
cd /home/abhijit.das/projects/TTA
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD/external/gym-pybullet-drones"
# Submission requires the root agent's resource check; this script requests no GPU.
environments/drone/.venv/bin/python scripts/extensions/collect_drone_branches.py collect --workers 4
