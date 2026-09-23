#!/bin/bash
#SBATCH --partition=h200
#SBATCH --comment="h200test2609"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=07:55:00
#SBATCH --chdir=/home/Test/abhijit.das/projects/WM-ICLR
#SBATCH --output=logs/slurm-%x-%j.out
# Generic lane job: runs the bash script given as $1 (a file under slurm_jebel/tasks/).
set -eo pipefail
source .venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "host $(hostname) start $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
bash "$1"
echo "end $(date -Is)"
