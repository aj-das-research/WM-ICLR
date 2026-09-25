#!/bin/bash
# Compute inputs of the appendix figures counterfactual.pdf, trails.pdf, winmap.pdf (held-out DROID, s0 checkpoints).
#   sbatch -J figs_new -p a100 --gres=gpu:1 --cpus-per-task=16 --mem=110G --time=01:00:00 \
#          slurm_jebel/job.sbatch slurm_jebel/tasks/figs_new.sh
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/winmap.py 2>&1 | tee results/v2/analysis/logs/winmap.log
python scripts/v2/counterfactual.py 2>&1 | tee results/v2/analysis/logs/counterfactual.log
python scripts/v2/trails.py 2>&1 | tee results/v2/analysis/logs/trails.log
