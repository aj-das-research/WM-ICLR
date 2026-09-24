#!/bin/bash
# Largest-advantage qualitative windows (appendix qual_best_<ds>.pdf) for every forecasting benchmark with finished
# ShiftWM/Direct/AR runs; others are recorded as "pending". Needs 1 GPU, ~5-30 min (feature loading dominates).
# Rerun when new datasets finish:
#   sbatch -J qual_select --nice=20 --time=01:30:00 --mem=64G slurm_jebel/job.sbatch slurm_jebel/tasks/qual_select.sh
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/qual_select.py 2>&1 | tee results/v2/analysis/logs/qual_select.log
python paper/submission_folder/figures/src/make_qual_best.py || echo "FAILED make_qual_best"
