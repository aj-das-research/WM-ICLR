#!/bin/bash
# E1 zero-shot camera transfer: DROID cam-1 checkpoints -> droid_cam2 cache (cam-1 normalisation stats).
# Writes results/v2/droid_cam2/dinov2s/<arm>/s<seed>/eval_test.npz, then refreshes the paper tables.
# Only finished runs (best.pt + summary.json) are used; finished transfer evals are skipped. ~5-15 min.
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/eval_transfer.py 2>&1 | tee results/v2/analysis/logs/eval_transfer.log
python scripts/v2/make_tables.py
