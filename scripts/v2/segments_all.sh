#!/usr/bin/env bash
# Segmentation study (scripts/v2/segments.py) on every dataset. Needs a GPU. Datasets whose ShiftWM/Direct checkpoints
# are missing get status "pending"; datasets already evaluated with the same checkpoints (path + mtime) are skipped.
# Usage: bash scripts/v2/segments_all.sh [dataset ...]      (default: all)
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate; export PYTHONPATH=src
DATASETS=${*:-"droid openh_hamlyn bridge fractal language_table iws_pusht iws_box iws_rope plan_pusht plan_reacher plan_tworoom"}
for ds in $DATASETS; do
  mkdir -p results/v2/analysis/segments/$ds
  python scripts/v2/segments.py --dataset "$ds" --skip-if-current >> results/v2/analysis/segments/$ds/run.log 2>&1 \
    && grep -a "^\[$ds\] \(pending\|up to date\|wrote\|rejected\)" results/v2/analysis/segments/$ds/run.log | tail -1 \
    || echo "[$ds] FAILED (see results/v2/analysis/segments/$ds/run.log)"
done
