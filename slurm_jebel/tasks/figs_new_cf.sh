#!/bin/bash
# Rerun only the counterfactual-actions analysis (appendix counterfactual.pdf).
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/counterfactual.py 2>&1 | tee results/v2/analysis/logs/counterfactual.log
