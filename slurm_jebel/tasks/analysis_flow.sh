#!/bin/bash
# E6 transport field vs RAFT optical flow (k=1,5,10) + gain-vs-motion (ShiftWM vs Direct, same seed).
# Writes results/v2/analysis/{flow,gain_vs_motion}/. ~20-40 min.
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/flow_agreement.py 2>&1 | tee results/v2/analysis/logs/flow_agreement.log
