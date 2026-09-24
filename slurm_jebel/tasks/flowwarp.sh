#!/bin/bash
# Non-learned feature-warping baselines (F2M-style causal flow extrapolation + oracle-flow warp) on DROID test
# windows (H=3, K=10, stride 2). RAFT-large on 224px frames. Writes results/v2/analysis/flowwarp/{summary.json,
# per_episode.npz}. Expected ~5-10 min on one H200 (CPU timing: ~1.1 s/window x 3923 windows).
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/flowwarp.py 2>&1 | tee results/v2/analysis/logs/flowwarp.log
