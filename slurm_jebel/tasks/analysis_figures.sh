#!/bin/bash
# Analysis + qualitative figures (flow_agreement, gain_vs_motion, tradeoff, qualitative_transport,
# gallery_droid, gallery_surgical, failures) -> paper/submission_folder/figures/. Run after the tasks above;
# panels with missing inputs render as "pending". ~5-10 min.
set -uo pipefail
mkdir -p results/v2/analysis/logs
python paper/submission_folder/figures/src/make_analysis_figures.py --device cuda 2>&1 \
  | tee results/v2/analysis/logs/figures.log
