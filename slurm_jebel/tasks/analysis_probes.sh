#!/bin/bash
# E6 ridge probes (DROID commanded EE xyz, Hamlyn tip xyz) on true vs predicted k=10 features.
# Writes results/v2/analysis/probes/ and tables/generated/probe_rows.tex. ~15-30 min.
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/probes.py 2>&1 | tee results/v2/analysis/logs/probes.log
