#!/usr/bin/env bash
# Compile paper/submission_main.tex (+ submission_folder/) to paper/submission_main.pdf with tectonic.
set -euo pipefail
cd "$(dirname "$0")/../../paper"
TECTONIC=${TECTONIC:-$HOME/.conda/envs/tex/bin/tectonic}
"$TECTONIC" -X compile --keep-logs --outdir build_submission submission_main.tex 2>&1 | grep -E "error|Error|warning: .*undefined|Overfull" | head -40 || true
cp build_submission/submission_main.pdf submission_main.pdf
echo "pages: $(grep -o "([0-9]* pages" build_submission/submission_main.log | tail -1)"
