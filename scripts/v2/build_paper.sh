#!/usr/bin/env bash
# Compile paper/submission_main.tex (+ submission_folder/) to paper/submission_main.pdf with tectonic.
set -euo pipefail
cd "$(dirname "$0")/../../paper"
TECTONIC=${TECTONIC:-$HOME/.conda/envs/tex/bin/tectonic}
rm -f build_submission/submission_main.pdf
if ! "$TECTONIC" -X compile --keep-logs --outdir build_submission submission_main.tex > build_submission/tectonic.out 2>&1; then
  grep -E "error|^!" build_submission/tectonic.out | head -20; echo "BUILD FAILED (PDF not updated)"; exit 1
fi
grep -E "warning: .*undefined|Overfull" build_submission/tectonic.out | head -20 || true
cp build_submission/submission_main.pdf submission_main.pdf
echo "pages: $(python3 -c "import re;print(len(re.findall(rb'/Type ?/Page\\b', open('submission_main.pdf','rb').read())))") (built OK)"
