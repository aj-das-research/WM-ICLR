#!/usr/bin/env bash
# Compile paper/iclr2027.tex (+ submission_folder/) to paper/iclr2027.pdf with tectonic.
set -euo pipefail
cd "$(dirname "$0")/../../paper"
# serialise builds across sessions/agents (wait up to 20 min)
exec 9>.build.lock; flock -w 1200 9 || { echo "build lock busy"; exit 1; }
TECTONIC=${TECTONIC:-$HOME/.conda/envs/tex/bin/tectonic}
rm -f build_submission/iclr2027.pdf
if ! "$TECTONIC" -X compile --keep-logs --outdir build_submission iclr2027.tex > build_submission/tectonic.out 2>&1; then
  grep -E "error|^!" build_submission/tectonic.out | head -20; echo "BUILD FAILED (PDF not updated)"; exit 1
fi
grep -E "warning: .*undefined|Overfull" build_submission/tectonic.out | head -20 || true
cat build_submission/iclr2027.pdf > iclr2027.pdf
echo "pages: $(python3 -c "import re;print(len(re.findall(rb'/Type ?/Page\\b', open('iclr2027.pdf','rb').read())))") (built OK)"
