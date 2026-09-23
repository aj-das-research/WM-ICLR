#!/usr/bin/env bash
# Regenerate tables/figures from completed runs, rebuild the PDF, push GitHub + Overleaf (only if build succeeds).
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate; export PYTHONPATH=src
python scripts/v2/make_tables.py || exit 1
python paper/submission_folder/figures/src/make_figures.py || exit 1
python paper/submission_folder/figures/src/make_mechanism.py || true
[ -f paper/submission_folder/figures/src/make_analysis_figures.py ] && python paper/submission_folder/figures/src/make_analysis_figures.py || true
scripts/v2/build_paper.sh || exit 1
git add -A paper/submission_folder paper/submission_main.pdf results/v2 2>/dev/null
git diff --cached --quiet || git commit -qm "Auto-refresh: tables/figures from completed runs ($(ls results/v2/*/*/*/s*/summary.json 2>/dev/null | wc -l) runs)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin main
scripts/v2/sync_overleaf_submission.sh "Auto-refresh from completed runs"
