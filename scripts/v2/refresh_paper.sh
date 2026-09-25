#!/usr/bin/env bash
# Regenerate tables/figures from completed runs, rebuild the PDF, push GitHub + Overleaf (only if build succeeds).
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate; export PYTHONPATH=src
python scripts/v2/make_tables.py || exit 1
python paper/submission_folder/figures/src/make_figures.py || exit 1
python paper/submission_folder/figures/src/make_mechanism.py || true
python paper/submission_folder/figures/src/make_interpret.py || true
python paper/submission_folder/figures/src/make_segments.py || true   # GPU part: bash scripts/v2/segments_all.sh
PYTHONPATH=src python paper/submission_folder/figures/src/make_setting.py || true
python paper/submission_folder/figures/src/make_benchmarks.py || true
python paper/submission_folder/figures/src/make_geometry.py || true
python paper/submission_folder/figures/src/make_anatomy.py || true
python paper/submission_folder/figures/src/make_vjepa_plugin.py || true
[ -f paper/submission_folder/figures/src/make_analysis_figures.py ] && python paper/submission_folder/figures/src/make_analysis_figures.py || true
python paper/submission_folder/figures/src/make_qual_best.py || true   # GPU selection: slurm_jebel/tasks/qual_select.sh
scripts/v2/build_paper.sh || exit 1
exec 8>paper/.push.lock; flock -w 600 8 || { echo "push lock busy"; exit 1; }
git pull -q --rebase --autostash || true
git add -A paper/submission_folder paper/iclr2027.pdf results/v2 2>/dev/null
git diff --cached --quiet || git commit -qm "Auto-refresh: tables/figures from completed runs ($(ls results/v2/*/*/*/s*/summary.json 2>/dev/null | wc -l) runs)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin main
scripts/v2/sync_overleaf_submission.sh "Auto-refresh from completed runs"
