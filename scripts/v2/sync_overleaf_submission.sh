#!/usr/bin/env bash
# Mirror paper/submission_folder -> Overleaf submission_folder/ and root main.tex, then push.
set -euo pipefail
REPO=${OVERLEAF_CHECKOUT:-$HOME/.local/share/shiftwm/overleaf/repo}
SRC=$(cd "$(dirname "$0")/../.." && pwd)/paper
cd "$REPO"
export GIT_TERMINAL_PROMPT=0
git pull -q --ff-only
# Root entry point is submission_main.tex; the previous draft stays as draft_v1_main.tex.
if [ -f main.tex ] && grep -q "submission_folder/" main.tex; then git rm -q main.tex; fi
rsync -a --delete --exclude 'figures/src/' "$SRC/submission_folder/" submission_folder/
cp "$SRC/submission_main.tex" submission_main.tex
[ -f "$SRC/submission_main.pdf" ] && cp "$SRC/submission_main.pdf" submission_main.pdf
git add -A
if git diff --cached --quiet; then echo "Overleaf: nothing to sync"; exit 0; fi
git -c user.name="Abhijit Das" -c user.email="aj.das.research@gmail.com" commit -qm "${1:-Sync ICLR submission sources}"
git push -q origin HEAD
echo "Overleaf synced: $(git log --oneline -1)"
