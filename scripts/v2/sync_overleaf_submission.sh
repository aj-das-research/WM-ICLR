#!/usr/bin/env bash
# Mirror paper/submission_folder -> Overleaf submission_folder/ and root main.tex, then push.
set -euo pipefail
REPO=${OVERLEAF_CHECKOUT:-$HOME/.local/share/shiftwm/overleaf/repo}
SRC=$(cd "$(dirname "$0")/../.." && pwd)/paper
cd "$REPO"
export GIT_TERMINAL_PROMPT=0
git pull -q --ff-only
if [ ! -f draft_v1_main.tex ] && ! grep -q "submission_folder/" main.tex; then
  git mv main.tex draft_v1_main.tex   # keep the previous draft compilable under a new name
fi
rsync -a --delete --exclude 'figures/src/' "$SRC/submission_folder/" submission_folder/
cp "$SRC/overleaf_root_main.tex" main.tex
git add -A main.tex submission_folder draft_v1_main.tex 2>/dev/null || git add -A
if git diff --cached --quiet; then echo "Overleaf: nothing to sync"; exit 0; fi
git -c user.name="Abhijit Das" -c user.email="aj.das.research@gmail.com" commit -qm "${1:-Sync ICLR submission sources}"
git push -q origin HEAD
echo "Overleaf synced: $(git log --oneline -1)"
