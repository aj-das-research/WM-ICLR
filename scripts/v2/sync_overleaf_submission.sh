#!/usr/bin/env bash
# Mirror paper/submission_folder -> Overleaf submission_folder/, root iclr2027.tex (the only root .tex) + iclr2027.pdf, push.
set -euo pipefail
REPO=${OVERLEAF_CHECKOUT:-$HOME/.local/share/shiftwm/overleaf/repo}
SRC=$(cd "$(dirname "$0")/../.." && pwd)/paper
cd "$REPO"
export GIT_TERMINAL_PROMPT=0
git pull -q --ff-only
rsync -a --delete --exclude 'figures/src/' "$SRC/submission_folder/" submission_folder/
cat "$SRC/iclr2027.tex" > iclr2027.tex
[ -f "$SRC/iclr2027.pdf" ] && cat "$SRC/iclr2027.pdf" > iclr2027.pdf
# keep the root clean: any other root .tex/.pdf goes to archive/
mkdir -p archive
for f in *.tex *.pdf; do [ "$f" = iclr2027.tex ] || [ "$f" = iclr2027.pdf ] || git mv -k "$f" archive/ 2>/dev/null || true; done
git add -A
if git diff --cached --quiet; then echo "Overleaf: nothing to sync"; exit 0; fi
git -c user.name="Abhijit Das" -c user.email="aj.das.research@gmail.com" commit -qm "${1:-Sync ICLR submission sources}"
git push -q origin HEAD
echo "Overleaf synced: $(git log --oneline -1)"
