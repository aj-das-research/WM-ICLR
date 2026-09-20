#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
exec 9>build/.build.lock
flock 9
# Generate the active-input recorder explicitly; never inspect a previous build.
rm -f build/main.fls
TEXINPUTS="$(pwd)/template/official/iclr2027//:${TEXINPUTS:-}" \
  pdflatex -recorder -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass1.stdout
(cd build && BIBINPUTS="..:${BIBINPUTS:-}" BSTINPUTS="../template/official/iclr2027//:${BSTINPUTS:-}" bibtex main > bibtex.stdout)
TEXINPUTS="$(pwd)/template/official/iclr2027//:${TEXINPUTS:-}" \
  pdflatex -recorder -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass2.stdout
TEXINPUTS="$(pwd)/template/official/iclr2027//:${TEXINPUTS:-}" \
  pdflatex -recorder -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass3.stdout
python scripts/check_table_completeness.py --fls build/main.fls --aux build/main.aux \
  --output build/table_completeness.json
cp build/main.pdf proposal.pdf
cp build/main.pdf world_model_draft.pdf
for figure in method split; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "figures/${figure}_standalone.tex" > "build/${figure}.stdout"
  cp "build/${figure}_standalone.pdf" "figures/${figure}.pdf"
  pdftocairo -svg "figures/${figure}.pdf" "figures/${figure}.svg"
  pdftoppm -r 300 -png -singlefile "figures/${figure}.pdf" "figures/${figure}"
done
python scripts/record_build.py
