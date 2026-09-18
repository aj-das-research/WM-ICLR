#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
shiftwm_uv="$(command -v uv || true)"
if [[ -z "$shiftwm_uv" && -x "$HOME/.local/bin/uv" ]]; then
  shiftwm_uv="$HOME/.local/bin/uv"
fi
if [[ -z "$shiftwm_uv" ]]; then
  echo "Install uv, then rerun this script: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
python3 scripts/bootstrap_sources.py
if [[ ! -x .venv/bin/python ]]; then
  "$shiftwm_uv" venv --python 3.11 .venv
fi
# Both indexes are explicitly trusted; every package/version is pinned.
"$shiftwm_uv" pip sync --python .venv/bin/python requirements.lock.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
"$shiftwm_uv" pip install --python .venv/bin/python --no-deps -e external/stable-worldmodel -e .
