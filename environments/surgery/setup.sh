#!/usr/bin/env bash
# Reproduce the CPU rendering environment without modifying the main .venv.
set -euo pipefail
surgery_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$surgery_dir/../.." && pwd)"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
sofa_commit=85bf7e05dd088b824794dda0046679df13b13e6e
sofa_sha=9d515e2f25f657c744821be8a5361e22803c18947b33af7a0b357c259202236a
if [[ ! -d "$project_dir/external/sofa_env/.git" ]]; then
  git clone --filter=blob:none https://github.com/ScheiklP/sofa_env.git "$project_dir/external/sofa_env"
  git -C "$project_dir/external/sofa_env" checkout "$sofa_commit"
fi
[[ "$(git -C "$project_dir/external/sofa_env" rev-parse HEAD)" == "$sofa_commit" ]]
if [[ ! -x "$surgery_dir/.venv/bin/python" ]]; then
  "$uv_bin" venv --python 3.10.21 "$surgery_dir/.venv"
fi
"$uv_bin" pip sync --python "$surgery_dir/.venv/bin/python" "$surgery_dir/requirements.lock.txt"
mkdir -p "$surgery_dir/downloads"
sofa_archive="$surgery_dir/downloads/SOFA_v24.06.00_Linux.zip"
if [[ ! -f "$sofa_archive" ]]; then
  curl -L --fail --retry 2 -o "$sofa_archive" https://github.com/sofa-framework/sofa/releases/download/v24.06.00/SOFA_v24.06.00_Linux.zip
fi
printf '%s  %s\n' "$sofa_sha" "$sofa_archive" | sha256sum --check
if [[ ! -d "$surgery_dir/SOFA_v24.06.00_Linux/lib" ]]; then
  unzip -q "$sofa_archive" -d "$surgery_dir"
fi
"$surgery_dir/run_python.sh" -c 'import Sofa; import Sofa.Core; import open3d; print("SOFA/Python imports successful")'
