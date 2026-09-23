#!/usr/bin/env bash
# Usage: launch_pack.sh CONFIG SEED "arm1 arm2 ..." [extra --set args...]
# Runs several arms concurrently on the allocated GPU (inside a Slurm job).
set -uo pipefail
CFG=$1; SEED=$2; ARMS=$3; shift 3
pids=()
for arm in $ARMS; do
  name=${arm%%:*}; realarm=${name}
  out=$(python -c "import json,sys;c=json.load(open('$CFG'));print(c['output'])" | sed "s/ARM/$name/;s/SEED/$SEED/")
  mkdir -p "$out"
  python -m shiftwm.v2.train --config "$CFG" --set seed=$SEED output="\"$out\"" model.arm="\"$realarm\"" "$@" > "$out/stdout.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done; exit $fail
