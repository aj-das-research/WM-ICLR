#!/usr/bin/env bash
# run_variant.sh CONFIG SEED NAME ARM [--set overrides...]  -> results/.../ablations/NAME/s<SEED>
set -uo pipefail
CFG=$1; SEED=$2; NAME=$3; ARM=$4; shift 4
base=$(python -c "import json;print(json.load(open('$CFG'))['output'])" | sed "s#/ARM/sSEED##")
out=$base/ablations/$NAME/s$SEED; mkdir -p $out
python -m shiftwm.v2.train --config $CFG --set seed=$SEED output="\"$out\"" model.arm="\"$ARM\"" "$@" > $out/stdout.log 2>&1
