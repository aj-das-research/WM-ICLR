#!/bin/bash
# Stage-1 frames + DINOv2-S 16x16 feature cache for a LeWM planning env.  PENV=pusht|tworoom|reacher
# (wrapped by plan_prep_<env>.sh). ~72k strided frames -> ~14 GB fp16 cache; est. 15-30 min/env.
set -eo pipefail
: "${PENV:?set PENV}"
export HDF5_USE_FILE_LOCKING=FALSE
F=data/v2/frames/plan_$PENV; X=data/v2/features/plan_$PENV/dinov2s
if [ ! -f $F/manifest.json ]; then
  python scripts/v2/prepare_plan_frames.py --env $PENV --out $F --max-frames ${MAX_FRAMES:-72000} --workers ${WORKERS:-24}
fi
if [ ! -f $X/stats.json ]; then
  python -m shiftwm.v2.extract --source $F --kind stage1 --output $X --encoder dinov2s
fi
du -sh $F $X
