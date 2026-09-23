#!/bin/bash
# Shared checks for the V-JEPA 2-AC plug-in tasks (sourced; job.sbatch already activated .venv, PYTHONPATH=src).
export OMP_NUM_THREADS=8
CK=data/pretrained/vjepa2_ac/vjepa2-ac-vitg.pt
REF=references/vjepa2_ac_sources.json
CACHE=data/v2/features/droid/vjepa2g
[ -f "$CK" ] || { echo "missing $CK (see $REF)"; exit 2; }
[ -d external/vjepa2/src ] || { echo "missing external/vjepa2"; exit 2; }
HEAD=$(git -C external/vjepa2 rev-parse HEAD)
PIN=$(python -c "import json;print(json.load(open('$REF'))['repo']['commit'])")
[ "$HEAD" = "$PIN" ] || { echo "external/vjepa2 at $HEAD, expected $PIN"; exit 2; }
# job limit 7h55: stop training (resumably, exit 75) so that ~15 min remain for saving
DEADLINE=$(( $(date +%s) + 7*3600 + 40*60 ))
cache_complete() { python -c "import json,sys;sys.exit(0 if json.load(open('$CACHE/index.json')).get('complete') else 1)" 2>/dev/null; }
