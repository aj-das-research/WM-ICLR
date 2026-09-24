# Sourced by lt_*.sh: Stage-2 DINOv2-S extraction for Language-Table, exactly once across the two lanes
# (flock on NFS; whichever lane starts first extracts, the other waits and then reuses the cache).
set -eo pipefail
cd /home/Test/abhijit.das/projects/WM-ICLR
test -f data/v2/frames/language_table/manifest.json || { echo "missing stage-1 manifest (scripts/v2/prepare_openx.py --dataset language_table)" >&2; exit 1; }
mkdir -p data/v2/features/language_table
(
  flock -w 7200 9 || { echo "extract lock timeout" >&2; exit 1; }
  if [ ! -f data/v2/features/language_table/dinov2s/stats.json ]; then
    python -m shiftwm.v2.extract --source data/v2/frames/language_table --kind stage1 \
      --output data/v2/features/language_table/dinov2s --encoder dinov2s
  fi
) 9> data/v2/features/language_table/.extract.lock
test -f data/v2/features/language_table/dinov2s/stats.json
