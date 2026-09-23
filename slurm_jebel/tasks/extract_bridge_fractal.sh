set -eo pipefail
# Stage-2 DINOv2-S feature extraction for the Open-X real-robot benchmarks
# (Stage-1 frames from scripts/v2/prepare_openx.py; GPU job, scheduled by the main session).
cd /home/Test/abhijit.das/projects/WM-ICLR; source .venv/bin/activate; export PYTHONPATH=src HF_HUB_OFFLINE=1
for ds in bridge fractal; do
  test -f data/v2/frames/$ds/manifest.json || { echo "missing stage-1 manifest for $ds" >&2; exit 1; }
  for enc in dinov2s; do
    python -m shiftwm.v2.extract --source data/v2/frames/$ds --kind stage1 --output data/v2/features/$ds/$enc --encoder $enc
  done
done
