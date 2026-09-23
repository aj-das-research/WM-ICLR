set -eo pipefail
cd /home/Test/abhijit.das/projects/WM-ICLR; source .venv/bin/activate; export PYTHONPATH=src HF_HUB_OFFLINE=1
for t in pusht box rope; do
  python -m shiftwm.v2.extract --source data/v2/frames/iws_$t --kind stage1 --output data/v2/features/iws_$t/dinov2s --encoder dinov2s
done
