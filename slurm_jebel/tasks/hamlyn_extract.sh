set -eo pipefail
cd /home/Test/abhijit.das/projects/WM-ICLR; source .venv/bin/activate; export PYTHONPATH=src HF_HUB_OFFLINE=1
for enc in dinov2s; do
python -m shiftwm.v2.extract --source data/v2/frames/openh_hamlyn --kind stage1 --output data/v2/features/openh_hamlyn/$enc --encoder $enc
done
