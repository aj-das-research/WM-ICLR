set -eo pipefail
if [ ! -f data/real_video/droid_selected/processed/manifest.json ]; then
  environments/real_video/.venv/bin/python scripts/real_video/prepare_droid.py --raw data/real_video/droid_selected/raw \
    --output data/real_video/droid_selected/processed --inventory reports/evidence/real_video/droid_selected_inventory.json \
    --expected-episodes 1126 --workers 28
fi
for enc in dinov2s dinov3s dinov2b; do
  python -m shiftwm.v2.extract --source data/real_video/droid_selected/processed --kind droid \
    --output data/v2/features/droid/$enc --encoder $enc
done
python -m shiftwm.v2.extract --source data/real_video/droid_selected/processed --kind droid \
    --output data/v2/features/droid_cam2/dinov2s --encoder dinov2s --camera exterior_image_2_left
