# Global-window ablation alone on an 80 GB GPU (with a packed neighbour it OOMs at eval batch 128 on 40 GB).
scripts/v2/run_variant.sh configs/v2/droid_v1s.json 0 global shiftwm model.window=0
