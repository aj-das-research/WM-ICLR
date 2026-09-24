#!/bin/bash
# Blind-review-2 control: V-JEPA 2-AC fine-tuned with a GATED PERSISTENCE head (window 1, 1 source = copy the same patch of
# the last measured frame), otherwise identical to arm C (finetune_shiftwm). Tests "transport vs. gated identity path".
set -uo pipefail
source slurm_jebel/tasks/_vjepa_common.sh
cache_complete || { echo "cache incomplete"; exit 3; }
python scripts/v2/vjepa_finetune.py --arm finetune_shiftwm --seed 0 --window 1 --sources 1 --tag _persist \
  --cache $CACHE --checkpoint $CK --deadline $DEADLINE
