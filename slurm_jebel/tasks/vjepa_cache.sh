#!/bin/bash
# V-JEPA 2-AC plug-in study, step 1: cache frozen ViT-g/16 tokens of all 65,952 DROID primary-camera frames
# (train 47,039 / val 9,556 / test 9,357; 256 tokens x 1408 fp16 = 47.5 GB) under data/v2/features/droid/vjepa2g,
# then run arm A (zero-shot) from the cache on val + test.
# Run: sbatch -J vjepa_cache slurm_jebel/job.sbatch slurm_jebel/tasks/vjepa_cache.sh      (resumable; re-submit if cut)
# Expected: ~3 min checkpoint load, ~25-45 min encoding+writing, ~15 min zero-shot eval (val + test w/ shuffled).
set -uo pipefail
source slurm_jebel/tasks/_vjepa_common.sh
python scripts/v2/vjepa_cache.py --out $CACHE || exit $?
cache_complete || { echo "cache incomplete"; exit 3; }
du -sh $CACHE
python scripts/v2/vjepa_finetune.py --arm zeroshot --cache $CACHE --checkpoint $CK
