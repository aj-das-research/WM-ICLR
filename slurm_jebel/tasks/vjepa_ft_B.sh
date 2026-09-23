#!/bin/bash
# V-JEPA 2-AC plug-in study, arm B (finetune): fine-tune the AC predictor on our DROID
# train split from the cached tokens (encoder frozen), their objective (TF L1 + auto_steps=2 rollout L1), 3000 steps x
# batch 64, their WSD schedule at LR 4.25e-4*64/256, val-selected; then val/test evaluation (eval_vjepa2ac protocol).
# Arms B and C use identical settings apart from the head. Seeds via SEEDS (default "0 1").
# Run: sbatch -J vjepa_ft_B slurm_jebel/job.sbatch slurm_jebel/tasks/vjepa_ft_B.sh
# Resumable: exits 75 before the job limit -> re-submit the same command; finished seeds are skipped.
set -uo pipefail
source slurm_jebel/tasks/_vjepa_common.sh
cache_complete || { echo "cache $CACHE incomplete: run vjepa_cache.sh first"; exit 3; }
for s in ${SEEDS:-0 1}; do
  python scripts/v2/vjepa_finetune.py --arm finetune --seed $s --cache $CACHE --checkpoint $CK --deadline $DEADLINE
  rc=$?
  [ $rc -eq 75 ] && { echo "== deadline: re-submit this task to resume (seed $s)"; exit 75; }
  [ $rc -ne 0 ] && { echo "== seed $s failed rc=$rc"; exit $rc; }
done
python scripts/v2/vjepa_finetune.py --compare || true
