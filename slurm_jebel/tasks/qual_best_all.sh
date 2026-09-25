#!/bin/bash
# Best-case qualitative examples for every benchmark not yet covered by qual_select.py's default set:
#   droid_cam2 (zero-shot transfer, DROID checkpoints), V-JEPA 2-AC plug-in (DROID), DINO-WM plug-in (PushT, Wall).
# Run: sbatch -J qual_best -p a100 --gres=gpu:1 --cpus-per-task=16 --mem=110G --time=01:30:00 \
#        slurm_jebel/job.sbatch slurm_jebel/tasks/qual_best_all.sh
set -uo pipefail
mkdir -p results/v2/analysis/logs
L=results/v2/analysis/logs
[ -f results/v2/analysis/qual_best/droid_cam2.npz ] || python scripts/v2/qual_select.py --datasets droid_cam2 --batch-size 32 2>&1 | tee $L/qual_select_cam2.log
[ -f results/v2/analysis/qual_best/vjepa2ac_droid.npz ] || python scripts/v2/qual_best_vjepa.py 2>&1 | tee $L/qual_best_vjepa.log
(
  source slurm_jebel/tasks/dinowm_common.sh
  for env in pusht wall; do
    [ -f results/v2/analysis/qual_best/dinowm_$env.npz ] || NUM_WORKERS=8 python scripts/external_dinowm_plugin/qual_best.py --env $env 2>&1 | tee $L/qual_best_dinowm_$env.log
  done
)
ls -la results/v2/analysis/qual_best/
