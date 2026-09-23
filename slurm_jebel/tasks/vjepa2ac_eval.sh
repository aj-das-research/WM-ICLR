#!/bin/bash
# E2 (P1): zero-shot V-JEPA 2-AC (official vjepa2-ac-vitg.pt, external/vjepa2 @ pinned commit) on our DROID
# test split, measured in its own layer-normed ViT-g feature space vs persistence / linear extrapolation.
# Run: sbatch -J v2-vjepa2ac slurm_jebel/job.sbatch slurm_jebel/tasks/vjepa2ac_eval.sh
# (job.sbatch activates .venv, sets PYTHONPATH=src HF_HUB_OFFLINE=1; no network needed.)
# Expected wall: ~5 min checkpoint load (11.8 GB) + ~2-5 min encoding 9357 frames + ~10-25 min rollouts
# (3923 windows x 10 AR steps, x2 with shuffled actions) for the main run; each ablation ~10-20 min.
# Budget ~1-1.5 h total; peak GPU ~25-40 GiB, host RAM < 40 GiB.
set -uo pipefail
export OMP_NUM_THREADS=8
CK=data/pretrained/vjepa2_ac/vjepa2-ac-vitg.pt
REF=references/vjepa2_ac_sources.json
OUT=results/v2/external/vjepa2ac
[ -f "$CK" ] || { echo "missing $CK (see $REF)"; exit 2; }
[ -d external/vjepa2/src ] || { echo "missing external/vjepa2 (clone + checkout commit in $REF)"; exit 2; }
WANT=$(python -c "import json;print(json.load(open('$REF'))['checkpoint']['sha256'])")
HEAD=$(git -C external/vjepa2 rev-parse HEAD)
PIN=$(python -c "import json;print(json.load(open('$REF'))['repo']['commit'])")
[ "$HEAD" = "$PIN" ] || { echo "external/vjepa2 at $HEAD, expected $PIN"; exit 2; }
mkdir -p $OUT
rc=0
run() { echo "== $(date -Is) eval_vjepa2ac.py $*"; python scripts/v2/eval_vjepa2ac.py --checkpoint $CK \
          --checkpoint-sha256 "$WANT" "$@" || rc=1; }
python scripts/v2/eval_vjepa2ac.py --selftest || exit 3
# main result: training-transform crop, 8-frame sliding context (trained clip length), shuffled-action check
run --output $OUT/droid_test.npz
# sensitivity: full (un-windowed) context up to 12 frames; notebook square-crop inference transform
run --max-context 0 --no-shuffled --output $OUT/droid_test_ctxfull.npz
run --crop square --no-shuffled --output $OUT/droid_test_square.npz
echo "== done rc=$rc"; exit $rc
