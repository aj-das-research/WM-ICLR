#!/bin/bash
# V-JEPA 2-AC plug-in smoke test (GPU): small cache (6 episodes/split), cache-vs-evaluator consistency on those test
# episodes, and short B / C fine-tuning runs at the real batch size (timings: s/step, eval s, peak memory).
# With SMOKE_THEN_CACHE=1 it continues with the full cache + zero-shot (vjepa_cache.sh) in the same job.
# Run: SMOKE_THEN_CACHE=1 sbatch -J vjepa_smoke slurm_jebel/job.sbatch slurm_jebel/tasks/vjepa_smoke.sh
set -uo pipefail
source slurm_jebel/tasks/_vjepa_common.sh
SC=data/v2/features/droid/vjepa2g_smoke
SR=results/v2/external/vjepa2ac_plugin_smoke
RC=0
rm -rf $SC data/v2/runs/vjepa2ac_plugin_smoke $SR
python scripts/v2/eval_vjepa2ac.py --selftest || { echo "== SMOKE FAIL 3"; RC=3; }
python scripts/v2/vjepa_cache.py --out $SC --limit-episodes 6 || { echo "== SMOKE FAIL 4"; RC=4; }
python scripts/v2/eval_vjepa2ac.py --checkpoint $CK --limit-episodes 6 --no-shuffled --output $SR/evaluator_test6.npz || { echo "== SMOKE FAIL 5"; RC=5; }
python scripts/v2/vjepa_finetune.py --arm zeroshot --cache $SC --limit-episodes 6 --eval-batch 32 --tag _smoke --no-shuffled || { echo "== SMOKE FAIL 6"; RC=6; }
python - <<'PY' || { echo "== SMOKE FAIL 7"; RC=7; }
import numpy as np
a = np.load("results/v2/external/vjepa2ac_plugin_smoke/evaluator_test6.npz")
b = np.load("results/v2/external/vjepa2ac_plugin_smoke/zeroshot/test.npz")
ids = [e for e in a["episodes"] if e in set(b["episodes"])]
ia = [list(a["episodes"]).index(e) for e in ids]; ib = [list(b["episodes"]).index(e) for e in ids]
for k in ("model_mse", "persistence_mse", "linear_mse", "model_cos"):
    x, y = a[k][ia], b[k][ib]
    print(f"[consistency] {k}: evaluator {x.mean():.6f} cache {y.mean():.6f} max rel diff {np.abs(x - y).max() / np.abs(x).mean():.2e}")
assert np.allclose(a["persistence_mse"][ia], b["persistence_mse"][ib], rtol=1e-4), "cache != evaluator encoding"
assert np.allclose(a["model_mse"][ia], b["model_mse"][ib], rtol=2e-2), "rollout from cache != evaluator"
print("[consistency] OK")
PY
for arm in finetune finetune_shiftwm; do
  python scripts/v2/vjepa_finetune.py --arm $arm --cache $SC --limit-episodes 6 --tag _smoke --steps 60 \
    --eval-every 30 --log-every 10 --no-shuffled || { echo "== SMOKE FAIL 8"; RC=8; }
done
python scripts/v2/vjepa_finetune.py --compare --results $SR || true
echo "== smoke done rc=$RC $(date -Is)"
if [ "${SMOKE_THEN_CACHE:-0}" = 1 ]; then bash slurm_jebel/tasks/vjepa_cache.sh || RC=$?; fi
exit $RC
