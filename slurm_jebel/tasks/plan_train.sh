#!/bin/bash
# Train <=2 learned arms concurrently on one GPU for planning env PENV (wrapped by plan_train_<env>.sh).
#   PENV=pusht|tworoom|reacher  SEED=0  ARMS="shiftwm ar"   (token NAME or NAME_ctr: *_ctr adds the
#   action-contrastive loss, contrastive_weight=${CTR_W:-0.1}; output results/v2/plan/<env>/<NAME>/s<SEED>)
# Est. (H=1,K=5, 16k steps, batch 48): ~0.5-1.5 h per pack; GPU mem ~2x(14 GB cache + activations).
set -uo pipefail
: "${PENV:?set PENV}"; SEED=${SEED:-0}; ARMS=${ARMS:-"shiftwm ar"}
CFG=configs/v2/plan_$PENV.json
pids=()
for name in $ARMS; do
  arm=${name%_ctr}; extra=()
  [ "$name" != "$arm" ] && extra=(contrastive_weight=${CTR_W:-0.1})
  out=results/v2/plan/$PENV/$name/s$SEED; mkdir -p $out
  echo "== $(date -Is) train $name (arm=$arm ${extra[*]}) -> $out"
  python -m shiftwm.v2.train --config $CFG --set seed=$SEED output="\"$out\"" model.arm="\"$arm\"" "${extra[@]}" \
    > $out/stdout.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
for name in $ARMS; do tail -n 3 results/v2/plan/$PENV/$name/s$SEED/stdout.log; done
exit $fail
