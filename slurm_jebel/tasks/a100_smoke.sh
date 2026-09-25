# Smoke test: two heaviest arms packed on one A100-40GB with host-resident features + micro-batching.
O=logs/smoke/smoke
for arm in ar ar_tf; do
  python -m shiftwm.v2.train --config configs/v2/droid_base.json --set seed=0 output="\"$O/$arm\"" model.arm="\"$arm\"" steps=200 eval_every=200 max_val_episodes=20 'report_splits=["val"]' > logs/smoke/smoke_$arm.log 2>&1 &
done
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader; done ) &
wait %1 %2
