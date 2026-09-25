#!/bin/bash
# Toy-world walkthrough (appendix): train the real V2WorldModel (tiny config: d=64, 2+2 blocks, window 11, H=3, K=5)
# on data/toy (scripts/v2/toy_world.py), identical budget for every arm. Outputs results/toy/<arm>/.
set -eo pipefail
mkdir -p logs/toy
for arm in shiftwm direct ar; do
  python scripts/v2/toy_train.py --arm $arm --steps 6000 --batch 64 --threads 4 --eval_every 1000 > logs/toy/$arm.log 2>&1 &
done
wait
