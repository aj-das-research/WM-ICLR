#!/bin/bash
# Toy-world walkthrough (appendix): train the real V2WorldModel (tiny config) arms on data/toy, same budget.
set -eo pipefail
for arm in shiftwm direct ar; do
  python scripts/v2/toy_train.py --arm $arm --steps 8000 --batch 64 --threads 4 --eval_every 1000 > logs/toy/$arm.log 2>&1 &
done
wait
