#!/bin/bash
# E7 feature->RGB decoders (true train features -> frames, L1 + LPIPS-VGG), DROID and Hamlyn packed on one GPU.
# Output results/v2/analysis/decoder/<ds>/dinov2s/best.pt; resumable from last.pt. ~1.5-2.5 h wall.
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/train_decoder.py --dataset droid > results/v2/analysis/logs/decoder_droid.log 2>&1 &
p1=$!
python scripts/v2/train_decoder.py --dataset openh_hamlyn > results/v2/analysis/logs/decoder_openh_hamlyn.log 2>&1 &
p2=$!
rc=0; wait $p1 || rc=1; wait $p2 || rc=1
tail -n 3 results/v2/analysis/logs/decoder_*.log; exit $rc
