#!/bin/bash
# E7 decoded-pixel metrics (PSNR/SSIM/LPIPS at k=1,5,10) for all finished arms/seeds; needs the decoders
# (analysis_train_decoder.sh). Writes results/v2/analysis/pixel/ and tables/generated/pixel_rows.tex. ~0.5-1 h.
set -uo pipefail
mkdir -p results/v2/analysis/logs
python scripts/v2/decode_eval.py 2>&1 | tee results/v2/analysis/logs/decode_eval.log
