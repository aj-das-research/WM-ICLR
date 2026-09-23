# ShiftWM-v2 analysis pipeline (E1 camera transfer, E6 analysis, E7 pixel decoding)

All scripts read Stage-2 caches (`docs/v2_data_format.md`) and trained runs under
`results/v2/<dataset>/dinov2s/<arm>/s<seed>/`, and write under `results/v2/analysis/` (except the
camera-transfer evals, which go next to the regular runs so `scripts/v2/make_tables.py` picks them up).
Shared helpers: `src/shiftwm/v2/analysis.py`.

**Only finished runs are used**: a learned arm counts when its run directory has both `best.pt` and
`summary.json` (written by `train.run` after the final test evaluation). `SHIFTWM_ALLOW_PARTIAL=1`
relaxes this for smoke tests only.

## Environment additions (not pinned in `requirements.lock.txt`)
- `~/.local/bin/uv pip install --python .venv/bin/python lpips` -> `lpips==0.1.4` (2026-09-23).
- Torchvision weights pre-downloaded on the login node into `~/.cache/torch/hub/checkpoints/`
  (compute jobs run with `HF_HUB_OFFLINE=1`): `alexnet-owt-7be5be79.pth` (LPIPS-Alex metric),
  `vgg16-397923af.pth` (LPIPS-VGG training loss), `raft_large_C_T_SKHT_V2-ff5fadd5.pth` (RAFT-large).

## Scripts, Slurm tasks and outputs
Run each with `sbatch -J NAME slurm_jebel/job.sbatch slurm_jebel/tasks/<task>.sh`.

| Script | Task file | Needs | Output | Est. H200 time |
|---|---|---|---|---|
| `scripts/v2/eval_transfer.py` | `analysis_transfer.sh` | DROID cam-1 runs, `droid_cam2` cache | `results/v2/droid_cam2/dinov2s/<arm>/s<seed>/eval_test.npz` (+ `transfer.json`), tables refreshed | 5-15 min |
| `scripts/v2/train_decoder.py` | `analysis_train_decoder.sh` | caches + frames | `results/v2/analysis/decoder/<ds>/dinov2s/best.pt` | 1.5-2.5 h (both packed) |
| `scripts/v2/decode_eval.py` | `analysis_decode_eval.sh` | decoders + runs | `results/v2/analysis/pixel/`, `tables/generated/pixel_rows.tex` | 0.5-1 h |
| `scripts/v2/probes.py` | `analysis_probes.sh` | runs | `results/v2/analysis/probes/`, `tables/generated/probe_rows.tex` | 15-30 min |
| `scripts/v2/flow_agreement.py` | `analysis_flow.sh` | ShiftWM (+Direct) runs, frames | `results/v2/analysis/{flow,gain_vs_motion}/` | 20-40 min |
| `figures/src/make_analysis_figures.py` | `analysis_figures.sh` | all of the above | `figures/{flow_agreement,gain_vs_motion,tradeoff,qualitative_transport,gallery_droid,gallery_surgical,failures}.pdf` | 5-10 min |

Order: transfer, decoder, and probes/flow are independent; `decode_eval` needs the decoders; figures last.
All scripts skip finished outputs (use `--overwrite` to redo) and accept `--device cpu --max-episodes N`
for smoke tests (login node: 5 GB RAM, keep batches small, e.g. `--batch-size 4`). `decode_eval.py` and
`probes.py` accept `--rows-only` to rebuild the LaTeX rows from saved results.

## Definitions
- **Windows**: test windows at stride 2 as in `train.evaluate` (flow: stride 4). Window start `s`
  observes frames `s..s+2`; `t0 = s+2`; step `k` targets frame `t0+k`.
- **Frames**: the real frames are resized exactly like the encoder input (224x224, full-frame
  antialiased bilinear, no crop), so patch `(i, j)` covers pixels `14i..14i+13`.
- **Camera transfer**: target-cache features and actions are standardised with the *training* cache's
  `stats.json` (cam 1); the target cache's own stats are never read.
- **Decoder**: `FeatureDecoder` (16x16xC -> 224x224 RGB, ~18M params), one per dataset, trained on true
  train features only (all train frames, 30k steps, batch 32, AdamW 2e-4, L1 + 1.0 x LPIPS-VGG), selected
  on val. Metrics: PSNR, SSIM (11x11 Gaussian, sigma 1.5), LPIPS-Alex, vs real future frames. Upper bound =
  decoder applied to the true future features.
- **Probes**: ridge on [mean-pool ++ 4x4-pool] of standardised features (6528-D), ridge strength chosen on
  true val features. DROID target: `actions[t, 0:3]` (commanded EE xyz of the first 7-D command at frame t,
  m -> cm). Hamlyn: measured tip xyz `proprio[t, 0:3]` (left PSM) and `proprio[t, 8:11]` (right PSM),
  m -> mm (`docs/v2_openh_hamlyn.md`). MAE per coordinate; Pearson r per coordinate, averaged.
- **Transport vs flow**: expected source offset `o = sum_j pi_j offset_j` (as in
  `make_figures.transport_arrows`); content motion is `-o`. Primary comparison uses RAFT *backward* flow
  (future -> observed), which is defined at the query patch like `o`; the forward-flow comparison at the
  same patch index is also saved (`*_fwd`). Flow is average-pooled to 16x16 and divided by 14 px. Metrics
  on patches with |flow| > 0.5 patch: cosine, EPE, EPE of the gate-weighted displacement, and the
  zero-motion EPE (= |flow|). Offsets are tiled over the S=3 source frames, so weight on older source
  frames is treated as displacement from `t0` (the saved `last_source_mass` quantifies this).
- **Gain vs motion**: per-patch squared error (mean over channels) of ShiftWM and Direct (same seed),
  binned by the true change `mean_c (z_{t0+k} - z_{t0})^2` with edges `0, 0.02 x 2^i (i=0..8), inf`.
- **Qualitative selection rules** are documented in the docstring of `make_analysis_figures.py`;
  the chosen episodes/windows are written to `results/v2/analysis/qualitative/`.
