# Corrected raw-coordinate DINO-WM adaptation (v2)

This separate study strengthens the completed v1 external baseline. It never
rewrites v1 results or training sources. Its motivation and read-only evidence
are `reports/external_dinowm_train_v1/diagnosis.md`. No reserved/test payloads
enter training, selection or final evaluation.

## Fixed predictor and declared adaptations

Official repository: https://github.com/gaoyuezhou/dino_wm, MIT, commit
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`. The reviewed profile adapter loads the
unchanged official ViT and kernel-one action Conv1d; its sole source portability
edit replaces a CUDA-created causal mask with the identical nonpersistent buffer.

The predictor has depth 6, 16 heads, head width 64, MLP width 2048, dropout 0.1,
no embedding dropout, and 19,412,420 trainable parameters. It preserves the
learned-affine terminal LayerNorm, random learned position embeddings and absolute
visual prediction. No extra projection, residual, initialization trick or model
size reduction is added. Our interface still differs from official benchmarks:
16 adaptively pooled DINOv2-small tokens of width384 (channel-major384x4x4),
three observed grids, no proprioception and 35-D action blocks comprising five
consecutive seven-dimensional commands. Those changes are declared adaptations,
not a reproduction of DINO-WM's native-patch benchmark accuracy.

Unlike v1, visual inputs, recursive intermediate predictions, outputs and
training targets remain in **raw cached DINO coordinates**. Shared-channel
training feature statistics are buffers only for standardized selection/scoring;
they never whiten the inputs or targets of the predictor. Actions retain the
same fixed training-only 35-D mean/std. Official code enables action normalization,
but these grouped-command statistics and DROID schema are our explicit adaptation.

Support grids0/1/2 have outgoing blocks0/1/2. Native predictions target grids1/2/3,
scoring only384 visual channels and excluding predicted action channels. Recursive
step k uses the three current visual grids and action blocks k:k+3, replacing
action embeddings with known supplied blocks each call. No future target is an
input; recursive forecasts remain connected for full backpropagation.

## Six complete runs and common selection

Seeds0/1/2 for each distinct mode:

- `official_raw_one_step`: mean squared error in raw visual coordinates for all
  three one-step-shifted slots (two observed next grids plus first future grid).
- `official_raw_recursive_h10`: an explicitly non-native training control with
  raw visual MSE over all ten recursive query grids3:13 and full BPTT.

Both use pinned upstream defaults of100 epochs, batch32, AdamW learning rate5e-4,
constant LR and weight decay0.01 (upstream implicit AdamW default), no gradient
clipping. One AdamW covers the disjoint predictor/action parameters; it is
algebraically equivalent to the upstream two AdamW instances with the same
settings and update schedule. Betas/epsilon remain PyTorch defaults. Training and
validation are explicitly FP32 with TF32 disabled; upstream Accelerator invocation
does not pin precision, so this is a declared choice rather than a claimed exact
reproduction of original runtime precision. There is no gradient accumulation,
precision fallback, batch reduction or early stopping.

The unchanged spatial loader gives18,660 H10-eligible train windows (stride2) and
1,631 validation windows (stride5,141 episodes,59 sessions), camera1 only. Native
training deliberately shares the H10-eligible population rather than adding its
shorter-only windows. Every run completes584 updates/epoch,58,400 total. Selection
is the earliest minimum **window-mean, all-ten, standardized FP32 recursive MSE**
over all100 epochs, common to both arms and existing controls. `standardized_mse`
is genuinely standardized even in training diagnostics; `raw_mse` and
`optimized_loss=raw_visual_mse` separately identify what was optimized. Native
and recursive raw training losses cover different target slots and are not
interchangeable. Development evaluation uses the unchanged spatial metric
arithmetic and all4x10 primitive arrays, returned in raw forecast coordinates.

`engine.py` checks the frozen atomic engine's SHA then transforms only `fit` and
`validate_completed`: ten epoch literals30/31 become100/101; the cosine scheduler
becomes an identity LambdaLR to retain optimizer/scheduler resume serialization.
Original sources remain untouched. Atomic generation packages preserve optimizer,
RNG, loader-generator and complete history. Resume replays only an interrupted
in-flight epoch; there is an exclusive lock per run.

## Resource and execution contract

Prior v1 measured RTX5000Ada runtimes were4.34–4.71min/native and15.28–16.23min/
recursive seed, at30epochs/B128/BF16. V2 has13.33x optimizer steps and3.33x data
exposures with FP32/B32. The previous profile supports capacity but is not an
exact v2 FP32 measurement. Planning estimates are30–90min/native and1.5–4h/
recursive seed;6–16.5 total GPU-hours, approximately2–5.5h under three balanced
lanes, plus queue time. These broad bounds will be replaced by actual execution
receipts, not presented as measured throughput or model quality.

Use one GPU,8 CPU,24GB host RAM,2h allocation, account`students`; continue at
completed-epoch boundaries after90min, at most five requeues. At most two `ws-ia`
GPU jobs and one `gpu` partition GPU job. Per-attempt receipts record hostname,
GPU model/memory, Torch/CUDA, scheduler IDs, UTC boundaries and training/evaluation
wall times. Scientific failures stop; no adaptive change to model/recipe occurs.

Run the synthetic contract tests, register all six configs/source bindings, then
obtain an independent source-review receipt **before** submitting. Registration
and review are required before cache payloads or selected checkpoints are loaded.

```bash
.venv/bin/python -m pytest -q --import-mode=importlib scripts/external_dinowm_raw_v2/test_contract.py
.venv/bin/python scripts/external_dinowm_raw_v2/campaign.py register
.venv/bin/python scripts/external_dinowm_raw_v2/campaign.py verify
sbatch --array=0-3%2 scripts/external_dinowm_raw_v2/run.slurm
sbatch --partition=gpu --array=4-5%1 scripts/external_dinowm_raw_v2/run.slurm
```

The seed-major roster is0/1 seed0,2/3 seed1,4/5 seed2; objectives alternate.
Final comparison is separately frozen in `external_dinowm_raw_reporting_v2` and
requires all six complete100-epoch results against the existing frozen internal
30-epoch controls. This is a stronger-baseline follow-up with unequal optimization
budgets, not a matched-budget ablation. Coordinates and recipe change jointly;
any improvement does not identify which v1 change caused its weak behavior.

A maximum of five requeues permits six90-minute training segments (approximately
nine hours/run, plus the final epoch and verification/evaluation overhead).
Requeue failure or exhausted allowance raises an error and creates no completion
marker; the all-six finalizer remains closed. Each attempt records requested,
submitted, or failed continuation. A scheduler may terminate the old process
immediately after accepting requeue, leaving its requested receipt; the next
attempt's Slurm restart count and resumed epoch then provide continuation evidence.
