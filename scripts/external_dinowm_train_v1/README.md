# Adapted official DINO-WM on matched DROID train/development data

This new, six-run study trains the official DINO-WM predictor from scratch under
two separately declared objectives and seeds 0, 1, and 2. It is an adapted external
implementation comparison on our existing development population, not a
reproduction of the published DINO-WM benchmark. No reserved or final-test data
may be loaded, used for model selection, or used for objective selection.

## Architecture and data interface

The MIT-licensed official source is `external/official-dino-wm`, revision
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`. The frozen, independently reviewed adapter
under `scripts/external_dinowm_profile_v1` preserves the predictor's computation:
six transformer layers, 16 heads of width 64, MLP width 2048, dropout 0.1, a
kernel-one 35-to-10 action encoder, and 19,412,420 trainable parameters. Its only
upstream source substitution makes the exact frame-causal attention mask a
nonpersistent device-portable buffer. Upstream checkout bytes are unchanged.

Inputs are the existing frozen DINOv2-small channel-major 384-by-4-by-4 grids,
sixteen pooled spatial tokens per frame, with train-only shared-channel
normalization. This differs from official native patch tokenization. Each
35-dimensional action block concatenates five consecutive seven-dimensional
commands; there is no proprioceptive input or proprioceptive prediction loss.
The same three observations and two past action blocks condition all forecasts.
Ten future action blocks are provided, without future feature input. Removing
proprioception, pooling tokens, using this command schema and using DROID are
explicit interface adaptations, not claims of an unchanged original experiment.

## Two objectives, one selection and evaluation rule

* `official_one_step_shifted`: one native predictor call on feature grids 0, 1,
  and 2 and their three outgoing action blocks. Visual predictions at all three
  slots are supervised against grids 1, 2, and 3. Thus the objective includes
  two observed next-slot targets plus the first future target. Predicted action
  channels are excluded from the loss. This preserves the upstream shifted
  one-step visual objective on the adapted interface.
* `matched_recursive_h10`: ten rolling predictor calls, with the last visual
  slot from each call fed into the next context and full backpropagation through
  the rollout. All ten future grids 3 through 12 are supervised. This changes
  the original training objective and is reported separately.

Both arms use precisely the same H10-eligible training windows: 18,660 windows
at stride two. The native arm does not silently gain additional short windows.
For both arms, every epoch is selected using the window-weighted mean normalized
FP32 recursive MSE across all ten future grids on 1,631 development windows at
stride five. CUDA TF32 and autocast are disabled for validation. Earliest minimum
over all 30 completed epochs is selected. Final reporting averages windows
within episodes, then 141 equally weighted eligible episodes (59 sessions),
then the three seeds. Selection and reporting therefore have different explicit
aggregation rules; their scalar losses need not be equal.

The unchanged original spatial evaluator computes every per-window/per-horizon
native MSE, native persistence MSE, original-2-by-2 MSE and original-2-by-2
persistence MSE. Coarse targets are the exact cached original coordinates, not
recomputed pooled targets. The predictor recurses in normalized coordinates and
returns raw-coordinate forecasts for final scoring, retaining the existing
metric arithmetic. No closed-loop planning or
external SOTA accuracy claim follows from this forecast-only experiment.

## Fixed training budget and continuation

Each arm completes 30 full epochs, batch 128 without dropping the final batch,
AdamW learning rate 1e-4, weight decay 0.01, cosine decay to 1e-6, gradient norm
clip 1, BF16 autocast during training and FP32 validation. Shared feature and
action statistics are recomputed from training episodes for validation of the
existing immutable statistics; nothing is fitted on development data. No
automatic reduced-model, batch-size, precision, early-stop, or loss fallback is
permitted. This recipe matches the existing spatial training protocol; it is
not advertised as the official paper's original optimizer schedule.

The reused atomic trainer stores model tensors/config, optimizer, scheduler,
RNG and loader-generator state in immutable generations. Resumption restores
the last fully completed epoch. An exclusive `.training.lock` covers training,
evaluation and completion publication. A 2-hour allocation checkpoints and
requeues at an epoch boundary after 90 minutes, at most five restarts. Source
hashes are checked before every epoch, before/after final evaluation, and at
completion. Logs report actual epoch losses; these are not benchmark gains.

## Freeze, run and report

`campaign.py register` creates all six exact configs and an immutable source
registration. It reads metadata and already completed synthetic capacity
evidence only. A separate independent review must bind the exact registration
and every dependency before any training or feature payload access. The
reporter implementation is part of this source closure; its own registration
can subsequently bind this registration without a circular dependency.

```
.venv/bin/python scripts/external_dinowm_train_v1/campaign.py register
.venv/bin/python scripts/external_dinowm_train_v1/campaign.py verify
sbatch --array=0-3%2 scripts/external_dinowm_train_v1/run.slurm
sbatch --partition=gpu --array=4-5%1 scripts/external_dinowm_train_v1/run.slurm
```

Only after independent review, the two disjoint arrays use account `students`:
seed-major indices 0/1 are seed 0, 2/3 seed 1, and 4/5 seed 2; each pair contains
native then recursive objectives. At most two `ws-ia` GPUs and one `gpu`
partition GPU run concurrently. The identical GPU family was confirmed on
both partitions. Per-attempt execution receipts record hostname, visible GPU
name/memory/capability, PyTorch/CUDA versions, Slurm job/array identity, UTC
start/end times and separate training/evaluation wall seconds. The training
timer includes data verification, loading and checkpoint writes. Failed and
interrupted attempts retain their own receipts; a resumed attempt is not
misreported as the whole run's runtime. The completion marker binds the final
attempt receipt, and its starting identity is also printed in the Slurm log.

Per-run outputs are `runs/external_dinowm_train_v1/{mode}_s{seed}` and
`reports/external_dinowm_train_v1/{mode}_s{seed}_{validation,completed}.json`.
`scripts/external_dinowm_reporting_v1/finalize.py --if-ready` checks all six
completed 30-epoch runs and the fifteen existing matched internal controls
before reporting any aggregate. Its independently reviewed protocol retains
all methods, all ten horizons, all four metric arrays and negative results.
Intervals are unadjusted exploratory paired session/seed bootstrap intervals.

Synthetic full-shape capacity job 201998 measured both objectives on an RTX
5000 Ada. Predictor-only projections exclude feature loading, hashing,
normalization checks, checkpoint serialization, final evaluation, scheduler
queue and other host work. They are not full-run timing or accuracy results.
