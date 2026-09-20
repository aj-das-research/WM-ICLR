# Fixed-checkpoint IWS reserved evaluation

This study evaluates all 36 existing development-selected predictors on the
original reserved upstream-validation handles: 200 per task, ten trajectories
per task, for recorded PushT, Box and Rope. It does not train or select models
on those outcomes. The no-tanh component follow-up remains secondary to the
original bounded-mixing versus additive-anchor comparison.

The original pre-access registration is
`configs/real_video_iws_reserved_v1/registration.json`. Cache job 201873
completed all 5,996 frames. Array 201874 stopped after 18 successful scores
and two Box autoregressive seeds failed the declared CPU prefix-consistency
check; finalizer 201875 was cancelled. Those partial outputs remain unchanged
and are not used as a completed result.

Three bounded numerical probes did not reproduce the initial failure. They
checked inputs and prediction consistency, including the original scorer with
synthetic targets, without scoring actual future targets. The initial cause
remains unresolved. A separately reviewed **post-access execution revision**
uses one native command row per GRU call. It retains identical weights,
normalization, examples, comparators, metrics and tolerance and recomputes
**all 36** predictors; it never stitches together passing rows from two attempts.

The recovery registration is
`configs/real_video_iws_reserved_recovery_v2/registration.json`
(SHA-256 `891207e1c88f758c45750a56a59bd5fa6b793391026f75085cb8263c1d2c21d4`).
Independent review binds all 408 dependencies. Array **201930** and dependent
finalizer **201931** were submitted on 20 September 2026 at 13:21 UTC.
Receipts are in `reports/real_video_iws_reserved_recovery_v2/`.
All 36 evaluations and finalizer 201931 completed successfully at 13:29 UTC.
Independent reconstruction of all primitive ledgers and paired intervals passed:
`reports/real_video_iws_reserved_recovery_v2/independent_result_review.json`.
Finalization SHA-256 is
`81e97ec2d1f225e35af84e83119f5f29781e8ecb0ca61489c759affa9abeaf0b`.
The original cached features were reused; every predictor ledger was newly
computed under the common execution setting.

## Completed findings

Removing the correction bound lowers reserved H60 MSE relative to bounded
ShiftWM by 7.16%, 9.14% and 11.69%, and relative to autoregression by 7.62%,
4.84% and 6.80%, for PushT, Box and Rope. All six paired 95% intervals favor
the ablation. The equal-task gain versus autoregression is 6.42% [4.90, 7.92].
These are relative error reductions, not percentage-point task-success gains.
The secondary component comparisons are unadjusted.

The original primary bounded-versus-additive comparison remains mixed:
+3.04% [2.28, 3.75] on PushT, +2.31% [-0.09, 4.80] on Box, and
-0.38% [-2.01, 1.08] on Rope. Its equal-task gain is +1.66% [0.70, 2.62].
Bounded ShiftWM loses to autoregression on Box and Rope. The full comparison
retains those results and does not retroactively choose a new primary method.

## What is scored

Each example supplies one observed DINOv2 feature grid and 60 native command
rows. The predictor returns the next 59 stored-frame feature grids. H15, H30
and H45 are independently invoked command prefixes at the same starting
observation; their scores are stored separately from the full-H60 trajectory.
They are not additional official handle sets or physical-time horizons.

Four feature-space errors are retained: standardized MSE and MAE, raw DINOv2
L1 and cosine distance. Equal-trajectory aggregation is primary; equal-handle
aggregation is also saved because trajectory handle counts differ. Paired
uncertainty resamples three training seeds and ten trajectory clusters per
task, with shared seed draws across tasks. All comparisons and signs are kept.

## Reproduction commands

Use the workspace environment and preserve registered dependencies. Existing
data access, encoder weights and the selected local predictor packages are
required; the public source tree alone does not contain those payloads.

```bash
# Validate the existing recovery registration and independent review.
.venv/bin/python -I scripts/real_video_iws_reserved_recovery_v2/register.py check
# Launch only in an unused evaluation namespace. The actual submitted chain
# is already recorded in submission.json; do not submit it a second time.
# sbatch scripts/real_video_iws_reserved_recovery_v2/evaluate.slurm
# sbatch --dependency=afterok:ARRAY_JOB_ID scripts/real_video_iws_reserved_recovery_v2/finalize.slurm
```


The cache uses one BF16-capable GPU and the frozen feature encoder. All learned
predictors use CPU FP32 with eight threads and batch size 64, matching the
reviewed prefix-consistency protocol. The scheduler array is capped at two
concurrent jobs. No GPU-accuracy equivalence is assumed.

The evaluator reuses existing model loaders and metric functions, writes each
per-handle primitive ledger once and rejects overwritten or incomplete
evidence. The finalizer independently rebuilds all 36 result aggregates,
checks matched persistence and input identities, and only then produces the
complete comparison report. It does not require reserved error to equal the
development checkpoint-selection score.

This is a recorded-video feature-forecasting comparison. It does not measure
RGB video generation quality, physical robot control or independently
reproduced external state-of-the-art methods. Negative results remain valid
observations, not a reason to change the frozen test protocol.
