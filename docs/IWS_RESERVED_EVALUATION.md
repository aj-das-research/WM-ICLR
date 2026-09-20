# Fixed-checkpoint IWS reserved evaluation

This study evaluates all 36 existing development-selected predictors on the
original reserved upstream-validation handles: 200 per task, ten trajectories
per task, for recorded PushT, Box and Rope. It does not train or select models
on those outcomes. The no-tanh component follow-up remains secondary to the
original bounded-mixing versus additive-anchor comparison.

The frozen registration is
`configs/real_video_iws_reserved_v1/registration.json`. The independent review,
job submission chain and completed evaluation ledgers are stored in
`reports/real_video_iws_reserved_v1/`. The first registered chain is cache job
201873, CPU array 201874 and finalizer 201875. Submission is not completion;
only a passing `finalization.json` represents the complete numerical study.

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
.venv/bin/python -I scripts/real_video_iws_reserved_v1/prepare_cache.py metadata --task all
.venv/bin/python -I scripts/real_video_iws_reserved_v1/register.py check
sbatch scripts/real_video_iws_reserved_v1/cache.slurm
# Submit the score array after the cache job succeeds, then the finalizer
# after the entire score array succeeds. See submission.json for the actual
# dependency arguments. Do not resubmit a previously completed evaluation.
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
