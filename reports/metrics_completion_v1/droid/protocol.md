# DROID complementary feature metrics: secondary completion protocol v1

This protocol adds measurements after the original DROID spatial and adapted
DINO-WM MSE findings were revealed. It does not change the selected checkpoints,
original endpoints, training, primary comparisons or data splits. The new metrics
and every new interval are post-hoc secondary exploratory analyses.

## Fixed population and models

Evaluate all 21 internal learned models (seven arms, seeds 0/1/2), both six-model
adapted DINO-WM campaigns (standardized-coordinate 30-epoch v1 and raw-coordinate
100-epoch v2), and deterministic persistence exactly once. All packages must have
passed their original full-budget training and complete-study finalizers before
this registration. The original validation population is 1,631 windows in 141
eligible episodes from 59 recording sessions, exterior camera 1 only. Use the
unchanged verified SpatialDataset with three observed grids, ten future targets
and stride five. Do not read original test/fresh-session/reserved data.

## Measurements

Raw prediction and target features have 6,144 coordinates (16 pooled DINOv2-small
patch locations x 384 channels). Use exactly the frozen IWS `feature_errors`
helper, with original DROID training-only per-coordinate standard deviations:

- Standardized MSE: mean squared coordinate difference divided by training variance.
- Standardized MAE: mean absolute coordinate difference divided by training std.
- Raw DINOv2 L1: mean absolute coordinate difference in original feature units.
- Raw cosine distance: one minus the clipped cosine of the flattened raw feature
  vectors; normalize each vector by max(norm, 1e-8), including zero vectors.

Report all h1–h10 endpoints. The original h10 standardized-MSE primary endpoint
remains unchanged. A separately named equal-horizon mean h1–h10 is descriptive,
not an endpoint or a new primary outcome. Feature scores do not measure RGB
quality, object trajectories, reward or control success.

Use the unchanged raw feature loader, checkpoint-specific model loader and
original evaluation batch size (128 internal/v1, 32 raw-v2). FP32 inference and
metric computation, autocast/TF32 disabled; save individual errors as FP64 and
aggregate windows within episode, then equally across episodes and learned seeds.
Persistence copies the last observed grid and has no training seed.

## Integrity and completion

Before any new scoring, write an immutable registration binding these sources,
each selected package, complete training records, original evaluation/finalization,
feature manifest and training normalization, and every validation cache payload.
The feature loader verifies cached payload hashes again. Reload the same selected
package and exactly reproduce the first complete inference batch. Save only scalar
per-window metrics, with NPZ round-trip equality checks; do not export raw features.

Recompute every original window/episode/summary MSE and require agreement with
its frozen finalized ledger at rtol=2e-5 and atol=2e-6, the same predeclared numeric
tolerance as the existing qualitative replay. Record maximum discrepancy. This
tolerance allows small numerical backend differences; it does not allow population,
checkpoint, normalization or model changes. Any missing row, failure, nonfinite
error, changed source or parity failure blocks complete-study aggregation. Keep
failure logs and failed models in the roster; do not substitute a favorable model.
No thresholds may be relaxed after observing a failure without a new disclosed
protocol version. Existing successful row outputs are verified, not overwritten.

## Paired uncertainty

Reuse the frozen DROID session x seed paired bootstrap unchanged, substituting
the four metric names in its private imported namespace. Resample three training
seeds with replacement and 59 whole sessions with replacement, preserving all
episodes in each selected session; use 10,000 draws, RNG seed 173 and percentile
95% intervals. Compare each learned arm and persistence with fixed AR and the
no-tanh ablation, omitting self-comparisons. Repeat identical persistence values
across matched seed slots solely for paired resampling, not as independent runs.
Intervals describe signed method-minus-reference absolute error, negative favorable.
Relative reductions are named-reference point effect sizes, not success percentage
points. The intervals are unadjusted and secondary across correlated metrics,
horizons and comparisons; no familywise confirmatory claim is permitted.

Bounded `transport` remains the primary ShiftWM identity. `unbounded_transport`
remains the no-tanh ablation, regardless of the new metric rankings. External
recipes and unequal training budgets remain explicit; weak adapted baselines do
not establish superiority over their published planning benchmarks.

## Execution

The root agent controls GPU submission and quota. After validation and registration:

```bash
.venv/bin/python scripts/metrics_completion_v1/droid_register.py verify
# First 21 internal models; final row is deterministic persistence.
sbatch --array=0-20,33%3 scripts/metrics_completion_v1/droid_run.slurm
# The two external recipes: six models each.
sbatch --array=21-32%3 scripts/metrics_completion_v1/droid_run.slurm
# Or one complete array, with concurrency set to the available allocation.
sbatch --array=0-33%3 scripts/metrics_completion_v1/droid_run.slurm
.venv/bin/python scripts/metrics_completion_v1/droid_finalize.py
```

Use one launch pattern, not both. The finalizer refuses partial campaigns with
exit code 75 and no aggregate. Its CPU Slurm wrapper can be submitted with an
`afterany` dependency on all evaluation arrays so failed conditions remain visible.
