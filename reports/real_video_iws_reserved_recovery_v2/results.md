# Reserved IWS results — independently verified

Snapshot: 2026-09-20T13:31:36.939639+00:00. Finalizer completed 2026-09-20T13:29:29.650358+00:00.

All **36/36 evaluations** completed under one reviewed CPU FP32 command-GRU backend. The fixed population contains **600 original handles from 30 trajectories**: 200 handles and ten trajectories each for PushT, Box and Rope. Each learned arm retains its three original training seeds and preselected checkpoint. Persistence is an analytical control. These are locally withheld **upstream official-validation** trajectories, not a new independent-session test or an external-SOTA comparison.

The registered primary result is mixed across tasks: bounded ShiftWM versus additive anchoring improves the equal-task standardized-MSE macro by **1.66% [0.70, 2.62]**, but only PushT has a task-specific interval wholly above zero. Box is positive in point estimate with an interval spanning zero; Rope is slightly negative with an interval spanning zero. Bounded ShiftWM is worse than the autoregressive comparator on Box and Rope, with both MSE intervals wholly negative; its macro versus autoregression is **−3.26% [−5.47, −1.02]**.

The strongest result is the **secondary no-tanh component study**, declared after development and before reserved access. It improves standardized MSE against both bounded ShiftWM and autoregression on all three tasks, with all six unadjusted paired intervals wholly positive. Across all four error metrics, all **24 no-tanh task/comparator contrasts** (three tasks × two named references × four metrics) have positive point estimates and positive intervals. No-tanh has the lowest point mean among the five evaluated methods in all **12 task/metric cells**. These related comparisons are not 24 independent experiments, and the secondary finding does not replace the original primary hypothesis.

## H60 standardized-MSE reductions

Positive values favor the first method. Each cell is relative error reduction in percent followed by its paired 95% interval; these are not percentage-point success changes.

| Comparison | PushT | Box | Rope | Equal-task macro |
|---|---:|---:|---:|---:|
| Bounded ShiftWM / additive (primary) | +3.04% [+2.28, +3.75] | +2.31% [-0.09, +4.80] | -0.38% [-2.01, +1.08] | +1.66% [+0.70, +2.62] |
| Bounded ShiftWM / autoregressive | +0.50% [-3.00, +4.39] | -4.74% [-9.33, -0.22] | -5.53% [-7.80, -3.30] | -3.26% [-5.47, -1.02] |
| Bounded ShiftWM / persistence | +54.83% [+51.23, +57.35] | +52.97% [+49.99, +55.52] | +62.91% [+60.98, +64.64] | +56.91% [+55.24, +58.27] |
| No tanh / bounded ShiftWM (secondary) | +7.16% [+5.53, +8.60] | +9.14% [+7.06, +11.15] | +11.69% [+10.45, +13.01] | +9.33% [+8.35, +10.24] |
| No tanh / autoregressive (secondary) | +7.62% [+5.22, +10.09] | +4.84% [+1.40, +8.00] | +6.80% [+5.72, +7.88] | +6.42% [+4.90, +7.92] |

The 53–63% reductions against persistence are useful sanity checks, but persistence is a weak control. The closest learned-comparator effects above should lead the scientific discussion.

## All-metric direction audit

Each metric below includes the same five declared comparisons across three tasks (15 overlapping contrasts). Macro summaries are excluded from these counts. “Positive interval” means both bounds exceed zero; “crosses zero” includes touching zero. Every interval is unadjusted for multiple comparisons.

| Metric | Positive point | Negative point | Positive interval | Negative interval | Crosses zero |
|---|---:|---:|---:|---:|---:|
| Standardized MSE (primary metric) | 12/15 | 3/15 | 10/15 | 2/15 | 3/15 |
| Standardized MAE | 13/15 | 2/15 | 12/15 | 1/15 | 2/15 |
| Raw DINOv2 L1 | 13/15 | 2/15 | 12/15 | 1/15 | 2/15 |
| Feature cosine distance | 12/15 | 3/15 | 10/15 | 1/15 | 4/15 |

Bounded versus additive is favorable on all three tasks for standardized MAE and raw L1, with positive intervals, but MSE and cosine distance remain mixed. Bounded versus autoregression is not consistently favorable: Rope is worse on all four metrics, with intervals below zero. These adverse results must remain in tables and figures.

## Estimation and execution scope

H60 is the forecast at stored native offset 59 from one observed frame and 60 supplied command rows. H15/H30/H45 endpoints come from separate command-prefix calls. Errors average handles within each trajectory, ten trajectories equally, then three learned seeds equally. Macro relative reductions average the three task-specific reductions; they are not reductions computed from pooled task error. Paired percentile intervals use 10,000 draws with seed 173, shared seed draws across tasks and paired trajectory draws across methods and metrics. Their 95% coverage is nominal and unadjusted; secondary contrasts and metrics are not confirmatory family-wise tests.

The first execution attempt preserved 18 completed rows and two Box autoregressive prefix-check failures. Remaining jobs were stopped. Three diagnostic versions were retained; the final two matched the ordinary-parameter loading path and synthetic-target scorer path. Native and row-wise diagnostic outputs passed and matched exactly, so **the original failure was not reproduced and its root cause remains unresolved**. The numerical recovery was explicitly registered after initial reserved access. It reran all 36 models with the same one-native-command-row-per-GRU-call backend; no original result row was reused. Checkpoints, normalization, examples, metrics, comparisons and relative/absolute prefix tolerances remained fixed. Both previously failing runs then passed on the real scoring path.

No new training or fine-tuned checkpoint was created by this evaluation. The 36 existing selected packages remain unchanged and reusable. The new backend is an inference execution wrapper, not a learned parameter update. The original resource table measures the earlier full-sequence GRU implementation (CPU two threads / GPU); it is not a latency measurement for the new eight-thread row-wise accuracy backend.

These results concern DINOv2 feature forecasting. They establish neither RGB video quality nor closed-loop robot-control, clinical, simulator-promotion, or external-SOTA superiority. The original bounded method and the no-tanh secondary ablation must retain their identities in the main narrative.

## Verification and source bindings

- Frozen recovery registration: `configs/real_video_iws_reserved_recovery_v2/registration.json`, SHA256 `891207e1c88f758c45750a56a59bd5fa6b793391026f75085cb8263c1d2c21d4`.
- Complete finalization: `reports/real_video_iws_reserved_recovery_v2/finalization.json`, SHA256 `81e97ec2d1f225e35af84e83119f5f29781e8ecb0ca61489c759affa9abeaf0b`.
- Independent review: `reports/real_video_iws_reserved_recovery_v2/independent_result_review.json`, SHA256 `d3e090923bacb982643271c24a17a738468faa54b903868175460e8362251e72`.
- The independent reconstruction checked 525 bound source files, all 36 primitive NPZ ledgers, 174,240 per-trajectory curve values, 7,080 final full-curve means, 1,920 horizon/per-seed cells and all 280 confidence-interval bounds. It regenerated the bootstrap independently; maximum numerical reassociation difference was 9.88e-14. No new model inference or reserved future feature loading was used for this result audit.
