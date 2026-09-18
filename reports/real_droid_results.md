# Real DROID forecasting results

Validated 12/12 full 30-epoch runs and 48/48 evaluations.

These are recorded-video feature forecasts on a prespecified DROID subset, not closed-loop robot success or the full published DROID benchmark. Checkpoint selection used validation only. Each test error first averages windows within an episode, then weights episodes equally. Learned rows show the three-seed mean ± sample SD; fixed support baselines have no training-seed SD. Paired intervals resample recording sessions and training seeds together; they are unadjusted for multiple comparisons.

## exterior_image_1_left, horizon 5 — primary

| Method | All-query standardized MSE | h1 MSE | h3 MSE | h5 MSE |
|---|---:|---:|---:|---:|
| Framewise | 0.0989419 ± 6.25e-05 | 0.0498783 ± 2.17e-05 | 0.103141 ± 3.95e-05 | 0.138479 ± 0.000152 |
| Constant dynamics | 0.0989157 ± 6.82e-05 | 0.049873 ± 2.25e-05 | 0.103117 ± 4.97e-05 | 0.138427 ± 0.000159 |
| ShiftWM real-video variant (ours) | 0.0987121 ± 5.38e-05 | 0.0497946 ± 4.18e-05 | 0.102875 ± 0.00011 | 0.138198 ± 0.000108 |
| Action-free | 0.0991536 ± 9.84e-05 | 0.0499299 ± 2.85e-05 | 0.103346 ± 0.000113 | 0.138861 ± 0.000132 |
| Persistence | 0.101112 | 0.0502939 | 0.105349 | 0.142392 |
| Constant feature velocity | 0.713096 | 0.120725 | 0.621502 | 1.48895 |

Explicit comparisons for the same matched population:

| Reference | Metric | Ours − reference MSE [95% paired CI] | Relative MSE reduction | Interval |
|---|---|---:|---:|---|
| Framewise | h1_standardized_mse | -8.37188e-05 [-0.000164887, -1.28066e-05] | +0.17% | interval below zero |
| Framewise | h3_standardized_mse | -0.000266108 [-0.000655145, +7.60043e-06] | +0.26% | interval includes zero |
| Framewise | h5_standardized_mse | -0.000280819 [-0.00103817, +0.000413019] | +0.20% | interval includes zero |
| Framewise | mean_standardized_mse | -0.000229775 [-0.000617632, +6.69177e-05] | +0.23% | interval includes zero |
| Constant dynamics | h1_standardized_mse | -7.84389e-05 [-0.000159576, -6.94718e-06] | +0.16% | interval below zero |
| Constant dynamics | h3_standardized_mse | -0.00024183 [-0.000631967, +2.32183e-05] | +0.23% | interval includes zero |
| Constant dynamics | h5_standardized_mse | -0.000228933 [-0.000987114, +0.000449778] | +0.17% | interval includes zero |
| Constant dynamics | mean_standardized_mse | -0.000203531 [-0.000593526, +8.6313e-05] | +0.21% | interval includes zero |
| Action-free | h1_standardized_mse | -0.000135368 [-0.000229481, -6.08991e-05] | +0.27% | interval below zero |
| Action-free | h3_standardized_mse | -0.000470862 [-0.000973223, -8.86721e-05] | +0.46% | interval below zero |
| Action-free | h5_standardized_mse | -0.000662487 [-0.00171893, +0.000320815] | +0.48% | interval includes zero |
| Action-free | mean_standardized_mse | -0.000441421 [-0.000956147, -1.88867e-05] | +0.45% | interval below zero |
| Persistence | h1_standardized_mse | -0.000499281 [-0.000660962, -0.000358462] | +0.99% | interval below zero |
| Persistence | h3_standardized_mse | -0.0024738 [-0.00357982, -0.00150201] | +2.35% | interval below zero |
| Persistence | h5_standardized_mse | -0.00419324 [-0.00671354, -0.00190827] | +2.94% | interval below zero |
| Persistence | mean_standardized_mse | -0.00239983 [-0.00356406, -0.00134855] | +2.37% | interval below zero |
| Constant feature velocity | h1_standardized_mse | -0.0709301 [-0.0795024, -0.0625131] | +58.75% | interval below zero |
| Constant feature velocity | h3_standardized_mse | -0.518627 [-0.581066, -0.457908] | +83.45% | interval below zero |
| Constant feature velocity | h5_standardized_mse | -1.35076 [-1.51483, -1.19249] | +90.72% | interval below zero |
| Constant feature velocity | mean_standardized_mse | -0.614384 [-0.688228, -0.542464] | +86.16% | interval below zero |

Negative MSE differences favor ours; positive relative reductions indicate lower error. The percentages are point-estimate ratios, not percentage points and not confidence intervals. Intervals that include zero are inconclusive. All signs are retained. Raw MSE, cosine error, per-seed values, and reversed-action diagnostics are preserved in the accompanying JSON.

## exterior_image_1_left, horizon 10 — horizon_extrapolation

| Method | All-query standardized MSE | h10 MSE |
|---|---:|---:|
| Framewise | 0.134787 ± 0.000281 | 0.189895 ± 0.000851 |
| Constant dynamics | 0.134745 ± 0.000295 | 0.189827 ± 0.000887 |
| ShiftWM real-video variant (ours) | 0.135562 ± 0.00111 | 0.193525 ± 0.00397 |
| Action-free | 0.134929 ± 0.000126 | 0.189516 ± 3.87e-05 |
| Persistence | 0.138468 | 0.193748 |
| Constant feature velocity | 2.07802 | 5.11407 |

Explicit comparisons for the same matched population:

| Reference | Metric | Ours − reference MSE [95% paired CI] | Relative MSE reduction | Interval |
|---|---|---:|---:|---|
| Framewise | h10_standardized_mse | +0.00363007 [-0.000386862, +0.00749276] | -1.91% | interval includes zero |
| Framewise | mean_standardized_mse | +0.000774449 [-0.000265891, +0.00207399] | -0.57% | interval includes zero |
| Constant dynamics | h10_standardized_mse | +0.00369838 [-0.000243542, +0.00751044] | -1.95% | interval includes zero |
| Constant dynamics | mean_standardized_mse | +0.00081682 [-0.000193075, +0.00209603] | -0.61% | interval includes zero |
| Action-free | h10_standardized_mse | +0.00400964 [-0.000312664, +0.00850229] | -2.12% | interval includes zero |
| Action-free | mean_standardized_mse | +0.00063328 [-0.000715372, +0.00222433] | -0.47% | interval includes zero |
| Persistence | h10_standardized_mse | -0.000222243 [-0.00768795, +0.00722305] | +0.11% | interval includes zero |
| Persistence | mean_standardized_mse | -0.00290657 [-0.006078, +0.000193317] | +2.10% | interval includes zero |
| Constant feature velocity | h10_standardized_mse | -4.92054 [-5.52568, -4.34672] | +96.22% | interval below zero |
| Constant feature velocity | mean_standardized_mse | -1.94246 [-2.17937, -1.71745] | +93.48% | interval below zero |

Negative MSE differences favor ours; positive relative reductions indicate lower error. The percentages are point-estimate ratios, not percentage points and not confidence intervals. Intervals that include zero are inconclusive. All signs are retained. Raw MSE, cosine error, per-seed values, and reversed-action diagnostics are preserved in the accompanying JSON.

## exterior_image_2_left, horizon 5 — camera_transfer

| Method | All-query standardized MSE | h1 MSE | h3 MSE | h5 MSE |
|---|---:|---:|---:|---:|
| Framewise | 0.10983 ± 3.19e-05 | 0.0549317 ± 8.55e-06 | 0.115341 ± 3.89e-05 | 0.154337 ± 8.86e-05 |
| Constant dynamics | 0.109808 ± 3.99e-05 | 0.0549277 ± 8.58e-06 | 0.115319 ± 4.88e-05 | 0.1543 ± 9.92e-05 |
| ShiftWM real-video variant (ours) | 0.109657 ± 0.000114 | 0.054832 ± 7.02e-05 | 0.115109 ± 0.000127 | 0.15419 ± 0.000315 |
| Action-free | 0.10991 ± 8.87e-05 | 0.0549576 ± 2e-05 | 0.115453 ± 0.000105 | 0.154367 ± 0.000155 |
| Persistence | 0.111581 | 0.0553071 | 0.117071 | 0.157372 |
| Constant feature velocity | 0.783453 | 0.131165 | 0.685322 | 1.6365 |

Explicit comparisons for the same matched population:

| Reference | Metric | Ours − reference MSE [95% paired CI] | Relative MSE reduction | Interval |
|---|---|---:|---:|---|
| Framewise | h1_standardized_mse | -9.97199e-05 [-0.000193781, -1.62237e-05] | +0.18% | interval below zero |
| Framewise | h3_standardized_mse | -0.00023263 [-0.0005861, +4.93225e-05] | +0.20% | interval includes zero |
| Framewise | h5_standardized_mse | -0.000146493 [-0.000830337, +0.000644295] | +0.09% | interval includes zero |
| Framewise | mean_standardized_mse | -0.000172893 [-0.000526316, +0.000152405] | +0.16% | interval includes zero |
| Constant dynamics | h1_standardized_mse | -9.56992e-05 [-0.000189875, -1.0948e-05] | +0.17% | interval below zero |
| Constant dynamics | h3_standardized_mse | -0.000210943 [-0.000569623, +6.83338e-05] | +0.18% | interval includes zero |
| Constant dynamics | h5_standardized_mse | -0.000110389 [-0.000793551, +0.000658934] | +0.07% | interval includes zero |
| Constant dynamics | mean_standardized_mse | -0.000150967 [-0.000502815, +0.00017098] | +0.14% | interval includes zero |
| Action-free | h1_standardized_mse | -0.000125591 [-0.000228685, -4.21637e-05] | +0.23% | interval below zero |
| Action-free | h3_standardized_mse | -0.000344586 [-0.000754696, +6.15514e-06] | +0.30% | interval includes zero |
| Action-free | h5_standardized_mse | -0.000177204 [-0.00104341, +0.000771374] | +0.11% | interval includes zero |
| Action-free | mean_standardized_mse | -0.000252302 [-0.000681194, +0.000160413] | +0.23% | interval includes zero |
| Persistence | h1_standardized_mse | -0.000475093 [-0.000686385, -0.000302841] | +0.86% | interval below zero |
| Persistence | h3_standardized_mse | -0.00196276 [-0.00298473, -0.00104694] | +1.68% | interval below zero |
| Persistence | h5_standardized_mse | -0.00318196 [-0.00539192, -0.00114177] | +2.02% | interval below zero |
| Persistence | mean_standardized_mse | -0.00192401 [-0.00300888, -0.000938915] | +1.72% | interval below zero |
| Constant feature velocity | h1_standardized_mse | -0.0763327 [-0.0856723, -0.0679898] | +58.20% | interval below zero |
| Constant feature velocity | h3_standardized_mse | -0.570214 [-0.637149, -0.509151] | +83.20% | interval below zero |
| Constant feature velocity | h5_standardized_mse | -1.48231 [-1.65631, -1.32316] | +90.58% | interval below zero |
| Constant feature velocity | mean_standardized_mse | -0.673795 [-0.752918, -0.601632] | +86.00% | interval below zero |

Negative MSE differences favor ours; positive relative reductions indicate lower error. The percentages are point-estimate ratios, not percentage points and not confidence intervals. Intervals that include zero are inconclusive. All signs are retained. Raw MSE, cosine error, per-seed values, and reversed-action diagnostics are preserved in the accompanying JSON.

## exterior_image_2_left, horizon 10 — camera_and_horizon_transfer

| Method | All-query standardized MSE | h10 MSE |
|---|---:|---:|
| Framewise | 0.151203 ± 0.000208 | 0.22002 ± 0.00066 |
| Constant dynamics | 0.151155 ± 0.000209 | 0.219942 ± 0.000662 |
| ShiftWM real-video variant (ours) | 0.151792 ± 0.000867 | 0.222748 ± 0.00309 |
| Action-free | 0.150936 ± 0.000215 | 0.218799 ± 0.000554 |
| Persistence | 0.153397 | 0.221364 |
| Constant feature velocity | 2.28596 | 5.63694 |

Explicit comparisons for the same matched population:

| Reference | Metric | Ours − reference MSE [95% paired CI] | Relative MSE reduction | Interval |
|---|---|---:|---:|---|
| Framewise | h10_standardized_mse | +0.0027271 [-0.000740872, +0.0061951] | -1.24% | interval includes zero |
| Framewise | mean_standardized_mse | +0.000589624 [-0.000389473, +0.00184009] | -0.39% | interval includes zero |
| Constant dynamics | h10_standardized_mse | +0.00280601 [-0.000608973, +0.00625078] | -1.28% | interval includes zero |
| Constant dynamics | mean_standardized_mse | +0.000636668 [-0.000325071, +0.00187021] | -0.42% | interval includes zero |
| Action-free | h10_standardized_mse | +0.00394848 [-5.52818e-05, +0.00790166] | -1.80% | interval includes zero |
| Action-free | mean_standardized_mse | +0.000856187 [-0.000435158, +0.00234675] | -0.57% | interval includes zero |
| Persistence | h10_standardized_mse | +0.00138385 [-0.00396614, +0.00693979] | -0.63% | interval includes zero |
| Persistence | mean_standardized_mse | -0.00160448 [-0.00407724, +0.000882734] | +1.05% | interval includes zero |
| Constant feature velocity | h10_standardized_mse | -5.41419 [-6.11157, -4.77908] | +96.05% | interval below zero |
| Constant feature velocity | mean_standardized_mse | -2.13417 [-2.40847, -1.88446] | +93.36% | interval below zero |

Negative MSE differences favor ours; positive relative reductions indicate lower error. The percentages are point-estimate ratios, not percentage points and not confidence intervals. Intervals that include zero are inconclusive. All signs are retained. Raw MSE, cosine error, per-seed values, and reversed-action diagnostics are preserved in the accompanying JSON.

## Scope and reuse

No test result was used to choose a model or training epoch. Context conditioning on observed support is not evidence of identified physical factors. Reversing future commands is an observational diagnostic; the recordings do not reveal the outcome of unexecuted actions. Camera transfer and ten-block extrapolation remain separate from the primary camera-one/five-block population.

The local release contains all twelve validation-selected predictors, a shared frozen DINOv2-small encoder, pinned source, statistics, protocol, model cards, and offline CPU verification. It contains feature predictors, not generated RGB-video models. No files were uploaded to GitHub, Hugging Face, or another service.
