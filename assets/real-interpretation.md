# Independent interpretation of the completed real DROID study

Audited 12 complete 30-epoch training histories and 48 completed evaluation files. The source is `reports/real_droid_results.json` (SHA256 `0a80aa3e1a190e14366f0c4d50d72cf162377374558eb1cf5c2848a7fe212b0e`). All raw evaluation hashes, episode-level means, matched episode/window identities, reported paired point differences, and the primary 10,000-draw crossed session/seed confidence interval were independently checked.

## What the results support

On the prespecified real DROID subset, all 12 runs completed 30 epochs, with checkpoints selected solely by validation error. At five action blocks on the primary camera (132 episodes from 58 recording sessions), ShiftWM achieved standardized feature MSE 0.138198, compared with 0.138479 for Framewise and 0.142392 for persistence. The 0.20% reduction relative to Framewise was inconclusive (paired 95% CI for the MSE difference: [-0.001038, 0.000413]), whereas the 2.94% reduction relative to persistence had an interval below zero ([-0.006714, -0.001908]). Camera transfer retained a 2.02% reduction relative to persistence, but ten-block extrapolation was 1.91% and 1.24% worse than Framewise on the two cameras, respectively, with both intervals crossing zero. All selected checkpoints occurred at epochs 1–2, and subsequent validation deterioration despite lower training loss indicates overfitting. These results establish a modest real-video forecasting benefit over persistence; they do not establish a substantial advantage over matched learned baselines or real-robot control performance. All intervals are unadjusted for multiple comparisons.

## Final-horizon comparisons

All numbers below are train-standardized frozen DINO feature MSE, averaged within episodes and then equally across episodes and three training seeds. Lower is better. Five-block and ten-block populations differ because some short recordings do not support the longer horizon. The two cameras use the same eligible held-out recordings and are not independent datasets.

| Population | Episodes / sessions / windows | Framewise | Constant dynamics | ShiftWM (ours) | Action-free | Persistence | Constant velocity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Camera 1, h5 | 132 / 58 / 1738 | 0.138479 | 0.138427 | 0.138198 | 0.138861 | 0.142392 | 1.488954 |
| Camera 1, h10 | 130 / 57 / 1606 | 0.189895 | 0.189827 | 0.193525 | 0.189516 | 0.193748 | 5.114068 |
| Camera 2, h5 | 132 / 58 / 1738 | 0.154337 | 0.154300 | 0.154190 | 0.154367 | 0.157372 | 1.636496 |
| Camera 2, h10 | 130 / 57 / 1606 | 0.220020 | 0.219942 | 0.222748 | 0.218799 | 0.221364 | 5.636942 |

| Population | Comparator | Relative MSE reduction (ours) | Ours minus comparator [95% paired CI] | Interpretation |
|---|---|---:|---:|---|
| Camera 1, h5 | Framewise | +0.203% | -0.000281 [-0.001038, +0.000413] | interval includes zero |
| Camera 1, h5 | Constant dynamics | +0.165% | -0.000229 [-0.000987, +0.000450] | interval includes zero |
| Camera 1, h5 | Action-free | +0.477% | -0.000662 [-0.001719, +0.000321] | interval includes zero |
| Camera 1, h5 | Persistence | +2.945% | -0.004193 [-0.006714, -0.001908] | interval below zero |
| Camera 1, h5 | Constant feature velocity | +90.718% | -1.350755 [-1.514835, -1.192494] | interval below zero |
| Camera 1, h10 | Framewise | -1.912% | +0.003630 [-0.000387, +0.007493] | interval includes zero |
| Camera 1, h10 | Constant dynamics | -1.948% | +0.003698 [-0.000244, +0.007510] | interval includes zero |
| Camera 1, h10 | Action-free | -2.116% | +0.004010 [-0.000313, +0.008502] | interval includes zero |
| Camera 1, h10 | Persistence | +0.115% | -0.000222 [-0.007688, +0.007223] | interval includes zero |
| Camera 1, h10 | Constant feature velocity | +96.216% | -4.920543 [-5.525683, -4.346723] | interval below zero |
| Camera 2, h5 | Framewise | +0.095% | -0.000146 [-0.000830, +0.000644] | interval includes zero |
| Camera 2, h5 | Constant dynamics | +0.072% | -0.000110 [-0.000794, +0.000659] | interval includes zero |
| Camera 2, h5 | Action-free | +0.115% | -0.000177 [-0.001043, +0.000771] | interval includes zero |
| Camera 2, h5 | Persistence | +2.022% | -0.003182 [-0.005392, -0.001142] | interval below zero |
| Camera 2, h5 | Constant feature velocity | +90.578% | -1.482306 [-1.656313, -1.323164] | interval below zero |
| Camera 2, h10 | Framewise | -1.239% | +0.002727 [-0.000741, +0.006195] | interval includes zero |
| Camera 2, h10 | Constant dynamics | -1.276% | +0.002806 [-0.000609, +0.006251] | interval includes zero |
| Camera 2, h10 | Action-free | -1.805% | +0.003948 [-0.000055, +0.007902] | interval includes zero |
| Camera 2, h10 | Persistence | -0.625% | +0.001384 [-0.003966, +0.006940] | interval includes zero |
| Camera 2, h10 | Constant feature velocity | +96.048% | -5.414195 [-6.111574, -4.779076] | interval below zero |

Across these 20 correlated final-horizon comparisons, 13 point estimates favor ours and 7 favor the comparator. Six intervals exclude zero in favor of ours: persistence on both cameras at h5, and constant velocity in all four populations. The other 14 include zero. None of the 12 final-horizon comparisons against the three learned baselines has an interval excluding zero. These counts are a descriptive inventory, not 20 independent experiments or multiplicity-adjusted discoveries.

The large 90.58–96.22% improvements over constant feature velocity reflect a weak extrapolation baseline whose error grows rapidly. They should not be used to suggest a similarly large gain over established learned world models. The primary h1 comparison against Framewise has a small 0.168% improvement with an unadjusted interval below zero; h3 and h5 intervals include zero. The primary all-query comparison with Action-free improves by 0.445% with an unadjusted interval below zero; its final-horizon interval includes zero. None justifies choosing a favorable horizon after seeing results.

## Training and validation behavior

| Run | Selected epoch | Best validation MSE | Epoch-30 train MSE | Epoch-30 validation MSE | Validation deterioration from best |
|---|---:|---:|---:|---:|---:|
| droid_action_free_s0 | 1 | 0.111610 | 0.099468 | 0.120769 | +8.21% |
| droid_action_free_s1 | 1 | 0.111588 | 0.099272 | 0.120478 | +7.97% |
| droid_action_free_s2 | 1 | 0.111637 | 0.099306 | 0.121295 | +8.65% |
| droid_constant_dynamics_s0 | 1 | 0.111167 | 0.088891 | 0.129910 | +16.86% |
| droid_constant_dynamics_s1 | 1 | 0.111378 | 0.088083 | 0.130889 | +17.52% |
| droid_constant_dynamics_s2 | 1 | 0.111238 | 0.088147 | 0.133339 | +19.87% |
| droid_factorized_s0 | 2 | 0.111104 | 0.089038 | 0.129591 | +16.64% |
| droid_factorized_s1 | 2 | 0.111159 | 0.088334 | 0.130890 | +17.75% |
| droid_factorized_s2 | 1 | 0.111139 | 0.088480 | 0.132744 | +19.44% |
| droid_framewise_s0 | 1 | 0.111211 | 0.089092 | 0.131350 | +18.11% |
| droid_framewise_s1 | 1 | 0.111414 | 0.088371 | 0.130924 | +17.51% |
| droid_framewise_s2 | 1 | 0.111286 | 0.088499 | 0.135975 | +22.18% |

Ten runs select epoch 1; the seed-0 and seed-1 ShiftWM runs select epoch 2. Training error falls 15.62–24.94% from epoch 1 to epoch 30, while validation error finishes 7.97–22.18% above its minimum. This is direct evidence of late overfitting in this configuration, not an incomplete-training failure. The registered validation-only selection rule prevents using those degraded last checkpoints for the reported test comparison. The experiment does not establish whether additional training diversity, a lower-capacity predictor, or stronger regularization would resolve the problem; those remain hypotheses to study on training/validation data with a newly locked evaluation protocol.

## Recorded-action diagnostic

This diagnostic reverses the order of future 35-dimensional action blocks while keeping their within-block chronological commands and observed support unchanged. Positive values below mean reversal increased error; negative values mean reversal reduced error. It observes errors on the originally recorded future, not ground truth for a different executed robot action.

| Population | Method | Original MSE | Reversed MSE | Relative error change |
|---|---|---:|---:|---:|
| Camera 1, h5 | Framewise | 0.13847923 | 0.13854705 | +0.0490% |
| Camera 1, h5 | Constant dynamics | 0.13842735 | 0.13849613 | +0.0497% |
| Camera 1, h5 | ShiftWM real-video variant (ours) | 0.13819841 | 0.13833309 | +0.0975% |
| Camera 1, h5 | Action-free | 0.13886090 | 0.13886090 | +0.0000% |
| Camera 1, h10 | Framewise | 0.18989538 | 0.19027020 | +0.1974% |
| Camera 1, h10 | Constant dynamics | 0.18982708 | 0.19021066 | +0.2021% |
| Camera 1, h10 | ShiftWM real-video variant (ours) | 0.19352546 | 0.19341815 | -0.0554% |
| Camera 1, h10 | Action-free | 0.18951582 | 0.18951582 | +0.0000% |
| Camera 2, h5 | Framewise | 0.15433657 | 0.15438850 | +0.0336% |
| Camera 2, h5 | Constant dynamics | 0.15430046 | 0.15435351 | +0.0344% |
| Camera 2, h5 | ShiftWM real-video variant (ours) | 0.15419007 | 0.15429107 | +0.0655% |
| Camera 2, h5 | Action-free | 0.15436728 | 0.15436728 | +0.0000% |
| Camera 2, h10 | Framewise | 0.22002043 | 0.22035179 | +0.1506% |
| Camera 2, h10 | Constant dynamics | 0.21994152 | 0.22028048 | +0.1541% |
| Camera 2, h10 | ShiftWM real-video variant (ours) | 0.22274753 | 0.22284163 | +0.0422% |
| Camera 2, h10 | Action-free | 0.21879905 | 0.21879905 | +0.0000% |

For ours, each final-horizon error changes by less than 0.10% under this reversal. This is limited evidence of sensitivity to the temporal ordering of the recorded commands. Smooth or nearly constant commands can make reversal a weak perturbation, so this alone does not prove the predictor ignores actions. Action-free has exactly zero change, as required by its implementation, and is competitive with the action-conditioned variants, especially at h10. In combination, these diagnostics do not support a claim of strong action-dependent prediction or reliable counterfactual planning.

## Scope and next decisions

- The footage and recorded commands are real. The study uses a fixed subset of DROID, not the complete published DROID policy benchmark. Session splits are disjoint; scene/object disjointness across recording dates is not established.
- The released checkpoints predict frozen image features. They do not generate RGB video, and the experiment contains no physical robot deployment or clinical validation.
- Keep small positive, null, and negative outcomes in the paper. Any selected qualitative case must identify the actual episode and selection rule; it is not evidence of average superiority.
- Preserve this completed campaign. Further model changes should be developed using training/validation evidence; repeated tuning against this revealed test set would invalidate its role as a confirmatory holdout.
- The method-level contribution remains to be established against closely related algorithms under matched resources. A new real dataset or frozen DINO backbone is not by itself a novel algorithm.

## Paper-ready paragraph

On the prespecified real DROID subset, all 12 runs completed 30 epochs, with checkpoints selected solely by validation error. At five action blocks on the primary camera (132 episodes from 58 recording sessions), ShiftWM achieved standardized feature MSE 0.138198, compared with 0.138479 for Framewise and 0.142392 for persistence. The 0.20% reduction relative to Framewise was inconclusive (paired 95% CI for the MSE difference: [-0.001038, 0.000413]), whereas the 2.94% reduction relative to persistence had an interval below zero ([-0.006714, -0.001908]). Camera transfer retained a 2.02% reduction relative to persistence, but ten-block extrapolation was 1.91% and 1.24% worse than Framewise on the two cameras, respectively, with both intervals crossing zero. All selected checkpoints occurred at epochs 1–2, and subsequent validation deterioration despite lower training loss indicates overfitting. These results establish a modest real-video forecasting benefit over persistence; they do not establish a substantial advantage over matched learned baselines or real-robot control performance. All intervals are unadjusted for multiple comparisons.
