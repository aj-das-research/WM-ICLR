# Full matched real-DROID residual-calibration development campaign

All 12 frozen best checkpoints were calibrated using all eligible training recordings, with one scalar fitted independently per checkpoint. No gradients, architecture changes, or test payloads were used. Every method received the same calibration opportunity; all results below are validation development evidence.

## All methods, three seeds

| Method | Horizon | Original error | Calibrated error | Reduction from its own original model |
|---|---:|---:|---:|---:|
| framewise | 5 | 0.159071 | 0.158942 | +0.081% |
| constant_dynamics | 5 | 0.158998 | 0.158880 | +0.074% |
| factorized | 5 | 0.158398 | 0.157704 | +0.438% |
| action_free | 5 | 0.159766 | 0.159718 | +0.030% |
| framewise | 10 | 0.216903 | 0.216206 | +0.321% |
| constant_dynamics | 10 | 0.216746 | 0.216099 | +0.298% |
| factorized | 10 | 0.219507 | 0.215874 | +1.655% |
| action_free | 10 | 0.217435 | 0.217144 | +0.134% |

Errors are train-standardized frozen-feature MSE, averaging windows within episodes, then equally across episodes and the three seeds. Positive reductions favor calibration. All gains and losses are shown.

## Fairly calibrated baselines

| Horizon | Comparator (also calibrated) | Ours relative error reduction | Ours minus comparator [paired 95% interval] |
|---:|---|---:|---:|
| 5 | framewise | +0.779% | -0.001238 [-0.002173, -0.000470] |
| 5 | constant_dynamics | +0.740% | -0.001175 [-0.002119, -0.000397] |
| 5 | action_free | +1.261% | -0.002014 [-0.003201, -0.001017] |
| 10 | framewise | +0.154% | -0.000332 [-0.001395, +0.001022] |
| 10 | constant_dynamics | +0.104% | -0.000225 [-0.001279, +0.001128] |
| 10 | action_free | +0.585% | -0.001269 [-0.002895, +0.000453] |

Intervals resample recording sessions and training seeds (10,000 draws), preserving matched episode differences. They are exploratory validation intervals, unadjusted for multiple comparisons, and do not turn validation into a new test set.

## Fitted scalar and complete per-seed outcomes

| Method | Seed | Alpha | Fitting episodes / windows | Original h5 | Calibrated h5 | Original h10 | Calibrated h10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| action_free | 0 | 1.000000 | 830 / 7721 | 0.159742 | 0.159742 | 0.217185 | 0.217185 |
| action_free | 1 | 0.963672 | 830 / 7721 | 0.159731 | 0.159614 | 0.217451 | 0.216737 |
| action_free | 2 | 0.992293 | 830 / 7721 | 0.159824 | 0.159796 | 0.217668 | 0.217509 |
| constant_dynamics | 0 | 1.000000 | 830 / 7721 | 0.158738 | 0.158738 | 0.215883 | 0.215883 |
| constant_dynamics | 1 | 0.930623 | 830 / 7721 | 0.159189 | 0.158854 | 0.217513 | 0.215687 |
| constant_dynamics | 2 | 0.995486 | 830 / 7721 | 0.159066 | 0.159046 | 0.216841 | 0.216728 |
| factorized | 0 | 0.907685 | 830 / 7721 | 0.158245 | 0.157389 | 0.220366 | 0.216026 |
| factorized | 1 | 0.875878 | 830 / 7721 | 0.158229 | 0.157056 | 0.221926 | 0.215705 |
| factorized | 2 | 0.986601 | 830 / 7721 | 0.158721 | 0.158667 | 0.216230 | 0.215891 |
| framewise | 0 | 1.000000 | 830 / 7721 | 0.158814 | 0.158814 | 0.216058 | 0.216058 |
| framewise | 1 | 0.928025 | 830 / 7721 | 0.159245 | 0.158895 | 0.217623 | 0.215727 |
| framewise | 2 | 0.992262 | 830 / 7721 | 0.159152 | 0.159118 | 0.217029 | 0.216834 |

All 851 train and 143 validation episode payloads were hash-verified against the immutable feature manifest at registration and before/after execution; short recordings remain in the audited manifests but cannot provide every horizon's windows.

- Validation h5: 141 episodes, 59 sessions, 1772 windows.
- Validation h10: 141 episodes, 59 sessions, 1631 windows.

## Reuse and interpretation

Each run's `calibration.json` records the fitted scalar, its training sufficient statistics, training payload hashes, source/protocol hashes, and immutable base checkpoint SHA256. `load_calibrated_package(path, device='cpu', base_checkpoint=relocated_base_directory)` supports relocation without changing the base weights. All 12 packages passed exact offline CPU prediction parity on four ten-step validation windows. The wrappers reuse trained checkpoints; no newly trained neural weights are claimed.

The scalar is a standard constrained least-squares correction around persistence, not a new architectural contribution. Any positive outcome should motivate a carefully controlled causal adaptation study, not a claim of novelty, SOTA, or physical robot control. Preserve the original completed campaign. Freeze the final method and a fresh recording-session-disjoint data manifest before confirmatory testing; the already inspected original test set is not eligible for new confirmatory claims.
