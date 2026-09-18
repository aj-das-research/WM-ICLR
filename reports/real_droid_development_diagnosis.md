# Real DROID development diagnosis

This report uses only training and validation recordings. No test feature/video payload was read by the diagnostic or this summary. The original completed campaign remains unchanged.

The validation population contains 141 episodes from 59 sessions and 1631 windows. All horizons below use the same ten-block-eligible recordings. Windows are averaged within episodes, then episodes and the three training seeds are weighted equally. Results are exploratory development evidence, without confirmatory intervals.

## Recursive forecasting and observed motion

| Model | h1 error | h5 error | h10 error | h10 predicted displacement | h10 observed displacement | h10 oracle-refresh error |
|---|---:|---:|---:|---:|---:|---:|
| framewise, best | 0.055117 | 0.156712 | 0.216903 | 0.034187 | 0.225622 | 0.061744 |
| constant_dynamics, best | 0.055111 | 0.156638 | 0.216746 | 0.034157 | 0.225622 | 0.061737 |
| factorized, best | 0.055003 | 0.156234 | 0.219507 | 0.051390 | 0.225622 | 0.061602 |
| action_free, best | 0.055189 | 0.157491 | 0.217435 | 0.027854 | 0.225622 | 0.061826 |
| factorized, last | 0.057116 | 0.188089 | 0.279258 | 0.168326 | 0.225622 | 0.063661 |
| Persistence | 0.055625 | 0.163156 | 0.225622 | 0 | 0.225622 | not applicable |

Errors and displacement magnitudes are standardized feature mean-square values, not physical distances. Oracle refresh supplies the actual three immediately preceding observations for each one-step prediction; it changes the available information and refreshes inferred contexts. Its lower error cannot be claimed as a deployable algorithm improvement or attributed solely to one source of drift.

## Stronger future-command sensitivity

Positive error change means the perturbation made prediction worse against the original recorded future. Prediction change measures output sensitivity separately from this error. None of these perturbed commands was executed on hardware.

| Model / checkpoint | Future-command perturbation | Standardized command change | h10 prediction change MSE | h5 error change | h10 error change |
|---|---|---:|---:|---:|---:|
| framewise, best | reversed | 0.47549 | 0.0000763 | -0.025% | +0.257% |
| framewise, best | permuted_recording | 1.67256 | 0.0021105 | +0.538% | +0.991% |
| framewise, best | raw_zero | 4.41265 | 0.0029406 | +0.687% | +0.842% |
| framewise, best | train_mean | 0.95703 | 0.0013204 | +0.398% | +0.605% |
| framewise, best | hold_support_command | 0.51068 | 0.0007553 | +0.358% | +0.981% |
| constant_dynamics, best | reversed | 0.47549 | 0.0000785 | -0.031% | +0.263% |
| constant_dynamics, best | permuted_recording | 1.67256 | 0.0021186 | +0.550% | +0.989% |
| constant_dynamics, best | raw_zero | 4.41265 | 0.0028948 | +0.715% | +0.918% |
| constant_dynamics, best | train_mean | 0.95703 | 0.0013224 | +0.413% | +0.632% |
| constant_dynamics, best | hold_support_command | 0.51068 | 0.0007644 | +0.369% | +1.006% |
| factorized, best | reversed | 0.47549 | 0.0009040 | +0.476% | +0.196% |
| factorized, best | permuted_recording | 1.67256 | 0.0146963 | +4.286% | +5.173% |
| factorized, best | raw_zero | 4.41265 | 0.0147904 | +3.862% | +5.415% |
| factorized, best | train_mean | 0.95703 | 0.0075870 | +2.529% | +2.921% |
| factorized, best | hold_support_command | 0.51068 | 0.0052802 | +0.819% | +1.413% |
| action_free, best | reversed | 0.47549 | 0.0000000 | +0.000% | +0.000% |
| action_free, best | permuted_recording | 1.67256 | 0.0000000 | +0.000% | +0.000% |
| action_free, best | raw_zero | 4.41265 | 0.0000000 | +0.000% | +0.000% |
| action_free, best | train_mean | 0.95703 | 0.0000000 | +0.000% | +0.000% |
| action_free, best | hold_support_command | 0.51068 | 0.0000000 | +0.000% | +0.000% |
| factorized, last | reversed | 0.47549 | 0.0114446 | +3.023% | +1.836% |
| factorized, last | permuted_recording | 1.67256 | 0.0718732 | +9.888% | +9.706% |
| factorized, last | raw_zero | 4.41265 | 0.1476583 | +25.939% | +35.543% |
| factorized, last | train_mean | 0.95703 | 0.0493710 | +9.753% | +6.969% |
| factorized, last | hold_support_command | 0.51068 | 0.0353441 | +2.992% | +3.527% |

Raw-zero and training-mean values may be out of context for absolute robot position commands. Donor commands come from different recording sessions. These stress tests can expose dependence on actions but cannot assess physical counterfactual accuracy.

## Where errors occur

Tertile boundaries are fixed from all training windows. Future-motion strata are descriptive uses of targets, not inputs to an online decision rule.

| Stratum | Eligible validation windows | Framewise h10 | Ours h10 | Persistence h10 | Ours h5 gain vs persistence |
|---|---:|---:|---:|---:|---:|
| action_variation_tertile_1 | 627 | 0.215826 | 0.218641 | 0.219870 | +2.07% |
| action_variation_tertile_2 | 523 | 0.209592 | 0.212594 | 0.212992 | +3.09% |
| action_variation_tertile_3 | 481 | 0.230898 | 0.232517 | 0.245108 | +5.40% |
| all | 1631 | 0.216903 | 0.219507 | 0.225622 | +4.24% |
| future_motion_tertile_1 | 581 | 0.117651 | 0.123713 | 0.106219 | -10.10% |
| future_motion_tertile_2 | 514 | 0.184155 | 0.189051 | 0.184358 | +0.85% |
| future_motion_tertile_3 | 536 | 0.323546 | 0.321605 | 0.356647 | +8.53% |
| support_motion_tertile_1 | 563 | 0.204925 | 0.206353 | 0.220453 | +4.67% |
| support_motion_tertile_2 | 536 | 0.226055 | 0.228200 | 0.233860 | +3.56% |
| support_motion_tertile_3 | 532 | 0.242890 | 0.245944 | 0.247915 | +3.76% |

## Training-versus-validation overfitting

| Seed-0 factorized checkpoint | Train-subset h5 | Train-subset h10 | Validation h5 | Validation h10 | Train-fitted residual scale | Validation h10 with this fixed scale |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.146494 | 0.194646 | 0.156231 | 0.220366 | 0.80933 | 0.212483 |
| last | 0.115563 | 0.155465 | 0.186416 | 0.277711 | 0.91686 | 0.260595 |

The 64 training episodes were selected deterministically without inspecting losses. The residual scale is the clipped least-squares coefficient fitted to train-subset predicted/observed displacement moments, pooled over all ten horizons. This is a numerical diagnosis, not a separately trained architecture or a reported test improvement.

## Pooling and spatial information

On 48 deterministic training frame pairs, median retained raw DINO feature-change energy was 100.00% at 16×16, 14.42% at 2×2, 26.20% at 4×4, 48.71% at 8×8.

This establishes strong contraction of spatial variation under pooling if retention is low. It does not establish that manipulated-object information is the particular lost signal, or that restoring patch resolution will improve validation forecasting. No object masks were used, and the current experiment did not train a competing patch model.

## Rules for the next controlled experiment

1. Preserve all original positive, null, and negative results and released checkpoints.
2. Choose a narrowly defined fix using this train/validation diagnosis; freeze its protocol and source before running new test evaluation.
3. Tune only on training/validation data. Maintain matched architectures, budgets, and validation selection for the corresponding baseline.
4. Obtain an untouched evaluation population from new recording sessions absent from the original 433 sessions. Exclude overlap before viewing frames or results; record deterministic inclusion and all exclusions.
5. Call any reuse of the already revealed original test set exploratory, not confirmatory. Do not claim SOTA or physical robot success from latent forecasting alone.
