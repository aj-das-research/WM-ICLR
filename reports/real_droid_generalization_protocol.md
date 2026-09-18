# Real-DROID generalization development campaign

Registered before the new training runs and before observing the separate fresh
holdout outcomes. This is an original-training/original-validation study. Neither
the original test payloads nor the newly acquired holdout may be read by this
campaign. Its results cannot alter the already frozen fresh-holdout comparison.

## Question and fixed interventions

The completed campaign selected epochs 1–2 and deteriorated later on validation.
We test three specific, matched changes to determine whether optimization rate,
weight decay or predictor capacity improves generalization:

| Arm | Predictor and context | Learning rate | Weight decay |
|---|---|---:|---:|
| slow | Original width 192, depth 4, context 32/128 | 0.00003 | 0.01 |
| decay | Original width 192, depth 4, context 32/128 | 0.0001 | 0.1 |
| compact | Width 96, depth 2, context 16/64 | 0.0001 | 0.01 |

The minimum cosine-scheduled learning rate is one hundredth of the initial rate.
The compact arm changes the capacity family as a group; it cannot identify the
separate effect of depth, width or context size. No new method novelty is claimed
for these ordinary architectural/optimization controls.

Every arm includes Framewise, Constant dynamics, ShiftWM (ours) and Action-free
at seeds 0, 1 and 2: **36 complete 30-epoch runs**. All modes within an arm receive
the same optimizer, data, initialization seed and validation opportunity. There
is no test-driven early termination or winner-only reporting. The original
baseline campaign remains intact and separately identified.

## Unchanged data and evaluation

Reuse the audited original training/validation camera-one DINO feature cache,
training-only normalization, three-frame support, five-block training targets,
training stride 2, validation stride 5, batch 128, gradient clipping 1, BF16
training and FP32 validation with TF32 disabled. Reuse the existing training
implementation without editing it. It selects the best checkpoint by the common
all-five-query validation MSE; all runs still execute 30 epochs.

After each run, evaluate its selected checkpoint at five and ten blocks on
original validation only, using the unchanged causal evaluator and equal episode
weighting. Preserve per-episode errors, persistence and constant-velocity
controls, and action-reversal diagnostics. Report every arm and method, three-seed
means, paired session/seed bootstrap intervals versus its matched Framewise,
selected epochs, parameter counts and actual runtime. Development intervals are
exploratory and unadjusted for multiple comparisons. Five-block error is the
primary development comparison; ten-block behavior is a separate stability check.

## Registration and execution

`scripts/real_video_development/generalization_campaign.py --register` writes
all configs and a registry pinning this protocol, the unchanged scientific
dependencies, feature manifest, normalization, and exact new runner/configs.
Registration refuses to overwrite an existing registry. Each job verifies these
hashes before and after training/evaluation. Nine one-GPU job groups each run the
four modes for one arm/seed, reusing a single allocation. Completed packages use
the original strict manifest and offline loader; optimizer/RNG state is retained.

No gain is promised by registration. If a revision improves validation, it still
requires its own frozen evaluation on an untouched population for a subsequent
confirmatory claim. The concurrently running fresh-holdout study concerns the
previously frozen base/calibrated checkpoints and is not a selection set here.
