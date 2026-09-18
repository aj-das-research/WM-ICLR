# Fresh real-DROID confirmatory evaluation, version 1

This protocol is frozen with code, checkpoints and train-fitted scalar packages
before decoding any of the 65 retained episodes in 52 fresh recording sessions.
It follows the independently registered acquisition protocol in
`reports/real_droid_fresh_holdout_protocol.md`. No original test-set result is
used to select a checkpoint, scalar, fresh recording or analysis setting here.
The development motivation and method choice are the completed original
training-only scalar fit and validation-only evaluation. Calibration is a known
control for excess residual magnitude, not a new architecture or novelty claim.

## Single primary claim

Primary population: exterior camera 1, horizon five action blocks. Primary
endpoint: **h5_standardized_mse**, the error at the fifth predicted frame,
standardized using the unchanged original training feature standard deviations.
The single primary comparison is calibrated ShiftWM (ours) versus equally
calibrated Framewise. A negative paired MSE difference favors ours; a 95% upper
confidence bound below zero supports improvement on this registered endpoint.
The percentage reduction is 100 times (comparator mean minus ours mean) divided
by comparator mean. An interval crossing zero is reported as inconclusive.
The primary endpoint is not the average error over all five query frames.

## All methods and unchanged scientific inputs

Evaluate all four modes (Framewise, constant-dynamics, factorized ShiftWM and
action-free), all three original training seeds, and both original and calibrated
variants: 24 model conditions. Use every existing validation-selected best
checkpoint from its completed 30-epoch run. No new optimization, late checkpoint
selection, extra context fitting or inference search is permitted.

For each of 12 checkpoints, the scalar comes unchanged from
`configs/real_video_development/calibrations/droid_<mode>_s<seed>.json`.
It was fitted using original training episodes only, camera 1, horizon ten,
stride five, with equal episode weighting and clipped least-squares residual
scale in [0,1]. Calibrated predictions equal final support observation plus
the fixed scalar times the base trajectory's displacement. The base autoregressive
trajectory is computed unchanged; contracted outputs are not fed back into it.
No fresh frame, outcome or error estimates the scalar. The same fitting procedure
applies to all methods, including baselines with a fitted scale of one.

Use original DINOv2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`,
weight SHA256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
Full RGB images resize to 224x224 with bilinear antialiasing and original ImageNet
mean/std. BF16 encoder outputs use float32 2x2 patch-grid mean pooling, giving
1536 features. Feature batch size is 32. Copy the original training normalization
JSON byte-for-byte; never refit moments on fresh recordings. Prediction is FP32,
autocast off and TF32 off, batch size 128, unchanged model architecture.

## Fresh episode population and temporal contract

Retain all 65 episodes in the frozen metadata manifest. They were selected only
through a deterministic 12-shard draw followed by exclusion of every original
site/date recording session. The 491 overlapping episodes were excluded before
decoding or evaluating any image. No additional success, motion, appearance,
length, clinical relevance or error-based filtering is allowed. This is a small
registered subset of DROID, not the entire published benchmark, and session
separation does not prove scene or object separation.

Reuse the original ingestion function for exact action-field equality, episode
boundaries, RGB decoding, temporal grouping and hashes. The only split adapter
assigns every retained fresh episode to test. Store observed frames at native
indices 0,5,... and concatenate the five intervening recorded 7D commands into
35D blocks. No physical timestamp or control latency is inferred. Use three
observed support frames and two prior action blocks; predict recursively from
future recorded command blocks. **The model receives no future image.** Future
images are encoded only as scoring targets, in the same frozen coordinates.

Window starts are 0,5,10,... in the stored-frame sequence, without crossing
episode boundaries. Horizon five requires eight stored frames; horizon ten
requires thirteen. Short episodes remain in the manifest and are explicitly
listed as having zero eligible windows for the corresponding horizon. No
padding, repeated target or invented query is used. A decoding, checksum,
nonfinite-output or schema failure aborts the stage instead of dropping examples.

The existing feature extractor and dataset loader require all three splits.
Narrow, scoped adapters replace only their manifest split-presence guard with
an exact test-only population guard and replace feature-extractor training
statistics fitting with the unchanged original training-statistics copy.
Pixel processing, features, payload validation, windows, model execution and
error scoring reuse the original frozen implementations. No artificial train
or validation episodes are inserted, and original source files remain unchanged.

## Secondary evaluations and diagnostics

Evaluate the same 24 conditions on all four separate populations: camera1/h5,
camera1/h10, camera2/h5 and camera2/h10, giving **96** complete evaluations.
Camera 2 is real viewpoint transfer of the same recorded episodes; h10 is
longer-horizon extrapolation. Never pool these populations into the primary.

For each population report endpoint standardized feature MSE, raw MSE and cosine
distance, plus means over the complete rollout. For h5 also report h1 and h3
errors as descriptive trajectory diagnostics. Include unchanged persistence and
constant feature velocity using the same three-frame support. Preserve the
original reversed-future-action diagnostic, clearly observational rather than
a physically executed counterfactual.

Within each original/calibrated variant, compare ours with Framewise,
constant-dynamics, action-free, persistence and constant velocity. Also compare
each calibrated method with its own original version. Report point estimates
and intervals for endpoint and mean standardized MSE. No result is hidden if
negative and no subgroup, extra metric or favorable subset is selected later.

## Aggregation and uncertainty

Average windows within episode, weight eligible episodes equally, then weight
the three training seeds equally. Reuse the paired bootstrap that resamples
recording sessions and training seeds, keeping all episodes from a resampled
session together. Use 10,000 draws, fixed bootstrap seed 20260919 and percentile
95% intervals. This preserves pairings across all methods and variants.
Only the designated primary comparison is confirmatory; every secondary
interval is descriptive and unadjusted for multiplicity.

## Integrity, execution and reporting

`configs/real_video_development/fresh_evaluation_v1.json` is created before any
decoding and pins this protocol, scripts, reused sources, original normalizers,
encoder, all checkpoint/scalar packages, the original training/validation
development record and fresh metadata/exclusion/download receipts. Verify these
identities at the beginning and end of extraction, caching, evaluation and
aggregation. Verify raw shard/record hashes on decoding and cached payload hashes
before/after inference. Persist every per-episode result and the complete source
ledger. Resume may reuse only fully completed results with the same identity.

Keep all acquisition metadata, original inputs and earlier results immutable.
The candidate metadata registry retains `eligible_for_evaluation: false`; this
separate freeze authorizes only the evaluation stated above, not arbitrary new
analysis or training on the fresh set. Do not infer physical robot control,
clinical performance, state-of-the-art superiority or a novel calibration method
from feature forecasting on these recordings.
