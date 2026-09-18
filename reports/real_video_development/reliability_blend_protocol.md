# Causal support reliability diagnostic v1 — before execution

This is an original-training/original-validation development study. No original
test or inspected fresh-test feature/image payload may be read. It is a bounded
calibration/ensemble control, not a novel world-model architecture or confirmatory
test. Frozen scientific files/checkpoints remain unchanged.

Use all three seed pairs of original, equally training-calibrated Framewise and
factorized checkpoints. Keep encoder, native frame/action grouping, normalization
and donors fixed. On camera one take every stride-five window with 23 stored
frames. All 13 prefix observations precede the ten query targets. Preserve all
episodes in the audit, including those too short to supply an eligible window.

For the gate, forecast prefix frames 3–12 using frames 0–2 only and action blocks
0–11. Infer donor contexts only from frames 0–2; do not insert prefix targets into
this ten-step rollout. Fit local alpha from A=mean((O−F)^2), B=mean((target−F)*(O−F))
in fixed standardized coordinates: clip((B+lambda*prior)/(A+lambda),0,1). Use the
global prior if the denominator is zero. Each actual query separately initializes
donors from observed frames 10–12 and actions 10–11, then uses recorded future
action blocks 12–21. Alpha stays fixed; no query image reaches inference.

Fit prior=clip(equal-episode mean B / equal-episode mean A,0,1), or zero for zero
energy, using training prefix moments. Set lambda=kappa*training mean A. Choose
kappa from {0,1,10,100} with deterministic three-fold training-session cross-
validation (SHA256("support-reliability-v1:"+session_id) modulo three). Priors and
energy are fit on fold-training; choose minimum out-of-fold h5 query endpoint MSE,
averaging windows within episodes and episodes equally. Ties choose larger kappa.
The frozen donors have seen original training; this cross-validation isolates gate
fitting, not end-to-end unseen-session model training. Lock every seed's fitted
parameters and artifact hashes before decoding or evaluating any validation
feature payload. Registration/integrity checks may hash validation bytes earlier.

Report both donors, equal blend, global-prior blend, unshrunk local gate, shrunk
local gate, and one-step-prefix gate. The latter uses ten one-step predictions,
each with only its actual preceding three frames, and receives the same train-only
fitting opportunity. Also report a stronger constant blend fitted directly to
training query h5 moments. The shuffled-gate diagnostic cyclically permutes all
local shrunk gate values across sorted recording-session blocks by the largest
block size; require every donor to be from another session, preserving the exact
gate distribution. If a session exceeds half the windows, mark that diagnostic
unavailable rather than silently filter. Include persistence and constant velocity
for context. All methods use the same windows and thirteen-frame observation
availability; existing donors still consume only three frames per forecast.

Cache FP64 sufficient statistics of fixed FP32 donor predictions; analytically
reconstruct every blend's per-step MSE and verify this against direct prediction
arithmetic. Do not perform per-method outcome filtering. Report per-episode and
three-seed h5/h10 endpoint errors, paired session×seed bootstrap intervals (10,000
draws, seed 20260919), and gate distributions. All intervals are exploratory.

Prespecified promotion: shrunk gate improves h5 by at least 1% over the stronger
individual donor and global-prior blend; paired interval vs global-prior blend is
strictly below zero; each seed improves over that blend; mean h10 does not worsen.
Additionally require a 1% gain over the training-query-fitted constant blend and
report comparisons to equal/one-step/shuffled controls. Failure of any criterion
means no promotion. A matched h10-training study is separate; this gate run cannot
establish a novel solution to horizon mismatch, physical dynamics, or robot control.

Run CPU-only in a scheduler allocation: eight CPUs, 16 GiB, no GPU requested.
Record all dependency, input, donor and output hashes, actual runtime/RSS, job ID,
portable donor+gate packages, and exact offline reload parity. Verify dependencies
at start and end. Freeze protocol/config/code/tests/Slurm and registration before
reading any model outputs. No results are promised by registration.
