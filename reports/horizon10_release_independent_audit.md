# Independent horizon-ten release contract audit

Read-only audit of the completed original-validation campaign. No new model
inference, training, image inspection, or test-data access was performed.
Machine-readable identity and arithmetic evidence is in
`reports/horizon10_release_independent_audit.json` (SHA256
`a026adf89f4e429538067fa40ca40b2975fa0ec1bbd2ae63e289273898251bba`).

**Observed:** all 12 registered runs contain exactly 30 journal epochs;
every first validation minimum and selected checkpoint epoch is **1**.
This means 30 full epochs were executed and the epoch-1 predictor was selected;
do not describe the selected weights themselves as epoch-30 weights.
The saved training summary equals the final report's training receipt for
every run. All frozen scientific and original-h5 evidence hashes matched.

All **60** source evaluation files matched their report hashes and embedded
summaries. Their episode/session/window identities matched manifest-derived
validation eligibility. Standard h5 uses **141 episodes, 59 sessions, 1,772
windows**; h10 and matched h5 prefixes use the same episode/session IDs but
**1,631 windows**. Equality of episode counts alone does not establish matched
windows. Every saved metric was finite/nonnegative, and every equal-episode
summary was independently recomputed from the saved episode aggregates.
All 20 comparison point means, signed differences and relative reductions
were independently checked. Bootstrap intervals were source-inspected but
not recomputed in this bounded audit.

## Contracts the publisher must preserve

1. Bind registry SHA256
   `91e9d9dbe269362e0b19364571216310b5b3710b8754f8fde1340814f278469c`,
   all four modes × seeds 0/1/2, every config/dependency/original-h5 hash.
   Require the distinct package kind
   `shiftwm_real_video_droid_horizon10_v1`. This is the original 1,536-feature
   2×2 DINO model architecture, not the new 6,144-feature spatial family.
2. Call the unchanged `horizon10_train.validate_completed` for each run.
   It checks best/last/optimizer-RNG package hashes, full epoch history,
   scientific identity/config, ten-query metrics, first minimum selection,
   counts and accumulated steps. Selection is window-weighted all-ten
   standardized MSE in FP32; the auxiliary equal-episode journal metric
   never selects checkpoints. Public inference packages must omit optimizer
   state while keeping source evidence for full completion.
3. Reopen all five evaluations per run. The final report deliberately removes
   episode arrays; its embedded summaries cannot replace the full source
   ledgers. Check expected kinds/horizons/checkpoint identities, populations,
   finite metrics, summary arithmetic and source hashes. Existing ledgers
   store episode-average errors and window-start lists, not individual
   per-window errors; do not claim per-window error recomputation.
4. Preserve all **20** comparisons. Four compare factorized versus Framewise
   under equally h10-trained models: endpoint and all-query mean at h5/h10,
   using each horizon's standard population. Sixteen compare new h10-trained
   versus original h5-trained checkpoints separately for all four modes,
   both horizons and endpoint/mean metrics. For h5, these use the exact
   h10-eligible window prefixes for both donors. Recompute their paired
   recording-session × training-seed bootstrap with 10,000 draws and seed
   5198010. The point estimate equally weights episodes and seeds; sampled
   sessions retain all constituent episodes rather than equally averaging
   session means.
5. The old receipt's CPU reload is a same-workspace model reload. It does not
   establish portability of a newly assembled source/encoder bundle. Run a
   new physical relocation with the actual released source, strict package
   loader, offline restrictions and all 12 selected checkpoints. Match the
   original DINO feature preprocessing and retain its encoder/data licenses.
6. Bind all finalizer, evaluation, receipt, journal, source, checkpoint,
   normalization and encoder identities in the new publisher's evidence
   ledger and verify them again before sealing/uploading. Preserve immutable
   publication behavior, safe archive paths and public-download hash checks.

## Interpretation

Fourteen of the 20 exploratory intervals lie wholly below zero in the
first-minus-second direction; six include zero. There is one unfavorable
point estimate: factorized h10-trained versus original h5-trained **matched
h5-prefix mean**, −0.116% relative reduction (an increase in error), with an
interval including zero. Do not omit it. The four equally h10-trained
factorized-versus-Framewise point reductions are small: about 0.226% h5
endpoint, 0.129% h5 mean, 0.405% h10 endpoint and 0.252% h10 mean.

The treatment changes training horizon and matching checkpoint-selection
horizon together. Matching inference windows does not isolate those two
changes. This is a conventional horizon-mismatch control on reused original
validation data, not a new algorithm or independent test confirmation.
Intervals are exploratory and unadjusted for 20 comparisons. Latent prediction
errors do not establish RGB-generation quality or deployed robot-control
benefit. No blocking inconsistency was found in the inspected evidence;
future bundle portability and public asset integrity remain exporter gates.
