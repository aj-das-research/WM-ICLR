# Technical qualitative comparison — figure brief

User request: explain, at several technical points, how a successful ShiftWM
execution differs from a failed baseline execution. The original four positive
cases are outcome-selected development evidence, not a benchmark-win sample.

Visual thesis: identical initial support leads to different executed controls;
measured physical errors show when each policy enters or misses the actual goal
tolerances. A same-transition prediction diagnostic tests a possible link to
forecast quality instead of assuming that link from the successful endpoint.

Three compositions considered:

1. Paired execution lanes above constraint trajectories and paired prediction
   errors (selected). Reading path: observed behavior, scoring rule, then a
   separate measurement of model prediction on the same transition.
2. A single pipeline with green checks on calibration, dynamics, ranking and
   success (rejected: candidate ranking is not archived, and calibration errors
   are worse for ours in all four positive cases).
3. A dense all-task table of physical errors and latent errors (retained as
   machine-readable supporting evidence, unsuitable as the visual explanation).

Use 5.5-inch-wide paper figures. The two detailed case studies are the first
positive task in each environment by the existing lexicographic selection rule;
all four positive cases remain in Figures 6/7 and all discordant cases are
retained in the diagnostic evidence, including Framewise-only successes.

Representation contract:

- Actual RGB observations retain method-matched image coordinates and scale.
  Dashed goal silhouettes are display annotations, not predictions or scores.
- At decision boundaries 10, 15 and 20 native actions, show the actual observed
  frame and, when available, physical error. Real endpoints retain their own
  stopping times, with no extension of a successful trajectory to action 50.
- Plot physical errors only after exact saved-action replay reproduces archived
  frames, goals, endpoint measurements and success/stopping behavior.
- PushT success requires both combined pusher/block position error below 20
  native pixels and wrapped block-angle error below pi/9. Plot these separately;
  do not substitute mixed-unit Euclidean distance or block position alone.
- Reacher success requires each unwrapped joint error below 0.05 radians.
  Plot individual joint errors or their maximum; never label L2<0.05 as success.
- Prediction comparisons give both checkpoints identical three-frame observed
  histories, two past action blocks, and one actual future block. Targets use
  the common frozen encoder on the canonical next observation. These privileged
  targets are diagnostic only, never context inputs. Partial terminal blocks
  have no one-full-block prediction score.
- Show both realized behavior-policy traces. Comparisons within each transition
  are paired; different policies visit different states after the common start.
- Separate a code-derived inference schematic from measured results. Framewise
  still uses temporal history and actions; its difference is per-image rather
  than support-conditioned calibration. Neither schematic color nor a positive
  episode establishes causal factorization or globally superior planning.

All numerical fields will be populated from source-validated diagnostic files.
No visual proof of superior calibration, candidate ranking, or model mechanism
is assumed. The complete results determine the final labels and captions.
