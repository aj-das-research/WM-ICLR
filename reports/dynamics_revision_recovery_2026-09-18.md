# Completed dynamics revision: recovery audit and next controlled experiment

Audit date: 2026-09-18. This report audits existing artifacts and source code;
it does not submit jobs, alter trained packages, or change scientific code.
The machine-readable record is
`reports/evidence/dynamics_revision_recovery_2026-09-18.json`.

## Resolution: the Reacher result is complete

No job recovery is needed. The earlier one-of-two result ledger was observed
while the campaign was still finishing. Both jobs completed successfully:

| Slurm job | State / exit | Node | Runtime | What happened |
|---|---|---|---|---|
| 199687 | COMPLETED / 0:0 | gpu-01 | 17m 57s | Trained and evaluated both full revisions |
| 199688 | COMPLETED / 0:0 | gpu-01 | 19s | Validated and reused both already completed runs |

The Slurm controller reports completion; `sacct` is unavailable because its
accounting connection to localhost:6819 is refused. The controller output,
logs, complete training histories, checkpoint metadata, and evaluation records
agree. `runs/dynamics_revision/campaign_state/status.json` is complete, and
`paper/generated/dynamics_revision.json` now contains two complete records.
No additional recovery allocation or rerun is warranted.

## Actual results

These are post-hoc development results from one training seed. They are not
new main-test results and do not establish a multi-seed improvement.
Eligible tasks exclude goals already achieved during initial support.

| Environment | Revision raw success | Frozen Framewise donor raw success | Revision eligible success | Donor eligible success | Eligible difference | Revision-only / donor-only wins |
|---|---:|---:|---:|---:|---:|---:|
| PushT | 2/32 | 2/32 | 1/31 | 1/31 | 0.00 pp | 1 / 1 |
| Reacher | 10/32 | 13/32 | 8/30 | 11/30 | -10.00 pp | 4 / 7 |

The new temporal residual has not improved development planning over its
complete frozen Framewise donor: it ties on PushT and loses on Reacher. These
outcomes should appear in the appendix and should not be promoted as a
successful replacement architecture.

| Environment | Trained epochs | Selected revision epoch | Revision validation prediction MSE | Donor selected epoch | Donor validation prediction MSE | Relative MSE reduction vs donor |
|---|---:|---:|---:|---:|---:|---:|
| PushT | 30 | 8 | 0.12079349 | 4 | 0.12012320 | -0.5580% |
| Reacher | 30 | 5 | 0.00295761 | 29 | 0.00297192 | +0.4815% |

Both runs completed 24,360 optimizer steps and exactly 30 unique epochs with
finite numeric train/validation metrics. Recorded training wall times were
387.23 seconds for PushT and 393.72 seconds for Reacher. These are training
logs, not controlled inference-efficiency comparisons. The Reacher revision
slightly improves the selected one-step validation error while degrading
development planning, demonstrating that this metric alone is insufficient
to establish a control benefit.

The initial donor at revision epoch zero was not an eligible checkpoint under
the fixed selection protocol. Therefore PushT can, and did, select a trained
revision with worse validation error than the original donor. Retrospectively
changing this selection protocol would not repair the completed experiment.

## What was actually learned and what remained fixed

The donor is the complete completed Framewise seed-zero package, selected at
epoch 4 on PushT and epoch 29 on Reacher. Its image encoder, reference encoder,
framewise calibration network, action encoder, predictor, prediction projector,
and buffers remained fixed. The donor stays in evaluation mode, disabling
dropout changes and batch-normalization updates during revision training.

Only a new TransitionContext GRU and action-embedding ResidualFiLM were trained:
205,600 context parameters plus 12,672 adapter parameters, totaling 218,272.
The GRU consumes three corrected observed support states and two already
executed action blocks. The adapter scales and shifts the frozen donor's
action embeddings. Its zero initialization exactly preserves the donor before
training. The available shifted goal uses the frozen Framewise calibration;
its embedding never depends on the new context.

This audit independently loaded both final best packages and both original
donor packages on CPU with strict package loaders. Every embedded donor state
tensor and buffer is bit-for-bit equal to its source donor. The trained
adapter's originally zero affine weights and biases are nonzero in both
packages, so these are trained revision checkpoints, not untrained wrappers.
The source identities of all six launch-pinned scientific files still match
`reports/evidence/dynamics_revision_allocations.json`.

Reusable best packages exist at:

- `runs/dynamics_revision/pusht_s0/best`
- `runs/dynamics_revision/reacher_s0/best`

They embed the entire donor plus new modules. The loader is
`shiftwm.dynamics_revision.load_revision_package`; the original campaign
loader does not recognize this separate package kind. These packages are
reusable research artifacts, but neither has demonstrated a planning gain
over its frozen donor.

## Concrete next intervention

The main code uses teacher-forced query windows to predict target indices
4 through 7 from an eight-frame sequence with three support frames. Its
recursive evaluation starts with the support/query boundary prediction at
index 3 and recursively feeds predictions back through index 7. The immediate
2-to-3 transition is excluded from the existing prediction loss. This is a
documented training-design mismatch, not evidence of corrupt results or an
off-by-one implementation failure.

The smallest informative next experiment is a separate, fixed-donor 2-by-2
development study:

| Training objective | Context from observed history | Shared constant context |
|---|---|---|
| Boundary-complete teacher-forced one-step loss, targets 3:8 | New controlled reference | Matched extra-training control |
| Recursive five-step loss, target frames 3:8 | Proposed intervention | Checks whether any gain needs temporal conditioning |

For the recursive objective, call the same `rollout_features` path used for
inference with support features `features[:, :3]`, observed actions
`actions[:, :2]`, and five future training action blocks `actions[:, 2:7]`.
Compare its five predictions with immutable canonical targets `target[:, 3:8]`
using a fixed mean MSE. The context must be inferred only from the observed
support and held fixed within the rollout. Future observed features must not
replace predicted states. Gradients must flow through the complete recursive
rollout into the trainable residual while donor parameters remain frozen.
The new teacher-forced arm should include the same target frames 3:8. Matching
target coverage isolates recursive feedback from the separate boundary-coverage
change. The old 4:8 objective remains the completed historical experiment.

Use the same data, optimizer, 30 epochs, initialization rule, and number of
optimizer updates for each arm. Select all new arms using one predeclared
validation metric, preferably the same five-step recursive MSE, so checkpoint
selection does not favor one training objective. The completed original
revision remains a separate historical baseline; do not overwrite its metrics
or silently reinterpret its one-step checkpoint selection.

The smallest constant-context control replaces observed temporal context with
one trainable global context vector and retains the same FiLM. This matches
additional optimization and removes history dependence but has fewer active
parameters, which must be reported. If equal nominal trainable parameter count
is required, preserve the exact GRU/readout/FiLM modules and feed the GRU a
fixed synthetic support/action sequence shared by every sample. The resulting
context can train but cannot encode actual observations,
conditions, targets, or candidate actions. Its effective function is still a
constant; equal nominal parameter count is not equal functional capacity.
The implemented new control uses zero normalized inputs, so some GRU input
weights receive zero gradients. The paper must state this effective-capacity
limitation even though both arms serialize the same trainable parameter count.

Because the original donor is frozen, a further optional diagnostic is the
exact zero-residual donor evaluated on the same recursive validation metric.
It should be part of a newly specified selection or safety rule only if that
rule is fixed before the new outcomes. No retrospective switching between
checkpoints based on development success should enter primary results.

Implement this work in new isolated source/config/output namespaces, leaving
the original campaign and completed revision hashes untouched. Meaningful
verification should cover: exact donor equality at initialization; no future
observations or targets in context; boundary target 3 included; five recursive
steps and non-detached gradients; invariant frozen donor tensors/buffers;
constant-context invariance to real support; strict portable loading; and
matching development task/support identities. These checks establish software
correctness; they are not experimental efficacy claims.

## Interpretation and resource limits

This diagnosis supports testing the training/rollout mismatch. It does not
prove that this mismatch caused the negative result or that changing it will
improve control. The frozen predictor may be difficult to steer with an
action-embedding affine residual, only two support transitions may be
insufficient for the environment, and optimized action sequences can differ
from recorded training trajectories. Those remain hypotheses. Raw context
standard deviation is not a reliable test of context identifiability or
collapse because downstream affine weights can rescale it.

The completed runs used one GPU, eight CPUs, and 32 GB host memory. New training
can reuse this allocation shape. Actual recursive-training time must be
measured before reserving a broad campaign; it cannot be inferred directly
from the six-minute one-step runs. Start the fixed development study, retain
all negative outcomes, and expand seeds/test evaluation only after a
predeclared development decision. Existing main-test tables and GPU jobs
should continue unchanged. The previously explored development tasks are no
longer an untouched estimate of generalization.
