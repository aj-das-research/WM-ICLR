# Frozen-framewise dynamics revision

This is a development-driven, separate training experiment. The original main
models, training/evaluation implementations, splits and results stay intact.
The completed goal-only intervention motivates this question: does adding a
trained dynamics-context residual help when the expressive framewise visual
calibration and the world-model predictor are held fixed?

The preceding intervention improved Reacher from 6/30 to 12/30 eligible
development successes, compared with 11/30 for the complete framewise model.
PushT declined from 1/31 to 0/31. Those exploratory seed0 outcomes are not a
factorization benefit or a guarantee that this revision will help. See
`reports/development_diagnosis.md` and its source-validated evidence files.

## Fixed and trainable computation

For each environment, the donor is the completed **framewise seed0
validation-best checkpoint**: PushT epoch4 and Reacher epoch29. All donor
parameters and buffers are fixed, including the framewise residual calibrator,
visual encoder/projector, action encoder, predictor and prediction projector.
The donor is forced into evaluation mode even while the wrapper is training;
dropout and batch-normalization state cannot change the frozen control.

Only these newly initialized modules train:

1. The existing `TransitionContext` GRU, width128 and output dimension32,
   receives the three corrected **observed** support features and two
   normalized, already executed action blocks. Each transition input is
   `[corrected state, corrected next-state minus state, normalized action]`.
2. The existing `ResidualFiLM` acts on frozen action-encoder embeddings:
   `a_emb * (1 + 0.1*tanh(scale(context))) + shift(context)`. Its final affine
   weight and bias start at zero. Its inference predictions therefore exactly
   match the framewise donor at initialization.

The new context/residual modules contain **218,272 trainable parameters**:
205,600 context parameters and12,672 residual parameters. The full loaded model
contains25,141,238 parameters, including the embedded24,922,966-parameter donor
with its reference encoder and inactive ablation modules. These are serialized
parameter counts, not runtime-memory measurements.

Both observed frames and the available shifted goal still use exactly the
donor's framewise calibration. Goal encoding cannot depend on the new context.
During a candidate rollout the inferred context remains fixed; it is refreshed
only from real observed support at the next planning call. Future observations,
future executed outcomes, condition IDs, paired images and canonical goal
images are not context inputs.

## Training and selection fixed before revision outcomes

Run seed0 for both PushT and Reacher, for all30 epochs, using the existing
complete training and validation splits. Reuse the donor's cached frozen
features, immutable canonical targets, official action statistics,
sequence length8, stride1, batch128, AdamW learning rate5e-5, cosine minimum
1e-6, weight decay0.001, clipping1.0 and bfloat16 configuration.
The initializer rejects changed data/cache/action-stat checksums, a different
encoder cache, incomplete donor training, a non-best donor, changed specified
training settings or dataset subsampling overrides.

The prediction objective and indexing are reused directly from
`ShiftWorldModel.forward`: H=3 support observations at indices0–2, executed
support actions0–1, teacher-forced query predictions at indices4–7 from
action-conditioned windows ending at indices3–6. The immediate support/query
boundary prediction at index3 remains excluded. This preserves the original
one-step objective and its exposure-bias limitation; the revision does not
silently introduce multi-step training.

Optimize prediction MSE only. Canonical alignment MSE is still logged but is
constant with respect to the new parameters because the calibrator is frozen.
Dropping this constant from total loss changes neither its gradient nor the
prediction-based selection rule. No paired-context or observation-ID
regularizers apply. Select the minimum validation prediction loss among the
30 completed epochs; initial donor epoch0 is not a selection candidate.
Development planning never selects training epochs.

## Development evaluation and interpretation

The separate evaluator calls the unchanged campaign evaluator. The fixed
control is the already completed corresponding framewise seed0 development
run. Its checkpoint, dataset, evaluator source, budget,32 unique selected tasks
and initial-support diagnostics are validated before comparison.

The protocol is fixed: development condition(o1,d1),32 existing trajectory
seeds, solver seed1701,300 candidates,30 CEM iterations,30 elites, horizon5,
five native actions per block,50 native calls including10 support calls, and
goal offset5 from the last support frame. Search uses the unchanged official
action z-scores and native clipping. No test/extrapolation CLI is exposed.
The revision verifies that **all embedded donor tensors** remain exactly equal
to the original completed donor before evaluation. Full and interrupted
results are stored in a separate namespace with source identities and
per-episode resumability. Missing/incomplete results are pending, not zero.

Report raw success, support success, eligible policy success and paired
revision-only/control-only successes for both environments, including any
negative result. No efficacy is claimed before these jobs complete. Seed0
development outcomes do not measure variation across training seeds and are
not primary held-out evidence.

A gain would support the added trained contextual residual over the frozen
donor. It would **not establish that temporal conditioning itself is necessary**:
the residual also adds parameters and30 epochs of optimization. A matched
trained residual with constant context would be required to isolate that
stronger claim. This experiment does not change the foundation architecture,
establish causal dynamics identification, or demonstrate medical/real-robot
generality.

## Code, commands and reusable packages

The new implementation is `src/shiftwm/dynamics_revision.py`; training and
evaluation are separate scripts. They reuse the existing model's feature,
query and rollout indexing, GRU/FiLM modules, dataset/device helpers and
weights-only checkpoint/RNG/optimizer functions. Original active campaign
files are unchanged.

```bash
.venv/bin/python scripts/train_dynamics_revision.py \
  --config configs/dynamics_revision/pusht_s0.json --resume-if-present
.venv/bin/python scripts/evaluate_dynamics_revision.py \
  --environment pusht --checkpoint runs/dynamics_revision/pusht_s0/best \
  --device cuda --save-video --max-runtime-seconds 2400
```

Use the analogous Reacher config/checkpoint/environment for the second run.
Training accepts an operational `--max-runtime-seconds` override. Both CLIs
return75 after safely recording an intentional interruption, and0 on completion.
Training resume restores optimizer, scheduler, RNG, epoch permutation and
completed-batch position. Scientific source/config changes reject resume;
runtime/device/worker/log controls may change. A per-output process lock prevents
two simultaneous writers. Completed training is reused only after its identity,
full epoch history, best/last metadata and strict package reconstruction pass.

Outputs are `runs/dynamics_revision/{environment}_s0/{best,last}` and
`results/development_dynamics_revision/{environment}_s0/planning_development.json`.
Packages contain the complete frozen donor plus new modules, without an
inference dependency on the donor checkpoint path or upstream weights download.
Use the new loader; the original campaign CLI does not recognize this distinct
package kind:

```python
from shiftwm.dynamics_revision import load_revision_package
model, state = load_revision_package("runs/dynamics_revision/reacher_s0/best", device="cpu")
# RGB inputs are floats in [0,1]; action blocks are raw native controls.
future_features = model.rollout(history_images, executed_action_blocks, candidate_action_blocks)
```

The package requires the ShiftWM source/package version containing this module
and the existing vendored LeWM dependency. Optimizer/RNG state is in the separate
training-state file and is not needed for inference. A later public release
should include source, model card and validation evidence, rather than describe
an untrained initialization as a trained revision checkpoint.

## Pre-launch verification

Targeted tests cover exact zero-initialized donor equality, corrected-support
and executed-action access, ignored future/paired/ID inputs, fixed goal
calibration, trainable partition, finite new-module gradients/updates, frozen
dropout/BN and donor tensors, strict portable loading, optimizer/RNG restoration,
and exact interrupted-versus-uninterrupted fixture training. Evaluation tests
check the development-only CLI, source/budget identity, support equality,
partial protocol signatures and paired count accounting.

Both real donors were also loaded on CPU, checked on one existing **training**
cache window, saved to temporary full revision packages and reloaded without
the donor path being used by the loader. Predictions matched bit-for-bit before
and after loading and matched the donor in both wrapper training/evaluation
modes. These are software/API checks, not training completion or research
benchmark results. No GPU job was launched for this verification.
