# Implementation and scientific review

Date: 2026-09-18. Scope: model, objective, data isolation, normalization,
checkpoint selection, and interpretation of the first training campaign.
This is a code/evidence review, not independent experimental replication.
No running experiment's mathematical definition was modified by this review.
The additive framewise control was introduced after inspecting validation
diagnostics; original experiments remain preserved.

## Assessment

The implementation supports a reproducible comparison, but the original
three-way comparison alone cannot establish that factorizing contexts improves
planning. The largest issues are the unaligned plain baseline's goal coordinates
and conflating architectural factorization with additional pairing losses.
Controls have been added for both. No future-query or simulator-label leakage
into deployment context inference was found. Current validation diagnostics
do not establish an advantage for factorization.

## Findings and status

### HIGH — Plain planning uses mismatched prediction and goal coordinates

Evidence: all learned predictors minimize distance to canonical frozen reference
features (`model.py`, `forward`, target at lines 208–225). In `plain` mode,
`correct_observations` returns the raw shifted feature; `goal_embedding` calls
that same function (lines 185–190 and 269–271). Consequently, the predictor learns
canonical outputs but its planning goal remains a shifted feature. Recursive
forecasting also replaces observed shifted-history features with predicted
canonical features, an additional change in input distribution.

This does not invalidate the recorded plain prediction MSE. It does mean that
planning improvements over plain alone can reflect coordinate alignment rather
than inference of separate physical and observational changes.

**Status:** original plain runs retained and designated an unaligned diagnostic.
An additive `framewise` mode now learns
`u + Linear(GELU(Linear(LayerNorm(u))))`, with widths 192→128→192 and a zero
initialized final layer. Exactly the same per-image calibration applies to
observations and goals. It uses neither history context nor dynamics modulation.
Framewise, shared-context, and factorized models are the meaningful primary
planning comparisons. Framewise has 49,856 calibration parameters, 18,084,334
required deployment parameters, and 11,790,190 trainable parameters. Its training
and evaluation results remain pending at the time of implementation.

### HIGH — Architecture and pairing supervision were initially confounded

Evidence: `model.py` lines 228–238 add cross-appearance dynamics-context
consistency and same-appearance observation-context consistency only for
`factorized`. `single` receives the same primary examples and canonical targets,
but does not consume these pairing constraints. Equal access to stored arrays
is not the same as an equal objective.

A factorized-versus-single difference therefore measures the complete methods,
not the isolated effect of splitting a context network.

**Status:** the controller has added `factorized_unpaired` runs for both
environments and all three seeds. The campaign expands from 18 to 30 trained
runs: the original runs plus six framewise and six unpaired-factorized runs.
Compare unpaired factorization with shared context to study architecture, and
factorized with unpaired factorization to study the added constraints. Do not
apply dynamics invariance directly to the entire shared context while also
requiring it to represent appearance: that would impose contradictory roles.

### HIGH — Total training loss is not a method-comparison metric

Evidence: in plain mode, the cached visual features are frozen and there is no
observation adapter. Its alignment term is a constant with respect to every
trainable parameter. The reviewed PushT seed-0 logs show validation alignment
loss exactly 0.3472623500624314 across the observed epochs. Factorized and shared
models can optimize that term; factorized additionally has two regularizers.

The constant does **not** affect plain's gradients, gradient clipping, AdamW
updates, or the epoch-based cosine schedule. It does inflate its displayed total
loss and makes cross-method ranking by total loss incorrect.

**Status: checkpoint selection already correct.** `train.py` lines 192–203 select
the best checkpoint using validation `prediction_loss` only. Paper and demo
plots must use the common fixed-coordinate prediction metric for comparisons,
with separate alignment and regularization diagnostics. Total loss is suitable
only for describing optimization within a fixed objective.

For example, the captured best-at-epoch-4 prediction losses for PushT seed 0
were plain 0.1180303, shared 0.1195910, and factorized 0.1207728. These are
validation-only diagnostics, not test or planning results, and do not support
claiming that factorization wins. The exact sampled rows are retained in
`implementation_review_metrics_snapshot.json`; later training may continue.

### HIGH interpretation risk — Small context MSE does not prove disentanglement

The context consistency loss has a scale ambiguity. For a context `c` and the
following affine FiLM map, replacing `c` by `epsilon*c` and its affine weights
by `W/epsilon` preserves the modulation but reduces squared context differences
by `epsilon²`. The dynamics context's final linear layer allows this change;
the following affine modulation makes it functionally invisible. Weight decay
discourages extreme weights but does not make the representation identifiable.

The observed dynamics-context standard deviation being much smaller than the
shared context's standard deviation is therefore not by itself evidence of
collapse, successful invariance, or causal identification. Batch-averaged
standard deviation also depends on batch composition; validation windows are
ordered and often highly correlated.

**Status:** no existing loss changed. Interpret the networks as functional
conditioning modules. If mechanistic claims are made, use effective predictions
under held-out paired renders, context replacement/shuffling, and adapter-output
changes. Report context statistics only as diagnostics. Current raw context
statistics cannot substantiate disentanglement claims.

### MEDIUM — Teacher-forced training and recursive inference differ

For eight images and history length three, context inference uses images 0–2 and
executed actions 0–1. Prediction training uses four teacher-forced windows ending
at images 3–6 to predict canonical targets 4–7. The first strictly future image,
index 3, is used by alignment but is not itself a prediction target in this
version. Forecast evaluation starts from history 0–2 and recursively predicts
images 3–7. MPC likewise begins immediately after observed history.

This is temporally valid, but it is a train/inference distribution difference.
One-step validation ranking need not match recursive forecasting or planning.
It is not correct to describe the existing runs as multi-step rollout training,
or as supervising every future query frame.

**Status:** preserve the current campaign definition. Report one-, three-, and
five-step recursive errors and closed-loop results. A future version could add
the support-boundary target and recursive training as separately identified
experiments; do not silently change them in the middle of these runs.

### MEDIUM — Canonical paired renders are privileged training supervision

The reference target uses a canonical render of the same physical trajectory.
This is more information than an unpaired observational dataset supplies. Every
learned baseline receives those canonical prediction targets and the same data
split. Context modules do not receive canonical targets, factor IDs, true
simulator state, or future query frames at deployment. The factorized objective
additionally uses observation IDs and paired views during training, as above.

**Status:** the model card and paper must state the training privilege. This is
a simulator-supervised world-model adaptation study, not unsupervised adaptation
from arbitrary deployment video. No medical or real-robot generalization follows
from this experiment.

### MEDIUM — Validation overfitting and condition aggregation require care

The reviewed logs show falling training prediction error while validation error
is already close to its best value around epoch four. Finishing all 30 epochs
and retaining the best validation checkpoint is correct; do not present the
last-epoch result as the selected checkpoint or treat lower training loss as a
generalization result. Hyperparameter changes after viewing test outcomes would
require a new untouched test protocol.

The aggregate test summary contains all nine observed/dynamics combinations,
including combinations seen during training. The designated unseen composition
is `(observation=2, dynamics=2)`; development composition `(1,1)` is separate.
An aggregate improvement alone does not establish compositional transfer.

**Status:** retain condition-level results, three training seeds, paired initial
states, and the predeclared held-out cell. Root's new controls address validation
audit findings; record that timing transparently rather than claiming all 30
runs were fixed before implementation review.

### MEDIUM — The frozen-base fine-tuning protocol differs from original training

The pinned upstream training code trains encoder/projector/predictor and uses
SIGReg. This campaign freezes the encoder and its canonical reference geometry,
does not use SIGReg, and fixes running statistics in both visual and prediction
projector BatchNorm modules. Prediction-projector affine parameters remain
trainable. All learned variants use the same convention.

This avoids cross-example/future-frame statistics entering support features and
allows batch-size-one MPC. It is a deliberate fine-tuning protocol, not an exact
reproduction of the upstream end-to-end optimization. Strict checkpoint loading
and reuse of its architecture should not be conflated with reproducing its
published planning scores.

**Status:** common across controls and explicitly documented. Do not change BN
behavior mid-campaign. Evaluation uses FP32 inference, while fitting/validation
use BF16 autocast when supported; report these precision settings with latency.

## Checks that passed

- Images enter as RGB floats in `[0,1]`; ImageNet normalization occurs once.
  Cached features and online frozen encoders use the same source weights and
  preprocessing. Cache/data manifest and encoder hashes are checked.
- Official native two-action mean/sample-standard-deviation values are tiled
  over five chronological actions, rather than recomputed from a test split.
  The actual PushT source statistics prove relative controls; the absolute-action
  exploratory corpus is excluded from the campaign. Reacher uses its own stats.
- Context inference takes only the support prefix and executed inter-frame
  actions. Query images and intervention labels are not API inputs. Projector
  running statistics cannot leak query frames into support features.
- Whole trajectory seeds remain in one split. Although all appearance features
  exist in the cache, the training loader returns only allowed combinations and
  paired training views; stored but unselected arrays do not become model inputs.
- Action-conditioned rollout uses exactly `H-1` executed history blocks followed
  by strictly future candidate blocks. Predicted states are appended without
  reapplying visual correction to already canonical predictions.
- Framewise calibration applies identically to goal and history images, starts
  at identity, and has no temporal or context dependence. Its alignment loss
  trains the calibration module without updating frozen visual coordinates.
- **24 model tests passed** after the framewise addition. Existing seven modes'
  checkpoint keys/values, constructor RNG, stochastic predictions, and losses
  are bitwise identical to a pre-addition snapshot. See
  `framewise_addition_compatibility.json`.
- Portable checkpoint loading, immutable reference targets, finite gradients,
  raw/cache equivalence, and exact interrupted training with stochastic dropout
  are covered by tests. No expensive new training was run for this review.

## Evidence and next decisions

Source: `src/shiftwm/model.py`, `train.py`, `data.py`, `evaluate.py`,
`scripts/prepare_action_stats.py`, pinned upstream `train.py`/`utils.py`, and
`configs/world/*`. Quantitative review evidence:
`implementation_review_metrics_snapshot.json`, `model_parameter_counts.json`,
`framewise_addition_compatibility.json`, and `portable_checkpoint_verification.json`.

Do not elevate the factorized method based on total loss, low context MSE, or
improvement over unaligned plain planning. The scientific decision depends on
held-out-composition recursive prediction and planning relative to framewise,
shared-context, and unpaired-factorized controls, together with honest uncertainty
and runtime comparisons. A simpler winning model is a valid outcome of this study.
