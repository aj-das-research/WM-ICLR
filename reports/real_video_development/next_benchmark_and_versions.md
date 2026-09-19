# Focused real manipulation and architectural versions

Decision recorded 19 September 2026, before inspecting any IWS validation video
or evaluating a model on its official validation handles.

## Why retain DROID and add IWS

DROID tests forecasting across varied real robot interactions with observed
images and recorded commands. The current task is future **visual-feature
prediction**, rather than RGB generation or physical robot control. Our figures
must distinguish observed inputs, supplied commands, predicted features and
withheld reference images. A reference RGB frame must never look like generated
model output. The study uses a fixed 1,126-episode subset, not the entire dataset.

The next focused benchmark is **IWS PushT, box manipulation and rope manipulation**,
starting with PushT. These real recordings have clear manipulated objects and
released action-conditioned evaluation handles. The choice is based on visible
task structure, public availability, existing local acquisition and compatibility
with forecasting, before observing method performance. It is not a search for a
favorable outcome. All selected tasks and outcomes will remain reported.

Sources: [DROID](https://droid-dataset.github.io/),
[RLA-WM code](https://github.com/mlzxy/rla-wm),
[official IWS evaluator documentation](https://github.com/mlzxy/rla-wm/blob/main/docs/evaluation.md),
[released data](https://huggingface.co/datasets/xyzhang368/RLA-WM).
The separate [BAIR robot pushing benchmark](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small)
is a possible later video-generation study. It requires an RGB output pipeline
and established RGB metrics; it is not added simply to increase task count.

## Data separation before model development

Use only the upstream training trajectories to construct an internal training /
development split, with 20% of trajectories held out by a deterministic salted
SHA256 ordering within each task. No frame, command value, performance or outcome
label enters that ordering. Preserve the upstream validation trajectories and
all 600 official handles for a later locked evaluation. They are upstream
validation data, not an independently collected test population; name them
accurately even though our pipeline reserves them until its final evaluation.
No session, scene or operator disjointness is established by this dataset.

The split preparer reads the previously audited metadata manifest, hashes the
train metadata and official split/handle files, and writes identities only. It
does not decode video, read official-validation command values or run a model.
Input archive and source revisions remain pinned to the acquisition audit.

The frozen split contains 480 / 120 / 10 PushT trajectories and 481 / 121 / 10
for each of box and rope (internal training / internal development / reserved
official validation). Its SHA256 is
`17be56426dbee136ec883c45813b092a0747d707f25ee6efddb12b8a351d29c9`.
Independent review reproduced the split and all 600 reserved handles, with a
runtime file-access guard confirming that no video or official-validation HDF5
was opened. One earlier dataset-identification preview, PushT trajectory 000010
frame zero, falls in internal development. That exposure is retained explicitly;
the trajectory is not moved, and internal development images are not claimed
to be entirely unseen. No model outcome informed the split.

## Comparison design to implement and freeze before training

Start with the released single-observation information budget: one actual start
frame plus the supplied command rows. Additional observed history, if studied,
must have a separately named equal-history control. Retain all official endpoint
indices and handle identities. The upstream horizon-60 convention pairs frames
at s and s+59 with 60 command rows; its rollout boundary ambiguity is documented
in `iws_temporal_semantics_review.md` and must not be silently repaired.

A compact DINOv2 study can reuse the local frozen encoder and reviewed spatial
modules, with task-specific command widths and statistics fitted on the internal
training partition only. The single-observation adapter, temporal contract,
full training budget, selection criterion, all comparison arms and final metric
definitions require a distinct reviewed registration before training. The
upstream DINOv3 RLA-WM checkpoint is a separate reproduction: its gated encoder
and compatibility issues remain unresolved. Do not label adapted DINOv2 controls
as an official RLA-WM reproduction or as state-of-the-art performance.

Primary evaluation should retain a fixed feature error and endpoint appropriate
to the registered model. Report all chosen horizons, all tasks, per-seed results,
paired trajectory-level uncertainty and the persistence control. Trajectory
bootstrap does not imply session-independent sampling. A common train-only RGB
decoder is a possible additional output for interpretation; all methods would
use the same decoder, and decoder error needs its own measured reference ceiling.
RGB reconstruction or generated footage is not available from the current
feature-only checkpoints. No unmeasured pixel metric or robot-success claim is
permitted.

## Versions are hypotheses, not a promised performance ordering

| Version / control | Mechanism | Current status |
|---|---|---|
| Matched autoregressive baseline | Recursively updates predicted visual history | Completed spatial control |
| Observation anchoring control | Predicts residuals around the last actual observation | Completed spatial control |
| Bounded additive (ours, ablation) | Anchor plus bounded feature correction, no mixing | Three full training runs being prepared |
| Unbounded transport (ours, ablation) | Gated feature mixing with unbounded correction | Three full training runs being prepared |
| Spatial transport (ours) | Gated feature mixing plus bounded correction | Completed: native h10 +5.30% vs autoregression, +3.55% vs anchoring |
| Context-off (ours, ablation) | Full decoder without learned support context | Completed; full context has no clear additional benefit |

The six new DROID component runs complete a 2×2 study of feature mixing and
innovation bounding using the already completed controls. Report the four
component contrasts and the difference-of-differences interaction, including
all null or negative outcomes. Mixing also adds a gate, identity initialization
and active parameters; do not attribute its effect exclusively to transport.
The follow-up is exploratory because the original controls have been observed.

Choose a final version using one declared development criterion and a declared
tie rule before the reserved IWS evaluation. Freeze it and evaluate it once.
Do not choose a different winning version for every final metric unless an
explicit, independently evaluated selection policy was registered. Keep all
versions, selected checkpoints and complete comparison tables available.

## How gains will be communicated

The current 5.30% is a **relative reduction in feature MSE**. It is not five
percentage points of accuracy or success. A target of 5–10% relative improvement
can guide development, but measured gains may differ by task, horizon and metric.
Percentage-point claims require an actual bounded rate such as measured task
success. Stronger results should come from better prediction and evaluation,
not changing denominators, omitting difficult tasks or tuning the final set.

Show genuine input footage, clearly marked withheld reference footage, measured
feature-error maps on a common scale, horizon curves and representative positive,
median and negative cases. An animation may explain the mechanism, but it must
be distinguished from measured model output. The existing spatial selection
rule keeps the median episode's slightly negative first window visible.
