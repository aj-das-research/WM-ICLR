# Recorded real-video study: DROID

Registered 19 September 2026 (Dubai), before feature extraction, optimization or
evaluation on this selection. This is an additional real-data study. It does not
replace or retroactively revise the registered simulation experiments.

## Data and scope

The source is the authors' [DROID release](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md),
containing physical-robot camera recordings and command/state logs. Download only
from the official `gresearch/robotics/droid/1.0.0` Google Cloud objects, pinning
each generation and verifying publisher MD5, size and local SHA256. Preserve the
provided CC-BY-4.0 license. The separately downloaded DROID-100 release is used
to audit ingestion, not to select models or training hyperparameters.

The experiment selects 24 of 2,048 full-release shards by the lowest SHA256 of
`shiftwm-real-v1-20260919:` concatenated with each object name. This gives
**1,126 expected episodes and 22,017,792,821 download bytes**, including metadata
and license. The exact selection is fixed in
`reports/evidence/real_video/droid_selected_inventory.json`. This is a
prespecified subset of DROID, not the complete benchmark or its published
real-robot policy evaluation. It includes the recorded outcomes without
performance-based selection.

Split by **site plus recording date** parsed from the source paths, using a
fixed salted session hash (`shiftwm-real-split-v1`) and 70/15/15 train/validation/test
thresholds. Record the precise hashing implementation and counts in the data
manifest. No episode or session can cross splits. These are session-disjoint
splits; absence of repeated scenes or objects across dates is not guaranteed.
Unparseable grouping metadata must fail the audit, not fall back to frame-level
or random-window splits. No ground-truth state, reward, language annotation,
success flag or future image is an input to the world model.

Decode real images from the released JPEG streams. Store frames at native
indices 0,5,10,... and the intervening five chronological seven-dimensional
commands as a 35-dimensional action block. Preserve the full raw archive,
native action/state records, camera identity and incomplete-tail accounting.
There are no per-step timestamps in the inspected RLDS schema: report horizons
in native recording steps, without inventing precise physical time alignment.
Enforce the command semantics by comparing all actual `action` arrays with
their `action_dict` components. The first inspected record suggests Cartesian
pose plus gripper position, contrary to one publisher description; the final
dataset audit must resolve and explicitly record this before training.

Primary train/validation/test input is `exterior_image_1_left`. As a separately
labeled natural camera-transfer evaluation, use `exterior_image_2_left` on the
same held-out test episodes. Camera 2 is not used in fitting, normalization or
checkpoint selection. Wrist footage is preserved for provenance/inspection.
The primary study applies **no synthetic appearance corruption**.

## Representation and model

Use the released [DINOv2-small](https://huggingface.co/facebook/dinov2-small)
encoder, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, frozen throughout.
Resize the complete recorded image to 224x224 with bilinear antialiasing, then
ImageNet normalization. This is an explicitly recorded resize convention,
not the model card's center-crop preprocessing. Pool the final 16x16 patch grid
to 2x2 spatial cells, preserving 1,536 channels. Store immutable features;
compute feature and action means/sample standard deviations on train/camera 1
only, flooring standard deviations at 1e-5 and reporting constant dimensions.
GPU encoding uses bfloat16 computation and float32 pooling/storage.

Reuse the pinned LeWM causal action-conditioned transformer implementation:
hidden width 192, four blocks, six heads, head width 32, MLP width 768,
dropout 0.1. A learned projection maps frozen features to predictor coordinates;
a learned residual output maps back to the fixed DINO coordinates. The shared
zero-initialized residual head starts at persistence. Target coordinates never
move with the predictor. This is an architecture adaptation to real footage,
not an official DINO-WM reproduction or a new-method claim by itself.

Compare four modes with matching common initialization, data and optimization:

| Mode | Available context |
|---|---|
| Framewise | Causal image/action history and per-frame residual adapter |
| Constant dynamics | Episode observation context; learned shared dynamics context |
| ShiftWM real-video variant (ours) | Observation and transition contexts from causal support |
| Action-free | Same causal image history, with command information removed |

Report trainable and total parameter counts. Constant-input modules do not
have the same effective capacity as input-dependent ones. Real footage has no
paired canonical appearance or known dynamics intervention: do not transplant
the simulator alignment/consistency losses or claim identified physical factors.

## Training and selection

Twelve runs: four modes times seeds 0,1,2. Each completes 30 epochs with AdamW,
learning rate 1e-4, cosine minimum 1e-6, weight decay .01, gradient clipping 1,
batch size 128, training-window stride 2. Support has three observed frames and
two preceding action blocks. Recursively predict all five subsequent frames,
conditioned only on their commanded actions; no teacher-forced query images.
Minimize mean squared error in train-standardized frozen feature coordinates.
Train with bfloat16 autocast; compute validation losses in float32 with autocast
disabled. Record TF32 settings rather than implying full IEEE operations.

Choose the checkpoint with lowest complete validation all-query standardized
MSE, using stride 5. Test access requires a complete 30-epoch run and a passed
metadata/payload audit. Save best/last weights, optimizer/RNG, config, source
identities, feature statistics and data/encoder hashes for offline reuse.

## Evaluation and interpretation

Report fixed horizons 1,3,5 blocks (5,15,25 native transitions), and separately
10-block/50-transition extrapolation. Evaluate all complete test windows with
stride 5, aggregating first within each episode. Compare recorded future
features using raw-feature MSE, standardized MSE and cosine error. Add
persistence and constant-feature-velocity extrapolation using only support.
Inspect the learned action-free model and reversed-query-action diagnostic;
action reversal is an observational sensitivity check, not real intervention
ground truth. All compared methods receive identical support and recorded
commands where appropriate.

Report paired differences and uncertainty clustered by source session, with
training-seed variation. Keep camera 1 and camera 2, and in-range and extrapolated
horizons, separate. Include failures, ties and confidence intervals. Do not
claim a large gain, clinical benefit, general disentanglement or state of the
art before matched evidence supports it.

Qualitative media must use actual source frames, exact episode/frame IDs and
recorded commands. Any future retrieval, decoder rendering or model-generated
image must be explicitly labeled; it cannot replace recorded ground truth.
Select representative and discordant cases by a documented rule, retaining
the full evaluation inventory. Offline recorded trajectories cannot measure
the real outcome of an unrecorded action or demonstrate closed-loop robot
success. Keep simulation control evidence and real-video forecasting separate.
