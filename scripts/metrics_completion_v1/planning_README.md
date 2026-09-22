# Current spatial simulator planning v1

This new study trains the current DINOv2 4x4 spatial architecture on released
PushT/Reacher RGB, then executes its plans in the unchanged historical
stable-worldmodel backend. It does not reuse the historical context-model
forecast features, learned weights, or success results.

Before payload access, `planning_register.py --freeze` fixes a seed-173 whole
episode split: 1,000 training, 100 development and 100 test episodes per task.
Only offsets, lengths, episode IDs and native frame indices are used to select
episodes. Complete HDF5 bytes are subsequently hashed for acquisition
provenance, without decoding RGB/action/state arrays. Test semantic payloads
remain closed until all 24 training runs finish.

The frozen DINOv2-small encoder receives actual 224x224 RGB with ImageNet
normalization. Its 16x16 patch grid is spatially averaged to 4x4; each of the
16 tokens retains all 384 channels. We never reshape the historical 192D LeWM
embedding into artificial spatial tokens. Five recorded native actions form
one 10D action block, aligned from one stored image to the next. Normalization
is fit only on training episodes and shared over patch positions per channel.

Bounded transport, autoregressive, bounded additive and no-tanh transport use
the exact existing spatial/component implementations, three seeds each, two
tasks, 30 full epochs, AdamW, BF16 training and FP32 development selection.
The only dataset-specific architecture field is action_dim=10. Selection is
equal-episode mean standardized MSE over every five-step development window.
The original atomic training engine provides strict reload and epoch-boundary
resume. A 90-minute boundary requests Slurm requeue, up to five restarts;
incomplete runs cannot satisfy the all-24 completion gate.
Planning similarly checkpoints each completed case and requests continuation
after 90 minutes at a case boundary, with the same five-restart cap. Requeue
submission failures or exhaustion fail the allocation; partial cases never
become complete results. Independent execution receipts preserve hostname,
GPU, software version, wall time and scheduler identity for every attempt.

Each paired planning case has three real observations acquired with ten fixed
recorded support controls. These controls count against the total budget of
50 native calls. CEM uses 128 samples, eight iterations and 16 elites, horizon
five action blocks, executing five blocks then at most three on a second
replan. These reduced search settings are shared across arms; they are not
the original LeWM 300x30/top30 budget. Candidate controls are optimized in the
common training action coordinates and clipped after conversion to native
[-1,1] controls. Recorded support controls are retained verbatim, not clipped.

Only RGB, executed controls and a goal image enter the model/cost. Simulator
state is used to restore paired starts, construct the desired goal image and
score outcomes. PushT's upstream initializer advances physics once; we encode
the actual post-restore render. The separate desired-goal renderer restores
the requested pose exactly without advancing it. Reacher restoration retains
qpos/qvel and calls the unchanged physics.forward implementation.

The second solve has horizon three when only 15 native calls remain; it never
optimizes a terminal goal beyond the remaining control budget. Both future
horizons use the same trained predictor and unchanged cost arithmetic.

Primary success is the native upstream criterion: PushT concatenated
pusher/block XY distance <20 pixels and circular block-angle error <pi/9;
Reacher each unwrapped joint error <0.05 rad. Any-call, endpoint and
support-only success remain distinct. Block polygon IoU is a PushT secondary
diagnostic, not a replacement success threshold. Every native state/action,
RGB frame and success value is saved locally, alongside physical endpoint
errors, candidate counts, actual and failure-penalized calls, wall time and GPU
peak. The online wall timer includes encoding, context, candidate optimization,
simulation, rendering and scoring; it excludes environment construction and
trace serialization. No outcomes are dropped because a method fails.

Commands (root owns scheduling; no automatic submission):

```bash
.venv/bin/python scripts/metrics_completion_v1/planning_register.py --freeze
# Independent source_review.json must bind the resulting registration.
sbatch scripts/metrics_completion_v1/planning_cache.slurm pusht
sbatch scripts/metrics_completion_v1/planning_cache.slurm reacher
# After both cache/profile jobs pass, using the actual allocated quota:
sbatch --array=0-23%2 scripts/metrics_completion_v1/planning_train.slurm
# Only after all24 full runs; the evaluator independently rechecks the gate:
sbatch --array=0-23%2 scripts/metrics_completion_v1/planning_evaluate.slurm
# Submit with afterok on all evaluation arrays; all24x100 rows are rechecked:
sbatch scripts/metrics_completion_v1/planning_finalize.slurm
```

The initial cache jobs profile disposable fresh models on training windows
only (three warmup plus five measured full B128/H5 optimization steps per
arm). These are resource measurements, not study training or accuracy scores.
The existing isolated simulator Python environment is used for planning;
the primary Python environment lacks historical loguru/stable_pretraining.
Before cache extraction, a training-only action/restoration audit verifies the
native row convention and relative-action interface, reports actual replay
deviations, and checks PushT polygon self-IoU. This audit does not choose any
hyperparameter or access test data.
