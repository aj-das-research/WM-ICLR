# World-model execution record

Implementation started 18 September 2026. Paths are relative to the project root.
Run logs and JSON summaries are authoritative for status and measured results.

## Data and sources

Sources: `references/world_upstream_revisions.json`,
`data/manifests/upstream_sources.json`, and
`data/pretrained/<environment>/provenance.json`. The isolated Python 3.11/CUDA 12.4
environment is recorded in `requirements.lock.txt`.

Official archives recover released-model action normalization. This exposed a
wrong assumption in the initial PushT collector: official actions are relative
displacements (std approximately 0.208), rather than absolute pixel coordinates.
No models were trained on the incorrect corpus. The corrected corpus is
`data/world/pusht_relative`; `data/world/pusht` is an excluded audit artifact.

```bash
python scripts/prepare_action_stats.py \
  --archive data/upstream/pusht_expert_train.h5.zst --output data/upstream/pusht
python scripts/prepare_action_stats.py \
  --archive data/upstream/reacher.tar.zst --output data/upstream/reacher
python -m shiftwm.generate --env pusht --output data/world/pusht_relative --workers 6
python -m shiftwm.generate --env reacher --output data/world/reacher --workers 6
python -m shiftwm.cache_features --data data/world/pusht_relative \
  --checkpoint data/pretrained/pusht --output data/features/pusht_relative --device cuda --batch-size 256
python -m shiftwm.cache_features --data data/world/reacher \
  --checkpoint data/pretrained/reacher --output data/features/reacher --device cuda --batch-size 256
```

Statistics use sample standard deviations as in pinned upstream training and
also record population standard deviations used by upstream evaluation. Archive
SHA256, action ranges, and row counts are recorded. An explicitly named training
partition is preferred; exact source files are listed in the statistics JSON.
Statistics are never recomputed on our held-out test trajectories.

## Prespecified training

30 runs: PushT/Reacher × five trainable variants × seeds 0/1/2. The independent
implementation review added framewise calibration and unpaired factorization
before inspecting planning results. Plain fine-tuning remains an unaligned
diagnostic; it does not support the main planning claim. Each run traverses every
training window for 30 epochs with batch 128, sequence 8, stride 1,
AdamW learning rate 5e-5, weight decay 1e-3, cosine decay to 1e-6, gradient clipping 1.
Every epoch evaluates the complete validation split. Checkpoint selection uses
validation prediction loss only; final test results do not choose checkpoints.

The pretrained visual encoder/projector stay frozen. Action encoder, predictor,
prediction projector, and active context/FiLM modules train. Immutable canonical
targets preserve a common coordinate system. Canonical images and factor IDs
are training supervision; inference context receives neither these nor future
query observations. Training is teacher-forced one-step prediction; evaluation
also tests autoregressive horizons 1/3/5.

Each run saves best/last complete model packages, optimizer/scheduler/RNG,
`metrics.jsonl`, `run_config.json`, and `training_summary.json`. Exact mid-epoch
continuation is tested with stochastic dropout. Active context parameters are
259,368 shared vs 259,776 factorized. Plain fine-tuning has fewer parameters;
all methods receive the same data and optimizer budget.

## Evaluation

64 independent initial seeds per physics setting. Methods share tasks, initial
states, appearance, warm-up actions, planner settings/seeds, and interaction
budgets within each comparison. Ten native steps acquire the three-frame context
and count toward the 50-step budget. CEM uses 300 candidates, 30 iterations, 30 elites,
five grouped-action horizon, and one grouped action before replanning.
Candidate controls are sampled in the original standardized action coordinates,
then converted using official action statistics and bounded in physical units.

Shared-context successes are separated from successes on policy-eligible tasks.
PushT also reports a block-movement subset. Random-action and replay-oracle
controls measure difficulty/reachability. Simulator states initialize/score
episodes, never enter the policy. Intervals cluster repeated conditions by seed.

These shifted tasks and bounded-action planner settings are a new controlled
benchmark, not a reproduction of published success numbers. Goals are paired
across methods within conditions but can differ across physics conditions;
between-condition scores alone do not identify a causal physics effect.

The main search budget matches the pinned upstream CEM configuration
(`external/le-wm/config/eval/solver/cem.yaml`) and was fixed before any
world-model-policy main-test evaluation. Historical complete development
evaluations used 128 candidates, 5 iterations, and 16 elites: factorized seed 0
achieved PushT 1/32 overall (0/31 excluding common-support successes), and Reacher
11/32 overall (9/30 excluding common-support successes). These do not establish
a benefit over a baseline. Shared-GPU timing is excluded from efficiency claims.
A separate dedicated comparison evaluates frozen, framewise, shared-context, and factorized
seed-0 models on all 32 development tasks per environment at the upstream budget;
outputs live in `results/development_official_budget/`.

## Allocations

Corrected collection 199376, corrected feature extraction 199378. Training workers
199379 (`gpu`), 199380 and 199381 (`ws-ia`), with the last dependent on feature
extraction. Workers claim distinct runs and checkpoint before walltime expires.
`runs/jobs/<job>/record.json`, `runs/world/campaign_state/`, and Slurm hold live
state; this document is not proof of completion. Control workers 199393, 199394,
and 199395 follow the original training workers. Dedicated development comparison
199416 follows training worker 199379. Goal-calibration diagnostic job 199461
follows successful development completion and precedes GPU control worker 199393.
The full evaluation allocation manifest is recorded in
`reports/evidence/full_evaluation_allocations.json`; earlier pending evaluation
workers may be superseded there. The 136-task evaluation manifest
includes 27,648 potential closed-loop episodes plus fixed-coordinate forecasts.
The replacement main workers are 199428--199439 in four continuation waves.
All wait for successful completion of development job 199416 and their preceding
training/evaluation stage. Two full-forecast workers, 199446 and 199447, follow
the workstation control workers and precede those slots' main workers. These
produce prediction results while dedicated development planning runs elsewhere.
The main worker time allowance is at most 90 GPU-hours (94 allocated GPU-hours,
including job startup/headroom); the two early forecast jobs reserve at most
four additional GPU-hours. Completed tasks are reused rather than repeated.

The implementation audit proposes an optional scientific development gate; it
has not been adopted as an automatic threshold. Slurm dependencies validate
process completion only. Any later algorithm revision must retain the original
results and receive distinct run identities, with model choices based on
development data rather than the final test results.

The isolated goal-calibration diagnostic evaluates the same 32 development tasks
for frozen, framewise, shared-context, and factorized seed-0 checkpoints in both
environments. It compares actual corrected goal features and recorded-action
terminal predictions with immutable canonical targets. Those targets remain
diagnostic supervision and never enter the deployable planner. The diagnostic
does not modify models, training, or the main evaluator; its one-hour allocation
is an upper bound, not measured GPU consumption. See
`reports/goal_calibration_diagnostic.md` and `scripts/run_goal_calibration_campaign.py`.

The full training-action audit in
`reports/evidence/training_action_distribution.json` covers all 768 physical
training episodes and 245,760 native action vectors per environment, without
duplicating appearance variants or overlapping windows. PushT action standard
deviations are [0.503869, 0.503598], versus [0.208467, 0.206749] in the released
expert training data. Reacher [0.577401, 0.577124] closely matches its upstream
statistics. This quantifies the collection-distribution difference; it does not
establish the cause of planning failures or change checkpoint normalization.

The first auxiliary evaluation launch incorrectly inherited multiple Slurm tasks
and exhausted its small host-memory request. Its job step alone was terminated;
training was unaffected. The corrected single-task launch completed all 12 frozen
forecast and random/replay control tasks. Replay controls reached all 768 goals
per environment across test and extrapolation. These controls verify replay and
reachability, not learned-policy quality.

## Interpretation

At the 18 September 2026 11:34 UTC checkpoint, all 30 configured training runs
have completed all 30 epochs, and all 30 local release packages have passed
loading verification. Both 50-task original-protocol upstream checks completed:
PushT 46/50 and Reacher 39/50. The original main evaluation workers 199428,
199429 and 199430 now run across the three entitled GPU slots. Forecast and
planning outputs retain separate completion records; training completion alone
does not mean the full experiment grid is finished.

The full goal-only intervention is also complete: PushT 0/31 policy-eligible
successes versus fixed/framewise 1/31; Reacher 12/30 versus fixed 6/30 and
framewise 11/30. Reacher hybrid-minus-fixed is +20 percentage points with an
exploratory paired trajectory-bootstrap interval [0,+40]; hybrid-minus-framewise
is +3.33 points [-20,+26.67]. These are development results at training seed zero,
not confirmatory evidence of a general factorization benefit. The reporter
`scripts/summarize_goal_intervention.py` validates source/protocol/support pairing
and writes the diagnostic tables incorporated by the paper watcher.

The full official-budget development comparison (job 199416) and all eight
goal-calibration diagnostics (job 199461) are complete. The four PushT modes
each reach 2/32 raw successes, with one common support-only success (1/31
policy-eligible). Reacher frozen/shared/factorized/framewise yield 5/8/8/13 raw
successes out of 32, respectively; excluding the same two support-only successes
gives 3/6/6/11 out of 30. These are seed-zero development measurements and do
not demonstrate a factorization benefit. See `reports/development_diagnosis.md`.

The diagnostic verifies exact archived/fresh Reacher pixel agreement on the
allocated NVIDIA renderer. Factorized Reacher goal-calibration MSE is 0.796745,
while recorded-action terminal prediction MSE is 0.011741 against the same
canonical goal. A separately versioned development intervention will test learned
goal correction with a fixed factorized predictor; the original results and main
protocol remain intact. Canonical goals are never policy inputs.

Full released-data upstream reproduction is queued as 199469 (PushT) and 199470
(Reacher), 50 official starts each, in a separate dependency environment and a
compatible historical simulator checkout. Neither source pin is claimed to be
the exact author training environment. The goal-only intervention job 199477
was moved to `ws-ia` after verified completion of worker 199394's five tasks.
The two upstream reproductions on `gpu` follow it and precede main GPU worker
199428; each new diagnostic job has a one-hour allocation limit,
not measured consumption. Workstation workers continue their existing schedules.
The original development job has completed successfully and aged out of Slurm
history; its completion record and all eight validated results preserve that gate.

Workers 199394 and 199395 were released only after all five tasks owned by each
were complete and a node-local process audit found no active training child.
Their coordinator processes were waiting for the final task on 199393. Cancelling
these idle allocations preserved every task checkpoint and completion record,
and allowed intervention 199477 and forecast worker 199447 to use the GPUs.
Forecast worker 199446 follows 199477. These allocation cancellations are not
failed training runs. New training workers can use `--exit-when-not-ready` to
release an allocation after a full scan finds no claimable task; default waiting
behavior and task-ownership protection are unchanged.

Separating observation and physical dynamics is not itself novel; related work
includes ContextWM and contemporary adaptation methods. This campaign tests a
concrete compositional-transfer hypothesis. Forecast improvement alone does not
establish useful planning or publishable novelty. Medical/general applicability
and conference acceptance remain unestablished.
