# Potential evaluation workload and table grain

These are counts implied by the current campaign, not completed evaluations or
experimental findings. Source: `configs/world/full_campaign.json`, the
generated dataset manifests, and `scripts/run_evaluation_campaign.py`.

There are two environments and sixteen checkpoints per environment: one frozen
released model plus three training seeds for each of five modes: plain,
single-context, factorized, framewise, and factorized without pairing. Every
checkpoint is evaluated on the ordinary test split and
the extrapolation split. The worker waits for completed training before using
the immutable validation-selected best checkpoint.

## Closed-loop workload

| Evaluation family | Tasks | Test episodes | Extrapolation episodes | Total episodes |
|---|---:|---:|---:|---:|
| Frozen and trained world-model planning | 64 | 18,432 | 6,144 | 24,576 |
| Random-action and privileged replay diagnostics | 8 | 2,304 | 768 | 3,072 |
| Total potential closed-loop workload | 72 | 20,736 | 6,912 | 27,648 |

A test task contains 64 initial-state seeds across nine appearance–dynamics
combinations: 576 episodes. An extrapolation task contains the same number of
seeds across three combinations: 192 episodes. Diagnostic policies run once
per environment and split; they are not duplicated for each trained model.

At fifty native interactions per episode, the total upper bound is 1,382,400
evaluation interactions, including at most 276,480 common support-history
interactions. Early success reduces actual usage. This excludes dataset
collection and must not be reported as observed compute.

The model-planning part permits at most eight CEM replans after the paid
ten-interaction history. Therefore its upper bound is 196,608 solves. With
300 candidates, thirty CEM iterations and five predicted grouped transitions,
this corresponds to at most 1,769,472,000 candidate sequences and 8,847,360,000
predicted latent transitions. These are workload counts, not latency estimates;
batching and early termination materially affect elapsed time.

The main search uses thirty elites, matching the pinned upstream sampling
budget in `external/le-wm/config/eval/solver/cem.yaml`. The earlier development
diagnostic used 128 candidates, five iterations and sixteen elites. The main
sampling count is 14.0625 times that diagnostic's count per solve. The correction
was locked before any main test-set world-model CEM planning ran; forecasts and
random/replay controls do not use these CEM samples. Their historical, unused
128/5/16 metadata is preserved and is not evidence of a different control policy.

Main timing claims additionally require the worker's explicit
`--execution-scope dedicated_gpu_campaign` declaration. Unknown or shared
resource provenance in any resumed segment suppresses the efficiency claim.
Slurm job membership by itself does not establish exclusive GPU use.

## Forecast workload and table grain

There are sixty-four forecast result files: thirty-two checkpoints times two splits.
Length-eight windows sampled with stride five give twelve windows from each
65-observation physical trajectory. Consequently a full test file contains
6,912 condition-window records and an extrapolation file contains 2,304.
Across the current sixty-four files this is 294,912 forecast condition-window
records. Each record contains metrics at horizons one, three and five grouped
transitions, corresponding to five, fifteen and twenty-five native actions.

The **raw-record key** is:

`environment, checkpoint, split, initial_state_seed, dynamics_id, observation_id, window_start`

The **long-form metric key** additionally contains `horizon` and `metric`.
The four metrics are prediction MSE, persistence MSE, zero-future-action MSE
and squared prediction change when future actions are zeroed. All prediction
errors use the same immutable canonical-reference encoder coordinates.

The **per-checkpoint table grain** is:

`environment, method, training_seed, split, appearance, dynamics, horizon, metric`

Each cell reports a mean and a bootstrap interval clustered by initial-state
seed. The effective resampling clusters are the 64 initial-state seeds, not
the hundreds of overlapping windows. The same seed is shared across physical
conditions, so it also remains one cluster in all-condition averages. Frozen
models have one checkpoint, not three independent training seeds.

Method-level tables should first retain or summarize the three training seeds
explicitly. Do not present pooled windows from all checkpoints as independent
replicates. Report the held-out `(appearance=2,dynamics=2)` composition separately
from the all-nine-condition average. The latter includes conditions also seen
in training or development, on independently held-out physical trajectories.

## Worker I/O

Each worker now caches SHA256 digests by resolved path, file size and nanosecond
modification time. It reads a stable checkpoint's bytes once per process and
checks metadata on subsequent scans. A changed file is hashed again and still
must match the identity recorded in a completed evaluation. Hashing uses
one-megabyte chunks, avoiding large temporary byte strings. A metadata change
during hashing fails rather than caching a potentially inconsistent digest.
