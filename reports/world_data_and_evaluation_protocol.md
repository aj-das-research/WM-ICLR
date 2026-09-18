# World-model data and evaluation protocol

This is the implemented protocol, not an assertion of benchmark scores. The
implementation reuses pinned `external/stable-worldmodel` environments and its
unmodified CEM solver. It defines a controlled adaptation benchmark; it is not
an exact replication of the original LeWM planning evaluation.

## Dataset identity and controls

Use `data/world/pusht_relative` and `data/world/reacher`. The earlier
`data/world/pusht` archive used absolute-target actions before official action
statistics became available. It is retained for audit and excluded from the
main campaign. Official PushT actions are relative displacements. The corrected
collector uses upstream relative control with an action scale of 100 pixels;
it follows the near-block sampling and clipping rules in upstream WeakPolicy.
Reacher controls are normalized torques. Both observations group five native
two-dimensional actions into ten-dimensional model actions. Reacher retains
the upstream two simulation-step action repeat inside each native step.

Each environment has 256 training, 32 validation, 32 development and 64 test
initial-state seeds. There are three physical conditions in the first three
splits and four in test: 1,216 episodes, each with 64 grouped transitions and
65 stored canonical RGB observations. This is 389,120 native interactions and
79,040 canonical images per environment. Appearance conditions are applied
deterministically to the same image and therefore do not require additional
simulator interactions. The test extrapolation uses the fourth condition.

Physical changes are real simulator interventions:

| Environment | Default | Condition 1 | Condition 2 | Extrapolation |
|---|---:|---:|---:|---:|
| PushT Pymunk space damping | 0 | 0.15 | 0.65 | 0.95 |
| Reacher arm and finger density | 1000 | 650 | 1350 | 1500 |

The observation changes are invertible-range RGB gain/bias transforms with
canonical, warm, cool and dim conditions. They are photometric interventions,
not new camera viewpoints or real-world acquisition domains. Exact channel
values are in `shiftwm.data.APPEARANCES` and each manifest.

Training and ordinary validation use seven of the nine combinations of the
first three appearance and physical conditions. `(appearance=1,dynamics=1)` is
reserved for development; `(2,2)` is the held-out test composition. The test
split also measures all nine combinations on independent initial states.
Thus the all-condition test average is not itself a wholly unseen-composition
score. Report the `(2,2)` result separately. Extrapolation consists of `(3,0)`,
`(0,3)` and `(3,3)`.

Trajectory seed families are disjoint across train, validation, development and
test. Different physical conditions share initial-state seeds within a split.
Sliding windows and all appearance versions inherit their physical
trajectory's split. Primary and paired training views are both restricted to
the seven training combinations. Although frozen feature caches contain all
appearance conditions, the dataset loader never samples forbidden training
combinations. Canonical renders are privileged training targets and offline
forecast targets; deployment receives shifted observations only.

## Forecasting

Use three chronological support observations and their two executed action
blocks to infer context. Predict one, three and five grouped transitions into
the future autoregressively. Compute MSE against the immutable released encoder
on canonical renders, in the same coordinate system for all methods. Report
persistence, zero-future-action prediction error and the prediction difference
caused by zeroing future actions. These diagnostics help distinguish useful
action-conditioned prediction from a slowly changing representation prior.

The forecasting loader uses length-eight windows and stride five. No labels,
future frames or simulator parameters enter context inference. Offline targets
are used only for scoring. Bootstrap by initial-state seed, preserving the
correlation among overlapping windows, appearances and physical conditions.

## Closed-loop planning

The environment starts from a saved test state. The goal is a real state five
grouped transitions after the end of the three-frame support history. A
separate simulator copy renders that state under the same task-goal settings;
only the resulting shifted RGB goal image reaches the model. Goal construction
does not perturb contact caches in the controlled environment.

The support history costs ten native interactions. Those interactions count
against a total fifty-interaction budget. Main CEM uses 300 candidate sequences,
thirty optimization iterations, thirty elites and a horizon of five grouped
transitions. Search variables use the checkpoint training action z-scores: a
unit Gaussian is transformed by the saved action mean and standard deviation.
Only the resulting native controls are clipped to physical action bounds;
z-scores and warm starts are not restricted to one standard deviation. It
executes one group and replans. Every method receives the same
seeds, history acquisition, goal states and planning budget. The world model
receives only observation features, past executed controls and proposed future
controls. Hidden simulator states supply initialization and evaluation scores.

The main sampling budget matches the pinned upstream
`external/le-wm/config/eval/solver/cem.yaml` and was locked before any main
test-set model-planning evaluation. Historical development diagnostics used
128 candidates, five iterations and sixteen elites and remain separately
labelled. Random/replay policies never invoke CEM; their existing sampling
metadata is unused and does not require rerunning those controls.

Terminate on the first successful native action. This avoids accidentally
auto-resetting a dm_control episode inside the remainder of an action block.
Record successes during the common support rollout separately. The
`eligible_summary` evaluates episodes requiring at least one policy decision.
For PushT, also prespecify a manipulation stratum: block displacement from the
post-support state to the goal is at least 20 pixels, or wrapped orientation
change is at least pi/9. This uses upstream success tolerances and is defined
before seeing method outcomes.

Preserve upstream task-success criteria. The upstream PushT state distance
mixes pixels, radians and agent velocity; it is not object translation error.
Report block translation, block angle and agent position errors separately.
Reacher additionally reports wrapped joint-angle error while preserving its
upstream unwrapped qpos-match success rule. Actual per-replan latency is total
solver time divided by total replans, excluding zero-replan episodes from the
denominator. Episode records preserve action blocks and interaction counts.
Efficiency tables require explicit dedicated GPU provenance from the campaign
worker for every resumed segment. The worker defaults to unspecified resource
scope, and a Slurm job ID alone cannot make an evaluation eligible for a main
efficiency claim. Shared development timings stay outside those tables.

Random-action and privileged future-action replay diagnostics use the same
paid context history. Replay is a reachability check, not a deployable baseline.
Main planning never receives recorded future actions. Each full test planning
evaluation comprises 576 episodes (64 seeds x 9 combinations); extrapolation
comprises 192 episodes (64 x 3 combinations).

## Execution and provenance

`scripts/run_evaluation_campaign.py` derives its grid from the training
campaign. It uses only checkpoints from completed training runs, selected by
ordinary held-out validation. Frozen packages use the released weights and
official training action normalization. Completed results record checkpoint,
data manifest and evaluator hashes. Planning resumes only when these and the
full protocol match; each completed episode is saved atomically. Per-task locks
are held through child evaluation processes to prevent duplicate execution.

The full campaign grid contains sixty-four forecasts, sixty-four learned/frozen planning
evaluations and eight random/replay diagnostics. Forecasts are prioritized,
followed by diagnostics and model planning as their inputs become available.
All results remain unreported until actual output files have status `complete`.
