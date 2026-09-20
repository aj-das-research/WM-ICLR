# Next development work after the reserved IWS implementation

Read-only audit, 20 September 2026. No held-out or reserved payload was opened,
no model was selected or changed, and no job was submitted. Original and new IWS
registrations remain untouched. The existing session authorizes development
work; the recommendations below still need a separate, source-bound diagnostic
namespace before execution, not another user-permission round.

## Recommended order

| Work package | Question answered | Existing material | Resources and conditional elapsed time |
|---|---|---|---|
| **1. Complete the registered wide-gain drone geometry check** | Did the larger observation-gain range actually restore goal separation, and does that track the modest planning change? | All six wide and six matched narrow transformer checkpoints; the original fixed 16 development goals/120 pairs and cached support features | CPU 2 threads, 8 GiB, no simulator/GPU. Allow 30–60 minutes for a small new diagnostic wrapper and source/identity tests, then an estimated 2–10 minutes execution and 15–30 minutes review. New execution time is unmeasured; request a 15-minute CPU allocation. |
| **2. Tissue native-failure and chosen-action calibration audit** | Are failures primarily missed goals, invalid commands, or poor correspondence between the planner's next-step cost and realized progress? | All 18 original models' completed development JSON records, eight fixed goals, recorded decision/native-step metrics and existing trace hashes | CPU 1–2 threads, 4 GiB, no inference/simulator/GPU for the first pass. Allow 20–40 minutes implementation/review plus under 5 minutes execution. Reading the 18 JSON records and counting native outcomes took 0.24 seconds in this audit; this is not a complete validation timing. |

Neither work package changes a checkpoint, objective, success criterion, task
roster or final-test promotion rule. They fill an explanatory gap before another
training campaign. Their completion is **not** permission to label the final-test
cells complete.

## Why these are the useful next measurements

The original 36-model development study remains mixed: drone transformer
ShiftWM scores 5/24 task–seed instances against 4/24 for each control, whereas
its GRU scores 4/24 against 6/24 Framewise and 7/24 Constant dynamics. Tissue
transformer ShiftWM scores 1/24 against 2/24 for both controls; tissue GRU scores
3/24 against 2/24. No original paired interval has a strictly positive lower
bound. There are eight physical goals per domain, repeatedly evaluated across
training seeds—not 24 independent goals.

The wide-gain drone follow-up reaches 7/24 against 5/24 for original ShiftWM
and 4/24 for wide Constant dynamics. Its respective paired differences are
+8.33 pp [−12.50,+29.17] and +12.50 pp [0.00,+37.50]. These are the same
previously observed development tasks. `reports/geometry_revision_protocol.md`
explicitly requires repeating the fixed 16-goal/120-pair geometry diagnostic
before interpreting the mechanism; `reports/completed_extension_results.md`
explicitly says that this measurement is absent.

The demonstrated original observation-correction limitation is precise: its
fixed-context gain in [0.9,1.1] can expand squared goal distances by at most
1.21×. The original warm features have only about 17.4% of canonical mean
pairwise separation. This capacity bound is real, but it does not prove why a
particular rollout failed. The wide adapter allows [0.25,4] gains; whether the
trained models use that capacity is the missing measurement.

Drone settling is not the leading recorded failure: 111/114 failures never
enter the 4 cm XY region. Only three enter it but fail the speed test. The
completed four-task action-ranking probe also finds that ShiftWM and Constant
dynamics pick the same candidate on all four transformer tasks. Repeating that
exact probe or simply adding a braking penalty would not address the largest
unresolved issue.

## Work package 1: exact execution contract

Keep the original selection from
`reports/evidence/extension_mechanism_probe.py`: all 48 drone development
trajectories at fixed cached starts 0/16; the warm/gain-0.75 subset provides 16
endpoint goals and 32 support windows. Hold each support context fixed across
all 16 goals, as the original analysis does. Retain all 120 goal pairs; do not
treat them as 120 independent samples.

Compare `runs/extensions/drone_transformer_{factorized,constant_dynamics}_s{0,1,2}/best`
with the six identically named `runs/geometry_revision/.../best` packages.
Use the unchanged `shiftwm.extensions.checkpoint.load_package` for original
models and `shiftwm.extensions.geometry_revision.load_geometry_package` for
version-2 wide models. Validate full 30-epoch completion and the selected package
hashes against `reports/completed_extension_results.json`. Reuse only the
existing `data/features/drone_v1` development cache and its matched physical
metadata for diagnostic distances, never as model inputs.

Report per model/seed: pairwise corrected-goal MSE; fraction of canonical
separation recovered; Spearman association with squared XY goal distance;
canonical goal-calibration MSE; attained observation-gain distribution. Retain
the original recorded/zero-action and fixed-roll context-shuffle descriptors if
reusing the full probe. Pair narrow/wide results by training seed and support
context. Do not infer closed-loop causation from these descriptive quantities.

Implementation is small but **not currently a complete CLI**: the old probe
hard-codes original seed-zero packages and overwrites its original report. Do
not execute it unchanged for this task. Put a thin wrapper in a new namespace
that imports the existing loaders and copies the documented calculation, with
new output paths and exact source bindings. `scripts/extensions/evaluate_geometry.py`
only repeats completed forecasting/planning; it does not supply this missing
geometry measurement. No new training, physics rollout or data collection is
required.

## Work package 2: tissue audit contract and current clues

Use only the 18 `planning_result` paths for domain `surgery` enumerated in
`reports/completed_extension_results.json`; all 18 JSON hashes were rechecked
against that report in this audit. They live under
`results/extensions_v1/surgery_*/development/planning/results.json`. Preserve
all eight fixed goals, all three model variants, both backbones and all three
training seeds. Separate common-support from planned actions.

The saved records contain 144 task–seed instances, 12 successes and 132
failures. All 132 failures never reach the native 2 mm region. Of those, 41
stop on an invalid action and 91 exhaust the budget; there are no unstable
native deformation flags. The 24,369 executed native steps include 41 invalid
steps. Of 4,563 complete five-command planning decisions, 1,816 reduce native
distance. These counts describe saved development executions; they establish
neither a unique failure mechanism nor independent-sample significance.

The first new report should recompute native success, stop reasons, closest
approach, command validity and progress per task/method/seed. Adapt the **logic**
of `reports/evidence/extension_planning_probe.py` into a new report namespace;
that old script is drone-specific and uses `goal_distance_m`, whereas tissue
uses `distance_m`. Align `predicted_next_goal_mse` with physical distance after
exactly five executed commands. Report partial/invalid terminal blocks
separately, retain them in the failure denominators, and never compare a
25-command terminal prediction against a five-command realized transition.
The existing `shiftwm.extensions.evaluate.distance` and `valid_success` helpers
state the domain-specific definitions. Chosen-action correlations are
**descriptive calibration, not candidate-ranking accuracy**.

If this audit implicates ranking rather than invalid-command handling, the
next intervention would be a separately frozen tissue counterpart of the
existing matched candidate replay. That runner does not yet exist: the current
`scripts/extensions/diagnose_action_ranking.py` is explicitly drone-only and
seed-zero. Do not redirect it to tissue or call it a ready tissue experiment.
Before any such replay, freeze the candidate library, original eligible task
roster, native safety rules and exact support/goal-marker replay checks. This
is a later conditional step, not a current job or promised result.

## Work that should remain gated

- Drone/tissue final tests: the registered requirement is improved closed-loop
  success over both within-family controls, consistently across seeds/domains
  with paired uncertainty. Current evidence does not meet it. No promoted final
  roster is frozen. Do not run default eight-task test evaluation and call it a
  completed final campaign.
- New branched-data training: the 96-context/3,072-branch drone training dataset
  is complete, with 2,821 safe full-horizon branches and 251 retained escapes.
  Its protocol explicitly does not register a model, loader, loss or checkpoint
  comparison. A branch-ranking loss must be separately specified and matched
  across controls. Do not train it opportunistically as a continuation of the
  wide-gain study.
- Core PushT/Reacher final results already exist; repeating them fills no gap.
- Matched external recorded-video SOTA remains a separate input/representation
  problem, not something solved by these simulator diagnostics.

Source anchors: `reports/domain_extension_protocol.md`;
`reports/completed_extension_results.{md,json}`;
`reports/geometry_revision_protocol.md`;
`reports/extensions_mechanism_diagnosis.md`;
`reports/evidence/extension_mechanism_probe.py`;
`reports/evidence/extensions_mechanism_probe.json`;
`reports/drone_failure_modes.md`;
`reports/evidence/drone_action_ranking.{md,json}`;
`reports/drone_branched_training_{protocol,completion}.md`;
`src/shiftwm/extensions/{evaluate,checkpoint,geometry_revision}.py`.
