# Exact replay and conditional-prediction diagnostics

The source archives support observed behavior comparisons, but not a retrospective reconstruction of CEM candidate rankings or claims that a particular inferred context caused a win. Original per-episode records store executed native action prefixes, native action counts, endpoint/support diagnostics and success; video NPZs store the shifted RGB goal and observations after support and every executed action block. Candidate actions/costs, full unused planned trajectories, latent predictions and context vectors were not archived.

## Source paths and bounded selection

- Original completed evaluations: `results/development_official_budget/{pusht,reacher}_{factorized,framewise,single}_s0/planning_development.json`.
- Original videos: the corresponding `videos/development-s{seed}-d1-o1.npz`, with `frames` and `goal_image`.
- Original simulator initialization/support/goal: `data/world/{pusht_relative,reacher}/episodes/development-s{seed}-d1.npz`.
- Reset, goal, support and execution semantics: `src/shiftwm/evaluate.py`, `src/shiftwm/generate.py` and the unchanged upstream environments under `external/stable-worldmodel/stable_worldmodel/envs/`.
- New standalone replay: `scripts/diagnose_qualitative_replay.py`; tests: `tests/test_qualitative_replay.py`.

The declared population is every policy-eligible outcome-discordant development task between ShiftWM and Framewise: all four ShiftWM-only successes and all nine Framewise-only successes. Replay includes all three methods on those 13 tasks (39 executions). This population is explicitly selected using outcome, so its summaries are not estimates of general benchmark performance.

## What exact replay establishes

No model or search is run. The replay uses the original reset and archived initial simulator state, constructs the goal in a separate environment, executes the original two five-action support blocks, then executes precisely the saved policy action prefixes in their original float32 dtype. It never fills a partial terminal block. Physical states are read after each native action without advancing the simulator; images are rendered at the original support/planning boundaries.

Every original shifted boundary frame and goal must match byte-for-byte. Canonical support images at native times 0, 5 and 10 and the canonical goal must also match the dataset archive exactly. Initial/terminal physical diagnostics, mixed-unit upstream distance, outcome, first-success time, eligibility, number of replans, and budget stopping must agree. Diagnostic thresholds are cross-checked against the upstream success result at every step. The output is usable only with top-level `status=complete` and every record marked `validation.status=exact` and `claim_eligible=true`. Hashes pin all source files, original results, videos, selected episodes, model packages and generated NPZs; input hashes are checked again after completion.

Per-native-step intermediate states are therefore **reconstructed by a verified replay**, not states originally stored in the video archive. A mismatch produces failed status rather than a plot-ready claim.

## Output contract

`results/qualitative_diagnostics/replay_diagnostics.json` contains `status`, `schema_version`, `selection`, `sources`, `expected_records`, `records` and `failures`. Each record provides environment, trajectory, seed, appearance, category, behavior mode, original record, NPZ path/hash, criterion metadata, validation and renderer details.

Each NPZ under `episodes/{environment}/{trajectory_id}-o{observation_id}/{behavior_mode}.npz` stores:

| Array | Meaning |
|---|---|
| `trace_native_times`, `task_states`, `simulator_states` | Initial state and state after every executed native action |
| `executed_native_actions` | Exact float32 support and policy actions; row k advances native time k to k+1 |
| `goal_task_state` | Archived physical goal used by the original scorer |
| `criterion_errors`, `criterion_margin`, `success_flags` | Exact per-component success operands; margin is their maximum threshold ratio |
| `native_times`, `canonical_frames`, `shifted_frames` | Observations at 0, 5, 10 and each saved policy-block endpoint, including final partial blocks |
| `goal_canonical_frame`, `goal_shifted_frame` | Separate goal render before/after the original appearance transform |

PushT uses the combined Euclidean position error over both blue pusher and gray block coordinates, strictly below 20 native pixels, AND wrapped T-block angle error strictly below pi/9. The native coordinate system is 512 by 512; rendered image pixels are 224 by 224. Reacher requires each raw, unwrapped joint error to be strictly below 0.05 radians. Its L2 norm is only a diagnostic and is not the threshold. `success_flags[0]` describes the initial state; the original evaluator starts checking stopping after the first executed native action.

## Additional model diagnostics and limits

The separate `scripts/diagnose_qualitative_prediction.py` can compare both original completed checkpoints on identical rolling three-frame histories and identical next full five-native-action blocks from these verified traces. Reconstruct floating shifted inputs from canonical uint8 images with the original transform; dividing archived quantized shifted frames by 255 would not reconstruct the exact original inputs. Canonical targets are scoring-only reference features, not policy inputs. Early terminal prefixes at 21 or 23 native actions are not padded or interpolated into nonexistent full blocks.

These conditional prediction errors can show whether one model represents the same realized transition more accurately. They do not establish which unrecorded CEM candidates were evaluated, their ranking, the counterfactual outcome of a different action, or a causal explanation for task success. New instrumented evaluation would be needed for those claims and would need a separate source-pinned protocol.

## Validation before allocation

Nine tests passed, including complete real-simulator pixel-exact replays of PushT ShiftWM-only seed 2031024 and Framewise-only seed 2031000, correct strict/unwrapped joint and combined-position thresholds, source mutation rejection, altered-frame and endpoint rejection, and rejection of an added action after first success. Reacher full validation requires the matching NVIDIA EGL renderer and is performed in the bounded GPU allocation, jointly with conditional prediction diagnostics.

## Completed allocation and independent output verification

Approved Slurm job **200037**, partition `gpu`, node `gpu-04`, completed with exit code 0 in **39.10 seconds**. It requested one GPU, four CPU cores and 32 GB RAM for at most 30 minutes; the two existing study jobs were unchanged. Replay finished **39/39 records, zero failures**. Conditional prediction finished **70 paired transitions / 140 model predictions**, with eight anchors excluded because no full next action block existed. All 310 replay input hashes and all 39 output NPZ hashes were independently rechecked after completion.

The following physical comparisons are verified replay states, not model predictions:

| Case and method | native10 | native15 | native20 | actual terminal |
|---|---|---|---|---|
| PushT 2031024, ours: combined px / angle rad | 284.475 / .97725 | 192.198 / .63680 | 51.757 / .45639 | t21: 16.258 / .15949 |
| PushT 2031024, Framewise | 284.475 / .97725 | 211.904 / .59630 | 106.477 / .30641 | t50: 457.022 / 1.38357 |
| Reacher 2031004, ours: raw joint0 / joint1 rad | .663330 / .019588 | .481310 / .238328 | .257536 / .300298 | t23: .010717 / .046594 |
| Reacher 2031004, Framewise | .663330 / .019588 | .450400 / .067700 | .590556 / .033577 | t50: .568675 / .056548 |

These cases do **not** show uniform componentwise superiority. At PushT native20 Framewise has the smaller angle error and already satisfies the angular tolerance; ours has roughly half the combined positional error, then meets both criteria at native21. In Reacher, Framewise has smaller errors in both joints at native15, and smaller joint1 error at native20. Ours ultimately meets both tolerances simultaneously at native23. Captions should describe this observed joint-criterion attainment, without claiming that context inference or a particular unarchived CEM ranking caused it.
