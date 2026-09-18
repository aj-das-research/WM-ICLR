# Planning correctness and scientific audit

Date: 2026-09-18. Scope: read-only inspection of current code, completed training logs, and already completed evaluations. No models were run, simulators stepped, jobs submitted, or runtime code changed for this audit. The small action-distribution calculation below reads existing arrays only.

**Conclusion:** no concrete action-normalization, history-ordering, action-repeat, or goal-restoration bug was found in the current path. The poor low-budget development control results are real, but their cause is not established. Residual goal-embedding calibration error is the strongest model concern; the PushT recursive prediction error and task/distribution differences are also material. Increasing the CEM budget is a necessary comparison, not evidence that planning will become useful.

## Source identity

- LeWM: `external/le-wm`, revision `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Stable World Model: `external/stable-worldmodel`, revision `4821c8e6a3f0f83b7e6a80da3a757e026ea9026b`.
- `src/shiftwm/evaluate.py` SHA256: `3ea56985014cc3277af534ad87f788104671f36a8a2bd3f425667a94d30ee3bf`.
- `src/shiftwm/model.py` SHA256: `2f783caf8eed699a8d024f689aa956efd126a2c8b6c67aea23e4ff5dfb8536f8`.
- `src/shiftwm/generate.py` SHA256: `4a9651964cc328b7c6b3d1bf0f6ddafe4a5d0ab2f24d70f1faf5485941b2a631`.

Line references below refer to these inspected files. Campaign-runner changes are outside this audit. The completed development results use the same evaluator hash above.

## Evidence already available

Source files: `results/development/{pusht,reacher}_factorized_s0/both_development.json`. These are 32 development seeds in condition `(observation=1, dynamics=1)`, with 128 candidates, 5 CEM iterations, 16 elites, 5 grouped future steps, and a 50-environment-step budget including 10 support steps. They are not the new 300/30/30 comparison.

| Development result | PushT factorized s0 | Reacher factorized s0 |
|---|---:|---:|
| Selected training epoch | 4 | 30 |
| Raw success | 1/32 | 11/32 |
| Success during support collection | 1/32 | 2/32 |
| Policy-eligible success | 0/31 | 9/30 |
| Forecast MSE at grouped horizon 1 | 0.174831 | 0.007368 |
| Forecast MSE at grouped horizon 5 | 0.503772 | 0.012391 |
| Zero-action forecast MSE at horizon 5 | 0.996099 | 1.518980 |

The eligible PushT cases all require block manipulation under the current prespecified stratum. Their mean block translation error increases from 203.54px after support to 231.77px finally, and mean agent position error increases from 193.59px to 333.64px. These physical-unit diagnostics are more interpretable than the mixed-unit full-state distance. Reacher's mean eligible joint-angle norm decreases from 0.6754 to 0.4735 radians; 21/30 cases improve that distance.

Existing replay-oracle results in `results/world/{pusht,reacher}_frozen_s0/planning_{test,extrapolation}_replay_oracle.json` reach **576/576 test and 192/192 extrapolation goals in each environment**, including support successes. This is evidence that recorded goals are reachable and state/action replay works to the success tolerance. It is not a pixel-exact rendering check, a learned-model result, or proof of canonical latent calibration.

Twelve completed 30-epoch summaries existed at inspection: two seeds each of plain/single/factorized in both environments. Validation prediction MSE is not evidence that factorization wins:

| Environment | Plain s0 / s1 | Single s0 / s1 | Factorized s0 / s1 |
|---|---:|---:|---:|
| PushT | .118030 / .117883 | .119591 / .119097 | .120773 / .120232 |
| Reacher | .002984 / .002972 | .003380 / .003382 | .003362 / .003368 |

These numbers come from `runs/world/*/training_summary.json`. Plain remains an unaligned diagnostic and must not be treated as the fair primary planning control. Framewise and factorized-unpaired controls are still needed for the main interpretation, as documented in `reports/implementation_review.md`.

## Findings and actions

### High scientific priority: residual goal calibration may dominate the planning cost

`model.py:225` trains predictor outputs against immutable canonical reference features. `model.py:226` separately penalizes observed-feature alignment. At deployment, `model.py:269` encodes a shifted goal and applies the observation adapter; `evaluate.py:153` compares the terminal predicted feature directly against that corrected goal. This is the intended conversion, not a discovered wrong-coordinate implementation.

However, the selected Reacher factorized s0 checkpoint has validation prediction MSE **0.003362** and alignment MSE **0.227241**. The selected PushT checkpoint has **0.120773** and **0.189830**, respectively. Values are from the selected epoch rows of each run's `metrics.jsonl`. Selection correctly uses prediction MSE rather than incomparable total loss (`train.py:197`), but it does not select for goal calibration or planning.

The errors share the same feature dimension and MSE convention, but their sampled quantities differ: prediction averages targets 4–7, alignment averages frames 3–7, across validation conditions; a planning goal is a particular future frame under a particular support context. **The ratio is a warning signal, not a measured per-goal error or a causal explanation.** Low rollout error alone does not bound the error of the feature used as the goal.

The observation adapter is also restricted to a shared diagonal scale in `[0.9,1.1]` plus a shared shift (`model.py:30–39`). A nonlinear pretrained encoder need not turn a pixel color transform into that class of latent transformation. That limitation can persist after successful optimization.

Action after the current development comparison, if needed: on development only, measure corrected-goal versus canonical-goal error using the actual planning support, and compare terminal cost rankings against actual reached goal distance on a bounded fixed action set. A canonical-goal cost may be used as an explicitly privileged diagnostic, never as the reported deployable method. Do not infer a coordinate bug from alignment MSE alone or silently substitute canonical goals in the main evaluation.

### Medium scientific priority: one-step training and recursive control are different objectives

`model.py:217–225` teacher-forces observed histories ending at indices 3,4,5,6 and predicts reference targets 4,5,6,7 for an eight-frame sample. The first unseen target at index3 is excluded from the prediction objective, although it enters alignment. `model.py:249–264` and `evaluate.py:102–106` begin by predicting index3 from support0–2 and recursively append predictions. After three predictions, a five-step rollout no longer uses observed features in its immediate predictor window.

This is ordinary exposure bias plus a deliberately excluded boundary transition, **not a proof of invalid training or indexing error**: later training windows still learn next-step transitions, and frozen reference coordinates prevent trivial moving targets. Nonetheless, the excluded boundary means training does not directly optimize the exact first inference configuration with context and prediction ending at the same support boundary. The observed PushT MSE growth from .1748 at horizon1 to .5038 at horizon5 makes recursive accuracy a concrete concern. Reacher's much smaller growth suggests that this alone does not explain its control gap.

Action: retain existing runs; report recursive forecast and control results separately from teacher-forced validation. A subsequent model revision could train the boundary transition and/or short recursive unrolls, but it must get a new configuration/run identity and a development-only comparison. Do not change live training math or call the current experiment an exact upstream training reproduction.

### Medium protocol priority: this is not the upstream planning benchmark even at 300/30/30

Both upstream eval configs specify horizon5, receding_horizon5, action_block5, goal_offset25, and budget50 (`external/le-wm/config/eval/pusht.yaml:21–32`, `reacher.yaml:20–31`). The pinned policy defaults to history_len1 (`external/stable-worldmodel/stable_worldmodel/policy.py:53–58`) and flattens receding_horizon into 25 executed environment steps (`policy.py:374–376,499–518`). LeWM's rollout takes that observed-history length from `info['pixels']` (`external/le-wm/jepa.py:69–104`).

Our evaluator acquires three real observations separated by five executed actions, charges the ten support steps, targets recorded frame7 (35 steps from reset, 25 from the end of support), executes only the first five-step block, and replans with the latest real support (`evaluate.py:178–180,215,250–268,278–315`). It uses a fixed terminal-only horizon on every replan (`evaluate.py:153`), consistent with upstream terminal-only latent cost (`jepa.py:112–124`) but at a different replanning interval. Near the end of the budget, the planner still optimizes a 25-step terminal prediction although fewer than25 actions remain executable.

These are defensible, matched choices for the new adaptation benchmark. They are not bugs. Frequent replanning can help correction; terminal-only fixed-horizon planning can also delay required actions, and increasing CEM samples cannot remove that risk. This causal mechanism is a hypothesis, not established from the current aggregate results.

Action: label results as the new controlled-shift protocol and report history budget, execution interval, actual remaining-budget behavior, and full CEM work. Any alternative receding horizon or shrinking horizon is a separate prespecified development comparison, applied equally to all methods. Do not claim numerical reproduction of upstream success rates merely because CEM uses 300/30/30.

### Medium distribution priority: PushT collection is substantially broader than pretrained expert actions

Upstream PushT evaluation uses `pusht_expert_train` (`config/eval/pusht.yaml:32`). New collection uses bounded random targets near the block (`generate.py:127–138`). Reading the first12 training episodes in manifest order gives3840 native action vectors with standard deviations **[.49851,.50371]**, compared with official expert-data **[.20847,.20675]** in `data/upstream/pusht/action_stats.json`; 36.76% of individual action coordinates in this small sample have magnitude greater than.5. This is a bounded descriptive sample, not a full-dataset estimate. Reacher's corresponding sample std `[.58506,.57390]` is close to official `[.57741,.57718]`.

Normalization correctly retains pretrained statistics. Changing the statistics would change the pretrained interface rather than fix the distribution shift. However, the initial CEM unit Gaussian in pretrained standardized coordinates is substantially narrower than this PushT collection policy, and contact/momentum trajectories may be harder than upstream expert-data tasks. Actual damping interventions are explicit (`generate.py:29–40`); Pymunk's default0 is also the upstream setting (`pusht/env.py:549–553`).

Action: keep the correct normalization; report the changed collection/task distribution, and use the same fixed development benchmark to judge useful control. A later solver-prior comparison should be explicit and common to methods, not a hidden method-specific variance change.

### Low compatibility priority: native action clipping differs from upstream unconstrained PushT execution

`evaluate.py:140–145` converts standardized controls back to native units and clips to native[-1,1]. `generate.py:135–136` likewise clips new relative PushT data. Upstream policy inverse-transforms its controls without an explicit clamp (`policy.py:533–539`), and PushT's step converts relative displacements without clipping (`pusht/env.py:324–326`). Official actions include values outside[-1,1], as recorded in the downloaded statistics. The current behavior is internally consistent with the newly generated bounded-control benchmark; it is another reason not to claim identical upstream evaluation.

Action: retain and disclose the bounded-control definition. Do not silently widen controls or clip standardized coordinates. Reacher's physical actuator scaling differs from PushT and already constrains native torque controls.

## Verified implementation semantics

- **Action ordering:** native two-dimensional actions are concatenated chronologically in five-step blocks (`generate.py:159–176`). Training pairs state windows and the corresponding outgoing action windows (`model.py:218–223`). Rollout concatenates the two executed history blocks with future candidate blocks and indexes both state/action windows identically (`model.py:253–259`). No shifted or reversed action history was found.
- **Normalization:** the CEM Gaussian is in checkpoint z-score coordinates. `to_native` applies `mean+std*z`, and `predict_features` applies `(raw-mean)/std` once at the model boundary (`evaluate.py:140–151`, `model.py:145–148,192–196`). Historical actions remain raw until this boundary. Upstream training uses sample std (`external/le-wm/utils.py:25–31`) while its evaluation scaler uses population std (`eval.py:75–79`); with over2million rows the relative difference is approximately2.5e-7, not a credible explanation for the failure.
- **Image preprocessing:** ImageNet normalization before bilinear antialiased resize matches the upstream transform ordering (`model.py:150–165`, `external/le-wm/eval.py:18–24`). Goal and current images receive the same appearance mapping (`evaluate.py:241,246,263,316`).
- **Goal state/render:** goal and controlled environments use the same environment, dynamics, seed, task goal, and visual settings; only the separate goal environment is restored to the recorded goal state (`evaluate.py:224–240`). PushT's angle is restored before its center-of-gravity-dependent position (`generate.py:83–94`). Its displayed background goal marker is separate from task `goal_state` upstream (`pusht/env.py:439–445,546–547`); it stays consistent across the current and goal rendering here.
- **Native time:** each grouped action contains five Gym environment calls. PushT internally integrates10 physics steps per call (`pusht/env.py:318–336`). Reacher internally repeats each control twice (`dmcontrol/dmcontrol.py:38,109–118`), so one grouped transition normally contains ten dm_control steps. Collection and evaluation use this same wrapper; there is no extra outer repetition. Report budgets in environment calls, not simulator integrator steps.
- **Support and privileged data:** learned context receives observed features and already executed actions only (`model.py:167–183`, `evaluate.py:280–286`). Future recorded actions enter only the explicitly labeled replay oracle (`evaluate.py:298–301`). Stored state sets the start and goal and scores success; it is not an input to learned planning. Canonical/paired references are training supervision and forecast targets, not inference context.
- **Success:** support successes are explicitly removed from eligible policy success; goals and initial support are shared across policies (`evaluate.py:250–268,324–325,354–375`). PushT uses upstream combined agent/block position tolerance plus wrapped orientation (`pusht/env.py:353–370`). Reacher matches upstream per-joint absolute threshold. A truncated final action block occurs only on success and is never used in a subsequent planning call.
- **CEM:** standardization, candidate expansion, warm-start shift, and terminal cost shapes are consistent with the unchanged pinned solver (`cem.py:165–185,194–253,280`; `evaluate.py:147–154,289–293`). Mean-vs-sum latent MSE differs from upstream only by the constant feature dimension and preserves rankings. More iterations can improve optimization but cannot establish cost calibration.

## Recommended development gate before costly main planning

This is a **proposed operational gate**, not a preregistration already in force, a significance test, or a claim that25% success is publication-quality. Lock it before reading the new 300/30/30 comparative results. Preserve all completed development results, including the initial unsuccessful low-budget run. Do not use test/extrapolation outcomes to select models, thresholds, checkpoints, or solver settings.

1. Complete the fixed32-seed development comparison at300 samples,30 iterations,30 elites with identical checkpoint-selection rules, goals, appearance/dynamics condition, support, action bounds, and budget. Include frozen, single, factorized and framewise; completed comparator outputs must identify the same evaluator and dataset hashes. Check nonfinite outputs and confirm identical support-success masks. Do not pool different CEM budgets.
2. Require at least24 policy-eligible cases in each environment. For **each environment separately**, require both (a) at least one fair learned comparator, single or framewise, and (b) factorized to obtain `max(8, ceil(0.25*N_eligible))` eligible successes. Thus the current31-case PushT denominator requires8 successes and the30-case Reacher denominator requires8. Exclude frozen/plain from satisfying the fair learned-comparator requirement. These counts guard against spending the main budget when useful learned control has not been demonstrated even on development.
3. If either environment fails that gate, block the full all-method/all-seed planning sweep. Continue the already specified forecast analysis and a bounded development diagnosis of goal calibration and action ranking; retain the same development split and preserve failures. Any revised model/protocol gets a new version and another explicit gate rather than overwriting results. A failed gate is a scientific outcome, not permission to discard a difficult benchmark silently.
4. If the operational gate passes, assess the proposed contribution before committing the entire main budget: compare paired eligible successes against the strongest fair comparator and report uncertainty. Passing25% establishes runnable control, not improvement. If factorized trails that comparator by more than10 percentage points, first complete the prespecified unpaired control and context-use diagnosis on development; do not characterize the mechanism as beneficial. This last10-point rule is a resource-allocation recommendation, not a statistical acceptance threshold.

The gate can be applied automatically by the orchestrator after development completes. A Slurm `afterok` dependency only proves process exit success; it does not enforce any scientific usefulness criterion. No runtime or scheduler changes were made by this audit.
