# Development diagnosis after full-budget planning

Date: 2026-09-18. This report analyzes completed seed0 development runs and the subsequent goal-calibration diagnostics. No models/evaluators were changed, research jobs launched, or additional trajectories evaluated to produce the analysis. Proposed follow-ups are post-hoc development diagnostics, not primary-method evidence.

**Conclusion:** the current factorized method does not show a planning advantage over the fair framewise baseline. PushT control is poor for every evaluated method. On Reacher, framewise is descriptively better, and the actual-goal diagnostic identifies a large residual goal-calibration error in the factorized model despite accurate recorded-action canonical predictions. A fixed-predictor, learned-goal-calibrator intervention is justified before retraining. It is not yet established that this intervention will improve control.

## Completed planning results

All methods use the same32 development seeds, condition `(observation=1,dynamics=1)`, solver seed1701,300 CEM samples,30 iterations,30 elites, horizon5, action_block5, and50 native environment calls including10 support calls. Files are `results/development_official_budget/{environment}_{mode}_s0/planning_development.json`. Support-success masks and selected seed sets agree across all four modes.

| Environment | Method | Raw success | Support success | Eligible policy success |
|---|---|---:|---:|---:|
| PushT | Frozen | 2/32 | 1/32 | 1/31 (3.23%) |
| PushT | Shared context | 2/32 | 1/32 | 1/31 (3.23%) |
| PushT | Factorized | 2/32 | 1/32 | 1/31 (3.23%) |
| PushT | Framewise | 2/32 | 1/32 | 1/31 (3.23%) |
| Reacher | Frozen | 5/32 | 2/32 | 3/30 (10.00%) |
| Reacher | Shared context | 8/32 | 2/32 | 6/30 (20.00%) |
| Reacher | Factorized | 8/32 | 2/32 | 6/30 (20.00%) |
| Reacher | Framewise | 13/32 | 2/32 | 11/30 (36.67%) |

PushT support-only success is seed2031026. Frozen/shared/framewise then succeed on seed2031000, whereas factorized succeeds on2031024. Thus equal totals are not identical outcomes. Reacher support-only successes are2031023 and2031031.

Paired Reacher framewise versus factorized:8 framewise-only successes,3 factorized-only successes,3 common successes and16 common failures. The eligible success difference is **+16.67 percentage points**; a20,000-resample paired-seed percentile interval is **[-3.33,+36.67] points**. The exploratory exact two-sided McNemar p-value is.2265625. These development comparisons were inspected after results, involve one training seed, and are not confirmatory tests or proof of superiority. Framewise versus shared has the same discordant counts. Factorized versus shared has5 successes unique to each and only1 shared success.

Reacher eligible mean final wrapped joint-error norm is.6684rad frozen,.7370 shared,.5112 factorized,.3271 framewise, from a common.6754rad initial mean. PushT factorized ends farther from the block goal on average:245.53px block translation and419.32px agent position error, versus framewise178.94px and307.38px; initial means are203.54px and193.59px. These support the practical concern but are not additional independent experiments.

Exact paired counts, seed lists, bootstrap settings and source-file hashes are saved in `reports/evidence/development_paired_diagnosis.json`. A degenerate[0,0] bootstrap interval for exactly identical sampled outcomes does not establish universal equivalence.

The optional operational gate proposed in `reports/planning_correctness_audit.md`
would not be met: PushT has one eligible success and Reacher factorized has six,
below that proposal's eight-success threshold. That threshold has not been
adopted as an automatic requirement. The original main sweep remains scheduled;
it is not held or cancelled. The bounded intervention uses development evidence
only and will retain a separate identity. No claim of accepted-paper readiness
is supported by these results.

## Actual-goal calibration, now measured on the planning support

All eight `results/diagnostics/{environment}_{mode}_s0/goal_calibration.json` files completed. Each selects the same32 cases and excludes support-only successes. Checkpoint/dataset/evaluator hashes, support-success masks, goal indices and initial-state errors match the corresponding planning result. Source hashes and a compact numerical snapshot are in `reports/evidence/development_goal_diagnosis.json`.

The following MSEs use the same actual goal state and actual observed support per case. Unlike the earlier training-log comparison, they directly evaluate the goal used at the first planning invocation.

| Environment/mode | Corrected goal vs canonical goal | Corrected support vs canonical support | Recorded-action terminal prediction vs canonical goal | Recorded-action terminal cost vs corrected goal | Recorded actions preferred to zero under corrected-goal cost |
|---|---:|---:|---:|---:|---:|
| PushT frozen | .731837 | .792340 | 1.124729 | .669984 | 22/31 |
| PushT shared | .450417 | .524221 | 1.062639 | .951906 | 25/31 |
| PushT factorized | .421438 | .547945 | .601551 | .600771 | 27/31 |
| PushT framewise | .168977 | .390589 | .608556 | .500427 | 27/31 |
| Reacher frozen | 1.107148 | 1.062932 | .839717 | .826942 | 10/30 |
| Reacher shared | .767218 | .577019 | .197315 | .812902 | 17/30 |
| Reacher factorized | .796745 | .650001 | .011741 | .807137 | 20/30 |
| Reacher framewise | .271204 | .274530 | .008553 | .281764 | 29/30 |

For Reacher factorized, the canonical prediction error under the recorded future actions is.011741, while the actual corrected-goal error is.796745 and the deployed goal cost for that sequence is.807137. All30 recorded sequences have lower predicted canonical-goal cost than the zero-action input; only20 retain that preference against the corrected goal. Framewise retains29/30 and reduces actual-goal calibration error to.271204. This is direct evidence that a good canonical predictor can be paired with an inaccurate goal embedding. It still does not prove the cause of individual planning failures or guarantee that a donor calibrator will improve CEM's selected actions.

The zero-action comparison is a **model-input/cost-ranking diagnostic**, not a measured zero-action simulator counterfactual. Only two fixed action sequences were compared. Low recorded-trajectory error does not establish accurate predictions for CEM's optimized action candidates. The corrected-support/goal distances and terminal prediction errors must not be interpreted as success probabilities.

PushT framewise reduces goal error substantially, but its recorded-action recursive prediction error remains approximately.61, close to factorized.60, and control success remains1/31. Goal calibration alone is therefore unlikely to establish useful PushT control; optimization under inaccurate contact dynamics and the changed action/task distribution remain plausible limitations.

## Rendering concern resolved for the actual allocated diagnostics

Every eligible Reacher goal and observed support frame matches its archived canonical image exactly before the appearance transform. Both current and goal renderers identify **NVIDIA RTX5000 Ada Generation / NVIDIA Corporation**. This rules out the proposed live renderer/source mismatch for these compared development frames. The earlier CPU llvmpipe/archive discrepancy in `reports/goal_metadata_render_audit.md` does not occur on this allocated path.

PushT's observed support images also match archives exactly. Goal rerenders differ slightly: mean absolute channel difference averaged over cases is.02032 on the0–255 scale and mean changed-spatial-pixel fraction.000559. Some edge pixels have larger individual differences, so this is not a pixel-perfect equivalence claim. The same fresh goals and discrepancies occur for all four modes. The task-goal setter itself was separately shown not to change pixels.

## What the architecture and losses do—and do not—explain

1. **The observation correction has limited functional capacity.** Factorized/shared correction is a diagonal feature scale constrained to[.9,1.1], plus a shift shared across the episode's observed/goal states (`src/shiftwm/model.py:30–39,179–190`). The appearance context is estimated from mean/variance over just three support features. A color shift passed through a nonlinear frozen visual encoder need not be invertible by a shared near-identity diagonal affine map. Framewise uses an image-dependent nonlinear residual MLP (`model.py:92–98,185–190`). Lower framewise goal error is consistent with this expressivity explanation; it does not identify it causally because predictor training and parameters also differ.
2. **Dynamics can compensate for visually miscalibrated inputs while the goal cannot.** Both learned predictors are trained to output fixed canonical targets (`model.py:225`), while goal embeddings only receive observation correction (`model.py:269–271`). A strong predictor can thus achieve low prediction loss without the goal branch achieving the same canonical accuracy. The actual Reacher factorized diagnostic exhibits precisely that separation. It is not a discovered arithmetic coordinate-conversion bug: both branches follow the intended implementation.
3. **Selected training losses support, but do not replace, the per-goal diagnostic.** Reacher factorized best epoch30 has validation prediction/alignment MSE.003362/.227241; framewise best epoch29 has.002972/.090837. PushT factorized best epoch4 has.120773/.189830; framewise epoch4 has.120123/.121286. Checkpoints were selected by prediction loss consistently. Alignment is a separate objective and is not the selection metric. Aggregate validation losses average different states and conditions from the actual-goal diagnostic.
4. **Consistency regularizers are not demonstrated to help.** Already completed factorized-unpaired s0 training has Reacher validation prediction/alignment.003384/.223518 and PushT.120874/.183659, close to paired factorized. Its planning/actual-goal behavior has not yet been evaluated here. Reacher paired dynamics-context std is.03366 versus unpaired.68385, but context scale can be traded against downstream weights. This is not proof of collapse or domain identification. An architecture-matched unpaired comparison remains necessary for interpreting regularization.
5. **History and rollout differences remain hypotheses, not errors.** Teacher-forced one-step training, exclusion of the support/query boundary prediction, five-step recursive rollout and repeated context estimation can affect control. They are documented in the planning audit and do not establish invalid indexing. Shared context's poorer actual recursive error despite similar teacher-forced validation emphasizes the need for the measured rollout diagnostic.

## Concrete next experiment: learned goal-only intervention, no retraining

The next bounded experiment should reuse the completed factorizeds0 predictor/context and the completed framewises0 visual calibrator. Change **only** the goal embedding: encode the available shifted goal through the shared frozen visual representation and apply the donor framewise residual correction. Keep the factorized observed-history correction, context inference, action statistics, dynamics, rollout, CEM budget, support acquisition, goal selection and seed unchanged. Run all32 development cases in both environments and store results in a new diagnostic namespace.

This hybrid requires no canonical goal at deployment and no simulator state as model input. Before running, require exact equality of frozen visual/reference weights, action statistics and relevant data identity; record both checkpoint hashes plus the intervention source hash. A zero-initialized or disabled wrapper must preserve all non-goal calculations, and tests must demonstrate identical predictor/history/context/rollout behavior. Existing main/evaluator/model source remains untouched.

Interpretation is locked before its outcome:

- If Reacher improves with the learned goal-only substitution, it supports the narrower explanation that goal calibration limits this fixed predictor under this development condition. It does not establish that factorization itself is beneficial or that the hybrid beats a complete framewise model.
- If it does not improve, stop treating goal calibration as sufficient. Examine optimized-action prediction reliability and planner behavior on development before adding architecture complexity.
- PushT remains a required reported negative result. Better latent goal error without useful control is not success; do not drop PushT silently or relabel the same failed outcome.

The hybrid is an explicitly post-hoc diagnostic assembled from two trained checkpoints. It is not a new independently trained primary method and should not be inserted into the original method table without that label. No canonical/privileged-goal oracle arm is needed for this immediate experiment.

## Only after that result: a bounded trainable revision

If the goal-only intervention supports the hypothesis, the clean next architectural question is whether temporal dynamics context adds value **after holding a sufficiently expressive visual calibrator fixed**. Use a shared frozen framewise calibrator/predictor initialization for both arms; one arm stays the framewise control, the other adds a zero-initialized dynamics residual inferred from corrected observed transitions/actions. Train only the new context/residual modules on the existing training split, retain the established checkpoint-selection rule, and compare on the unchanged development split before any test use. This separates the dynamics-context question from replacing the visual calibration function.

Do not simultaneously change the visual encoder, action normalization, goal budget, losses, dataset and rollout objective: such a result would not explain the observed failure. Complete the already planned unpaired consistency control as a distinct test rather than claiming that small context variance proves success. Any revision needs a new run/model identity and full disclosure of development-driven design; a publishable novelty/efficacy claim remains contingent on new evidence and independent training seeds.

## Goal-only implementation readiness

`scripts/evaluate_goal_calibration_intervention.py` implements the specified unprivileged intervention in a separate inference wrapper. Every encoding/history/context/prediction/rollout call delegates to the fixed factorized model; only `goal_embedding` delegates to the framewise donor. Canonical goals are never supplied. Runtime model/evaluator/generator files remain unchanged.

The original thirteen targeted tests passed in 1.90 seconds. Both real PushT and Reacher checkpoint pairs were loaded and checked on CPU: frozen encoder/projector/reference tensors, action buffers, preprocessing buffers and data/statistics provenance agree exactly; all sources are completed validation-best seed0 models. These CPU checks used source SHA256 `cafbff396e399db05a8ff4a8dd9eb875ddc03c462a155bf3489acad7c08697ab` and do not evaluate planning efficacy.

Before queued job 199477 began, the wrapper gained a nonblocking per-output file lock spanning reuse checks, computation and writes, plus exact validation of search coordinates and planner-summary metadata. All twenty-one targeted tests now pass. They retain the original checks for unchanged recipient weights and bit-identical history/context/predictor/rollout behavior, donor context independence, frozen-coordinate compatibility and fixed development protocol, and additionally verify concurrent-writer exclusion, lock release after exceptions, and rejection of contradictory metadata. Model operations, task selection and budgets are unchanged, so the real checkpoint pairs were not reloaded. The current script SHA256 is `986f364d00fa7f5d025d137d942753dd553c643e06ce252041308d31b40ed959`. This hash and the test XML checksum are recorded in `reports/evidence/goal_intervention_validation.json`; the earlier CPU-validation source hash is retained separately there.

Queued execution command (or use `--environment pusht` / `--environment reacher` for separate allocations):

```bash
.venv/bin/python scripts/evaluate_goal_calibration_intervention.py \
  --environment both --device cuda --save-video --max-runtime-seconds 2400
```

Defaults write to `results/development_goal_intervention/{environment}_factorized_s0_framewise_goal_s0/planning_development.json`, with per-episode resumable journals. Both model/config/training-metadata hashes, dataset/action-statistics provenance, source hashes and complete planner protocol enter the result identity. No GPU job was launched during implementation; the later full evaluation is recorded below.

## Completed controlled intervention

Job 199477 completed both full 32-task development evaluations without changing
the source model, primary evaluator or original outputs. The measured results are
in `reports/evidence/goal_intervention_results.json`, generated by
`scripts/summarize_goal_intervention.py` after strict source, budget, episode and
support-state validation.

- PushT hybrid: raw 1/32, entirely the common support success, hence 0/31 eligible
  versus 1/31 for the fixed factorized and full framewise models. The paired
  difference versus either is -3.23 percentage points, interval [-9.68,0].
- Reacher hybrid: raw 14/32, including two support successes, hence 12/30 eligible
  versus fixed factorized 6/30 and full framewise 11/30. Hybrid-minus-fixed has
  nine wins and three losses, +20 points [0,+40]. Hybrid-minus-framewise has seven
  wins and six losses, +3.33 points [-20,+26.67].

Intervals are exploratory paired trajectory-seed percentile bootstraps with
20,000 resamples at fixed training seed zero. They do not measure variability
across training seeds. This controlled sample shows a Reacher benefit from the
goal-only change and no PushT benefit. It does not establish a population-level
advantage, factorization benefit, or superiority over the simpler full framewise
model. It supports further development of well-calibrated goal/prediction
interfaces while retaining the negative PushT result. The main multi-seed study
remains separate and in progress.
