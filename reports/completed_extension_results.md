# Completed simulator extensions and official AdaJEPA reproduction

2026-09-18T22:21:42.778605+00:00

All 36 original simulator models completed 30 epochs, forecasting and closed-loop evaluation. The six observation-gain revision models also completed all three stages. All results below are included, including negative comparisons. Final-test outcomes are not inspected in this report.

## Original simulator study

Each table entry aggregates training seeds 0/1/2 on the same eight development tasks. Success totals therefore describe 24 matched task–seed instances, not 24 independent tasks. Forecast MSE is the mean across three training seeds on identical development windows, with lower values preferred.

| Domain | Predictor | Method | Successes per seed /8 | Total /24 | Success | Forecast MSE@1 | MSE@3 | MSE@5 |
|---|---|---|---|---:|---:|---:|---:|---:|
| drone | transformer | Framewise | 1/2/1 | 4/24 | 16.67% | 0.006403 | 0.005975 | 0.005650 |
| drone | transformer | Constant dynamics | 1/1/2 | 4/24 | 16.67% | 0.006479 | 0.006049 | 0.005718 |
| drone | transformer | ShiftWM (ours) | 1/2/2 | 5/24 | 20.83% | 0.006478 | 0.006047 | 0.005711 |
| drone | gru | Framewise | 1/3/2 | 6/24 | 25.00% | 0.006509 | 0.006996 | 0.006578 |
| drone | gru | Constant dynamics | 1/4/2 | 7/24 | 29.17% | 0.006571 | 0.007027 | 0.006591 |
| drone | gru | ShiftWM (ours) | 1/2/1 | 4/24 | 16.67% | 0.006597 | 0.006975 | 0.006503 |
| surgery | transformer | Framewise | 0/1/1 | 2/24 | 8.33% | 0.002823 | 0.003540 | 0.003588 |
| surgery | transformer | Constant dynamics | 1/1/0 | 2/24 | 8.33% | 0.002851 | 0.003529 | 0.003563 |
| surgery | transformer | ShiftWM (ours) | 1/0/0 | 1/24 | 4.17% | 0.002853 | 0.003542 | 0.003573 |
| surgery | gru | Framewise | 0/1/1 | 2/24 | 8.33% | 0.003696 | 0.004462 | 0.004405 |
| surgery | gru | Constant dynamics | 0/1/1 | 2/24 | 8.33% | 0.003736 | 0.004488 | 0.004417 |
| surgery | gru | ShiftWM (ours) | 0/1/2 | 3/24 | 12.50% | 0.003726 | 0.004470 | 0.004389 |

Paired planning effects are ShiftWM (ours) minus the named comparator; positive numbers favor ours. The unmodified registered summary function resamples matched training seeds and physical tasks in 5,000 crossed draws. These are unadjusted exploratory intervals with only eight physical tasks.

| Domain | Predictor | Comparator | Difference, pp | 95% interval, pp | Ours-only / baseline-only |
|---|---|---|---:|---|---:|
| drone | transformer | Framewise | +4.17 | [-25.00, +33.44] | 3 / 2 |
| drone | transformer | Constant dynamics | +4.17 | [-25.00, +33.33] | 4 / 3 |
| drone | gru | Framewise | -8.33 | [-29.17, +12.50] | 1 / 3 |
| drone | gru | Constant dynamics | -12.50 | [-37.50, +12.50] | 1 / 4 |
| surgery | transformer | Framewise | -4.17 | [-25.00, +12.50] | 1 / 2 |
| surgery | transformer | Constant dynamics | -4.17 | [-20.83, +0.00] | 0 / 1 |
| surgery | gru | Framewise | +4.17 | [+0.00, +25.00] | 1 / 0 |
| surgery | gru | Constant dynamics | +4.17 | [+0.00, +25.00] | 1 / 0 |

The original transformer gains on drone tasks are small (+4.17 pp); GRU drone results favor both baselines. Surgery transformer results favor both baselines, while surgery GRU gains are +4.17 pp. No original paired interval has a strictly positive lower bound. Lower forecast error alone does not establish better control. The GRU is our in-house recurrent predictor, not Dreamer.

| Domain | Reference policy | Successes | Interpretation |
|---|---|---:|---|
| drone | random | 2/8 | One common seeded random control policy |
| drone | replay_oracle | 8/8 | Recorded-command replay reachability check; privileged reference, not a learned baseline |
| surgery | random | 1/8 | One common seeded random control policy |
| surgery | replay_oracle | 8/8 | Recorded-command replay reachability check; privileged reference, not a learned baseline |

The drone random reference reaches 2/8 goals (25%); original drone transformer ShiftWM reaches 5/24 (20.83%). These denominators differ because random has no trained model seeds. The surgical task is SOFA tissue manipulation simulation, not a clinical or real surgical experiment.

## Observation-gain revision: development ablation

Only the observation FiLM gain function/range changes from [0.9,1.1] to [0.25,4], preserving identity value and first derivative at initialization. Translation, action adapter, parameter shapes, losses, data, validation selection and planner budget remain fixed. Higher-order curvature and optimization paths also change. No causal claim about recovered geometry follows from these counts.

| Variant | Method | Successes seeds0/1/2 /8 | Total /24 | Success | Mean forecast MSE@5 |
|---|---|---|---:|---:|---:|
| Original narrow | Constant dynamics | 1/1/2 | 4/24 | 16.67% | 0.005718 |
| Original narrow | ShiftWM (ours) | 1/2/2 | 5/24 | 20.83% | 0.005711 |
| Revised wide | Constant dynamics | 0/2/2 | 4/24 | 16.67% | 0.005724 |
| Revised wide | ShiftWM (ours) | 1/3/3 | 7/24 | 29.17% | 0.005713 |

| Paired comparison | Difference, pp | 95% interval, pp | Per-seed difference, pp | Ours-only / comparator-only |
|---|---:|---|---|---:|
| Wide ShiftWM minus wide constant | +12.50 | [+0.00, +37.50] | +12.50, +12.50, +12.50 | 3 / 0 |
| Wide ShiftWM minus original ShiftWM | +8.33 | [-12.50, +29.17] | +0.00, +12.50, +12.50 | 3 / 1 |
| Wide constant minus original constant | +0.00 | [-25.00, +33.33] | -12.50, +12.50, +0.00 | 3 / 3 |

Task-level inventory follows. Each three-character string gives success(1)/failure(0) for training seeds 0,1,2 on the identical physical task, gain0.75 and warm appearance. Every comparison uses the same ten native support commands and total 200-command budget.

| Physical task seed | Original constant | Wide constant | Original ShiftWM (ours) | Wide ShiftWM (ours) |
|---:|---|---|---|---|
| 2053001 | 000 | 000 | 000 | 000 |
| 2053002 | 101 | 011 | 010 | 011 |
| 2053003 | 000 | 011 | 011 | 111 |
| 2053004 | 011 | 000 | 101 | 011 |
| 2053005 | 000 | 000 | 000 | 000 |
| 2053006 | 000 | 000 | 000 | 000 |
| 2053009 | 000 | 000 | 000 | 000 |
| 2053010 | 000 | 000 | 000 | 000 |

The revision is a follow-up on the same development tasks. Its paired effects and any plotted increases remain exploratory. Repeat the fixed 16-goal/120-pair geometry diagnostic on these revised checkpoints before claiming reduced goal-distance compression, and evaluate independent tasks before claiming generalization. No new geometry mechanism measurement is included here.

## Separate official AdaJEPA reproduction

This table uses the authors' released PushT checkpoint, goal set, RGB plus agent proprioception and official CEM configuration. It is not input-, data-, checkpoint-, or budget-matched to ShiftWM, so it must not be used for a direct ranking against our simulator or RGB-only results.

| Condition | Official method | Successes | Success rate | Elapsed seconds |
|---|---|---:|---:|---:|
| clean | AdaJEPA frozen | 34/50 | 68% | 721.1 |
| clean | AdaJEPA adaptive | 46/50 | 92% | 402.2 |
| blur | AdaJEPA frozen | 29/50 | 58% | 726.3 |
| blur | AdaJEPA adaptive | 39/50 | 78% | 495.6 |

Within the official protocol, adaptation increases aggregate success by 24 pp on clean observations and 20 pp under blur. There is one evaluation seed (100); aggregate logs do not support a paired per-task interval. Official CEM uses horizon25, 200 candidates, 30 elites, ten optimization iterations, at most20 replans and five grouped actions per replan (frameskip5). These numbers reproduce the published implementation locally; they do not establish state of the art.

The clean frozen stage started before launcher/runtime hashes and hardware fields were added to the run record. Its checkpoint, goals, official configuration and scientific command were pinned; a supplemental GPU observation documents hardware. Later stages include the added fields. Current launcher/runtime hashes cannot retroactively prove their first-stage bytes.

## Audit and reusable artifacts

The machine-readable report pins 1491 input artifacts by SHA256. All 42 training runs pass their existing completion validator, including 30 chronological epochs, complete optimizer/model packages, validation-best selection and strict model loading. Result identities match their sidecars; recorded code, protocol, manifest, checkpoint and trace hashes match current files. Planning counts are recomputed, saved task records agree with summary records, and paired initial states/support prefixes match. Forecast means are recomputed from all recorded windows. AdaJEPA's pinned upstream checkout is clean, and all checkpoint, goal set, launcher, runtime and configuration hashes present in its records match; the first-stage metadata gap is disclosed above.

The original 36 portable local checkpoint packages remain in `artifacts/releases/extensions_v1`; the six geometry checkpoints remain in their separate `runs/geometry_revision` namespace and require the version2 geometry loader. Nothing is externally published by this report. This report does not summarize the separate ongoing real DROID model study.

- [Machine-readable results and hash ledger](completed_extension_results.json)
- [Original development protocol](domain_extension_protocol.md)
- [Geometry revision protocol](geometry_revision_protocol.md)
- [Report generator](../scripts/extensions/report_completed_results.py)
- [Drone failure-mode audit](drone_failure_modes.md)
