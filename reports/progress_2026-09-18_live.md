# Research progress — 18 September 2026

Validated snapshot: 2026-09-18T19:41:27.483382+00:00. Counts and job states can advance after this snapshot.

**Main planning: 61/64 complete. Original training and forecasting are finished. The first dynamics revisions did not improve their donor; eight controlled follow-up arms are implemented and queued.**

| Experiment family | Complete | Remaining | New weights |
|---|---:|---:|---|
| Original 30-epoch training | 30/30 | 0 | 30 trained models |
| Original forecasting | 64/64 | 0 | No |
| Original main planning | 61/64 | 3: 3 running, 0 requeued, 0 waiting | No |
| Random/replay controls | 8/8 | 0 | No |
| First dynamics revision training | 2/2, 30 epochs each | 0 | Two development checkpoints |
| First dynamics revision planning | 2/2, 32 tasks each | 0 | No |
| New objective/context training | 0/8 | 8 | None yet |
| New objective/context planning | 0/8 | 8 | No |

Two unchanged frozen upstream exports are not newly trained models. The 30 existing release packages and two trained development revisions remain separate. Queued new arms do not yet provide checkpoints or results.

## Scheduler and capacity

| Job | Scheduler state | Node | Current/resumable task | Saved episodes |
|---|---|---|---|---:|
| 199432 | RUNNING | ws-l1-001 | pusht_framewise_s2_planning_test | 442/576 |
| 199433 | RUNNING | ws-l1-007 | pusht_factorized_unpaired_s2_planning_test | 226/576 |
| 199431 | RUNNING | gpu-01 | pusht_factorized_unpaired_s2_planning_extrapolation | 162/192 |

Job 199432 was automatically requeued once. Its former node ws-l1-001 reports a new boot at 23:36:58 Dubai and Slurmd start at 23:39:32, consistent with a node restart. The temporary BeginTime delay ended at 23:39:58; the job resumed automatically at 23:40:11 and is running at this snapshot. There is no manual hold, and saved progress permits continuation. Accounting is unavailable, so the precise scheduler cause is not asserted beyond this evidence.

Existing chains remain 199431 → 199434 → 199437, 199432 → 199435 → 199438 and 199433 → 199436 → 199439. Completed outputs are reused; remaining continuation allocations do not imply new seeds. No duplicate active task was identified.

| New job | Study allocation | State | Dependency |
|---|---|---|---|
| 199979 | swm-rollout-reacher-continuation | PENDING | afterany:199977(unfulfilled) |
| 199978 | swm-rollout-pusht-continuation | PENDING | afterany:199976(unfulfilled) |
| 199977 | swm-rollout-reacher-initial | PENDING | afterany:199439(unfulfilled) |
| 199976 | swm-rollout-pusht-initial | PENDING | afterany:199438(unfulfilled) |

Four arms run sequentially per environment, with one initial and one resumable continuation allocation each. New jobs preserve original evaluation priority. Effective capacity remains three single-GPU allocations: two workstation jobs plus one GPU-partition GPU (RTX 5000 Ada, 32,760 MiB each). Extra idle nodes do not override partition QoS. Earlier GPU utilization samples are historical, not current sustained utilization.

## Completed planning evidence

**5/9 completed learned-control contrasts have favorable mean raw success**; 4 are unfavorable, 0 are ties and 3 remain incomplete. All completed conditional 95% intervals include zero. These are comparator contrasts, not independent datasets; green does not imply significance.

| Setting | Comparator | Difference, pp | Conditional 95% interval, pp |
|---|---|---:|---:|
| pusht / test | single | +0.52 | [-1.04, +2.08] |
| pusht / extrapolation | framewise | +1.74 | [+0.00, +3.65] |
| pusht / extrapolation | single | -0.87 | [-2.95, +1.22] |
| reacher / test | framewise | -2.08 | [-11.46, +7.29] |
| reacher / test | single | +4.17 | [-4.17, +12.50] |
| reacher / test | factorized_unpaired | +5.73 | [-2.60, +14.06] |
| reacher / extrapolation | framewise | -1.56 | [-6.25, +3.12] |
| reacher / extrapolation | single | +2.78 | [-1.91, +7.64] |
| reacher / extrapolation | factorized_unpaired | -1.22 | [-5.73, +3.30] |

Forecast gains remain 6/8 against Framewise and Unpaired, with both Reacher extrapolation contrasts negative. PushT planning remains below the fixed random-policy mean.

| First revision, development seed 0 | Frozen donor, eligible | Revision, eligible | Difference |
|---|---:|---:|---:|
| PushT | 1/31 | 1/31 | 0 pp |
| Reacher | 11/30 | 8/30 | −10 pp |

Both first revisions completed 30 epochs, selecting validation epochs 8/5. Their checkpoints exist under `runs/dynamics_revision/{pusht,reacher}_s0/best`. They are not promoted as improved models.

## New study and paper artifacts

The eight-arm study crosses teacher-forced/recursive prediction with inferred/constant-input context in both environments, at seed 0 for 30 epochs. The same Framewise donor and goal pathway remain frozen. Both objectives predict targets 3–7 including the first boundary; a common FP32 recursive validation criterion selects checkpoints. Constant-input contexts can learn nonzero responses; nominal parameter matching does not match effective capacity. All arms use the existing 32 development tasks and 300/30/30 CEM budget. The reporter distinguishes planned/trained/complete and suppresses partial planning outcomes. Protocol: `reports/rollout_revision_protocol.md`.

The local gallery at `artifacts/qualitative/index.html` covers **64 matched development tasks × 3 methods using 192 saved observation archives**. Four paper panels show eight deterministic outcome-stratified examples, including wins, losses, joint failures and labeled support-only/joint successes. All 241 source dependencies, including 192 archive hashes, match the refreshed evidence ledger. The independent data audit counted 1,565 observed frames and passed 42 reporting tests. These are actual observed executions, not predicted videos; selected category proportions are not performance estimates. The reviewed manuscript has 29 pages: Figures 6–9 appear on pages 16–19, and rollout-study Table 12 is on page 28. Integrated pages, paper-width and enlarged exports, grayscale, and the headless-Chrome gallery were inspected with no critical issues. Review records: `paper/figures/qualitative_review.md` and `paper/build/qualitative_review/review.json`.

## Next work

1. Finish the remaining original evaluations and refresh all main/appendix comparisons.
2. Execute every queued study arm; report all eight outcomes. A candidate proceeds only if inferred context beats both matched constant input and its frozen donor in both environments on eligible development success.
3. Require multiple seeds and a fresh untouched final holdout before broader claims. A failed screen remains a negative result.
4. The current manuscript build and integrated visual review are complete; refresh measured tables as new experiments finish. Keep the main architecture tied to established evidence.
5. Support length, online-gradient baseline, measured compute and further backbone/environment coverage remain missing; narrow claims where these are unexecuted.

Exact source hashes, validation records and scheduler output are in `reports/evidence/progress_2026-09-18_live.json`. No jobs, scientific sources, main manuscript or figure files were altered by this audit.

Visual-review completion was added after the dated experiment snapshot; experiment counts and scheduler states above were not re-audited for this editorial update. Exact reviewed PDF SHA256: `df71050cee177711d33e8331366f2aaab5901bf5e201e31cf96a0bb6b51d1a9e`.
