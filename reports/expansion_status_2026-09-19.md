# Expansion readiness: 2026-09-19

Audited 2026-09-18T21:04:26.944281+00:00. **All eight rollout-study arms completed 30 training epochs and all 32 development planning tasks. The registered scientific screen failed.** No active user jobs appeared in `squeue` at this snapshot.

The existing strict reporter was called in memory, without updating the paper. It revalidated source/config/data/cache/donor/checkpoint identities, common FP32 recursive validation selection, task support pairing and outcomes. All 15 frozen scientific/protocol hashes and eight config hashes match launch. Tensor reconstruction was already required by evaluator preflight; this audit did not rerun inference.

| Environment | Objective | Context | Eligible success | Donor | Δ vs donor (pp) | Paired wins/losses | Selected epoch | Recursive val MSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pusht | teacher_forced | inferred | 1/31 | 1/31 | +0.00 | 0/0 | 4 | 0.28759487 |
| pusht | teacher_forced | constant | 0/31 | 1/31 | -3.23 | 0/1 | 12 | 0.28791332 |
| pusht | recursive | inferred | 1/31 | 1/31 | +0.00 | 1/1 | 29 | 0.26388249 |
| pusht | recursive | constant | 0/31 | 1/31 | -3.23 | 0/1 | 15 | 0.26876193 |
| reacher | teacher_forced | inferred | 11/30 | 11/30 | +0.00 | 6/6 | 6 | 0.00566411 |
| reacher | teacher_forced | constant | 15/30 | 11/30 | +13.33 | 8/4 | 7 | 0.00565410 |
| reacher | recursive | inferred | 13/30 | 11/30 | +6.67 | 6/4 | 6 | 0.00521244 |
| reacher | recursive | constant | 13/30 | 11/30 | +6.67 | 5/3 | 21 | 0.00524834 |

All rows are seed-zero, post-hoc development results. Eligible success excludes one support-only PushT success and two support-only Reacher successes; raw denominators are 32. No confidence interval or significance claim is supplied by this one-seed screen.

| Contrast (left minus right) | Environment | Eligible Δ (pp) | Paired wins/losses |
|---|---|---:|---:|
| pusht_teacher_forced_inferred − pusht_teacher_forced_constant | pusht | +3.23 | 1/0 |
| pusht_recursive_inferred − pusht_recursive_constant | pusht | +3.23 | 1/0 |
| pusht_recursive_inferred − pusht_teacher_forced_inferred | pusht | +0.00 | 1/1 |
| pusht_recursive_constant − pusht_teacher_forced_constant | pusht | +0.00 | 0/0 |
| reacher_teacher_forced_inferred − reacher_teacher_forced_constant | reacher | -13.33 | 4/8 |
| reacher_recursive_inferred − reacher_recursive_constant | reacher | +0.00 | 6/6 |
| reacher_recursive_inferred − reacher_teacher_forced_inferred | reacher | +6.67 | 7/5 |
| reacher_recursive_constant − reacher_teacher_forced_constant | reacher | -6.67 | 4/6 |

## Scientific decision

**Neither objective passes the registered screening rule.** Both inferred variants tie the frozen donor on PushT. Teacher-forced inferred context is worse than constant context on Reacher; recursive inferred context ties its constant control there. Thus the positive Reacher donor deltas do not demonstrate that episode information helps. Recursive validation MSE improves in all four environment/context comparisons, but planning improves only for Reacher inferred context and regresses for Reacher constant context. Forecasting and closed-loop control remain distinct outcomes.

The protocol calls for retaining the null/negative result and diagnosing failures before further search or stronger claims. A justified next step is criterion-level and planner-ranking diagnosis with matched constant/Framewise controls. Any newly authorized domain/model expansion should use a new fixed namespace/protocol and treat transfer as an untested hypothesis. Do not promote the best Reacher arm selectively. A confirmatory follow-up requires multiple training seeds and a newly specified untouched final evaluation set; original test results already informed development.

## Compute and job completion

| Job | Scope | Durable exit | Runtime (s) | Node |
|---|---|---:|---:|---|
| 199976 | initial / pusht | 0 | 2876.8 | ws-l1-001 |
| 199977 | initial / reacher | 0 | 2873.6 | ws-l1-007 |
| 199978 | continuation / pusht | 0 | 36.4 | ws-l1-001 |
| 199979 | continuation / reacher | 0 | 36.3 | ws-l1-007 |

All four wrappers recorded successful completion and logs contain campaign-complete events. Initial jobs finished all four arms in each environment; continuations revalidated completed artifacts in approximately 36 s. No failure markers occurred in these allocation logs. Each allocation used one RTX 5000 Ada (32 GB). `sacct` is currently unavailable because its database connection is refused; finished job IDs have expired from `scontrol`. These exit claims rely on saved wrapper records plus validated artifacts, not current scheduler accounting.

## Local checkpoints and release readiness

| Group | Completed trained models | Best packages | Last packages | Resume states |
|---|---:|---:|---:|---:|
| world | 30 | 32 | 30 | 30 |
| dynamics_revision | 2 | 2 | 2 | 2 |
| rollout_revision | 8 | 8 | 8 | 8 |

**40 trained model runs** are available locally: 30 original, 2 earlier residual revisions, 8 objective/context revisions. Two additional `runs/world/*_frozen*/best` packages are unchanged upstream exports, bringing distinct best packages to 42. Best/last checkpoints are alternative snapshots, not additional independently trained models. The eight new best/last packages include the full frozen donor and load with `shiftwm.rollout_revision.load_rollout_package`; no network is needed.

**30 original inference releases** exist under `artifacts/releases`; all 388 manifest-listed file hashes and live source model hashes were revalidated. Their recorded portable CPU API checks passed. 28 include a source wheel; the other two require installing compatible project source separately. These 30 are five methods × two environments × three seeds, including controls/diagnostics. There are no packaged release exports for the two earlier or eight new residual revisions yet.

The current `scripts/build_checkpoint_release.py` assumes the original `ShiftWorldModel` loader and one-step training/model-card language. It must not package the distinct revision format unchanged. A separate or explicitly versioned release path needs the corresponding strict loader, frozen donor/provenance, correct common recursive model-selection description, portable verification and honest development-only model card. API verification is not scientific transfer evidence. Nothing has been published by this audit.

## Evidence

- Compact auditable results, factorial pairs, all checkpoint identities and source hashes: `reports/evidence/expansion_status_2026-09-19.json`.
- Registered screen and full study contract: `reports/rollout_revision_protocol.md`.
- Validated result sources: `results/development_rollout_revision/*/planning_development.json`.
- Completed histories/packages: `runs/rollout_revision/*/`.
- Durable job exits: `runs/jobs/{199976,199977,199978,199979}/record.json`.

Paper status prose may lag this completed snapshot; no paper or generated table was edited in this audit.
