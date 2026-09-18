# Rollout revision status — 19 September 2026, 00:42 Dubai

Source-validated snapshot at **2026-09-19 00:42:00 Dubai**, equivalent to **2026-09-18 20:42:00 UTC**. The established `paper/scripts/render_rollout_revision.py` reporter was invoked through its read-only `collect(root)` function. No generated paper files, scientific sources, configurations, jobs or watcher state were changed.

**Six of eight arms completed all 30 training epochs and all 32 development tasks. Two arms were still training.** These are seed-zero, post-hoc development comparisons; favorable differences are not significance estimates or proof that inferred context is necessary.

| Environment | Training objective | Context input | Eligible successes | Frozen Framewise donor | Difference, pp | Revision-only / donor-only successes |
|---|---|---|---:|---:|---:|---:|
| PushT | Teacher-forced | Inferred | 1/31 | 1/31 | 0.00 | 0 / 0 |
| PushT | Teacher-forced | Constant | 0/31 | 1/31 | −3.23 | 0 / 1 |
| PushT | Recursive | Inferred | 1/31 | 1/31 | 0.00 | 1 / 1 |
| Reacher | Teacher-forced | Inferred | 11/30 | 11/30 | 0.00 | 6 / 6 |
| Reacher | Teacher-forced | Constant | 15/30 | 11/30 | +13.33 | 8 / 4 |
| Reacher | Recursive | Inferred | 13/30 | 11/30 | +6.67 | 6 / 4 |

PushT has one support-success task and Reacher has two; the denominator excludes these from policy-eligible comparisons. The newly completed recursive/inferred result files were last modified at 20:36:26 UTC (PushT) and 20:36:20 UTC (Reacher). File modification time is an artifact timestamp, not a claim about the exact final simulator transition time.

| Running arm | Completed metric epochs | Development result |
|---|---:|---|
| pusht_recursive_constant_s0 | 13/30 | Pending |
| reacher_recursive_constant_s0 | 13/30 | Pending |

The completion reporter calls unfinished arms `planned`; metric histories independently show that these two were actively training. No partial success rate is reported.

| Job | Live state | Node / reason |
|---|---|---|
| 199976 | RUNNING | ws-l1-001; PushT environment campaign |
| 199977 | RUNNING | ws-l1-007; Reacher environment campaign |
| 199978 | PENDING | Dependency on 199976; resumable continuation |
| 199979 | PENDING | Dependency on 199977; resumable continuation |

Thus these four job IDs comprise two running, two pending and zero completed jobs in this live snapshot. `squeue` returned their states and dependency chains. `sacct` could not connect to its accounting database, so accounting history was not verified.

Source paths follow `runs/rollout_revision/{run_id}/` for training summaries, full metrics and validation-selected packages, and `results/development_rollout_revision/{run_id}/planning_development.json` for completed evaluations. Frozen donors are in `runs/world/{environment}_framewise_s0/`, with original control evaluations in `results/development_official_budget/{environment}_framewise_s0/planning_development.json`. Configurations are in `configs/rollout_revision/`; the frozen campaign manifest is `runs/rollout_revision/campaign_state/protocol_identity.json`. The read-only reporter validates source/configuration/checkpoint identities, full completed metric history, common recursive validation selection and exact paired development protocol before counting a result.

The recursive/inferred Reacher arm improves over its donor, but the previously completed teacher-forced constant-input arm has a larger aggregate gain. The unfinished recursive/constant arm is essential to assessing the matched context contrast. No method promotion or surgical generalization claim follows from this snapshot.
