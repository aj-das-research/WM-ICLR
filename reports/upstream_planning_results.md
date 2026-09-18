# Official-protocol LeWM reproduction results

Updated 2026-09-18T11:34:30.777517+00:00.

Original released data/protocol on a compatible historical snapshot; not an author-established training runtime pin.

| Environment | Job | Status | Successes / tasks | Success (%) | Evaluation seconds* |
|---|---:|---|---:|---:|---:|
| pusht | 199469 | complete | 46/50 | 92.00 | 39.39 |
| reacher | 199470 | complete | 39/50 | 78.00 | 57.33 |

Current scheduled dependencies (read from the authoritative allocation ledger):
- pusht job 199469: `afterany:199477`.
- reacher job 199470: `afterany:199469`.

*Runtime includes simulator execution and video encoding inside `world.evaluate`; it is not solver-only GPU latency.

Missing, failed, truncated, ambiguous, or unverified runs have no numerical score. Completion requires a successful matching job receipt, complete output, the exact full official configuration and 50 Boolean outcomes, percentage consistency, and logged task identities. No published score is assumed. These results are separate from adaptation-benchmark comparisons.

Full configurations, per-task outcomes when available, receipt/log identities, and source/checkpoint/preflight/schedule hashes are in `reports/evidence/upstream_planning_results.json`.
