# Completed raw-coordinate DINO-WM follow-up: independent handoff

All six external predictors completed 100 epochs; all fifteen fixed internal controls completed 30 epochs. The complete-only finalizer passed. Re-running its full package/source/selection gate and numerical collector reproduced the saved report exactly. Independent arithmetic additionally reconstructed all 34,251 primitive window records, all episode/seed/aggregate curves, and the four reported primary native h5/h10 paired intervals.

| Method | Native h5 MSE | Native h10 MSE |
|---|---:|---:|
| ShiftWM bounded (ours) | 0.145074184 | 0.193141489 |
| Fixed autoregressive control | 0.149573024 | 0.203950362 |
| Persistence | 0.164028392 | 0.226128836 |
| Adapted DINO-WM: raw one-step, 100 epochs | 0.231162609 | 0.380522498 |
| Adapted DINO-WM: raw recursive H10, 100 epochs | 0.243827880 | 0.353244109 |

Both new adapted objectives remain **worse than persistence**, and their endpoints are worse than their earlier 30-epoch standardized-coordinate versions. The raw-coordinate/recipe follow-up did not repair baseline transfer performance. The resulting ShiftWM h10 reductions of 49.24% and 45.32% must not become a SOTA headline. Keep this result as an honest adapted-baseline diagnostic and investigate representation/interface/optimization before drawing an external-method conclusion.

The audit adds four explicitly **posthoc** external-minus-persistence intervals using the same paired session/seed bootstrap (10,000 draws, seed 20260919). All are positive, supporting the baseline weakness at these endpoints; they were not prespecified comparisons in the original reporter. Existing primary-versus-external intervals were independently reproduced rather than newly invented.

The actual all-method nearest component remains no-tanh; the established fixed-AR qualitative cases and scores do not change. Keep v1 and raw-v2 studies separately identified, with pooled 16-token features, 35-D grouped actions, absent proprioception, original development scope and unequal training budgets explicit. Coordinate and recipe changes were joint, so the difference does not identify a causal factor.

Scheduler queue is empty. Slurm accounting was unavailable (database connection refused) and completed jobs had expired from `scontrol`; no exact scheduler exit code is claimed. All application completion/execution receipts and finalization source gates passed; the last training receipt ended at 2026-09-20 18:09:06 UTC.

Audit: `reports/qualitative_temporal_v1/new_training_completion_audit.json`.
Finalization SHA-256: `f8fe4e30ff12fec8903b784ec271d8492e75f35da5321e8426613cfed9b624d8`.
Audit SHA-256: `2ef7ee742b047192e8a6ddba2496d209df8e2c0db9949e788e90b10dd0a21b37`.
