# Drone branched training data: completed audit

Job 200092 produced all 96 registered training contexts and 3072 branches in 301.85 seconds, using four CPU workers and no requested GPU. The independent full payload audit passed in 5.37 seconds.

| Hidden command gain | Branches | Full-horizon safe | Retained workspace escapes |
|---|---:|---:|---:|
| 1.0 | 1,024 | 948 | 76 |
| 0.75 | 1,024 | 1,012 | 12 |
| 1.25 | 1,024 | 861 | 163 |
| Total | 3,072 | 2,821 | 251 |

There were no recorded crashes. All 251 workspace escapes retain their actual terminal frames, executed action prefixes, future-frame availability masks, and separate physical audit. Do not call missing future frames valid training targets, and do not interpret gain-dependent invalid counts as learned-policy results.

The final dataset has 75,798 future calls, 30,720 repeated support calls, and 15,064 observed future boundary RGB targets. All 96 contexts derive exclusively from original training seeds 53000–53031 across the three original gains. No validation, development, or test trajectory entered collection.

Every original/generated payload hash, source ID/split, all 48 frozen source hashes, exact common physical support, candidate identity, gain-scaled executed prefix, RPM/state consistency, model-field allowlist, future-mask/padding contract, and per-step physical metric/flag was checked. Recomputed distance, speed and altitude error differed by exactly zero. Thirteen corruption tests independently exercised these checks.

Final manifest SHA256: `5cb5c7d3c24e009ccac5e62accecc6252b5b9cf3c9385803f1ee460778fe53f2`.
Registration SHA256: `b93d847a90077e8e8d496dc89fa93fa4f86f2dd3bf745fdc6c1968a52e09482e`.

Evidence: `reports/evidence/drone_branched_data_audit.json`, `reports/evidence/drone_branched_full_audit.log`, and `reports/evidence/drone_branched_auditor_tests.log`. Scheduler accounting (`sacct`) was unavailable because its database connection was refused; completion is established by the collector's final manifest and complete independent artifact audit. No new model or loss was trained by this collection.
