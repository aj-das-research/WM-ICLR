# IWS operational recovery and preliminary seed0 scores

Snapshot: 2026-09-19T15:09:15.532047+00:00

**Scope: one completed seed per task; incomplete study, not a finalized comparison.** All values are internal-development H60 feature MSE; lower is better. Positive signed gain favors bounded mixing. No uncertainty claim is made.

| Task | AR | Additive anchor | Bounded mixing | Gain vs anchor | Gain vs AR |
|---|---:|---:|---:|---:|---:|
| pusht | 0.260041701 | 0.267543679 | 0.261378058 | +2.305% | -0.514% |
| bimanual_box | 0.279299562 | 0.297180172 | 0.291082097 | +2.052% | -4.219% |
| bimanual_rope | 0.211717123 | 0.222361606 | 0.223616883 | -0.565% | -5.621% |

Current counts: 10/27 trained for all30epochs; 10/27 development receipts; 3 training jobs running and 14 pending.

All nine seed0 scores were independently reconstructed from hash-bound primitive window ledgers using equal-trajectory aggregation. Existing selected checkpoint epochs are retained. The source-bound JSON provides exact denominators, hashes, and job identities.

After30epochs all three seed0 AR GPU evaluations failed the existing H15/30/H45 causal-prefix consistency test; original logs preserved. No OOM is evidenced.
All three exact frozen CPU evaluations completed all windows, passed identical prefix tolerances and reproduced the already-selected training score within the existing gate.
Registered evaluator/model/config, selected epochs, microbatch64, all59offset metrics, population, denominators and every tolerance unchanged. Only existing --device cpu option used.
Actual GPU discrepancy magnitude and earliest differing operation remain unmeasured; CPU success supports a backend-sensitive numerical explanation but does not prove CUDA correctness.

Manuscript protocol/setup and completed DROID evidence are updated. Numerical IWS table/plot ingestion remains pending until the complete27run study finalizer; these preliminary values have not been inserted into paper result tables.

Future CPU continuation200835 waits for both training arrays200640/200645 (afterany), preserving all frozen gates. It does not preempt training.
