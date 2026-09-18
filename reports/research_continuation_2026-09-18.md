# Research continuation — 18 September 2026

Snapshot: 2026-09-18T14:34:25.147455+00:00. Runs and outputs continue to advance after this record.

| Experiment family | Complete | Remaining |
|---|---:|---:|
| Forecast | 64/64 | 0 |
| Controls | 8/8 | 0 |
| Main Planning | 23/64 | 41 |
| Original full training | 30/30 | 0 |
| New development revision training | 0/2 | 2 queued |

The three original main evaluation workers remain active. New jobs 199687 and 199688 form a resumable development pipeline after 199428 and before its existing successor 199431; the two workstation chains continue independently. Only three GPUs can be active. The two new allocations have a combined four-GPU-hour ceiling, not measured consumption. Their exact commands/dependencies/source hashes are in `reports/evidence/dynamics_revision_allocations.json`.

## Implemented

- A portable full world-model package that freezes the completed framewise donor and learns 218,272 dynamics-context/residual parameters. The two seed-zero runs use all 30 epochs, original training/validation splits and validation-based selection. The development CLI permits only the original 32 matched cases. Exact continuation, zero-initialized donor equality and frozen weights were checked. No revision checkpoint has been trained at this snapshot.
- A paired planning reporter requiring all three training seeds, matched protocol/task/support records, conditional trajectory-cluster intervals and separate training-seed SD. All incomplete contrasts remain blank.
- A source-validated development-revision table that recomputes outcomes and preserves the extra-optimization/post-hoc qualification.
- Completed-output caching in future evaluation workers. Existing live workers were unchanged. Root validated 93 then-completed real outputs and their cache reuse; operational I/O timing is separate from model efficiency.
- Expanded exact manuscript methods, negative-result interpretation and limitations. Both reporters now run in the manuscript refresh pipeline. Current PDF is 17 pages; new tables and existing method/setup pages were inspected.

All 119 targeted tests passed (`reports/evidence/research_continuation_tests.xml`). Independent reviews checked reporting/runtime compatibility and the paired bootstrap. New checkpoints and improved performance are not claimed before full runs finish. A learned context-free residual and other mechanism/baseline controls remain unexecuted; they are not silently represented as queued.

## Scientific interpretation

The completed development comparisons do not establish a factorization benefit. The goal-only intervention improves the fixed factorized Reacher model but does not establish superiority over full framewise calibration, and fails to improve PushT. The new revision tests a narrower development hypothesis with stronger visual calibration held fixed. Beating its frozen donor would still not prove that temporal conditioning is necessary; a matched context-free trainable residual would be needed for that attribution. The original test study remains unchanged.

The project remains local. Only the separately requested paper-figure skill was published to its own repository.
