# Evaluation campaign audit

Snapshot: **2026-09-18 14:17 UTC / 18:17 Dubai**. This is an operational audit, not a comparison selected from test performance. No jobs, campaign files, evaluator, or model were changed. Sources were live Slurm `squeue`/`scontrol`/`sstat`, the fixed task grid, task states, completed result records and worker logs. Files advance during collection, so this is a timestamped snapshot rather than an atomic scheduler transaction.

## Progress and validity

| Family | Complete | Running | Unclaimed | Total |
|---|---:|---:|---:|---:|
| Forecast | 64 | 0 | 0 | 64 |
| Random/replay diagnostics | 8 | 0 | 0 | 8 |
| Main world-model planning | 21 | 3 | 40 | 64 |
| All tasks | **93** | **3** | **40** | **136** |

Main planning has **7,872 episodes in completed files**, plus **419 saved episodes in active files**, out of 24,576. Thus 16,285 remain. All 21 completed main planning files and eight completed diagnostic files pass the existing planner/protocol validator and an independent exact record-population check against their real manifests: 64 initial-state seeds times nine test or three extrapolation combinations, with no duplicate/missing/substituted identities. Partial record keys are valid subsets. These checks do not reread every large checkpoint or certify statistical conclusions; artifact aggregation retains that separate role.

No failed task states, worker exceptions, CUDA out-of-memory messages or stalled progress were found. The three progress files were freshly updated within about two seconds of inspection. Active tasks were Reacher single-context seed 1 test (310/576 saved), PushT factorized seed 1 test (92/576), and its extrapolation task (17/192). These are progress counts, not outcome summaries.

## Resources and continuation safety

Jobs 199428 (`gpu-01`), 199429 (`ws-l1-001`) and 199430 (`ws-l1-009`) each have one GPU, eight CPUs and 32 GiB, with a 7 h 50 min allocation and a 7 h 30 min worker deadline. That is **three GPUs total: one `gpu` and two `ws-ia`**, within the stated entitlement. Each was running for approximately 2 h 43 min; no restarts were recorded. Batch maximum resident memory reported roughly 1.7–3.3 GiB; no host-memory pressure was evident. GPU utilization itself was not sampled, so no utilization percentage is claimed.

The nine queued continuation jobs form the correct independent `afterany` chains:

- 199428 → 199431 → 199434 → 199437 (`gpu`).
- 199429 → 199432 → 199435 → 199438 (`ws-ia`).
- 199430 → 199433 → 199436 → 199439 (`ws-ia`).

Each successor requests one GPU and cannot overlap its predecessor in that slot. Continuations enable bounded failed-task retries; inherited task locks prevent two workers evaluating the same task. Completed immutable outputs are reused, and the fixed grid is independent of test scores. The fulfilled development gate has disappeared from live pending dependency strings; the allocation ledger preserves its original scheduling history.

## Measured throughput and remaining time

| Environment/split | Completed files | Episodes/GPU-hour |
|---|---:|---:|
| PushT test | 4 | 1,036 |
| PushT extrapolation | 4 | 1,047 |
| Reacher test | 6 | 1,029 |
| Reacher extrapolation | 7 | 972 |

Rates use completed `attempt_wall_seconds`, including simulator/planning work in each evaluator attempt. A 576-episode file takes about 33–35 minutes; a 192-episode file about 11–12 minutes. At three workers, the simple estimate is **5.32 further wall-clock hours**; allow **5–6 hours plus scheduler delay** for unseen variants and balancing. Completion therefore likely reaches the first continuation wave, rather than needing all four waves. This is an estimate from observed runtime, not a promise or a reason to truncate the grid. The configured ceiling remains 90 worker GPU-hours / 94 allocation GPU-hours.

## Concrete efficiency and robustness follow-ups

1. **Repeated completed-output parsing:** every claim scan rereads/parses all 64 completed forecast JSONs, totaling **219,219,718 bytes**, then rewrites their completed task-state files. Checkpoint digest caching already prevents the larger repeated weight reads, but it does not cover these JSON results. A per-process cache keyed by output path/size/mtime and expected source/protocol identities could safely avoid this work while rejecting changed artifacts. Profiling its isolated latency was outside this audit; it is not currently a stall.
2. **Completion-state rewrites erase scheduler history:** the reuse branch replaces a state with `reused: true`, discarding original job/node/start/end/returncode fields. The authoritative output retains execution context and the logs retain task transitions, so no measured result is lost. Preserve original ownership/timing fields when adding reuse metadata in a future runner revision.
3. **Idle tails:** current workers lack `--exit-when-not-ready`; a worker with no claimable tasks waits while another owns the remaining task. This can waste up to a task's remaining runtime near campaign completion. Future continuations can use the existing flag, and unused queued waves can be removed after complete-grid validation; do not interrupt an active evaluation for this optimization.
4. **Hard-stop provenance remains conservative:** per-episode progress carries the protocol/signature, but explicit dedicated-resource provenance is stamped by the coordinator after the child returns. A hard-killed segment can therefore resume correctly while losing eligibility for an efficiency claim. Graceful deadlines have margin and no such failure occurred here. Preserve the conservative exclusion rather than retrospectively assuming dedicated timing.

The primary pending work is the remaining 43 planning tasks (three active). Observation/dynamics pathway removals, context shuffling/constant-context controls, support-length sweeps, online-gradient/AdaJEPA controls and broader environment/medical extensions remain outside this launched grid. Do not present those as queued. Complete matched three-seed cells before numerical paired-method reporting.

Source SHA256 at audit: full campaign `257802e8fada6d43449560d59157238f174a430681d1dd94b8c1c704850b7824`; runner `7e99b8479909656d1e066b89aaf9070b6f6bae07dc73a65feb123843eee45836`; evaluator `3ea56985014cc3277af534ad87f788104671f36a8a2bd3f425667a94d30ee3bf`. Detailed operational sources: `runs/world/evaluation_campaign_state/task_grid.json`, its per-task states, `results/world/*/planning_*.json`, `logs/swm-eval-199428.out`, `logs/swm-eval-199429.out`, `logs/swm-eval-199430.out`, and `reports/evidence/full_evaluation_allocations.json`.
