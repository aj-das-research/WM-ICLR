## Latest two-lane recovery schedule

Seed2 laneA:200266→200268→200270 on ws-l1-002 after200219. LaneB:200267→200269
on ws-l5-004 after200224. Finalizer200271 waits for both lane ends plus200219
and200224. Qualitative200236, release200254 and manuscript200278 wait for200271.
All scientific and exporter identities remain unchanged; only pending scheduler
dependencies and placement changed. The amendment is
`scheduling_amendment_seed2_two_lanes.json`. Earlier schedules below are historical.

## Current scheduler amendment

Two gpu-partition attempts failed before the first epoch because the allocated devices were already nearly full. No scientific settings changed. Seed2 now runs as jobs200266–200270 on ws-l1-002 after200219. Finalizer200271 replaces canceled200230/200264; qualitative200236 and release200254 wait for200271. See `scheduling_amendment_seed2_ws_fallback.json` and `gpu04_memory_audit_200265.json`. Historical job IDs below remain for provenance.

# Spatial architecture campaign handoff

Execution update: `scheduling_amendment_early_ws.json` supersedes the original
cache scheduling order below. After h10 seeds 0/1 completed, cache job 200214
moved to `ws-ia`; spatial seed 0/1 may follow its resource gate immediately.
Spatial seed 2 job 200225 explicitly waits for both 200214 and h10 finalizer
200200. The registration, science and three-GPU concurrency limit are unchanged.

The architecture, cache builder, full training, evaluation, checkpoint reload, independent ledger validator, and scheduler pipeline are implemented. **Production execution is queued behind the existing horizon-ten study; no spatial candidate has yet produced a trained result or public checkpoint.** This is original train/validation development only.

## Frozen scientific identity

- Registry: `configs/real_video_spatial/v1/registration.json`
- SHA256: `ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337`
- Protocol: `reports/real_video_development/spatial_protocol.md`
- Motivation/evidence: `reports/real_video_development/architecture_diagnosis.md` and `.sources.json`
- Independent review: `reports/real_video_development/spatial_independent_review.md`
- Engineering evidence: `reports/real_video_spatial/preflight_tests.json`; 49 combined tests pass, including nonzero-head causal checks, exact original-target scoring, full30-epoch CPU lifecycle/exact resume, isolated offline reload, and29 ledger/paired-statistic checks. Independent learned-weight gradient and requeue probes also passed.

Do not edit any registry dependency or registered configuration. Source verification is:

```bash
.venv/bin/python scripts/real_video_spatial/campaign.py verify
```

The exact scheduler submission and live verification are recorded in `scheduled_status.json` and `scheduler_verification.json`. The latter confirms all dependencies, partitions, resources, and no GPU request on the CPU finalizer.

## Complete job DAG

The five modes are always ordered: `autoregressive`, `anchored_additive`, `transport`, `context_off`, `action_free`.

| Stage | Job IDs | Partition/resources | Dependency |
|---|---|---|---|
| Existing horizon-ten finalizer | 200200 | Existing study | Existing study |
| Complete4×4 train/val extraction + full-batch GPU budget measurement | 200214 | gpu;1GPU,8CPU,40GB,4h | afterok200200 |
| Seed0, five complete30-epoch models | 200215→200216→200217→200218→200219 | ws-ia;1GPU/job,8CPU,40GB | first afterok200214; then sequential |
| Seed1, five complete30-epoch models | 200220→200221→200222→200223→200224 | ws-ia;1GPU/job,8CPU,40GB | first afterok200214; then sequential |
| Seed2, five complete30-epoch models | 200225→200226→200227→200228→200229 | gpu;1GPU/job,8CPU,40GB | first afterok200214; then sequential |
| Independent15-model completion/evaluation/export finalizer | 200230 | gpu partition; **0GPUs**,4CPU,16GB,4h | afterok200219:200224:200229 |

At submission verification all17new jobs were pending their dependencies. A maximum of three training GPUs can run simultaneously. Every model allocation lasts7h50m; if full training is unfinished at an epoch boundary after6hours, the job checkpoints and requeues itself. It resumes optimizer, scheduler, RNG, and loader state under the same identity. `training_summary.json` with `status: interrupted` means an epoch-boundary resumable state, not automatically a failed experiment. Five restart attempts are the limit.

The cache job must first pass the measured budget: projected total training for15models ≤80GPU-hours, and each complete30-epoch model ≤20hours. The ideal three-GPU wall-time cap is therefore about26.7hours before IO, evaluation, and queue delays. An out-of-memory error or exceeded budget fails closed; dependent jobs do not silently train smaller or shorter variants. Throughput has **not yet been measured**, so no completion-time estimate is claimed.

## Completion gates and artifacts

1. `data/features/droid_spatial_v1/manifest.json`, `identity.json`, `training_statistics.json`, and per-episode receipts: original train/val only, payload hashes, exact frame/action alignment, frozen DINO provenance,4×4 layout, shared-channel training statistics, and2×2pooling parity checks. Test payload access is rejected.
2. `reports/real_video_spatial/throughput.json` and `cache_and_budget_gate.json`: actual GPU throughput/memory and source/cache/statistics-bound passing budget receipt.
3. `runs/real_video_spatial/v1/<mode>_s<seed>/`: complete30-epoch journal, immutable best/last checkpoint generations, optimizer/RNG resumption, selection metadata, and exact active parameter counts.
4. `reports/real_video_spatial/<mode>_s<seed>_validation.json`: every expected validation window's ten native and original-coordinate errors, episode/session IDs, per-episode means, model/cache/source hashes, and selected epoch. Original2×2 forecasts are scored against **exact original cached targets**, not recomputed pooled targets.
5. `reports/real_video_spatial/finalization.json` and `results.md`: all15models pass authoritative30-epoch/checkpoint-selection gates, independent complete-population/arithmetic validation, and15 relocated isolated-process CPU inference parity checks. Includes every positive and negative arm, all-horizon means, and predefined h5/h10 paired session/seed intervals. Intervals are exploratory validation evidence, unadjusted for multiplicity.
6. `artifacts/releases/spatial_v1/models/<mode>_s<seed>/`: local inference-only predictor exports with `model.pt`, `config.json`, and hash manifest; optimizer state stays in the training runs. Encoder/source provenance remains in package metadata. Publishing a unified source/encoder/license/model-card release bundle is a separate next step; this campaign does **not** automatically upload checkpoints to GitHub.

The qualitative agent can depend on job200230 and must additionally verify finalization and matching hashes. Candidate panels still require numerical and visual review before manuscript promotion.

## Interpretation and pending work

The new candidate has a single observed transition context, not the original separate observation/dynamics factorization. It should be labeled a spatial architecture candidate, not an unchanged original model. Full transport has1,026,305 active parameters versus1,007,776 for either additive control (+1.84%); context-off has878,497 and action-free988,001. All arms instantiate1,026,305 total parameters, but inactive heads are frozen/excluded from the optimizer. Transport begins near persistence with a differentiable gate; additive outputs begin exactly at persistence.

The joint transport/innovation experiment does not isolate each ingredient. Semantic token mixing is not physical flow. Outputs are latent features, not RGB generators, robot-control success, medical predictions, or a SOTA claim. No test or fresh holdout is used to choose this candidate. Finalize and review every registered outcome before paper claims or any new confirmatory evaluation.

If a dependency fails, inspect its log and receipt; do not bypass a failed gate or edit the frozen files. If extraction/measurement fails, register any revised resource/scientific design in a new namespace. If training is preempted, preserve checkpoints and resume under the same identity. If finalization fails after beginning `artifacts/releases/spatial_v1`, it intentionally refuses an existing export directory on rerun; review and quarantine the incomplete export before rerunning the unchanged finalizer. The trained runs need not be repeated for that export recovery.

Relevant logs: `logs/droid-spatial-cache-200214.log`, `logs/droid-spatial-<job>.log`, and `logs/droid-spatial-finalize-200230.log`.
