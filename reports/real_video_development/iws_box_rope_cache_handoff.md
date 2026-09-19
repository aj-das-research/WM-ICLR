# Box/Rope native cache handoff

Implementation, actual-data compatibility, independent review and immutable
registration passed. Full GPU extraction is complete: Box job **200465** on
**ws-l1-006**, Rope job **200466** on **ws-l5-004**. Both logs contain every
episode-verification event and the final all-packages-verified receipt. A fresh
CPU validation independently rechecked all **1,204 packages** successfully.
The completion receipt is
`reports/evidence/iws_box_rope_cache_completion.json`.
Current source/test hashes are in
`reports/evidence/iws_box_rope_cache_implementation_readiness.json`.
No model training or official-validation evaluation has been performed.
All ten reserved official-validation IDs per task are absent from the caches,
logs and normalization inputs. No jobs were listed by `squeue` at the completion
audit. Historical Slurm exit codes are unavailable because `scontrol` no longer
retains these IDs and accounting is unreachable; no exit code is inferred.

| Task | Upstream-training trajectories | Internal train/development | Native rows | Command width | Output |
|---|---:|---:|---:|---:|---|
| bimanual_box | 602 | 481 / 121 | 120,274 | 14 | data/features/iws_bimanual_box_spatial_v1 |
| bimanual_rope | 602 | 481 / 121 | 120,312 | 8 | data/features/iws_bimanual_rope_spatial_v1 |

The frozen PushT cache remains complete at 600 trajectories / 119,887 frames and its
registration still verifies. The task extension imports the exact frozen encoder,
decoder, command loader, Welford and atomic-transaction helpers; explicit task/width
adapters live in `src/shiftwm/real_video_iws_tasks/`. The launcher/CLI lives in
`scripts/real_video_iws_tasks/`. No task globals are changed.

All 84 combined tests passed in 6.30 s. The actual-data preflight loaded all 1,204
allowed command arrays; decoded five prespecified internal-training videos across
199/200/201 native lengths; and reproduced first/last-frame CPU encoder outputs
exactly for both first-training examples. No official-validation payload was opened.
See `reports/evidence/iws_box_rope_cache_actual_preflight.json` for every identity/hash.

Exact idempotent registration commands (already completed):

```bash
.venv/bin/python scripts/real_video_iws_tasks/prepare_cache.py register --task bimanual_box
.venv/bin/python scripts/real_video_iws_tasks/prepare_cache.py register --task bimanual_rope
```

Schedule no more than two simultaneous 1 GPU / 8 CPU / 24G jobs. The reviewed launcher
requires at least 4 GiB actually free CUDA memory on its allocated BF16-capable GPU before
model loading; insufficient memory causes failure without fallback. It validates
every package again after full extraction. Intended nodes are ws-l1-006/ws-l5-004,
subject to current scheduler availability. The historical submission receipt at
`reports/evidence/iws_box_rope_cache_submission.json` records both job IDs,
registration hashes and verified scheduler resource requests. Failed jobs preserve verified packages for exact-identity
resume; do not delete corrupt or mismatched packages to conceal an integrity failure.
