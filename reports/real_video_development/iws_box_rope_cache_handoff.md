# Box/Rope native cache handoff

Implementation and actual-data compatibility are complete; independent review,
immutable registration and full GPU extraction are pending in this snapshot.
Current source/test hashes are in
`reports/evidence/iws_box_rope_cache_implementation_readiness.json`.
No model training or official-validation evaluation has been performed.

| Task | Upstream-training trajectories | Internal train/development | Native rows | Command width | Output |
|---|---:|---:|---:|---:|---|
| bimanual_box | 602 | 481 / 121 | 120,274 | 14 | data/features/iws_bimanual_box_spatial_v1 |
| bimanual_rope | 602 | 481 / 121 | 120,312 | 8 | data/features/iws_bimanual_rope_spatial_v1 |

The frozen PushT cache remains complete at600trajectories/119,887frames and its
registration still verifies. The task extension imports the exact frozen encoder,
decoder, command loader, Welford and atomic-transaction helpers; explicit task/width
adapters live in `src/shiftwm/real_video_iws_tasks/`. The launcher/CLI lives in
`scripts/real_video_iws_tasks/`. No task globals are changed.

All84combined tests passed in6.30s. The actual-data preflight loaded all1,204
allowed command arrays; decoded five prespecified internal-training videos across
199/200/201native lengths; and reproduced first/last-frame CPU encoder outputs
exactly for both first-training examples. No official-validation payload was opened.
See `reports/evidence/iws_box_rope_cache_actual_preflight.json` for every identity/hash.

After independent review passes, register each exact task using:

```bash
.venv/bin/python scripts/real_video_iws_tasks/prepare_cache.py register --task bimanual_box
.venv/bin/python scripts/real_video_iws_tasks/prepare_cache.py register --task bimanual_rope
```

Schedule no more than two simultaneous1GPU/8CPU/24Gjobs. The reviewed launcher
requires at least4GiBactuallyfreeCUDAmemory on its allocatedBF16-capableGPU before
model loading; insufficient memory causes failure without fallback. It validates
every package again after full extraction. Intended nodes are ws-l1-006/ws-l5-004,
subject to current scheduler availability. Later submission evidence records the
actual job IDs and source/registration hashes; that evidence supersedes this pending
snapshot's execution status. Failed jobs preserve verified packages for exact-identity
resume; do not delete corrupt or mismatched packages to conceal an integrity failure.
