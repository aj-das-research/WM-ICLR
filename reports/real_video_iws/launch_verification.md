# IWS launch verification

Checked 2026-09-19T12:05:15.907725+00:00. Launch checks passed; training is still in progress.

All 27 registered configurations are submitted exactly once: 3 running and 24 pending. The two workstation slots and one GPU-partition slot use 3 GPUs and 24 CPUs total.

| Arm (PushT, seed 0) | Array task | Committed epochs / 30 | Train / development windows per epoch |
|---|---|---:|---:|
| autoregressive | 200640_0 | 3 / 30 | 13,440 / 3,360 |
| anchored_additive | 200640_1 | 3 / 30 | 13,440 / 3,360 |
| bounded_spatial_mix | 200645_2 | 2 / 30 | 13,440 / 3,360 |

Each saved epoch includes all 480 training and 120 development trajectories, all 59 future offsets, and 210 optimizer updates. Selected and last checkpoint generations load and their recipe, populations, journal, and hash manifests agree. The incomplete-journal status text interrupted does not indicate a stopped Slurm job.

Read-only steps inside the existing allocations verified the inherited GPU assignment. Recorded memory was 8,110 MiB for autoregression, 7,214 MiB for anchored addition, and 7,279 MiB for bounded spatial mixing. The GPU-partition model uses physical GPU 1 on gpu-04; other GPU inventory entries are not attributed to this job. No visibility override or additional GPU allocation was used.

The GPU-partition allocation uses the documented 07:50:00 operational walltime override to satisfy its 8-hour QOS. Model, data, epoch budget, optimizer, batch size and selection remain unchanged.

These are launch and partial-training checks. Early epoch losses are recorded only as health monitoring in the JSON receipt; no final ranking or result is inferred. Official validation remains reserved.

Receipt: reports/real_video_iws/launch_verification.json
