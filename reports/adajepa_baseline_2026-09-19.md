# AdaJEPA official baseline: assets, validation, and scheduled evaluation

We have prepared an actual released adaptive-world-model comparison from
[AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa), rather than naming
an in-house gradient-update control as the published algorithm. The official
repository is locally pinned at `51d8665b7978824bd218decab9e05ddb6eb1f47b`
(2026-09-09, MIT license). Its upstream source is unmodified.

## Verified assets and model

The two relevant archives linked in the official README were downloaded from
public Google Drive: PushT visual-shift weights (310,210,277 bytes) and all
PushObj evaluation goal segments (4,147,919 bytes). Download sizes, ZIP CRCs,
and SHA256 identities were verified. The extracted model contains **34,295,767
parameters**, with a SmallResNetGeM visual encoder and six-layer, 16-head
transformer predictor. It was saved at epoch 3. The T-shape evaluation file has
**1,000 segments of 25 native actions**; upstream's standard configuration
samples 50 segments with seed 100.

The checkpoint file SHA256 is
`9098082f1d40bfa7b4f4884571db30a0fed4a0aa26ad7dab8db11415437b8b32`.
Its architecture uses three observed frames, five native controls per grouped
transition, 384-dimensional visual embeddings, and **four-dimensional agent
proprioception (XY position and velocity)**. That extra input must be disclosed
when discussing our RGB-based method.

## Engineering validation completed

The isolated Python 3.9 / Torch 2.3 runtime loaded the real checkpoint, replayed
25 genuine released commands with exactly repeated RGB and states, produced
finite latents, performed the real AdaJEPA update, and verified that its reset
restores every parameter and buffer exactly. This took 14.52 seconds on CPU,
including the first auxiliary model download. These are functionality checks,
not learned planning results or evidence of improvement.

The official loader initializes DINOv2 even though the released visual-shift
model uses SmallResNetGeM. Current DINOv2 main has Python-3.10-only annotations,
which failed under AdaJEPA's declared Python 3.9. We pinned this auxiliary
Torch Hub source to official revision
`e1277af2ba9496fbadf7aec6eba56e8d882d1e35` (2024-02-22), recording its archive
and full Python-source tree hashes. The AdaJEPA code and weights were unchanged.
This adjustment is recorded in each run's provenance.

## Scheduled official evaluation

Slurm **200074**, partition `ws-ia`, one GPU, eight CPU cores, 40 GB host memory,
six-hour limit, dependency `afterany:200065`:

1. Clean frozen model: 50 sampled goals.
2. Clean AdaJEPA: the same 50 goals and seed.
3. Frozen and adaptive blur pair only if measured clean runtimes leave enough
   wall-time margin; each completed run saves its results independently.

The job has started on `ws-l1-007` and is executing the full 50-goal frozen
planning stage. An overlapping read-only query within the allocation observed
an NVIDIA RTX 5000 Ada, 32,760 MiB, at 99% GPU utilization. Initial CEM rounds
took approximately 3.3 seconds each across the 50 tasks; this is an early runtime
observation, not a completed-campaign estimate. A supplemental hardware record
documents that the first launcher instance began just before metadata-only
hardware fields were added; scientific settings were unchanged.

Official CEM settings are retained: 25-step horizon, 200 candidates, 30 elites,
ten optimization rounds, 20 maximum replans, five grouped actions executed per
replan. Adaptive mode uses one update per replan, recent-five-segment replay,
predictor LR 5e-4 and encoder-head LR 1e-5, with per-episode weight reset.
Frozen mode uses the upstream `MPCPlanner` and removes the adaptation block.

This stage verifies the published method on its own released task protocol.
It is **not yet an input-, data-, or compute-matched comparison with ShiftWM**.
The user-facing main comparison must keep that distinction visible. We have not
claimed a reproduced score until the full runs finish and are audited.

Evidence and reproducibility:

- `reports/evidence/adajepa_release_inventory.json`: public file metadata.
- `reports/evidence/adajepa_downloads.json`: download identities and archive checks.
- `reports/evidence/adajepa_cpu_validation.json`: real-model update/reset validation.
- `reports/evidence/adajepa_gpu_start.json`: allocated GPU observation and first-stage metadata note.
- `environments/adajepa/README.md` and `requirements.lock.txt`: isolated runtime.
- `scripts/baselines/adajepa_official.slurm`: allocated campaign entry point.
- `scripts/baselines/adajepa_run.py`: source-backed run records and full budgets.
- `results/baselines/adajepa/campaign.json`: current campaign state once launched.
- `logs/adajepa-official-200074.log`: scheduler output.

Primary references: [official repository](https://github.com/agentic-learning-ai-lab/adajepa),
[author project page](https://agenticlearning.ai/adajepa/), and
[paper](https://arxiv.org/abs/2606.32026).
