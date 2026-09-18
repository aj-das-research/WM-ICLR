# Official AdaJEPA baseline runtime

This isolated Python 3.9 runtime follows the control/model dependency versions
of [the official AdaJEPA release](https://github.com/agentic-learning-ai-lab/adajepa),
pinned at `51d8665b7978824bd218decab9e05ddb6eb1f47b` (2026-09-09). The upstream
source is unchanged and MIT-licensed. It is a separately tracked reproduction,
not yet an input-, data-, or compute-matched comparison against ShiftWM.

The official release has a 310,210,277-byte PushT checkpoint ZIP and a
4,147,919-byte PushObj evaluation-goal ZIP. Both were downloaded from the Drive
folder linked by the official README, with expected-size and ZIP CRC checks.
SHA256 identities and extracted paths are recorded in
`reports/evidence/adajepa_downloads.json`. The extracted PushT model is
336,945,193 bytes. No offline training dataset is needed for the provided
`goal_source=segments` evaluation; the segment file supplies initial states and
executed action sequences that upstream replays to construct observed goals.

## Protocol differences that must remain visible

- Released visual-shift architecture: **SmallResNetGeM**, 384-dimensional
  embedding, with a six-layer, 16-head transformer predictor, three-frame
  history, five native controls per predicted transition.
- The model receives RGB **and four-dimensional agent proprioception** (XY
  position and velocity). Our current ShiftWM tasks are RGB-based. Official
  reproduction numbers cannot establish a fair direct ranking by themselves.
- Default CEM: horizon 25, 200 sampled sequences, top 30 elites, ten optimization
  rounds, at most 20 MPC replans, five grouped actions executed per replan.
- Adaptive mode makes one gradient step per replan on the predictor's last
  transformer layer and visual encoder head, using the last five observed
  segments. Predictor LR is 5e-4, encoder LR 1e-5; weights and relevant buffers
  reset per episode.
- Frozen mode uses the official `MPCPlanner` and removes the adaptation block.
- Initial-state/goal sampling, action normalization, simulator, and success
  criteria are upstream's. Comparisons to our method require a separately
  validated common task/input/planning interface and budget accounting.

## Local commands

```bash
.venv/bin/python scripts/baselines/adajepa_download.py
environments/adajepa/.venv/bin/python scripts/baselines/adajepa_pin_dinov2.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  environments/adajepa/.venv/bin/python scripts/baselines/adajepa_validate.py
environments/adajepa/.venv/bin/python scripts/baselines/adajepa_run.py \
  --method adaptive --condition clean
```

Run the final command on an allocated GPU node. `--method frozen` changes only
the official planner choice and adaptation removal. `--condition blur` selects
the released image-corruption implementation. The launcher records all
checkpoint, goal, source-config hashes, commands, budgets, and final metrics.
Defaults retain 50 evaluation episodes; a smaller explicit count is recorded
as such and must not be called the complete published protocol.

The upstream loader initializes a DINOv2 model even for this SmallResNet
checkpoint, before loading serialized modules. That cache is isolated under
`environments/adajepa/torch_cache`; it is a loader prerequisite, not the actual
visual-shift model architecture. Model assets and data remain local.

The current unpinned DINOv2 main branch uses Python 3.10 type annotations and
fails in AdaJEPA's declared Python 3.9 environment. The setup pins that auxiliary
source to official commit `e1277af2ba9496fbadf7aec6eba56e8d882d1e35` (2024-02-22)
and records its archive and Python-source tree SHA256. AdaJEPA itself remains
unchanged. This compatibility adjustment is disclosed in every run record.
The released T-shaped goal file contains 1,000 segments of 25 native actions;
the default official evaluation samples 50 segments using its specified seed.
