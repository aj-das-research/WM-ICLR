# Registered drone branch supervision, version 1

This TRAIN-only data extension is motivated by the completed development action-ranking diagnostic: feature prediction and action ranking remain imperfect, and the existing episode-context method did not consistently improve action selection. This extension collects controlled action-conditioned five-step future observations. It does not register a new model, loss, improvement claim, or closed-loop benchmark.

## Fixed selection and intervention

- Source: `data/extensions/drone_v1`, the existing immutable trajectory payloads.
- Contexts: first 32 sorted unique training seeds, 53000–53031, crossed with original command gains 1.0, 0.75, and 1.25. Exactly 96 contexts. No success, distance, model error, or goal-eligibility filtering.
- Each context reuses its own ten original native support commands and observations (1 s). The original support differs across gains because the original collector used a common state feedback rule; those differences remain visible in commanded actions and images. We do not claim identical support commands across gains.
- Each of 32 futures starts from a fresh simulator reset followed by **all ten recorded support commands**. Every original RGB image and 20-dimensional native physical state, at reset and after each support command, must match exactly. Matching the reset seed alone is insufficient. Physical state is used only to validate the intervention and for separate audit/scoring.
- Fixed candidate library: one zero sequence, four held axial commands, four normalized held diagonal commands, four five-call axial pulses followed by zeros, and 19 random five-block commands from NumPy `default_rng(20260919)`. All candidates have 25 native calls (2.5 s), commands in `[-1,1]`, and five groups of five two-dimensional calls. This is byte-identical to the library fixed before the prior development diagnostic; **candidate design, not development pixels, states, or outcomes**, is reused.
- All methods must subsequently use this same dataset and same candidate library. No model-specific or outcome-adaptive candidate collection.
- The support and future use 35 calls per branch, at the original command frequency of 10 Hz, upstream PID 120 Hz, and PyBullet physics 240 Hz. The existing frozen adapter and PID are unchanged. No GPU is requested; PyBullet TinyRenderer produces the canonical RGB on CPU.

## Model-facing and separate audit data

Each `contexts/<trajectory_id>.npz` has exactly these fields:

| Field | Shape | Type | Meaning |
|---|---|---|---|
| `support_images` | 3 × 128 × 128 × 3 | uint8 | Actual RGB at native calls 0, 5, 10 |
| `support_actions` | 2 × 10 | float32 | Original commanded support actions |
| `candidate_actions` | 32 × 5 × 10 | float32 | Prespecified future commanded actions |
| `future_images` | 32 × 5 × 128 × 128 × 3 | uint8 | Actual RGB after future calls 5, 10, 15, 20, 25 |
| `future_image_mask` | 32 × 5 | bool | True only at actually executed five-call boundaries |
| `terminal_images` | 32 × 128 × 128 × 3 | uint8 | Last actual RGB of each branch |
| `executed_lengths` | 32 | int16 | Number of actual future calls, at most 25 |
| `executed_mask` | 32 × 25 | bool | True exactly on executed action prefixes |
| `valid_mask` | 32 | bool | Full 25-call branch with no crash or workspace escape |

`future_images` and `terminal_images` are future **supervision targets**, never inputs used to infer the current context. Unavailable future images are zero-filled with a false availability mask; no padded image is a valid target. Availability means observed, not necessarily safe: a failure exactly on a block boundary still has its actual observed image, while `valid_mask` is false. The five target boundaries enable later recursive feature supervision and visual-velocity diagnostics. These targets are not necessarily settled goal images, and must not be presented as native successful goal references. Any later loader, target encoding, ranking objective, model comparison, or checkpoint is separately specified and tested. Canonical RGB is stored; any appearance augmentation is applied later consistently with the training split.

The separate `audit/` NPZ stores original support native images/states, branch native states, gain-scaled requests, rotor RPMs, and the original trajectory's recorded endpoint state used for diagnostic scoring. Separate audit JSON stores per-step distances, speed, altitude error, success, crash/escape flags, stop reasons, hidden gain, and replay checks. Audit padding after a stopped branch is NaN, explicitly marked by the executed prefix. No synthetic continuation is created. Hidden state/gain/distance fields never occur in model NPZs. The manifest and sidecars contain provenance labels for auditing and split assignment, not learned input features.

The native goal criterion includes low speed and altitude tolerance as well as XY position and safety. Original diagnostic XY ranking is a partial-objective diagnostic, not a complete success objective. Goal success does not stop collection. Crash or workspace escape stops a branch; **all unsafe and short branches are retained**, with their actual last image, exact action count, validity flag, and audit reason. Even a failure on call 25 is invalid. A runtime exception, nonfinite physics, source mismatch, or support replay mismatch aborts collection and prevents a completed manifest rather than silently dropping a branch.

## Registration, identity, and splits

`scripts/extensions/collect_drone_branches.py register` validates all 96 original episode/audit hashes and writes candidate bytes, source/runtime checksums, every selected source identity, support checksums, and this protocol's checksum before simulating any branch outcome. It pins all files in the upstream simulator package, the adapter, its own collector, and the isolated runtime lock. Collection checks this identity again before/after contexts and at completion. Resuming requires the same registration and unchanged existing main/audit payloads; corrupt shards fail explicitly.

Only the original training split is consumed. All gains and all branches sharing one source seed belong to the same training family. There is no development, validation, or test collection for training, and no random per-branch split. The original validation/development/test sets retain their existing roles. Development diagnostic findings motivated the next hypothesis; this extension is not a preregistration made before those findings.

## Execution gate and estimate

Preparation and synthetic contract testing are authorized now. **The root agent must check current Slurm capacity and authorize the full collection before submission.** No collection job is launched by registration.

After authorization, the intended command from the project root is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  environments/drone/.venv/bin/python scripts/extensions/collect_drone_branches.py collect --workers 4
```

Request one CPU-only allocation with four CPUs, 16 GB memory, no GPU, on a root-approved partition. The completed 128-branch CPU diagnostic took 118.8 s including model scoring. Linear single-worker extrapolation is about 47.5 min for 3,072 branches; four-worker execution is approximately 12 min before shared CPU/I/O variability. A **10–20 min planning estimate** is appropriate, with a 30 min allocation limit. This is an estimate, not measured collection throughput. Up to 107,520 native calls (3,072 × 35) are required. Model-facing raw arrays occupy approximately 926 MB before compression, plus separate native physical diagnostics. Each worker owns an independent simulator; the processes share no live physics state.
