# Released LeWM planning reproduction: local setup audit

Updated 2026-09-18. **Both full official-protocol evaluations completed successfully:** PushT 46/50 (92%) and Reacher 39/50 (78%). The collector verified job receipts, all selected task identities, complete configurations, source/checkpoint hashes, and all Boolean outcomes. See `reports/upstream_planning_results.md` and its evidence JSON. These are results on the official 50-task selections in a compatible historical runtime, not a claim to match a published aggregate score or the authors' exact training runtime. Active training/evaluation imports and packages were not changed. The setup and queue audit below preserves the preparation history.

The reproduction must run the released benchmark data and upstream evaluator, not `shiftwm.evaluate` or the new adaptation dataset. This distinguishes failure on the new task distribution from an upstream integration problem. It is not a search over planner hyperparameters.

## Source versions and scope

- LeWM evaluator/config/model source: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, the existing May 22 checkout.
- Current live stable-worldmodel: `4821c8e6a3f0f83b7e6a80da3a757e026ea9026b`, September 8.
- Isolated **compatible historical snapshot**: `abdced49809d5eae38e24b27dc7b635c502c4812`, May 20. This is the last available commit before the LeWM source date and has the evaluator's expected API. It is **not established as the authors' exact training/runtime revision**: the LeWM README requests `stable-worldmodel[train,env]` without a version lock, and the release config does not record simulator dependency versions.

Both source trees were exported from existing local Git objects into `artifacts/upstream_planning_reproduction/{le-wm,stable-worldmodel}`. No live upstream file was edited. `.source_revision` markers identify each snapshot. The historical evaluator and YAML files are unmodified. The runtime wrapper changes only local paths, output location, and isolated import paths.

Why the September backend is unsuitable for an unqualified original reproduction:

- The May evaluator instantiates `stable_worldmodel.solver.CEMSolver(model=...)`. September removes that namespace and uses `stable_worldmodel.planning.solver.CEMSolver(cost=...)`, with a separate objective adapter.
- August changes elite standard deviation from the earlier correction to population standard deviation. Substituting the newer solver would change the optimizer, despite identical 300/30/30 settings.
- Later versions refactor world/dataset execution and model rollout. A compatibility rewrite using the current API would need separate validation and should be labelled accordingly.

**Geometry clarification:** source changes alone do not establish a nominal PushT geometry mismatch. The default shape is index 2, `T`. Historically the scale metadata was 40 but `add_tee` hard-coded scale 30. September changes metadata to 30 and honors that value. The effective default T geometry is therefore unchanged. Changed `small_tee` vertices and symmetry checks concern other shapes. The Reacher wrapper/task source files compared here are byte-identical between the two snapshots. Do not claim these differences explain our low development success without direct evidence.

## Unchanged official protocol

| Setting | PushT | Reacher |
|---|---|---|
| Upstream config | `config/eval/pusht.yaml` | `config/eval/reacher.yaml` |
| Environment | `swm/PushT-v1` | `swm/ReacherDMControl-v0`, `qpos_match` |
| Dataset name | `pusht_expert_train` | `dmc/reacher_random` |
| Sample count | 50 starts | 50 starts |
| Sampling/CEM seed | 42 | 42 |
| Goal | Dataset state 25 native steps after selected start | Dataset qpos 25 native steps after selected start |
| Interaction budget | 50 native steps | 50 native steps |
| Planning | Horizon 5, action block 5 | Horizon 5, action block 5 |
| Replanning | Receding horizon 5 = 25 native steps | Receding horizon 5 = 25 native steps |
| CEM | 300 candidates, 30 iterations, 30 elites, scale 1, batch 1 | Same |
| Images | 224×224, upstream ImageNet normalization | Same |
| Support | No imposed 10-step support phase | Same |

The evaluator samples valid **rows**, without replacement, rather than prescribing an independent new simulator-seed list. It sorts sampled row indices for HDF5 access. Its `choice(len(valid_indices) - 1, ...)` excludes the final valid row; the audit intentionally preserves that behavior. Seed 42 controls row selection and CEM; the two HDF5 files have no `seed` column, so this is not a claim that every simulator reset is explicitly seeded.

The upstream evaluator fits `sklearn.preprocessing.StandardScaler` on the finite rows of its configured columns, using population variance. PushT fits action, proprio, and state; Reacher fits action. It does not consume our stored sample-standard-deviation statistics. Initial/goal states are installed by the upstream callable configuration. No new shift transforms, random-trajectory goals, context acquisition, or our success-denominator convention are injected.

## Released data and checkpoints already available

The local compressed archives were previously verified against the pinned manifest in `references/world_artifact_sources.json`. This audit reads the extracted HDF5 schema and all 50 selected start/goal image pairs per environment; it does not rehash the entire 145 GB of extracted data.

| Input | Local original file | Size |
|---|---|---:|
| PushT data | `data/upstream/pusht/pusht_expert_train.h5` | 46,300,921,856 bytes |
| Reacher data | `data/upstream/reacher/reacher.h5` | 98,905,882,624 bytes |
| PushT weights | `data/pretrained/pusht/weights.pt` | 72,290,721 bytes |
| Reacher weights | `data/pretrained/reacher/weights.pt` | 72,290,849 bytes |

The PushT HDF5 file contains 18,685 physical episodes and 2,336,736 rows. Reacher contains 10,000 episodes and 2,010,000 rows, including terminal rows. Both have `ep_len`, `ep_offset`, action, step indices, and RGB images. PushT has the required state/proprio columns; Reacher has qpos/qvel. All selected goals remain in the same episode and have the exact requested +25 offset.

Official selection validation completed:

- PushT: 1,869,611 valid starting rows; the selected 50 starts happen to belong to 50 distinct episodes. Selection/image digest `77a1410e441e9922d07359d2e4c44d004aff9377521e3649f8a4cd99b4bbaee7`.
- Reacher: 1,760,000 valid starting rows; the selected 50 starts belong to 50 distinct episodes. Digest `004e28d04206a04cf16d0f3f7ee0bf99cf4f90a63e858709b9bf27b9d5c2cd4c`.

Exact identities, image-pair checksums, model hashes, schemas, and resolved configs are in `reports/evidence/upstream_planning_preflight.json`. Released weights and config files were freshly SHA256-checked against the pinned artifact manifest. These are the original released checkpoints, not our fine-tuned or frozen wrapper packages.

Dataset aliases are symlinks under `artifacts/upstream_planning_reproduction/cache/datasets/`, including `dmc/reacher_random.h5` pointing to the already extracted `reacher.h5`. Checkpoints are linked under `cache/checkpoints/{pusht,reacher}/lewm/`. The README's object-checkpoint conversion instructions are stale relative to the inspected `eval.py`: this revision calls `swm.wm.utils.load_pretrained` and accepts the actual `weights.pt` plus Hydra `config.json` layout. No object conversion or additional checkpoint download is needed.

## Commands and isolated dependencies

Recreate/verify the staged inputs with the existing project environment:

```bash
.venv/bin/python scripts/prepare_upstream_planning_reproduction.py
```

This command performs CPU reads and local source export only. It does not install packages or launch planning. The full launch commands for the prepared environment are:

```bash
bash scripts/run_upstream_planning_reproduction.sh pusht
bash scripts/run_upstream_planning_reproduction.sh reacher
```

The wrapper uses `artifacts/upstream_planning_reproduction/.venv/bin/python`, sets the historical source trees ahead of installed packages, verifies the actual imported stable-worldmodel path, disables HF network access, and executes the unchanged historical LeWM `eval.py`. No episode count, seed, goal offset, planner budget, action parameters, or task configuration is overridden. The exact underlying commands are `eval.py --config-name pusht policy=pusht/lewm cache_dir=<isolated cache>` and the corresponding Reacher configuration, with separate output/log directories. Append `--config-only` to invoke the real upstream entry point with Hydra `--cfg job`, with CUDA disabled and without evaluating.

The active project environment lacks `stable_pretraining`, `sklearn`, `loguru`, `lancedb`, `lance`, and `pyarrow`; the new isolated environment supplies them. The historical stable-worldmodel eagerly imports its Lance reader even for HDF5 data, so these were actual import dependencies, not a change of data format. `decord` remains absent but its video reader is optional for these HDF5 runs. The isolated installation includes stable-pretraining 0.1.8, scikit-learn 1.9.1, loguru 0.7.3, lancedb 0.39.0, pylance 12.0.0, and pyarrow 25.0.1. The complete 153-package lock is `reports/evidence/upstream_reproduction_requirements.lock.txt`; `uv pip check` passes.

All original third-party pins are retained except **fsspec**, changed from 2026.7.0 to 2026.6.0 only in the isolated environment because stable-pretraining's datasets dependency rejects the newer version. Torch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.2.6, MuJoCo 3.11.0, and pymunk 7.3.0 are unchanged. Existing torch/CUDA wheels were reused from the uv cache; no duplicate torch download was needed. The active environment still has fsspec 2026.7.0. This is a documented compatible runtime, not an assertion that these package versions match the authors' original machine.

To reconstruct the isolated dependencies elsewhere after staging the source/data paths:

```bash
~/.local/bin/uv venv --python 3.11 artifacts/upstream_planning_reproduction/.venv
~/.local/bin/uv pip sync \
  --python artifacts/upstream_planning_reproduction/.venv/bin/python \
  reports/evidence/upstream_reproduction_requirements.lock.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
artifacts/upstream_planning_reproduction/.venv/bin/python scripts/validate_upstream_reproduction.py
```

Do not run `scripts/setup_environment.sh` for this purpose because it targets the active `.venv`.

Validation performed: local archive exports, original weight/config hashes, Hydra composition, both data schemas, all 100 sampled start/goal pairs, Python/shell syntax, historical imports, both real `eval.py --cfg job` entry points, strict loading of both 18,034,478-parameter released models through stable-pretraining and upstream `load_pretrained`, finite CPU goal costs using the original `get_cost`, and StandardScaler population statistics matching the released archives. Detailed model/package evidence is `reports/evidence/upstream_runtime_validation.json`; entry-point configuration captures are `artifacts/upstream_planning_reproduction/configs/{pusht,reacher}_entrypoint.txt`. CUDA was disabled during validation. CUDA inference, simulator execution, and video writing remain to be evaluated by the scheduled jobs. This is a runnable setup, **not yet a successful full reproduction**.

## Queue budget and required outputs

Scheduler status was checked at **2026-09-18 11:20 UTC / 15:20 Dubai**. Both reproduction jobs were `PENDING (Dependency)` at that observation; completion and scores remain pending.

| Job | Work | Dependency | Requested resources |
|---|---|---|---|
| 199477 | Separate development goal-only intervention | `afterany:199394` | Separate `ws-ia` diagnostic allocation |
| 199469 | Full PushT upstream reproduction, 50 tasks | `afterany:199477` | 1 GPU, 8 CPUs, 32 GB RAM, 1 hour |
| 199470 | Full Reacher upstream reproduction, 50 tasks | `afterany:199469` | 1 GPU, 8 CPUs, 32 GB RAM, 1 hour |
| 199428 | Main evaluation worker, separate protocol | `afterany:199470` | Existing 7:50-hour allocation request |

Logs are `logs/swm-upstream-pusht-199469.out` and `logs/swm-upstream-reacher-199470.out`. The machine-readable scheduler snapshot is `reports/evidence/upstream_reproduction_schedule.json`; the authoritative allocation history and current dependency overrides are in `reports/evidence/full_evaluation_allocations.json`. The separate goal-only intervention was subsequently inserted before PushT, then moved to `ws-ia` after the released coordinator job 199394; forecast job 199446 follows that intervention. The upstream sequence remains 199477 → 199469 → 199470 → 199428, subject to GPU availability. Dependencies enforce queue ordering; `afterany` does not assert that the predecessor succeeded. Queue submission does not upgrade the CPU validation evidence into a successful GPU reproduction. The scheduled jobs will establish CUDA/simulator/video compatibility and actual outcomes.

Each full environment run has at most 100 task-specific replans: 50 tasks × 2 replans. With 300×30 candidates, that is at most 900,000 candidate action sequences and 4,500,000 predicted latent transitions per environment before early termination. Both environments total at most 1.8 million sequences / 9 million latent transitions. There is no training.

Reserve **one GPU, 8 CPU cores, 32 GB host RAM, and up to one hour per environment** for the first historical run; the two environments can run independently if allocations permit. This is a conservative queue reservation, not a measured runtime prediction. Reading/normalizing the full HDF5 metadata, the original per-episode sampling loop, EGL initialization, and 50 video encodes also cost time. GPU capacity and actual elapsed time must be recorded after the first full run; do not promote our development timings to reproduction efficiency claims.

Keep the resolved dependency lock, source revisions, copied effective YAML, selected task identities, checkpoint/data provenance, complete stdout/stderr, upstream `*_results.txt`, and all per-episode videos. Record the 50-element upstream success vector and its denominator directly; do not infer results from expected published scores. A subsequent comparison with our new benchmark can diagnose protocol/distribution effects only after this reproduction succeeds.

### Eventual structured result summary

The historical evaluator writes an append-only text file under `cache/pusht/pusht_results.txt` or `cache/reacher/dmc_results.txt`; the launch wrapper explicitly overrides the latter basename to `reacher_results.txt`, so the actual scheduled Reacher destination is `cache/reacher/reacher_results.txt`. Each run block contains `==== CONFIG ====`, YAML, `==== RESULTS ====`, a multiline Python representation of `metrics`, and `evaluation_time: ... seconds`. The returned metrics are `success_rate` in **percent**, `episode_successes` as a 50-element NumPy boolean array, and `seeds` (expected `None` because the released HDF5 files contain no seed column).

`scripts/summarize_upstream_planning.py` implements a read-only collector. It requires a matching successful durable job receipt, a complete run block, the full locked effective YAML, exactly 50 Boolean outcomes, agreement with `100 * sum(episode_successes) / 50`, and the exact sorted 50-row selection in scheduled stdout. It reads current job identities from the authoritative allocation ledger, verifies archived source/config files against their pinned Git blobs, verifies released checkpoint/config hashes, and preserves source/log/preflight/schedule/lock hashes. It safely parses a restricted literal/`array([...])` grammar without `eval` and refuses multiple appended blocks as ambiguous. Missing, failed, incomplete, or unverified results remain null, never numerical zero. The available scheduler accounting database is unreachable; live `scontrol` plus durable `runs/jobs/<job>/record.json` receipts support status/completion checks without relying on `sacct`.

```bash
.venv/bin/python scripts/summarize_upstream_planning.py
.venv/bin/python scripts/summarize_upstream_planning.py --watch --interval 60 --max-hours 120
```

Outputs are `reports/evidence/upstream_planning_results.json` and `reports/upstream_planning_results.md`. Watch mode holds a singleton lock, caches large-file hashes by metadata, runs on CPU without an allocation, and stops after both successful results or the bounded duration. It does not alter the evaluator or any model. Parser and receipt guards are tested with `.venv/bin/python -m pytest -q tests/test_upstream_summary.py`. Runtime includes video generation and is not solver-only latency. No upstream confidence interval or published reference score is invented; the reproduction remains separate from the adaptation benchmark and its three-training-seed tables.
