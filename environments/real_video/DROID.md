# DROID real-video extraction

The isolated environment is shared with Open-H ingestion. Its combined exact dependencies are in `requirements.lock.txt`; `requirements.openh_initial.lock.txt` and `runtime_migration_2026-09-19.json` preserve the earlier Open-H environment. No packages in the project `.venv` were changed.

Use the existing verified download receipt with `scripts/real_video/prepare_droid.py --raw ... --output ... --inventory ... --expected-episodes ...`. The extractor reads the official TensorFlow Example format through TensorFlow's record reader and parser, including TFRecord CRC verification, and decodes selected native JPEGs with Pillow. It does not use simulated data or generate actions/images.

The first DROID100 episode was inspected solely as an ingestion audit. Its manifest is intentionally `schema_audit_only`, and its audit status is `schema_audit_passed`, so it cannot be confused with the complete training dataset. The main registered subset comprises 24 hash-selected full-release shards and exactly 1,126 episodes; see `reports/evidence/real_video/droid_selected_inventory.json`.

## Data contract

- Three camera payloads per episode: exterior1, exterior2 and wrist, retaining native 180×320 RGB resolution. Every fifth frame is exported; the complete native JPEG streams remain in the immutable raw TFRecord archive.
- Each model action concatenates five chronological 7D recorded commands into 35D. Actual serialized `steps/action` is checked for exact equality with `action_dict/cartesian_position` (6 values) plus `gripper_position` (1). This contradicts the upstream description calling them joint velocities; no relabeling is based on that description.
- Native action_dict, robot observations, RLDS boundary flags, reward/discount and unused terminal/tail command indices are retained separately. Simulator state is not present. Model camera NPZs contain only images, actions and original frame indices.
- No released per-step clock is available. Native frame indices are ordinal; the preview video's chosen 6fps playback is not original timing.
- Session grouping uses the released metadata's collection site and calendar date, deliberately coarser than an episode. Both recording and trajectory paths must agree. Unknown layouts fail rather than silently fall back to per-episode splitting. Hash salt `shiftwm-real-split-v1:` deterministically assigns 70%/15%/15% probability to train/validation/test sessions, without using observations, outcomes or motion.
- Entire short episodes are retained, including those that yield no complete five-command block or no eight-frame prediction window; window eligibility is recorded. No filtering by task success is added.
- A complete manifest is emitted only after exact episode counts, all selected raw shard SHA256 values, output payload hashes, action semantics, frame/action alignment and session separation pass. `data_audit.json` pins the final manifest hash. No artificial appearance/dynamics changes are introduced.

## Local commands

```bash
environments/real_video/.venv/bin/python -m pytest scripts/real_video/test_prepare_droid.py -q
sbatch scripts/real_video/prepare_droid.slurm
```

The Slurm extractor requests 4 CPUs and 16GB memory with no GPU. Model encoding/training are separate, coordinated jobs using the original project environment. A source/runtime/receipt identity prevents silently resuming extraction with changed code or inputs.

Each processed dataset has `preview/recorded_three_camera_episode.mp4`, a contact sheet and their provenance. These are recorded real-robot observations and carry no claim of learned prediction quality or robot-control performance.
