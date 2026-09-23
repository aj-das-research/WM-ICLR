# Open-H Hamlyn dVRK → ShiftWM-v2 Stage 1

Adapter: `scripts/v2/prepare_openh_hamlyn.py` → `data/v2/frames/openh_hamlyn/`
(format: `docs/v2_data_format.md`). Tests: `tests/test_v2_openh_adapter.py`.

## Source
- HF dataset `nvidia/PhysicalAI-Robotics-Open-H-Embodiment`, `Surgical/hamlyn/<task>/`,
  LeRobot v2.1, CC-BY-4.0. Hamlyn Centre, Imperial College London (Su, Deng, Hu,
  Rodriguez y Baena, Giannarou 2026). Bimanual dVRK (PSM1/PSM2) teleoperated with Sigma-7
  haptic devices, ex-vivo porcine tissue / phantoms.
- Tasks: knot_tying, needle_grasp_and_handover, peg_transfer, suturing_1, suturing_2,
  tissue_lifting, tissue_retraction (1019 episodes). All tasks are 30 Hz **except
  tissue_lifting (15 Hz)** (kinematics 100 Hz, nearest-neighbour aligned to the video timeline by
  the dataset authors; clutch pauses removed).
- Raw download: `data/medical/open_h/Surgical/hamlyn/` (log `logs/dl_openh_hamlyn.log`).

## Camera
Video keys per task: `observation.images.{color, depth, wrist_left, wrist_right}`.
The Hamlyn subset has **no stereo endoscope**: its `meta/modality.json` maps the
`endoscope` modality to `observation.images.color`, which is the primary scene camera
(Intel RealSense D405 RGB, 848×480 @ 30 fps). We use that stream only (it plays the role of
the "primary endoscope left camera"); depth and the INSKAM wrist cameras (640×480) are ignored.
Codec differs per task (h264 or AV1); both decode via PyAV (AV1 via bundled `libdav1d`).

## Temporal subsampling / actions
- One model step = 1/3 s for every task (DROID: 5 frames @ 15 Hz):
  `frame_stride = round(fps/3)` = 10 for 30 Hz tasks, 5 for tissue_lifting (15 Hz).
  Kept frames: raw indices `0, s, 2s, …` (< episode length).
- Raw per-frame action (`action` == `action.cartesian_absolute`, 16-D): per arm
  `[x, y, z (m, PSM base frame), qx, qy, qz, qw, gripper (1 open / 0 closed)]`, left arm dims
  0–7, right arm 8–15. Absolute set-points; empirically `action[i] ≈ observation.state[i+1]`
  on the 30 Hz tasks (on 15 Hz tissue_lifting it is closer to `state[i]`).
- Action block: K = 5 sub-commands per step at 15 Hz (exactly DROID's 5-per-step layout),
  `actions[k] = concat_j action[k*s + ceil((j+1)*fps/15) - 1]`, i.e. raw rows
  `10k+1,3,5,7,9` for 30 Hz tasks and `5k+0..4` for tissue_lifting (the set-point at the end of
  each 1/15 s sub-interval) → `action_dim = 5 × 16 = 80` for all tasks.
  **Deviation from "stride × raw_dim"**: stride × 16 would be 160 on 30 Hz tasks but 80 on
  tissue_lifting, so a single trainer could not use them together; resampling to 15 Hz keeps
  one dim. Every raw row can be kept with `--action-hz 0` (only valid without tissue_lifting,
  e.g. `--tasks` minus it → 160-D), or `--action-hz 30` (160-D, tissue_lifting commands
  repeated, zero-order hold).
  Absolute, not deltas; quaternion sign not canonicalised. Deltas can be derived from `proprio`.
- `proprio[k]` (30-D) at the kept raw frame: `observation.state` (16, same layout as action,
  measured) ++ `observation.state.left_arm_joint` (7: j1–j6 rad + gripper) ++
  `observation.state.right_arm_joint` (7).
- Manifest top-level `frame_stride`/`fps_source` = 10/30 (majority); per-task values in
  `frame_stride_per_task`, `fps_source_per_task`, `tasks[*]`. Each episode also carries its
  language `instruction` (peg_transfer / needle_grasp have several variants).

## Images
Full-frame resize to 224×224 (no crop, aspect ratio not preserved: 848×480 → 224×224), PIL
`Image.resize(BILINEAR)` (antialiased when downsampling) from PyAV `rgb24` decode.

## Splits
Per task, episodes with ≥ 14 kept frames (from `meta/episodes.jsonl` length) are sorted by
`sha256(episode_id)` (`episode_id = <task>__<episode_index:06d>`); first `round(0.70 n)` →
train, next `round(0.15 n)` → val, rest → test (val/test forced non-empty when n ≥ 3).
The dataset's own `info.json` splits are ignored. Episode-disjoint, deterministic, frozen.
Counts per task/split are in `manifest.json["counts"]`; skipped episodes in `["skipped"]`.

## Environment
- System ffmpeg is absent on the login node. Added to `.venv` (not pinned in
  `requirements.lock.txt`): `~/.local/bin/uv pip install --python .venv/bin/python av pyarrow`
  (installed av==18.1.0, pyarrow==25.0.1).
- Download note: the repo-wide `snapshot_download` listing hit anonymous HF 429 limits twice;
  it was completed with a per-task scoped `list_repo_tree(path_in_repo=Surgical/hamlyn/<task>)`
  + `hf_hub_download` into the same `local_dir` (see `logs/dl_openh_hamlyn.log`).
- Run (login node, CPU only, no Slurm; ~1.4 GB RSS with 3 workers, ~10 min, 5.0 GB output):
  `source .venv/bin/activate; export PYTHONPATH=src;
  nice -n 10 python scripts/v2/prepare_openh_hamlyn.py --workers 3`
  (resumable: existing episode files are reused unless `--overwrite`).
