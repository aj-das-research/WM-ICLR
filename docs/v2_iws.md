# IWS (RLA-WM real bimanual) → ShiftWM-v2 Stage 1

Adapter: `scripts/v2/prepare_iws.py` → `data/v2/frames/iws_{pusht,box,rope}/`
(format: `docs/v2_data_format.md`). Tests: `tests/test_v2_iws_adapter.py`.

## Source
- `data/real_video/iws_public_v1/extracted/iws_converted/{pusht,bimanual_box,bimanual_rope}/traj_<id>/`
  (HF `xyzhang368/RLA-WM` `iws_converted.tar`; revision pinned in
  `configs/real_video_development/iws_acquisition_v1.json`; fetched by
  `scripts/real_video_development/fetch_iws.py`).
- Per recording: `camera_0_rgb.mp4` (640×480, 30 fps, h264, 199–201 frames) and
  `metadata.h5` with `target_qpos` [N, w] (recorded command rows; w = 4 pusht, 14 box, 8 rope),
  `qpos` [N, 14], `root_poses`, camera intrinsics/extrinsics. Video frames = h5 rows.
- Official evaluation handles: `data/real_video/iws_public_v1/download/eval_handles/iws/handles.{pusht,box,rope}.json`
  — 200 handles per task, each `(traj_id, frame_id=s)` with `sampled_horizon = 60`, on the 10
  upstream **val** recordings (`split_ranges.json` val = traj 0–9).

## Model step and actions
- `frame_stride = 5` native rows (1/6 s). The 60-row official horizon is **K = 12 steps**
  (the 10–20 range asked for). The v1 prefix horizons 15/30/45/60 rows map to steps 3/6/9/12,
  and v1's window-start stride of 5 lines up with the grid.
- `actions[k] = concat(target_qpos[r_k], …, target_qpos[r_k+4])`, where `r_k` is the kept row.
  Command row i drives native frame i → i+1, as in v1 `IWSWindowDataset`
  (`commands[s:s+H]`, targets `features[s+1:s+H]`). This gives an action dim of
  **pusht 20 (5×4), box 70 (5×14), rope 40 (5×8)**. The raw command rows are kept unchanged:
  no resampling, no deltas, and units are unverified (as in v1).
- `proprio` = `qpos` (14-D) at each kept frame.
- Images are resized from the full 640×480 frame to 224×224 (no crop), with antialiased
  bilinear resizing (PIL), decoded with PyAV.

## Splits (per task)
| split | contents | IWS recordings |
|---|---|---|
| train | upstream *train* recordings (traj 10+), full, kept rows 0,5,…; T = 40 | 510 / 512 / 512 |
| val | 15 % of upstream train recordings, picked deterministically by `sha256("<task>__<traj>")` | 90 / 90 / 90 |
| test | **one episode per official handle** (200): kept rows s, s+5, …, s+60 (T = 13), 12 action blocks = exactly the handle's 60 command rows s…s+59 | 10 (= upstream val = v1 `reserved_official_validation`) |
| test_recordings | the same 10 reserved recordings, full grid from row 0 (T = 40). Diagnostic only (real history); never used for official numbers | 10 |

Train, val and test use disjoint recordings. Test ids are
`<task>__h<handle_index>__<traj>_f<start>`; `session` = `<task>__<traj>`, so bootstrap can group
by recording. Manifest fields per test episode: `handle_index` and `handle_start`. The real
history available before the handle is `handle_start // 5` kept frames.
Our val set is a fresh 15 % draw; it overlaps v1 `internal_development` in only
16–21 of 90 recordings. v1 used 20 % for development, and v2 does not reuse that split.

## How our horizon maps onto the official protocol
- **Official:** given the single RGB frame s and command rows s…s+59, predict frames s+1…s+59
  (59 offsets).
- **Ours:** context frame s. Step k (1…12) takes action block rows s+5(k−1)…s+5k−1 and predicts
  frame s+5k. The offsets we score are 5, 10, …, 60:
  - 5…55 (11 steps) are official targets.
  - 60 (step 12) is one row past the official last target. It always exists, because the
    upstream feasibility rule requires s+60 < N, and it only uses the handle's own 60 command
    rows.
- If an exact official-offset comparison is needed, report steps 1–11 (offsets 5–55)
  separately from step 12.
- **Single-image evaluation:** the official protocol gives one observation. At eval the first
  frame is replicated into the H = 3 history, with neutral past actions (zeros in standardised
  space, i.e. the train-mean action), the same way v1 used a single observation. Suggested
  trainer flag: `single_image=True`. For `FeatureSplit` this means building windows from
  `T ≥ 1 + K` (not `H + K`), using `hist = f[0].repeat(H)` and `past = 0`. To avoid a
  train/test mismatch, train or fine-tune with the same replication, e.g. with probability p per
  batch or always for IWS runs. The trainer currently requires `T ≥ history + horizon` (15), so
  the T = 13 test episodes are skipped until this flag exists.

## Run
```
source .venv/bin/activate; export PYTHONPATH=src
nice -n 10 python scripts/v2/prepare_iws.py --workers 3     # ~4 min, ~0.35 GB RSS, 8.7 GB out
```
Optional `--train-phases N` (≤ 5) stores each train recording at grid phases 0…N−1 as separate
episodes (`…__p<k>`). The test handles start at every phase (s mod 5 ∈ {0…4}), while the
default train grid only starts at phase 0. Existing outputs are reused unless `--overwrite`.
Dependencies are the same as for Open-H (`av`, `h5py`, PIL); no ffmpeg or cv2 is needed.
