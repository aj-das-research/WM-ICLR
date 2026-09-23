# ShiftWM-v2 unified data format

All v2 datasets (DROID, Open-H Hamlyn, IWS, planning suites) are converted into one
two-stage format so that a single encoder-extraction script and a single trainer serve all.

## Stage 1: processed frames (dataset adapter output)
```
data/v2/frames/<dataset>/
  manifest.json
  episodes/<episode_id>.npz
```
`<episode_id>.npz` (np.savez, uncompressed or compressed):
- `images`: uint8 `[T, 224, 224, 3]` RGB, already resized (antialiased bilinear, center-crop or
  full-frame resize — record choice in manifest `resize_policy`).
- `actions`: float32 `[T-1, A]` — action applied between frame t and t+1 (block of raw commands
  concatenated if subsampled, as DROID's 5×7=35).
- `proprio` (optional): float32 `[T, P]` robot state at each frame (used only for probes).

`manifest.json`:
```json
{"dataset": "openh_hamlyn", "version": 1, "frame_stride": 5, "fps_source": 30,
 "action_dim": 35, "proprio_dim": 14, "resize_policy": "...",
 "action_semantics": "...", "license": "CC-BY-4.0", "source": "...",
 "splits_policy": "episode-disjoint, seed 0, 70/15/15 per task, frozen before any training",
 "episodes": [{"id": "suturing_1__000012", "task": "suturing_1", "split": "train|val|test",
               "file": "episodes/suturing_1__000012.npz", "T": 57}]}
```
Splits are assigned once (deterministic hash/seed) and never changed after training starts.

## Stage 2: encoder features (`python -m shiftwm.v2.extract`)
```
data/v2/features/<dataset>/<encoder_tag>/
  manifest.json   # copies stage-1 manifest + encoder identity, grid, channels
  episodes/<episode_id>.npz   # features float16 [T, G, G, C]; actions; proprio
  stats.json      # train-split per-channel mean/std (shared over positions), action mean/std
```
encoder_tag examples: `dinov2s16` (DINOv2-S/14 @224 → 16×16×384), `dinov2b16` (768), `dinov3s16`.
