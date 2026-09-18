# Completed DROID real-video ingestion

The prespecified full-release subset passed extraction and data auditing on 2026-09-19. This is real recorded robot data. It is neither the full DROID benchmark nor a result of our forecasting algorithm. DROID100 was used only for the separate schema inspection.

| Quantity | Verified value |
|---|---:|
| Full-release shards | 24 of 2,048 |
| Complete episodes retained | 1,126 |
| Native observation/action records | 327,498 |
| Five-command grouped transitions | 64,826 |
| Train / validation / test episodes | 851 / 143 / 132 |
| Train / validation / test site-day sessions | 316 / 59 / 58 |
| Distinct collection sites | 14 |
| Sessions shared across splits | 0 |
| Episodes without a complete eight-frame window | 11, retained |
| Terminal/incomplete-tail command records omitted from model blocks | 3,368, retained in audit |
| Episodes filtered by outcome or motion | 0 |
| Downloaded input bytes, including metadata and license | 22,017,792,821 |
| Exported three-camera NPZ bytes | 21,510,455,291 |

Sessions are the released site plus recording date. Deterministic hashing with salt `shiftwm-real-split-v1:` assigns each session using 70/15/15 probability thresholds; the realized episode proportions differ because sessions vary in size. This split prevents same-site/same-day leakage but does not guarantee new physical scenes or objects across days.

## Observation and action contract

Three cameras remain available at native 180×320 RGB resolution: `exterior_image_1_left`, `exterior_image_2_left`, and `wrist_image_left`. Each camera NPZ contains only `images` (uint8), `actions` (float32), and `frame_indices` (int64). For N complete groups, shapes are `[N+1,180,320,3]`, `[N,35]`, and `[N+1]`. Exported frame indices are 0,5,10,…; each 35D model command concatenates the five intervening native 7D commands chronologically. The immutable TFRecord source retains all native JPEG images. Separate audit payloads retain native commands, action dictionaries, robot observations, RLDS flags, reward/discount, and tail indices. Original paths and language strings remain provenance, not model inputs.

Every native `steps/action` record is exactly equal to the released `action_dict/cartesian_position` (6 components) concatenated with `action_dict/gripper_position` (1 component). The main manifest therefore labels the action `commanded_cartesian_position_6_plus_gripper_position_1`. This array-level equality holds across all 327,498 records; the upstream description calling these joint velocities is inconsistent with the observed release. We do not infer a more specific coordinate ordering or unit convention. The metadata declares float64, but the TensorFlow Example stores float lists as float32; the audit preserves their exact decoded values.

No per-step timestamp is present. Native indices are ordinal observations, and the preview's chosen 6 fps is display playback only. There are no generated images, synthetic commands, or imposed image corruptions. All outcomes are retained. External camera 1 is the registered training/primary view; external camera 2 is reserved for the held-out camera-transfer evaluation. Wrist images are retained for inspection and future work.

## Verification and reproducibility

CPU-only Slurm extraction job `200105` used four CPUs and a 16GB request; the audited extraction took 277.1s. The four extractor tests cover session agreement, invalid session rejection, chronological action grouping and tail exclusion, and malformed/nonfinite action rejection. The completed audit verifies all 24 raw shard SHA256 values, TensorFlow record CRCs, metadata/receipt identity, all exported payload hashes, every native action mapping, RLDS boundaries, native image resolutions, action/target/frame alignment, session separation, and the exact expected episode count. Final status is `complete` in the dataset manifest and `passed` in the data audit. Short episodes stay in the manifest and contribute only where complete windows exist.

- Extractor: [prepare_droid.py](../scripts/real_video/prepare_droid.py)
- CPU job: [prepare_droid.slurm](../scripts/real_video/prepare_droid.slurm)
- Tests: [test_prepare_droid.py](../scripts/real_video/test_prepare_droid.py)
- Runtime and usage: [DROID.md](../environments/real_video/DROID.md)
- Completed audit: [data_audit.json](../data/real_video/droid_selected/processed/data_audit.json)
- Dataset manifest: [manifest.json](../data/real_video/droid_selected/processed/manifest.json)
- Raw download receipt: [download_receipt.json](../data/real_video/droid_selected/raw/download_receipt.json)
- Publisher license: [CC-BY-4.0](../data/real_video/droid_selected/raw/1.0.0/CC-BY-4.0)

Exact final manifest SHA256: `0f89fa0f3ddfcc27f01e4926c1b1e5e1a039aac1111c4c13be9e454c57f34280`. Extractor SHA256: `6bd753dad9cf1b744bdc3a7367f071db4c37ca53ffa00bc0b37ac3c25298661d`. Combined isolated runtime lock SHA256: `858961e8d3725538752a84f4e85d88c154e1bc57ef26fb4510f90a0cb1ab8a86`. Downloaded license SHA256: `7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661`. The runtime migration preserves the original Open-H lock; the project's main `.venv` was unchanged.

## Real recordings available for inspection

The selected-subset preview uses its first deterministically sorted episode and shows all three actual camera streams. No success or model outcome was used to choose it. Its contact sheet was visually inspected after extraction.

- [Recorded three-camera MP4](../data/real_video/droid_selected/processed/preview/recorded_three_camera_episode.mp4)
- [Recorded contact sheet](../data/real_video/droid_selected/processed/preview/recorded_three_camera_contact_sheet.png)
- [Preview provenance](../data/real_video/droid_selected/processed/preview/provenance.json)

The separate [DROID100 ingestion preview](../data/real_video/droid_100/schema_audit/preview/recorded_three_camera_episode.mp4) decodes one 166-record example and is labeled `schema_audit_only`; it is not included in the registered training subset. Forecasting comparisons require the subsequent feature cache, matched training, and held-out evaluation. Recorded images alone do not establish model quality or real-robot control success.
