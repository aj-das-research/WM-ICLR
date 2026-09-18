# IWS temporal and command semantics: independent source/metadata audit

**The released 600 evaluation windows are valid, but “60-step” means 60 stored command rows between image indices `s` and `s+59`. The official four-chunk rollout has an unresolved boundary convention.** Preserve the official evaluator when reproducing its scores; do not silently alter chunk overlap, horizon labels, actions, or endpoints.

Scope: clean upstream commit `6f19048758699bf9a152eaed5ac6dbf1caa07c18`, the verified downloaded IWS archive, all 3,489 training HDF5 metadata records, shapes/names for 60 validation metadata records, and all three official handle files. No video frames were decoded, validation command values read, or models loaded/evaluated. Container headers were inspected for the 30 evaluation trajectories' RGB and mask streams. Source and input hashes are in `reports/evidence/iws_temporal_semantics_audit.json`; per-episode metadata identities and shapes are in `reports/evidence/iws_temporal_metadata_manifest.json`. Reproduce with `scripts/real_video_development/audit_iws_temporal_semantics.py`.

## Exact upstream convention

The following is directly established by the pinned source, including execution of its indexing methods with a recording stub that cannot access video observations:

| Operation | Observed/target image indices | Command rows passed |
|---|---|---|
| Training example at maximum `H=15`, stride 1 | `s`, `s+14` | `s` through `s+14`, inclusive: 15 rows |
| Released evaluation handle `H=60`, stride 1 | `s`, `s+59` | `s` through `s+59`, inclusive: 60 rows |
| Official rollout chunk 1 | Starts from the observed latent; predicts a latent | Relative command rows 0–14 |
| Official rollout chunk 2 | Starts from chunk 1's prediction | Relative command rows 15–29 |
| Official rollout chunk 3 | Starts from chunk 2's prediction | Relative command rows 30–44 |
| Official rollout chunk 4 | Starts from chunk 3's prediction; scored against image `s+59` | Relative command rows 45–59 |

`TrajectoryDataset.frame_index_to_trajectory_data` constructs `range(s, s+(H-1)*stride+1, stride)`. `_compute_sampled_frame_indices(2,H)` selects `[0,H-1]`. The IWS configs use stride 1, two RGB frames, training horizons 2–15, and `directly_use_target_qpos: true`. The trainer uses the first and second image as start/target, and passes the complete `target_qpos` tensor directly to the model without the alternative normalizer branch's final-row removal. The action encoder accepts up to 15 rows. The evaluator slices disjoint 15-row chunks, resets the action-row offset by 15 each time, and feeds each predicted latent into the next chunk. Its introductory “120 / eight chunks” comment is stale: the actual released handles specify 60 / four chunks.

Primary source locations: [dataset indexing](../../external/rla-wm/src/datasets/trajectory_dataset.py), [IWS predictor](../../external/rla-wm/eval/predictors/rla_wm_predictor_iws.py), [trainer conditioning](../../external/rla-wm/src/trainers/rla_wm_trainer.py), [action encoder](../../external/rla-wm/src/models/rla_wm.py), and the three [IWS configurations](../../external/rla-wm/configs/rla_wm).

**Inference, not an established physical-time correction:** composing four maps with the maximum training endpoint span would nominally cover `4×14=56` stored intervals, while the evaluation endpoint spans 59. Also, the next action block starts at row 15 after the preceding trained map's endpoint row 14. The model outputs no timestamp, and command/image acquisition lead–lag is undocumented here. Therefore this arithmetic demonstrates a boundary-alignment ambiguity, not proof that the final prediction corresponds to a particular earlier physical time. Overlapping rows, changing chunk sizes, or changing the target would create a different evaluation protocol.

The upstream indexing additionally reserves one unused tail frame: at stride 1 its feasible horizon is `N-s-1`, not `N-s`. All released handles match this exact legacy rule. `datalib` clips video indices to valid bounds, so an explicit external bounds audit is useful; the released handles need no clipping.

## Completed metadata and bounds checks

| Task | Training trajectories | Validation trajectories | Command width (`target_qpos`) | State width (`qpos`) | Training stored rows |
|---|---:|---:|---:|---:|---:|
| Box | 602 | 10 | 14 | 14 | 120,274 |
| Rope | 602 | 10 | 8 | 14 | 120,312 |
| Sweep | 575 | 10 | 4 | 14 | 114,890 |
| PushT | 600 | 10 | 4 | 14 | 119,887 |
| Chain in box | 601 | 10 | 4 | 7 | 120,093 |
| Grasp | 509 | 10 | 4 | 7 | 101,729 |

All training command, qpos, root-pose, camera-intrinsic, and camera-extrinsic arrays are finite; their first dimensions agree within every metadata file. The declared command/state dimensions agree with the downloaded conversion summary. The generic `target_qpos` name must not be interpreted as a documented full joint-position vector: for example, PushT has four command coordinates and fourteen state coordinates. Coordinate meanings and physical units remain unverified.

Each official task has 200 distinct windows covering all 10 of its validation trajectories. All 600 windows passed exact index reconstruction, official split/scene/camera identity checks, command/image endpoint bounds, and the legacy feasible-horizon check. Box starts range from 2 to 139; PushT and rope from 0 to 139. Endpoints remain at or before index 198. At least one stored frame remains after every endpoint. The 60 relevant RGB/mask stream headers match their trajectory's metadata length, are 640×480, and report a 30/1 frame rate. Header inspection requested neither frame decoding nor frame enumeration.

## Discrepancies retained, not repaired

Four published conversion-summary entries differ from the actual downloaded HDF5 lengths:

| Task/split | Published frame-count mean | Actual metadata mean | Consequence |
|---|---:|---:|---|
| Rope/train | 199.855481728 | 199.853820598 | Actual aggregate is one row lower |
| Rope/validation | 199.900000000 | 200.000000000 | All ten actual records and video streams have 200 rows/frames |
| Chain in box/train | 199.820299501 | 199.821963394 | Actual aggregate is one row higher |
| Chain in box/validation | 199.700000000 | 199.600000000 | Actual aggregate is one row lower |

The archive/acquisition identities are unchanged, and no trajectories or windows were dropped. These discrepancies concern the upstream summary; the official box/PushT/rope handles still fit the actual metadata and video containers. All-training video frame counts were not audited; training metadata finiteness does not establish that additional contract.

The handle files also preserve historical configuration paths under `configs/generation/final/wm_frontend/` that are absent from the pinned checkout. The released evaluation script instead points at `configs/rla_wm/iws_*.yaml`. Keep both identities in any reproduction record; do not claim that the unavailable historical configuration was inspected.

No metadata key or attribute supplies physical timestamps, frame rate, or a command timebase. The 30-fps container header alone does not establish the capture rate or conversion subsampling. No IWS conversion script or timestamp mapping was located in the pinned source. “59 stored intervals” is supported; a duration in physical seconds is not. The official trajectory split does not establish session, operator, or scene independence.

## Comparison contract to retain

An upstream reproduction should retain every released handle, its exact `[s,s+59]` endpoint pair, all 60 command rows, the four disjoint chunks, and the published metrics; separately disclose the indexing ambiguity and compatibility repairs. An alternative temporal alignment may be evaluated only as a separately named and registered variant, never silently substituted for the official baseline. Equal-information comparisons must also account for the official predictor conditioning on one initial image: extra observed-prefix images for an adaptation method would require a matched control. A DINOv2 experiment on these recordings remains distinct from reproducing the released DINOv3 RLA-WM checkpoint.

This audit supplies no new model performance result and selects no method based on validation outcomes.
