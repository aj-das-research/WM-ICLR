# Open-H real endoscopy sample: downloaded and checked

The public CUHK `find_greater_curvature` subset is a practical real-video extension: **recorded endoscopic camera video paired with two motor-speed commands**. It is footage from a physical stomach phantom, not rendered simulation, patient surgery, or generated video. The subset's own [pinned README](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/blob/e29dda7cabf2a2626634c7822db27695553ae523/Endoscopy/cuhk/openh_dataset_full/find_greater_curvature/README.md) identifies its human teleoperation and phantom setup. The [hosting card](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) lists CC-BY-4.0; the public API confirmed `gated:false` and `private:false` without credentials.

Revision pinned on 2026-09-19: `e29dda7cabf2a2626634c7822db27695553ae523`. No model was trained and no full-subset download was started.

| Measured item | Result |
|---|---|
| Entire named subset inventory | 462 MP4s + 462 Parquets + six metadata/card files |
| Exact remote bytes | **2,233,086,405 bytes** (2.23 GB; 2.08 GiB) |
| Released metadata | 462 episodes, 107,488 frames, 20 Hz, 640×480 RGB |
| Downloaded scope | All subset metadata + episode 0 MP4 and Parquet; **7,978,572 bytes** |
| Episode 0 | **257 decoded frames / 257 action-state rows**, 12.85 s video |
| Nominal alignment | PTS 0.00–12.80 s; maximum PTS–row timestamp error **0.382 microseconds** |
| Command dimensions | `m1_spd`, `m2_spd`; sample range −100 to +99 |
| Command variation | 175/257 rows have nonzero commands; 41 consecutive command changes |
| Full sample decode at 128×128 | 257 actual frames in **1.002 s**, about 256 FPS including ffmpeg startup; one CPU invocation |
| Real source video / decoded data | Download size and remote LFS SHA256 verified; all sampled numeric values finite |

The metadata totals were independently summed across all 462 episode entries and match `info.json`. Episode lengths range from 7 to 943 frames, so any later fixed-window loader must account for short episodes. The complete original video was decoded; an eight-frame contact sheet was also visually inspected and shows camera movement toward the phantom's target region. Decode throughput is a one-episode engineering measurement, not a training or whole-subset throughput claim. The 128×128 decode is an audit artifact; the original MP4 remains untouched at 640×480.

## View the actual evidence

- [Original episode 0 MP4, unchanged bytes](/home/abhijit.das/projects/TTA/data/real_video/openh_sample_v1/derived/episode_000000_original.mp4)
- [Contact sheet of eight actual source frames](/home/abhijit.das/projects/TTA/data/real_video/openh_sample_v1/derived/episode_000000_contact_sheet.jpg)
- [Machine-readable audit, source URLs, checksums and full schema](/home/abhijit.das/projects/TTA/reports/real_openh_video_audit.json)
- [Pinned remote inventory](/home/abhijit.das/projects/TTA/data/real_video/openh_sample_v1/remote_inventory.json)
- [Downloaded original Parquet](/home/abhijit.das/projects/TTA/data/real_video/openh_sample_v1/original/data/chunk-000/episode_000000.parquet)

## What the sample establishes—and what still needs checking

The frame indices are contiguous from zero and the original video has exactly one frame per Parquet row. However, nominal container timestamps agreeing with row timestamps **does not establish actual camera–motor acquisition synchronization, action-before/after-frame convention, or physical actuation latency**. The source README separately warns that its release timestamp checks do not establish physical video–kinematic alignment. Its small manual delay review is not a guarantee for every episode.

The published motor fields describe speed commands, but the physical unit/scaling is not fully specified. Do not label raw −100…99 values as radians/s or reuse the simulator's action normalization. The state has 13 tracker/motor/light fields; keep these out of image/action model inputs unless a separately declared privileged baseline explicitly uses them. Offline video can test action-conditioned prediction and action sensitivity; it cannot verify success of actions that were never executed.

This episode's `light_val` is constant at 100, so it provides **no measured real illumination-shift evidence**. The subset metadata lists every episode under training with empty validation and test ranges. Inspected episode metadata contains only episode ID, task, length and path; no operator/session identifier was found there. The README describes two operators, but a session/operator-disjoint split cannot be claimed from the inspected metadata.

A schema issue was found and handled: Arrow episode/frame/task/index columns are singleton fixed-size lists, whereas `timestamp` is scalar float32. The camera directory name contains a literal backslash-u escape; the downloader uses the exact remote inventory path and URL encoding rather than assuming a normalized Unicode name. These are loader compatibility issues, not evidence of corrupt video.

## Appropriate next experiment, after approval of its protocol

A bounded offline real-video evaluation is feasible. First download the 2.23 GB subset, validate every video/row count and timing sequence, and establish fixed episode-level splits with duplicate/adjacency checks. Report any inability to separate recording sessions. Then compare the same compact predictor with RGB-history-only, action-shuffled, constant-context and inferred-context controls, using identical encoder training and compute. Test future feature/image prediction and action sensitivity; keep the true motor history separate from state-based scoring. Any actual appearance/context shift must be measured from data rather than assumed from the dataset description.

No closed-loop phantom controller, clinical performance, cross-operator generalization, or improved model result follows from this sample audit.

## Reproduce the audit

The isolated runtime is `environments/real_video/.venv`, with packages pinned in `environments/real_video/requirements.lock.txt`. The existing project and simulator environments were not modified. The downloader refuses a changed remote revision in the same output directory, verifies original file lengths and remote LFS hashes, and uses no authentication.

```bash
environments/real_video/.venv/bin/python scripts/real_video/fetch_openh_sample.py
```

The run log is `reports/evidence/real_openh_fetch_final.log`. An initial schema inspection caught the singleton-list index representation before the completed report was generated; the final run succeeds with original bytes preserved.
