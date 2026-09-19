# Complete manuscript input gallery

The minimum complete task gallery has **eight representatives**, plus **one
separately labeled ingestion-only example**. These are input illustrations.

| Task | Image path (repository relative) | Native size | Scientific scope |
|---|---|---:|---|
| PushT simulation | `paper/figures/split_assets/pusht.png` | 224×224 | Completed historical context-model study |
| Reacher simulation | `paper/figures/split_assets/reacher.png` | 224×224 | Completed historical context-model study |
| Drone simulation | `paper/figure_sources/benchmark_gallery/assets/drone_drone-train-s53000-d0_frame0.png` | 128×128 | Completed historical development study; final test pending |
| Tissue manipulation simulation | `paper/figure_sources/benchmark_gallery/assets/surgery_train-s4200000-d0_frame0.png` | 128×128 | Completed historical development study; final test pending |
| DROID | `paper/figure_sources/spatial_qualitative/case1_recorded_frame_10.png` | 320×180 | Completed current spatial-decoder development evidence |
| IWS PushT | `paper/figure_sources/teaser_gallery/assets/pusht_train_000011_frame0.png` | 640×480 | Ongoing single-observation decoder transfer |
| IWS Box | `paper/figure_sources/teaser_gallery/assets/bimanual_box_train_000011_frame0.png` | 640×480 | Ongoing single-observation decoder transfer |
| IWS Rope | `paper/figure_sources/teaser_gallery/assets/bimanual_rope_train_000011_frame0.png` | 640×480 | Ongoing single-observation decoder transfer |
| Open-H · input only | `paper/figure_sources/benchmark_gallery/assets/openh_episode_000000_frame160.png` | 640×480 | Physical phantom ingestion sample; no model trained/evaluated |

`benchmark_inventory.json` records exact image hashes, manuscript locations,
selection, split, aspect ratio and provenance for all nine. The drone and tissue
PNGs are frame0 of the first lexicographic training trajectory; no validation,
test or performance-selected payload was opened to create them. Open-H uses a
visually clear native-resolution frame160 of the already audited episode0,
selected from nine uniformly spaced frames without model or outcome information. No image
was cropped or enhanced. Its source is a physical phantom, not patient surgery.

Keep the four simulation studies, current DROID result route, ongoing IWS study
and Open-H input-only scope visibly distinct. Do not imply all samples were
processed by one checkpoint. Camera views, photometric transformations, seeds
and dataset-split revisions are not additional datasets needing duplicate tiles.

The new three images can be re-extracted with `extract_training_examples.py`
when the pinned local collections exist; rendering only requires the public
PNGs. See `ATTRIBUTION.md` and the original DROID/IWS provenance. IWS's dataset
card does not specify a license; its separate model license is not substituted.
