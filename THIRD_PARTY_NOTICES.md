# Third-party notices

This document covers the upstream code, pretrained checkpoints, and official
source datasets used by the current compact-world-model implementation. Exact
artifact commits, file SHA256 values, byte sizes, and license-verification URLs
are recorded in [world_artifact_sources.json](references/world_artifact_sources.json).

## LeWorldModel source

- Project: [LeWorldModel](https://github.com/lucas-maes/le-wm).
- Pinned revision: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- License: [MIT at the pinned revision](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/LICENSE).
- Copyright notice: Copyright (c) 2026 Lucas Maes.

The `jepa.py` and `module.py` files in
[`src/shiftwm/vendor/lewm`](src/shiftwm/vendor/lewm) are copied without
modification from that revision. The complete upstream license is preserved in
[`LICENSE`](src/shiftwm/vendor/lewm/LICENSE); individual source hashes and the
copying record are preserved in [`NOTICE.json`](src/shiftwm/vendor/lewm/NOTICE.json).
The package loader verifies these hashes. The original notice must accompany
redistributed copies or substantial portions of that source.

The work is credited as *LeWorldModel: Stable End-to-End Joint-Embedding
Predictive Architecture from Pixels*, Lucas Maes, Quentin Le Lidec, Damien
Scieur, Yann LeCun, and Randall Balestriero (2026). Its architecture, released
weights, and source implementation are upstream contributions; the present
project does not claim them as new work.

## stable-worldmodel source and simulator/planner infrastructure

- Project: [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).
- Pinned revision: `4821c8e6a3f0f83b7e6a80da3a757e026ea9026b`.
- License: [MIT at the pinned revision](https://github.com/galilai-group/stable-worldmodel/blob/4821c8e6a3f0f83b7e6a80da3a757e026ea9026b/LICENSE).
- Copyright notice: Copyright (c) 2026 GalilAI-group.

This dependency supplies the PushT/Reacher environment infrastructure and the
unchanged CEM solver. The bootstrap script fetches the exact upstream checkout,
including its license. Simulator engines and assets installed through their
own packages retain their respective notices. Reusing this implementation does
not imply that the project's custom shift experiments reproduce published scores.

## Released LeWorldModel weights

The following model cards declare **MIT**. Each declaration was checked through
the Hugging Face API at the exact pinned revision on 2026-09-18; the API-reported
SHA matches the requested revision. The evidence URLs are included in the
versioned artifact manifest.

| Model | Pinned model-card revision | License |
|---|---|---|
| `quentinll/lewm-pusht` | [22b330c28c27ead4bfd1888615af1340e3fe9052](https://huggingface.co/quentinll/lewm-pusht/blob/22b330c28c27ead4bfd1888615af1340e3fe9052/README.md) | MIT |
| `quentinll/lewm-reacher` | [62adae4b71dc474ddf8f794c476ebfe737a743ca](https://huggingface.co/quentinll/lewm-reacher/blob/62adae4b71dc474ddf8f794c476ebfe737a743ca/README.md) | MIT |

Both packages contain `weights.pt` and `config.json`. This project loads them with
strict state-dictionary checks. Frozen exports are labeled unchanged upstream
weights; learned exports identify the specific base checkpoint and subsequent
training configuration. Preserve the base attribution and license notices when
redistributing either type of checkpoint. The original card declaration is the
source of the model license; a separate upstream model-repository LICENSE file
is not assumed to exist.

## Official source datasets and derived normalization statistics

These dataset cards also declare **MIT** at their pinned revisions, verified
through their revision-specific Hugging Face APIs on 2026-09-18.

| Dataset archive | Pinned dataset-card revision | License |
|---|---|---|
| `quentinll/lewm-pusht` / `pusht_expert_train.h5.zst` | [655cd446b9929369d7d406001da85c15d1457850](https://huggingface.co/datasets/quentinll/lewm-pusht/blob/655cd446b9929369d7d406001da85c15d1457850/README.md) | MIT |
| `quentinll/lewm-reacher` / `reacher.tar.zst` | [e70a080d0d04c6072123c9ebd343acf7fff28dbf](https://huggingface.co/datasets/quentinll/lewm-reacher/blob/e70a080d0d04c6072123c9ebd343acf7fff28dbf/README.md) | MIT |

The original archive SHA256 values and sizes were measured during the completed
statistics-extraction stage. The versioned manifest retains those values and
the exact derived action-statistic JSON files, including source archive hashes,
row counts, means, and sample standard deviations. This permits restoring the
verified normalization without re-downloading the large archives. The optional
`--datasets` flag downloads and verifies the original archives when needed.

The project's newly collected shifted trajectories are separately identified by
their collection manifests. They are not presented as the original released
dataset, and no medical data is included in that original simulator campaign.

## Other runtime dependencies

The tested package versions are recorded in
[`requirements.lock.txt`](requirements.lock.txt). PyTorch, Transformers, einops,
NumPy, Gymnasium, MuJoCo/dm-control, Pymunk, and other installed dependencies retain
their own licenses and notices. The source project license does not replace
those licenses. No third-party license text or attribution is removed by the
model packaging scripts.

## Real-video extension

- **DROID**: the selected recordings and recorded commands come from the official
  [DROID dataset](https://droid-dataset.github.io/) under its CC BY 4.0 license.
  The source inventory and immutable object generations are recorded in
  `reports/evidence/real_video/droid_selected_inventory.json`. The 1,126-episode
  study is a prespecified subset, not the full DROID policy benchmark. Qualitative
  real images retain DROID attribution in captions and source ledgers.
- **DINOv2-small**: the frozen encoder uses
  [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056` (Apache 2.0). Exact downloaded-file
  hashes are in `references/real_dinov2_sources.json`. Source Git includes the
  manifest and verified downloader, not the encoder weight file.
- **Open-H**: the separate ingestion illustration is from the CUHK physical
  endoscopy phantom portion of
  [PhysicalAI-Robotics-Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment),
  revision `e29dda7cabf2a2626634c7822db27695553ae523`, declared CC BY 4.0. It is a
  recorded physical phantom, not patient surgery, and is not included as a model
  performance benchmark. The exact sample audit is `reports/real_openh_video_audit.md`.

## Simulator extensions and external baselines

The teaser includes one unchanged internal-training frame per IWS task
(PushT, bimanual box and bimanual rope) from the
[RLA-WM dataset release](https://huggingface.co/datasets/xyzhang368/RLA-WM/tree/34bd8a8cbf3fa68e09ebd69aa35cb673279f4fc2).
Selection, source-video hashes and pixel hashes are recorded in
`paper/figure_sources/teaser_gallery/asset_manifest.json`. These are attributed
input illustrations, not predicted images or performance evidence. The pinned
dataset card does not specify a license; the separate model release's license
is not applied to these data. No raw IWS video archive is included.

Drone and surgical experiments depend on separately installed upstream projects.
Their environment-specific setup documents preserve repository and license
provenance. No external source checkout, engine binary or copied third-party
research-paper PDF is included in this public snapshot. The official AdaJEPA
reproduction is recorded under `scripts/baselines/` and `environments/adajepa/`;
its observation and planning protocol differs from the ShiftWM experiments.

The paper-wide teaser also includes project-rendered input observations from
PushT, Reacher, the adapted drone task and LapGym tissue manipulation. These
belong to separately documented historical simulation studies. Drone/tissue
frame extraction and the Open-H input illustration are recorded in
`paper/figure_sources/benchmark_gallery/asset_manifest.json` and
`paper/figure_sources/benchmark_gallery/ATTRIBUTION.md`. The Open-H image is
an ingestion illustration, not a trained-model result. The camera foreground
is explicitly labeled generated conceptual artwork; its prompt and provenance
are in `paper/figure_sources/visual_story_references_v1/`.

The ICLR manuscript style files are distributed with their original template
notices and provenance. Figure or paper release does not replace data/model
license obligations. No patient dataset is redistributed through this source
repository.
