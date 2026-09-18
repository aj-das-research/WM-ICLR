# ShiftWM: compact world models under distribution shifts

Research code, reproducible experiment configurations, measured results and
manuscript sources for separating observation and dynamics context in compact
world models. Built on pinned LeWorldModel modules, released encoders and
established simulator/planning implementations.

[Paper PDF](paper/world_model_draft.pdf) · [Reproduce](REPRODUCING.md) ·
[Project page](https://aj-das-research.github.io/WM-ICLR/) · [Local inference demo](demo/README.md) ·
[Download models](https://github.com/aj-das-research/WM-ICLR/releases/tag/real-droid-v1) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

## Current evidence

The completed real-video campaign uses **1,126 actual DROID robot recordings**,
with session-disjoint training/validation/test splits, recorded robot commands
and three cameras. Twelve predictors completed 30 training epochs each; 48
held-out evaluations cover two camera views and two forecast lengths. These
models predict frozen DINO features, rather than generating RGB video.

| Real DROID setting | ShiftWM (ours) | Framewise | Persistence | Interpretation |
|---|---:|---:|---:|---|
| Primary camera, five action blocks | 0.138198 | 0.138479 | 0.142392 | 2.94% lower MSE than persistence; 0.20% vs Framewise is inconclusive |
| Second camera, five action blocks | 0.154190 | 0.154337 | 0.157372 | 2.02% lower MSE than persistence; no clear learned-baseline advantage |
| Primary camera, ten action blocks | 0.193525 | 0.189895 | 0.193748 | Worse point estimate than Framewise; interval includes zero |
| Second camera, ten action blocks | 0.222748 | 0.220020 | 0.221364 | Worse point estimates; intervals include zero |

Lower train-standardized feature MSE is better. Tables average within episodes,
then over episodes and three training seeds. The two cameras share recordings.
The complete [results](reports/real_droid_results.md) and
[independent interpretation](reports/real_droid_interpretation.md) retain
uncertainty, negative results, overfitting and weak action-order sensitivity.
These results do not establish state-of-the-art performance or physical robot
control. The research hypothesis remains under evaluation.

Simulation studies cover PushT, Reacher, drone navigation and a surgical
simulator. They are explicitly distinguished from recorded real videos and from
clinical data. Full [extension results](reports/completed_extension_results.md)
include positive, null and negative comparisons; exploratory revisions remain
separate from the original confirmatory protocols.

## Repository map

- `src/shiftwm/`: predictors, context modules, feature extraction, training,
  evaluation and pinned vendored LeWM components.
- `configs/`: frozen campaign configurations and separately named revisions.
- `scripts/`: downloads, data audits, campaigns, checkpoint exports and reporting.
- `environments/`: pinned isolated runtimes for data ingestion and simulators.
- `tests/`: causal data boundaries, training/resume integrity, model loading,
  evaluation and reporting checks.
- `reports/`: prespecified protocols, completed measurements and limitations.
- `paper/`: LaTeX source, generated tables/figures and compiled draft.
- `demo/` and `site/`: local inference application and public project page.

Start with [REPRODUCING.md](REPRODUCING.md). Raw datasets, downloaded encoders,
trained weights and environments are deliberately stored outside source Git.
Download scripts verify pinned public artifacts. Trained model packages have
been published in the [real-DROID model release](https://github.com/aj-das-research/WM-ICLR/releases/tag/real-droid-v1):
12 validation-selected predictors, their shared DINO encoder and 12 training-fitted
calibration wrappers, with source, model cards, licenses and checksums. The
245.7 MB archive passed exact offline inference parity and public download
verification. It supports feature forecasting from recorded video and commands.

The [calibration development study](reports/real_droid_residual_calibration_results.md)
reports a 0.779% five-block improvement over equally calibrated Framewise on
validation recordings. The ten-block difference remains inconclusive; these
development results are separate from the original held-out table above.

The paper and source are synchronized with Overleaf through the server's
[publishing bridge](docs/PUBLISHING.md), which merges remote source edits and
updates GitHub Pages after the paper build and publication checks.

## Method and scope

The simulator implementation freezes the pretrained visual encoder and adapts
small predictors and contexts. Canonical simulation renders supply privileged
training targets, whereas inference consumes shifted observations and executed
actions. The real-video variant uses no paired canonical renders and makes no
factor-identification claim. Its frozen image coordinates and recorded commands
support offline forecasting, not counterfactual physical-control validation.

Source identities, validation-only checkpoint selection, equal seed budgets,
matched task records and paired uncertainty are preserved. Missing results are
not filled with estimates. A [closest-work audit](reports/world_model_novelty_update_2026-09-19.md)
documents related approaches and limits novelty claims.

## Citation and licenses

This repository is an in-progress research artifact, not an accepted conference
paper. Author and publication metadata will be updated with the final manuscript.
Original project code is MIT; upstream code, datasets and models retain their
own licenses and required notices. See [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and pinned source manifests.
