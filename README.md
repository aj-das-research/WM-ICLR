# ShiftWM: observation-anchored spatial mixing for compact world models

Research code, reproducible experiment configurations, measured results and
manuscript sources for observation-anchored spatial prediction, with the earlier
context-learning studies preserved for reproducibility. Built on pinned LeWorldModel modules, released encoders and
established simulator/planning implementations.

[Paper PDF](paper/world_model_draft.pdf) · [Reproduce](REPRODUCING.md) ·
[Project page](https://aj-das-research.github.io/WM-ICLR/) · [Local inference demo](demo/README.md) ·
[Download models](https://github.com/aj-das-research/WM-ICLR/releases) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

## Current proposed method

**ShiftWM (ours)** keeps the last observed feature grid fixed, mixes its patches
using the supplied action prefix, and adds a bounded correction. The main paper
presents one algorithm; no-mixing, no-bounding, no-context and no-action arms
are component ablations. Stable checkpoint IDs remain unchanged (`transport`,
historically Ours-5).

On 141 DROID development episodes, the completed three-seed study gives **5.30%
lower ten-step endpoint feature error versus matched autoregression**, and
**3.55% versus an additive anchor**. Mixing helps with and without bounding;
the incremental native-error effect of bounding is inconclusive. These are
validation findings, with fresh held-out confirmation still pending.
See the [spatial results](reports/real_video_spatial/results.md),
[component study](reports/real_video_spatial_components/finalization.json), and
[detailed architecture](paper/generated/editorial/spatial_architecture_main.pdf).
All 21 trained spatial/component models completed 30 epochs; 15 are in the
public spatial release and six additional component models are local.

## Historical context-model evidence

The results below concern the distinct earlier context model. They are preserved
with their original protocols and are not performance claims for the spatial
method proposed in the current main paper.

The completed real-video campaign uses **1,126 actual DROID robot recordings**,
with session-disjoint training/validation/test splits, recorded robot commands
and three cameras. Twelve predictors completed 30 training epochs each; 48
held-out evaluations cover two camera views and two forecast lengths. These
models predict frozen DINO features, rather than generating RGB video.

| Real DROID setting | Original context model | Framewise | Persistence | Interpretation |
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

A subsequent [frozen confirmation](reports/real_droid_fresh_evaluation_results.md)
on 65 recordings from 52 additional sessions finds **0.742% lower five-block
error than equally calibrated Framewise**, with a paired interval excluding
zero. Both methods use the same training-fitted calibration procedure;
longer-horizon differences remain inconclusive. These sessions are now revealed
and are not used to select new revisions.

The completed [generalization study](reports/real_droid_generalization_results.md)
adds 36 full training runs and 72 validation evaluations. Slower learning gives
0.29% / 0.32% gains against matched Framewise at five / ten blocks; stronger
weight decay and smaller capacity retain ten-block regressions. All outcomes
and all 36 selected models are included in the development release.

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

The separate [generalization development release](https://github.com/aj-das-research/WM-ICLR/releases/tag/generalization-v1)
adds **36 predictors**, their shared encoder, model cards and complete controls.
Its 443.3 MB archive and all 36 isolated offline reloads were verified. The
[simulation development release](https://github.com/aj-das-research/WM-ICLR/releases/tag/simulator-development-v1)
adds **42 predictors** for the drone and surgical-proxy simulators, including
six geometry-revision models. All 42 match their original cached-feature and
image-input forecasts exactly after offline relocation. Simulation results retain all mixed outcomes.

The [ten-step training control](reports/real_droid_horizon10_results.md) is complete:
12 further models trained for 30 epochs. On development validation, the earlier context model
reduces final-step error by 2.77% against its five-step-trained version and by
0.40% against equally ten-step-trained Framewise. All models selected epoch one;
longer training still overfits. All twelve are now in the [ten-step development release](https://github.com/aj-das-research/WM-ICLR/releases/tag/horizon10-development-v1),
with exact offline parity and verified public downloads.

The [spatial architecture study](reports/real_video_spatial/results.md) is also
complete: 15 models trained for 30 epochs. At step ten, spatial transport (ours)
reduces native feature error by **5.30%** against matched autoregression and
**3.55%** against observation anchoring alone. The separate context module has
no clear benefit in this study. These are development-validation comparisons,
with all outcomes and paired intervals retained. All 15 models are available in
the [spatial model release](https://github.com/aj-das-research/WM-ICLR/releases/tag/spatial-world-models-v1),
with exact offline prediction parity and verified public downloads.
**117 trained predictors** are public across five releases.

The [calibration development study](reports/real_droid_residual_calibration_results.md)
reports a 0.779% five-block improvement over equally calibrated Framewise on
validation recordings. The ten-block difference remains inconclusive; these
development results are separate from the original held-out table above.

The paper and source are synchronized with Overleaf through the server's
[publishing bridge](docs/PUBLISHING.md), which merges remote source edits and
updates GitHub Pages after the paper build and publication checks.

## Method and scope

### What is the real-video task?

**Forecast what the robot will see after a given sequence of commands.**
On our 1,126-episode DROID subset, the model receives three observed frames and
the recorded command sequence. A frozen image encoder turns the frames into
visual features; the world model predicts the features of later frames. During
evaluation, we compare those predictions with features extracted from the
withheld future frames. Smaller error means a more accurate forecast in that
encoder's coordinates.

Every RGB frame shown in our DROID figures is a real recorded input or reference
frame. The current released models predict visual features; they do not generate
future RGB video. Error maps show where feature predictions disagree with the
reference, rather than pixel reconstruction errors or object segmentations.
This tests a component needed by predictive robot planners; physical closed-loop
control has not been evaluated on DROID. The simulator experiments separately
evaluate goal-directed planning.

DROID supplies paired observations and commands across varied robot interactions.
A separate, completed IWS study trains 27 predictors on PushT, Box and Rope
recordings, using one observed image and native command sequences. At the
long-horizon development endpoint, ShiftWM reduces standardized feature MSE
versus additive anchoring by **2.36% on PushT** and **1.90% on Box**; Rope is
effectively tied (**−0.01%**). Autoregression has lower endpoint MSE on all three
tasks in that original bounded-model comparison. Its results and paired
intervals remain in the development appendix; main Table 2 now reports the
complete reserved study. The IWS protocol remains separate from DROID.

All 27 IWS selected checkpoints have [local inference exports and exact relocated
CPU checks](reports/real_video_iws/release/README.md). Their weights are not yet
part of the 117 public predictors above. A separately registered nine-run
component study removes only the innovation bound, retaining matched inputs,
initialization, seeds and the full 30-epoch training budget. Its complete-result
gate has passed for all nine models. Removing `tanh` lowers H60 development
MSE by **7.47%, 9.50% and 11.60%** against bounded ShiftWM on PushT, Box and
Rope, and improves over matched autoregression on all three tasks. The
[complete follow-up](reports/real_video_iws_unbounded/development_finalization.json)
and [separate local inference bundle](reports/real_video_iws_unbounded/release/README.md)
retain every seed and the original study. A separate, complete reserved study
now evaluates all 36 frozen predictors on 600 original handles from 30
trajectories. The no-tanh ablation reduces endpoint MSE against autoregression
by **7.62%, 4.84% and 6.80%** on PushT, Box and Rope, with all three paired
95% intervals favoring it. The equal-task gain is **6.42% [4.90, 7.92]**.
The original bounded-versus-additive comparison is mixed across tasks and
remains visible. The [complete protocol and findings](docs/IWS_RESERVED_EVALUATION.md)
disclose the post-access numerical execution revision, unchanged checkpoints
and tolerance, and independent reconstruction of every ledger and interval.

A completed [range diagnostic](reports/iws_bound_diagnostic_v1/findings.md)
checks all 10,136 development windows: 0.94–1.36% of endpoint feature coordinates
lie outside the bounded decoder's permitted channel range. The resulting relaxed
error floors identify a representational restriction; they do not attribute the
entire no-tanh improvement to that restriction. Equation 5 and Table 10 document
the calculation and all three task results.

The paper now combines forecast curves and paired component intervals in one
main figure and puts all three IWS tasks beside paired endpoint gains in another.
Historical controls use compact horizontal interval plots. Source packs,
editable SVGs, PDFs and numerical reviews accompany these figures under `paper/`.

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

Publishing status and conflict-recovery commands are documented in [the synchronization guide](docs/PUBLISHING.md).
