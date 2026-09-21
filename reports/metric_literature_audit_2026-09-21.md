# Metrics for ShiftWM: literature audit and completion plan

Checked 21 September 2026 against original papers, official implementations and
the project's completed evaluation records. This audit adds no experimental
scores and does not change the already declared primary comparisons or select
models using additional test metrics. The 20 September metric inventory is a
historical snapshot: its pending IWS and external-baseline counts are outdated.

## Main finding

Absolute feature MSE is a useful prediction metric when methods share the same
encoder, target coordinates, normalization, observations, actions and horizons.
Relative MSE reduction, `100 * (reference - method) / reference`, is a derived
effect size whose denominator must be named. It is not an additional independent
metric, a success rate, or a percentage-point improvement. For example, current
DROID h10 standardized MSE changes from 0.203950 (AR) to 0.193141 (bounded
ShiftWM), corresponding to a 5.30% reduction under that matched protocol.

Prediction error, decoded image quality and executed task outcome answer
different questions. The paper needs complementary evidence appropriate to its
claims. Our current spatial decoder has recorded-video forecasting results;
the existing simulator control experiments use a different historical model.

## What closely related papers measure

| Primary source | Setting | Reported metrics and protocol | Relevance |
|---|---|---|---|
| [DINO-WM, Table 1 and Appendix A.7](https://arxiv.org/html/2411.04983v2) | Simulated Maze, Wall, Reach, PushT; deformable Rope/Granular | Planning success for the first four tasks; Chamfer distance for particle-set goals. Separately, decoder-based LPIPS/SSIM. | A squared latent planning objective is not the resulting task score. |
| [LeWorldModel, §4 and Appendices E–F](https://arxiv.org/html/2603.19312v1) | Two-Room, Reacher, PushT, OGBench-Cube | Planning success and planning runtime; physical-state probes use MSE and Pearson correlation. | Report physical success under matched goal sampling, input access and action budgets. |
| [Fast LeWorldModel, §4.2–4.4](https://arxiv.org/html/2606.26217v1) | Same four task families | Success, dynamics runtime, full CEM solve time, open-loop latent loss and physical-state probe error. | A useful comparison design for action-prefix predictors; it is an additional relevant external method, not a completed ShiftWM comparison. |
| [RLA-WM, Table 1, §4.1 and Table A2](https://arxiv.org/html/2605.07079v1) | Recorded IWS PushT, Box and Rope | Final-frame DINO-token L1, LPIPS, SSIM and inference FLOPs; 60-step IWS forecast through four action chunks of 15. RGB metrics require its pretrained token-to-image decoder. | Directly relevant prediction metrics, but representation and evaluation preprocessing must match before comparing numbers. |
| [Interactive World Simulator, Table I and §IV-C](https://arxiv.org/html/2603.08546) | Original IWS video prediction and policy evaluation | RGB MSE, LPIPS, FID, PSNR, SSIM, UIQI and FVD over 192-step predictions; separate task scores and simulator/real policy comparisons | Its RGB MSE is not our feature MSE; downloading its recordings does not reproduce its policy experiments. |
| [DROID dataset paper, §V](https://arxiv.org/html/2403.12945v2) and [V-JEPA 2, §4](https://arxiv.org/html/2506.09985v1) | Physical robot policies; DROID-trained planning models | Executed manipulation success and robustness; V-JEPA 2 also compares planning time. | Downloading DROID recordings does not reproduce those robot evaluations. |
| [JEPA-WMs, §5.1 and Appendices E/G.2](https://arxiv.org/html/2512.24497) | Simulated tasks and a separate DROID-like recorded robot evaluation | Success in executable environments; offline planned-action L1 error and a rescaled Action Score for recordings; rollout embedding error, state-decoding error and LPIPS diagnostics. | An offline action-agreement evaluation is possible without a robot, but is still a proxy and needs a new planner evaluation. |

The JEPA-WM action-score study uses 16 videos collected by its authors, not our
DROID split. Its action error emphasizes the first three end-effector translation
coordinates; the score is `800 * (0.1 - E)` when `E < 0.1`, otherwise zero. We
should report interpretable action errors first, and use that score only under a
matched definition. Multiple action sequences can reach a goal, so disagreement
with one recorded demonstration does not by itself prove a plan is wrong.

[Decision-Metric Alignment, §3 and §5](https://arxiv.org/html/2608.18746)
additionally evaluates agreement between a model's ranking of candidate plans
and their simulated physical outcomes with Plan-Real and CEM-stage Spearman
correlations. This is a useful mechanistic diagnostic for us, but requires
executing candidate plans from matched simulator states. It cannot be obtained
from the existing recorded-action feature-MSE tables alone.

## Exact benchmark definitions matter

- Simulated DINO-WM PushT uses the joint pusher-plus-block position norm below
  20 native units and orientation error below pi/9. It is not an image-overlap
  score. [Pinned official implementation](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/env/pusht/pusht_wrapper.py#L54-L68).
- LeWM Reacher's configuration-matching task requires each joint error below
  0.05 radians. A wrapped joint-distance diagnostic is not an interchangeable
  success definition. [Environment implementation](https://github.com/galilai-group/stable-worldmodel/blob/4821c8e6a3f0f83b7e6a80da3a757e026ea9026b/stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py).
- DINO-WM deformable Chamfer distance sums the two mean nearest-neighbor
  Euclidean distances between achieved and goal particle sets; the distances
  are unsquared. It requires particle ground truth, unavailable from an RGB
  rope recording alone. [Official implementation](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/env/deformable_env/FlexEnvWrapper.py#L25-L50).

These simulator definitions must remain distinct from real bimanual IWS task
scoring and from the full action, goal and dataset contracts of each study.
Original IWS real T-pushing uses maximum block/target intersection-over-union
within 600 steps, while rope routing counts threaded clips within 200 steps.
These require executed policy outcomes or a validated simulator evaluation;
forecasting the recorded actions does not measure them. The paper includes Box
Packing in prediction experiments but does not define it among those four
policy-score tasks. [IWS §IV-A–C](https://arxiv.org/html/2603.08546).

### Reproduction details checked in official code

- Original IWS's RGB PSNR/SSIM use `data_range=2.0`; the metric pipeline
  operates on images in [-1,1], with separate conversion for FID.
  [Pinned metric implementation](https://github.com/WangYixuan12/interactive_world_sim/blob/3ba69b41d070d7a56e79c287da5b24a7d90742aa/interactive_world_sim/utils/logging_utils.py#L105-L170).
- RLA-WM directly compares DINO tokens with `F.l1_loss`; its preprocessing uses
  masked images and its RGB decoder outputs are clamped to [0,1]. The
  [pinned predictor](https://github.com/mlzxy/rla-wm/blob/6f19048758699bf9a152eaed5ac6dbf1caa07c18/eval/predictors/rla_wm_predictor_iws.py#L216-L325)
  and [LPIPS normalization](https://github.com/mlzxy/rla-wm/blob/6f19048758699bf9a152eaed5ac6dbf1caa07c18/src/utils/loss_utils.py#L322-L344)
  differ from our full-image DINOv2-small forecasting interface. Published
  DINOv3-Large scores cannot be copied into a common-number comparison.
- RLA-WM's current predictor header says 120 steps while the paper's experiment
  specifies 60; its handle averaging also differs from our equal-trajectory
  aggregation. Pin actual evaluated handles/horizons and reconcile these
  differences before claiming reproduction. Task folder names alone do not
  establish the same physical success criterion.

## What we already have, and what is missing

| Our study | Existing completed measurements | Additional evidence and cost |
|---|---|---|
| Current DROID spatial decoder | Absolute standardized feature MSE on native 4x4 and separately standardized pooled 2x2 grids, all ten horizons, episode/session records and paired intervals | Complete error curves and explicit mean-over-horizons summaries need only existing ledgers. MAE, raw-feature L1 and cosine require an additional frozen-checkpoint evaluation; scalar MSE records cannot recover them. |
| Current IWS decoder, all three tasks | Standardized MSE, standardized MAE, raw DINOv2 L1 and cosine distance at all 59 forecast offsets, plus separately evaluated prefix calls | Absolute four-metric scorecards already exist. Further temporal summaries can reuse the saved arrays. Keep full-H60 offsets distinct from separate H15/H30/H45 calls. |
| Historical simulator PushT | Success counts/rates, final pusher/block position and block angle errors, interaction costs; feature MSE secondary | Promote the saved physical-error summaries and success-versus-budget results. Dense state paths or overlap scores need further valid reconstruction/evaluation. |
| Historical simulator Reacher | Success, final joint errors, commands to success/budget; feature MSE secondary | Report physical-error distributions and budget curves with the exact existing success criterion. |
| Historical drone/tissue extensions | Development success, task-native final distances, native-step traces and failures | Summarize distances and failure modes, explicitly retaining development scope; these are not current spatial-model or clinical results. |
| Current spatial decoder: closed-loop control | No completed simulator/robot control study for this model | A new matched experiment must measure success and physical goal errors for this decoder and external controls. Historical model results cannot fill these rows. |
| RGB prediction | No validated RGB decoder for current ShiftWM | A shared train-only decoder and its target-feature reconstruction reference are needed before LPIPS/SSIM/PSNR. FVD additionally requires adequate predicted clips and a fixed feature-extraction/sampling protocol. |

## Recommended paper metric hierarchy

1. **Keep the existing primary forecast endpoints.** Main cells are absolute
   scores with units, encoder, normalization and horizon stated; show relative
   gain as a secondary annotation against a named reference. Preserve every
   declared comparison and its unfavorable outcomes.
2. **Show complementary absolute prediction errors.** IWS already supports
   MSE, MAE, raw-feature L1 and cosine. Complete the equivalent DROID secondary
   evaluation using frozen checkpoints and the same population. These are
   related diagnostics, not four independent proofs of success.
3. **Measure the current model's utility.** For executable simulator tasks,
   prioritize closed-loop success, final physical goal error and interaction
   budget. For recorded DROID, consider planned-action errors as an explicitly
   offline auxiliary experiment. This does not establish physical robot success.
4. **Add decoded-image quality only with a validated decoder.** The existing
   real RGB frames in figures are recorded observations, not generated forecasts.
   Use a shared decoder per representation, no target-conditioned decoding,
   train-only fitting, and target-feature reconstruction as a reference. Our
   pooled DINOv2 features do not directly match RLA-WM's DINOv3 decoder inputs.
5. **Support efficiency and reliability claims directly.** Measure predictor
   latency, encoder-inclusive latency, complete CEM time when relevant, peak
   memory and training budget on matched hardware/precision. Use independent
   episodes or sessions and model seeds for uncertainty; do not count nearby
   video windows as independent trials.

Existing primary metrics are not retrospectively replaced. Newly selected
diagnostics are secondary; a future confirmatory experiment needs its metric,
model-selection and split decisions fixed before evaluating its test outcomes.

## Local evidence inspected

- `paper/sections/main_evaluation.tex` and `paper/sections/main_results.tex`
- `paper/generated/qualitative_closest_v1/closest_table.tex`
- `paper/generated/iws_reserved_evidence/scores.tex`
- `scripts/real_video_spatial/evaluate.py` and
  `reports/real_video_spatial/transport_s0_validation.json`
- `scripts/real_video_iws/evaluate.py` and
  `reports/real_video_iws_reserved_recovery_v2/evaluations/*.npz`
- `src/shiftwm/evaluate.py`, `results/world/*/planning_test.json`
- `src/shiftwm/extensions/evaluate.py`,
  `results/extensions_v1/*/development/planning/results.json`
- `reports/upstream_planning_reproduction.md`

Research/code audit only: no new training, metric scoring or model promotion
was performed for this document. The manuscript wording update clarifies the
existing measurements; it does not populate missing results.
