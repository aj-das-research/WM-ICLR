# Completing the current model's evaluation

This is a working measurement plan, not a report of completed experiments. It
extends the 2026-09-21 literature audit. New measurements on already inspected
forecasting populations are secondary analyses; they do not change checkpoint
selection, existing primary endpoints, or previously published results.

## Scientific questions and ordering

1. **Decision quality:** does the current spatial architecture improve executed
   goal reaching under the same observation, action and computation budgets?
2. **Prediction quality:** do improvements persist across absolute feature
   errors, forecast horizons, seeds and trajectories?
3. **Visual fidelity:** can a shared decoder recover useful RGB forecasts, and
   how much error comes from the decoder rather than the predictor?
4. **Practical cost:** what accuracy is obtained for a measured inference and
   complete planning cost on the same hardware?
5. **Mechanism:** where do spatial mixing, correction and action conditioning
   change predictions and decisions? Attention-like source weights are model
   quantities, not proof of causal explanation or semantic localization.

Historical context-model simulations remain a separate study. The new simulator
campaign trains the current spatial architecture on simulator training episodes;
it is not zero-shot deployment of a DROID checkpoint.

## Complete metric coverage and prerequisites

| Family | Metrics and direction | Current applicable tasks | Required measurement contract |
|---|---|---|---|
| Feature forecasting | Standardized MSE/MAE ↓, raw DINO-feature L1 ↓, cosine distance ↓ | DROID and IWS PushT/Box/Rope; current simulator forecasts | Same encoder, coordinate grid, normalization, action access, horizons and aggregation; full paired uncertainty, not just relative changes. |
| Goal reaching | Executed success ↑, terminal physical goal error ↓, success-by-budget ↑, actions to first success ↓, failures | Simulated PushT/Reacher, drone, tissue | Actual simulator transitions. Include support cost, unchanged native success, failure-censored effort; separate task units and historical/current architectures. |
| PushT object geometry | Final/max block overlap (IoU) ↑, translation and orientation errors ↓ | Simulator PushT when geometry is reconstructed correctly | Additional geometry diagnostic, not a replacement for the upstream compound pusher+block success criterion. Recorded IWS photos alone do not supply physical IoU. |
| Deformable state | Chamfer distance ↓; rope clips threaded ↑ | Particle-state Rope/Granular and physical rope-routing environments, respectively | These are different tasks. Need matched particle states or actual instrumented routing execution; neither follows from IWS Rope feature error. |
| RGB frame quality | RGB MSE ↓, PSNR ↑, SSIM ↑, LPIPS ↓, UIQI ↑ | IWS and DROID only after shared-decoder validation | Train decoder on training frames only, select on development, freeze once for all methods; report reconstruction of ground-truth features as decoder reference and persistence. Pin resolution, pixel range, LPIPS backbone and temporal aggregation. |
| Distribution/video quality | FID ↓, FVD ↓ | Decoder-generated frames/clips | Pin feature-network weights, preprocessing, clip length/fps, sample count and covariance convention. Account for correlated overlapping windows; sparse trajectory populations limit interpretation. Not the mean of per-frame FID/FVD values. |
| Efficiency | Predictor and encoder-inclusive latency ↓, complete CEM time ↓, peak memory ↓, FLOPs ↓, parameters, training GPU-hours | All current model comparisons | Same batch/precision/hardware; synchronize CUDA, warm up, repeated measurements; include encoder and decoder when reported. Expose FLOP counter coverage, including unsupported/recurrent operators. |
| Physical-state probes | State prediction MSE ↓ and Pearson correlation ↑ | Simulators with state labels | Same training-only probe, development selection, held-out labels used only for scoring. State units/coordinates explicit; a probe score is not control success. |
| Offline inverse planning | Planned-action error ↓ | DROID if an independently registered offline plan protocol is implemented | Common action normalization, control horizon and dimensions. Demonstration actions are one feasible solution, not a unique optimum; no robot-success claim. |
| Executed real robot | Robot task success ↑ | DROID/V-JEPA-like physical execution | Requires access to a robot and a matched task protocol; unavailable from recorded video or this cluster alone. |

## Sources checked

- RLA-WM: https://arxiv.org/html/2605.07079v1#S4.SS1
- Original IWS: https://arxiv.org/html/2603.08546
- DINO-WM: https://arxiv.org/html/2411.04983v2
- LeWM: https://arxiv.org/html/2603.19312v1
- Fast-LeWM: https://arxiv.org/html/2606.26217v1
- V-JEPA 2: https://arxiv.org/html/2506.09985v1
- JEPA-WM: https://arxiv.org/html/2512.24497

Exact implementation differences and source pins are recorded in
`reports/metric_literature_audit_2026-09-21.md`.

## Executable work streams

- `droid_*`: all 21 internal learned checkpoints, 12 adapted DINO-WM checkpoints
  and persistence; all ten query offsets, unchanged 1,631 development windows.
  Recomputed MSE must agree with its original finalized per-window records before
  any new metric is accepted. No new training or model selection.
- `planning_*`: new current-spatial PushT/Reacher training and closed-loop
  evaluation. Exact splits, support budget, CEM settings, checkpoint selection,
  source hashes and completion gates are in its immutable registration.
- `physical_*`: read-only aggregation of previously completed historical
  simulation traces. Negative as well as positive comparisons are retained.
- Shared RGB decoder: separate training and scoring stream; no photographs or
  interpolated images can be counted as model-generated predictions.

## Presentation and completion

Report completed measurements beside explicitly marked pending measurements.
Never use zero for an unavailable metric. Each numerical report needs a complete
population/seed gate, source/checkpoint/data identities and units. A partial job
can update progress, but cannot silently replace a complete comparison table.
Use paired physical trajectory panels and success-by-budget curves for control;
matched RGB prediction/error panels only after decoding; common-scale signed
feature-error and source-weight maps for current forecast diagnostics. Keep
failure cases and uncertainty. New planning gains, decoded quality and faster
execution are hypotheses until the corresponding completed measurements exist.
