# Architecture diagnosis from original training and validation

This is a development diagnosis, not a new test result or a novelty claim. Only the original training/validation diagnostics, model code, and registered generalization training summaries were consulted. No original test or fresh holdout results were used to choose the proposal. Exact source hashes, derived numbers, and the time-stamped campaign snapshot are in `architecture_diagnosis.sources.json` alongside this report.

**Recommendation:** first finish the matched ten-step training control, then test a spatial-token decoder that anchors predictions to actual observed features and learns bounded transport/innovation. A cheap reliability blend is useful as a baseline, but cannot recover information lost in the existing representation and is unlikely by itself to establish a major architectural contribution. Neither proposal guarantees large gains.

## What the evidence supports

The original diagnostic uses the same 141 validation episodes, 59 recording sessions, and 1,631 ten-step-eligible windows for all methods. Windows are averaged within episodes, episodes equally, and the three training seeds equally. Its five-step numbers differ from the calibration report because the latter admits additional five-step-eligible windows.

| Model/checkpoint | h1 error | h5 error | h10 error | h10 predicted displacement | h10 actual displacement |
|---|---:|---:|---:|---:|---:|
| Framewise, best | 0.055117 | 0.156712 | 0.216903 | 0.034187 | 0.225622 |
| Constant dynamics, best | 0.055111 | 0.156638 | 0.216746 | 0.034157 | 0.225622 |
| Factorized, best | 0.055003 | 0.156234 | 0.219507 | 0.051390 | 0.225622 |
| Factorized, last | 0.057116 | 0.188089 | 0.279258 | 0.168326 | 0.225622 |
| Action-free, best | 0.055189 | 0.157491 | 0.217435 | 0.027854 | 0.225622 |
| Persistence | 0.055625 | 0.163156 | 0.225622 | 0 | 0.225622 |

Errors and displacements are mean squared distances in frozen train-standardized feature coordinates. These are feature forecasts, not generated RGB videos or robot-control success rates.

1. **Later training amplifies unreliable displacement.** Factorized best-to-last h10 error rises 27.2%, while predicted displacement rises 3.28 times. In the separately recorded seed-0, 64-training-episode pilot, training h10 error falls from 0.194646 to 0.155465 while validation h10 rises from 0.220366 to 0.277711. This is evidence of overfitting, not merely an insufficient training budget. The pilot is not the full calibration campaign.
2. **Smaller predicted motion does not mean we should simply amplify it.** For prediction displacement `d`, true displacement `v`, and identical averaging, `E||d-v||² = A+B-2C`, where `A=E||d||²`, `B=E||v||²`, and `C=E<d,v>`. Factorized-best h10 gives `A=0.051390`, `B=0.225622`, `C=0.028752`, with aggregate alignment `C/sqrt(AB)=0.2670`. The validation-oracle common scale `C/A=0.5595` is below one. This oracle is a descriptive diagnostic only: it is not selected, deployed, or counted as a result. The problem includes inaccurate direction, not only amplitude. The exact calculation is recorded in the source ledger.
3. **The model does use actions.** Replacing future commands with commands from another recording changes Factorized-best h10 predictions by MSE 0.014696 and worsens h10 error by 5.17%; the corresponding Framewise prediction change is 0.002111 and error increase 0.99%. Action-free predictions remain unchanged. Reversal alone was a weak perturbation. These are sensitivity tests against the original observed future, not executed counterfactual robot experiments.
4. **Motion reliability varies, but an observable gate is not yet validated.** In the target-defined lowest future-motion tertile, Factorized-best h10 error is 0.123713 versus persistence 0.106219; in the highest tertile, it is 0.321605 versus persistence 0.356647. Future-motion strata use targets and cannot be gate inputs. Observed-support-motion tertiles yield h5 gains over persistence of approximately 4.67%, 3.56%, and 3.76%, which do not establish a useful monotonic reliability signal.
5. **Spatial compression is substantial.** On the registered 48 training frame pairs, median retained DINO patch-change energy is 14.42% at 2×2 pooling, 26.20% at 4×4, and 48.71% at 8×8, relative to the 16×16 patch grid. This is a pooling measurement, not proof of object localization or a causal explanation of forecast error. A trained, matched higher-resolution baseline is required.
6. **Five-step training does not establish ten-step robustness.** The existing loss supervises five recursively forecast steps. Parent-coordinated ten-step training of the original four modes, three seeds, is the first control. It should remain a distinct registered campaign; do not credit a new decoder for a gain produced by additional horizon supervision.

Oracle observation refresh lowers Factorized-best h10 error to 0.061602, but supplies true preceding observations and refreshes context. It changes the information available and cannot isolate recursive error from stale context, serve as an inference method, or support a deployable gain claim.

The full matched training-only scalar calibration is a useful cheap baseline: h10 Factorized improves from 0.219507 to 0.215874, but the fairly calibrated Framewise comparator is 0.216206; the paired improvement is only 0.154% and its exploratory interval includes zero. The h5 improvement over calibrated Framewise is 0.779%, with paired difference −0.001238 and interval [−0.002173, −0.000470]. All methods receive the same fitting opportunity. Scalar calibration contracts the completed trajectory around the last observation; it does not feed corrected predictions into recursive inference.

At this report's time-stamped snapshot, 22 of the 36 registered generalization runs had completed 30 epochs. The source ledger preserves each run's status and selected epoch. Partial arms must not be ranked; the finalizer requires all 36 models and 72 evaluations. Slower optimization moving checkpoint selection away from epoch one is not, by itself, a better model or evidence of generalization.

## Code-level constraints

`src/shiftwm/real_video/features.py` extracts a 16×16×384 frozen DINO grid after resizing each frame to 224×224, then average-pools to 2×2 and flattens to 1,536 coordinates. `src/shiftwm/real_video/model.py` projects this entire frame into 192 dimensions. Three observed frames and two observed action transitions initialize the contexts. Each action vector contains five recorded seven-dimensional commands. Contexts remain fixed as predicted features are recursively fed back and additive residuals accumulate.

The current dynamics-context GRU has only two observed transitions. It cannot identify an unrestricted action-to-dynamics map per episode. The loss is absolute standardized feature MSE; subtracting the same anchor from both target and prediction yields an algebraically identical delta loss unless its weighting, coordinates, or conditioning also change.

The existing feature extractor iterates the full manifest and handles additional test cameras. A development extractor must therefore explicitly filter original train/validation metadata **before any payload access**, reject test requests, and write a new immutable cache namespace. It must not reuse the current extractor's full-manifest loop unchanged.

## Ranked development candidates

### 1. Observation-anchored spatial transport with bounded innovation

Preserve a 4×4 grid of frozen 384-dimensional DINO tokens. Use a shared 384→96 projection, positional embeddings that preserve the exact grid layout, one small spatial-attention block, and the already pinned LeWM temporal predictor applied over each patch's three-frame history. Share a causal action-prefix encoder across all decoder variants. Reuse established DINO extraction, normalization/audit conventions, LeWM attention, package loading, and evaluation aggregation; implement only the decoder and necessary token-preserving interfaces in a new namespace.

For observed anchor tokens `Z_t`, action-prefix-conditioned transport `P_h`, gate `g_h`, and bounded innovation `R_h`, a candidate output is

`Zhat_h = (1-g_h) ⊙ Z_t + g_h ⊙ (P_h Z_t) + ε ⊙ tanh(R_h)`.

Each row of `P_h` sums to one, and `0 ≤ g_h ≤ 1`. All heads depend only on observed support and commands available through horizon h. Transport always reads the last **actual** observation, so repeated resampling of predicted tokens is avoided. Innovation permits new appearance and occlusion effects outside the anchor's convex hull. Features are semantic tokens; this is not a verified physical image warp or a correspondence ground truth.

Transport must mix channels in a common coordinate system, such as raw DINO coordinates or shared per-channel training normalization. Mixing patch-specific standardized coordinates without undoing their different offsets/scales would be incorrect. A train-defined, documented positive per-channel innovation budget gives `max_abs(Zhat_h) ≤ max_abs(Z_t)+max(ε)` in that coordinate system, independently of h. This elementary bounded-output statement does **not** prove forecast accuracy, physical safety, or global Lipschitz stability; `P_h` itself depends on the input.

The mechanism to test is reduced accumulated feature corruption while retaining local action-conditioned changes. It can materially change the representation/decoder ceiling, unlike blending two highly correlated completed trajectories. It can also fail: transport may blur patch features; spatial attention can overfit; bounded innovation can suppress genuine changes; a single anchor may omit disoccluded content. These are testable limitations, not reasons to hide negative outcomes.

**Matched campaign, at most 15 full runs:** five decoder/context arms, three seeds, 30 epochs, ten-step training in all arms:

| Arm | Same 4×4 input and shared temporal/action trunk | Decoder or removed component | Purpose |
|---|---|---|---|
| A | Yes | Additive autoregression | Token-preserving baseline; exposes gains from representation alone |
| B | Yes | Direct observation-anchored additive output | Separates removal of recursive feedback from transport constraints |
| C | Yes | Anchored transport plus bounded innovation | Proposed candidate |
| D | Yes | C with support-context conditioning disabled | Tests whether support-dependent adaptation adds value |
| E | Yes | C with all action inputs disabled | Tests action-dependent forecasting |

All arms need the same spatial mixing opportunity, loss, support frames, horizon, optimizer schedule, checkpoint selection rule, train/validation windows, and approximately matched trainable parameters. Report exact counts, runtime, and any residual mismatch. A transport/bound-specific further ablation is necessary before attributing gains separately to those two ingredients; it can replace an uninformative follow-up, but must not be silently selected using a test set. Frozen scalar calibration is available equally to every arm, with uncalibrated and calibrated rows both retained.

Evaluate native 4×4 error and deterministically pool **raw predictions** to 2×2 before using the unchanged original mean/std. The latter is a cross-resolution diagnostic. Do not compare native 4×4 and native 2×2 scalar errors directly, and do not attribute a resolution advantage to the transport algorithm. Verify the new ground-truth pooling reproduces old coordinates within documented floating-point tolerance. Keep the original cache untouched.

A 4×4 float32 train/validation cache for 56,595 existing frames is approximately 1.391 GB of feature payload, four times the old 2×2 feature payload; full 16×16 storage would be approximately 22.25 GB. There is no need to retain the full grid. The temporal predictor processes three tokens per patch with shared parameters; spatial attention processes sixteen tokens. Three GPUs can run three seeds concurrently. Exact throughput and memory must be measured on an allocated GPU before promising a 15-run completion time. Extraction/training submission needs parent coordination with the existing three-GPU quota.

### 2. Causal reliability adaptation from already observed prediction errors

A small ridge/Bayesian residual calibrator or learned reliability gate can use prequential errors from earlier observed frames. This is a useful, inexpensive comparator to static train-fitted shrinkage and persistence. It is more directly an adaptation mechanism, but generic residual fitting is not new and correlated donor forecasts can cap the improvement.

There are only three observed frames in the current protocol: that does not supply even one completely observed three-input→next-frame backtest for the same base predictor. A ten-step backtest needs thirteen observed frames. Every comparator must receive the same longer prefix, query start, eligible episodes, and action history; otherwise the gain is additional information. No oracle query frames or target-defined future-motion labels may enter the gate. A longer-prefix experiment is a separate protocol, not a replacement score for the current three-frame setting.

### 3. Multi-view supervision for dynamic factors

Training with synchronized real external-camera views could help separate view changes from controllable motion. It also introduces new preprocessing, correspondence/occlusion ambiguity, and privileged training information. Every control must receive the same views. This is a larger next-stage study, not the best immediate intervention under the current deadline.

Increasing backbone size, adding a full diffusion generator, or transplanting a large new action/world-model stack is lower priority until these controlled failures are understood.

## Novelty boundaries and established code

- [DINO-WM, official project](https://dino-wm.github.io/) and [official MIT-licensed code](https://github.com/gaoyuezhou/dino_wm) already establish frozen spatial visual features for action-conditioned world modeling. Preserving a patch grid is an established control, not a novelty claim. Pin and audit any imported module rather than copying an unversioned repository.
- [Finn et al., Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157) establishes image-transformation-based video prediction. Transport/warping alone is established.
- [ReDRAW](https://arxiv.org/abs/2504.02252) already studies residual dynamics learning for world-model adaptation. A generic residual adapter cannot be the claimed contribution.
- [RLA-WM official code](https://github.com/mlzxy/rla-wm) uses residual latent actions and spatial visual features. Its dependency/license audit is required before code reuse; its public availability does not automatically grant permission to copy every component.
- [Shortcut World Models](https://openreview.net/pdf?id=O8nGPjY7Tj), identified by the parent agent, is a specific prior to check for multi-horizon rollout claims. This agent could not retrieve the PDF through the browser challenge and does not claim a full independent review.

The plausible paper contribution is a precisely specified causal reliability/anchoring mechanism, supported by matched representation, horizon, action, and context controls across domains. The current evidence does not establish that contribution yet. Direct horizon conditioning, residual correction, a larger feature grid, or combining known modules is insufficient on its own.

## Artifact and validation contract for any new model

1. New module/config/cache/run namespaces; immutable original models, protocols, registrations, training records, and test artifacts.
2. Explicit `predict(support_features, support_actions, future_actions)` interface returning raw feature-coordinate forecasts. No query-image argument; horizon-prefix invariance and action-free invariance must be tested.
3. Cache manifest with original episode/session IDs, train/validation-only split proof, source image hashes, DINO revision/weights/license, preprocessing, grid ordering, dtype, and training-only statistics.
4. A package containing versioned architecture config, selected weights, normalization, coordinate schema, source and encoder hashes, parent registration, exact checkpoint-selection epoch/metric, and license/model card. State clearly that outputs are latent features and commands are recorded commands, not guaranteed reachable robot states.
5. Independent relocated CPU loading and prediction parity, deterministic grid-pooling parity, finite bounded-output checks where promised, loss/backpropagation checks, and future-action-prefix causality tests before full training.
6. Publish every registered model and completed positive/negative aggregate, plus reproducible preprocessing and evaluation. Freeze the final method before any new confirmatory evaluation; an already examined test cannot become a fresh test by relabeling it.

No experiments or scheduler jobs were launched for this diagnosis. The architecture implementation is a subsequent authorized development task; its outcomes are unknown.
