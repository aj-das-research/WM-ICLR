# Algorithm review: causal support reliability for real-video adaptation

Status: design recommendation, 2026-09-19. No training, evaluation, checkpoint selection, or frozen-file modification was performed for this audit. The evidence below comes from original training/validation diagnostics and source inspection. Neither original-test nor fresh-test outcomes are used to choose the proposal. This document is not an executed experiment or a novelty claim.

## What the existing evidence supports

| Evidence | Supported interpretation | Not established |
|---|---|---|
| The real-video forward pass recursively feeds its predictions back and optimizes all five query frames. Training enforces three support plus five query frames. [model.py lines 99–150](../../src/shiftwm/real_video/model.py), [train.py lines 261–293](../../scripts/real_video/train.py). | Five-step rollout training already exists; h10 lies outside the training horizon. | “The model was trained only one step” is incorrect. More rollout training alone is not a new algorithm. |
| On the common h10-eligible validation population, Framewise/factorized errors are 0.055117/0.055003 at h1, 0.156712/0.156234 at h5, and 0.216903/0.219507 at h10. Oracle refresh reduces factorized h10 error to 0.061602. [diagnosis lines 5–18](../real_droid_development_diagnosis.md). | A small short-horizon advantage reverses over longer recursion. Actual intervening observations are informative. | Oracle refresh changes information and contexts; it does not isolate one causal source of drift or represent deployable open-loop forecasting. |
| Donor commands change factorized predictions by MSE 0.0146963 and worsen validation h10 error by 5.173%; Action-free changes exactly zero. [diagnosis lines 20–52](../real_droid_development_diagnosis.md). | The model uses actions. Donor-command sensitivity is stronger than reversal sensitivity. | Action blindness is not the supported diagnosis; sensitivity does not prove correct physical counterfactual dynamics. |
| Factorized seed 0 train-subset h10 improves 0.194646→0.155465 from best to last, while validation worsens 0.220366→0.277711. [diagnosis lines 71–78](../real_droid_development_diagnosis.md). | Strong overfitting is directly observed. | It is not proved to originate specifically in the dynamics adapter. Existing slow/decay/compact controls address this separately. |
| Observation context uses support mean/variance; dynamics context sees only two transitions, after observation FiLM. Contexts remain fixed during the query. The real-video objective is plain feature MSE. [model.py lines 87–147](../../src/shiftwm/real_video/model.py), [TransitionContext lines 42–54](../../src/shiftwm/model.py). | The architecture separates two modules, but real-data training supplies no explicit factorization supervision. | Two transitions do not identify physical dynamics independently of appearance, scene, policy, or motion. Simulator consistency losses must not be attributed to this real-video model. |
| FiLM scale is bounded but its additive shift is not. [ResidualFiLM lines 30–39](../../src/shiftwm/model.py). | There is no architectural bound on additive corrections. | This is a possible contributor, not a diagnosed instability mechanism. |
| Training selection averages all query elements/windows; final scoring first averages within each episode. [train.py lines 277–293](../../scripts/real_video/train.py), [evaluate.py lines 86–118](../../scripts/real_video/evaluate.py). | Checkpoint selection and reported aggregation differ. | A ranking change from this mismatch has not been measured. |

The original validation low-future-motion stratum is harmed relative to persistence, whereas the high-motion stratum benefits. Those strata use future targets and cannot drive an online gate. Support-motion strata alone do not establish a useful gate. The full training-fitted scalar calibration already improves factorized validation h10 by 1.655%, but its advantage over equally calibrated Framewise at h10 has an interval crossing zero. These motivate testing reliability, not assuming success. [Strata](../real_droid_development_diagnosis.md), [matched calibration](../real_droid_residual_calibration_results.md).

## One bounded modification: support-validated blending of frozen forecasts

**Hypothesis:** errors on an earlier, fully observed causal prefix predict whether the factorized forecast is more reliable than a matched Framewise forecast on the next query. A training-prior shrinkage term can reduce the variance of this episode-specific decision. If that relationship is absent, the method should collapse to a global blend and the hypothesis should be rejected.

Use two frozen, already trained donors in identical standardized feature coordinates: `F` = Framewise and `O` = factorized, each with its existing training-fitted displacement calibration. The proposed correction is along `O−F`, not the earlier displacement-to-persistence direction. It needs no image decoder or new neural weights. Both donors and all calibration dependencies must ship with the wrapper; it is a two-model system, not a free single-model improvement.

Use **13 observed stored frames** `z[0:13]`, giving one genuine ten-step prefix backtest: both donors predict `z[3:13]` from only `z[0:3]` and the recorded connecting commands. Contexts for that backtest must be inferred from `z[0:3]` only. The ten later observations are scoring targets for the gate; they must never be inserted into those predictions or contexts. Thirteen stored frames span 60 native frame intervals under the unchanged stride five. This is explicitly a longer-prefix task, not a replacement for the original three-frame protocol.

Let `pF_h`, `pO_h` denote those prefix forecasts, `d_h = pO_h−pF_h`, and `e_h = z[2+h]−pF_h` for h=1,…,10. Compute means over horizons and feature dimensions:

```
A = mean(d_h ** 2)
B = mean(e_h * d_h)
alpha = clip((B + lambda * alpha_prior) / (A + lambda), 0, 1)
```

If the denominator is zero, use `alpha_prior`. Fit `alpha_prior = clip(mean_training(B)/mean_training(A), 0, 1)` on training prefix moments, weighting windows within episodes and then episodes equally; use 0 when the denominator is zero. Set `lambda = kappa * mean_training(A)`; choose `kappa` only by training-session cross-validation from the fixed grid `{0, 1, 10, 100}`, with ties choosing the larger shrinkage. Refit the scalar prior on all training episodes after choosing kappa. This is a proposed finite search; it must be registered before executing it.

At the real query boundary, reset both donors, infer their contexts from the **last three observed prefix frames** `z[10:13]`, and forecast from the same recorded future command sequence. Return `pF_query + alpha * (pO_query−pF_query)`. Keep alpha fixed for that entire query; never update it with query targets or self-generated observations. The earlier backtest context and the final query context differ, so transfer of reliability is a substantive assumption to test.

The formula minimizes a one-dimensional support loss plus `lambda*(alpha−alpha_prior)^2`. Consequently its unregularized loss on that same prefix cannot exceed the prior blend's prefix loss, up to numerical tolerance. This elementary in-prefix statement is **not a guarantee about the next query**, and the correlated ten frames/1536 dimensions must not be treated as independent statistical replicates. A convex blend stays between donor predictions but does not ensure physically correct dynamics or prevent both donors from failing.

## Required controls and decision rule

All arms use the identical 13-frame prefix, query windows, donor packages, normalization, cameras and command indexing. Every mixture/control is charged both donor parameter counts, both prefix and future rollouts, peak memory and measured latency. Single-donor cost is also reported separately. Do not claim a gain over the shorter-prefix published table without re-evaluating matched controls on this new population.

| Arm | What it isolates |
|---|---|
| Calibrated Framewise; calibrated factorized; persistence; constant velocity | Individual donor quality and simple video controls, on identical eligible windows. |
| Fixed equal blend, alpha=0.5 | Ordinary ensemble benefit. |
| Training-fitted constant alpha_prior | Global mixture calibration without episode adaptation; the main comparator. |
| Unshrunk support gate, kappa=0 | Whether noisy local fitting needs the training prior. |
| Proposed shrunk support gate | Whether observed prefix errors predict subsequent donor reliability. |
| One-step-backtest gate | Same 13 observed frames, ten separately causal one-step predictions from the actual preceding three observations; separates local errors from open-loop reliability. Its scalar prior/shrinkage gets the same train-only fitting opportunity. |
| Deterministically session-shuffled gate | Diagnostic: preserve the gate-value distribution but break the correspondence to the current episode. No donor chosen by losses. Not a deployable baseline. |

**Horizon mismatch control:** after the inexpensive frozen-donor gate study, a positive candidate still needs a separate matched `training horizon {5,10} × donor mode {Framewise,factorized} × seed {0,1,2}` campaign (12 complete runs). Use the same capacity, optimizer, epochs, train/validation inclusion and checkpoint-selection criterion (equal-episode validation h5 endpoint MSE, ties to the earlier epoch) across those new arms; report the extra ten-step compute. Train all ten outputs recursively in h10 arms, with no query-frame teacher forcing. Refit the identical gate procedure for each donor pair. Keep this separate from the existing 36-run slow/decay/compact campaign, which leaves training horizon five unchanged. [Existing campaign](../real_droid_generalization_protocol.md).

Proposed development gate, to register before implementation:

1. Original camera-one train/validation only. Common windows must have all 13 prefix plus ten future frames; report every length exclusion and resulting episode/session/window count. Primary endpoint is h5 standardized feature MSE; h10 is a separate stability endpoint. Average windows within episodes, then episodes and three seeds equally. Preserve all failures. These populations differ from previous tables.
2. Choose kappa by three deterministic training-session folds using `SHA256("support-reliability-v1:" + session_id) mod 3`. Fit priors/energy on fold-training only, score subsequent query h5 on fold-held-out training sessions, aggregate out-of-fold episodes equally, and refit on all training. Do not search on original validation. Stored base checkpoints already learned from original training, so this isolates gate fitting rather than claiming end-to-end unseen-session cross-validation.
3. Promote only if the full gate improves validation h5 by at least **1% relative** over both the stronger individual donor and the training-fitted constant blend, its paired session/seed 95% interval versus the constant blend excludes zero, every seed improves over that blend, and mean h10 is no worse than that blend. The 1% threshold is an operational relevance threshold, not a statistical theorem. These remain exploratory development criteria.
4. Reject the adaptation explanation if the constant or shuffled gate matches the full gate; reject the multi-step explanation if the one-step gate matches it. Report such outcomes rather than renaming ordinary averaging as a new method. Before a strong paper claim, add a genuinely long-history learned control with the same prefix access and matched training opportunity; equal input availability alone does not prove the old three-frame donors exploited all available information.
5. Any new confirmatory study needs a newly frozen session-disjoint population excluding **all original and subsequently inspected fresh sessions**. Neither already revealed test population is a selection or confirmatory set for this proposal. Camera-two transfer and a second backbone are later tests after method freezing, not extra tuning opportunities.

The frozen-donor stage requires feature-cache replay and scalar fitting, not training or new encoder extraction. Cache exact causal prefix/query donor predictions once and reuse them for all scalar arms. It uses roughly two 10-step prefix and two 10-step query donor rollouts per window before caching; measure runtime in an allocated job rather than promise a wall time. Release donor manifests plus a small gate package containing alpha_prior, kappa, energy scale, protocol and source hashes. New neural checkpoints arise only from separately executed horizon-control training.

## Novelty boundary and present recommendation

Latent residual adaptation is already central to [ReDRAW](https://arxiv.org/abs/2504.02252). Multiple-horizon prediction is already proposed in [Shortcut World Models](https://openreview.net/pdf?id=O8nGPjY7Tj); the retrieved manuscript is labelled under review, so no acceptance claim is made here. [RLA-WM](https://arxiv.org/abs/2605.07079) models residual latent actions, with an [official implementation](https://github.com/mlzxy/rla-wm). None of “residual”, “multi-horizon”, “calibration” or “two-model blending” by itself establishes novelty. These primary sources were checked on 2026-09-19; this bounded audit is not an exhaustive novelty search.

The defensible research question is whether **causal multi-step prefix errors can identify when episodic dynamics conditioning is transferable to a subsequent unseen query**, above ordinary calibration and ensembling. It is technically distinct from the current fixed-context model and global displacement scalar, but may still be an incremental mixture method. Existing evidence justifies the cheap controlled experiment; it does not yet justify a new headline method, SOTA claim, generalized physics interpretation, or an acceptance promise. If reliability fails to transfer, prioritize the ordinary generalization/horizon controls and report the limitation rather than expanding the gate's complexity on held-out outcomes.

## Inspected evidence identities

SHA256 at audit completion; no source below was edited by this audit.

| Path | SHA256 |
|---|---|
| `src/shiftwm/real_video/model.py` | `b73ae74255a4de96355aa4708fba236c7f2b265a06ad7ca0e65063622da17ad5` |
| `src/shiftwm/model.py` | `2f783caf8eed699a8d024f689aa956efd126a2c8b6c67aea23e4ff5dfb8536f8` |
| `src/shiftwm/real_video/data.py` | `965a6f706bd7858b1eb0971510accd62d8194df7c063054a4202257ccc21795a` |
| `scripts/real_video/train.py` | `90f26847fa0cb7915f40880187358e4cb0fbf3d65670292e5ba910503318bafe` |
| `scripts/real_video/evaluate.py` | `8e357304d5617d3c1afdd74847763a3143fe53ca83e1d182cf94f6fb8a27ca0b` |
| `reports/real_droid_development_diagnosis.md` | `605fe99c0ff60dd3d3ecdd12ad6e62cd572044223c1f5f2fefffca5778c77a07` |
| `reports/real_droid_development_diagnosis.json` | `1e128be89264cf945e22c7f37281a0b0574feb084c9a11c3af5a292699f44c40` |
| `reports/real_droid_residual_calibration_results.md` | `f9afdf079952faa29eba5e469de4fd722e3f26f27793d44c7ad921bf4ffd535c` |
| `reports/real_droid_generalization_protocol.md` | `10a5121c135be9f73c3e2464dc5a60e476c229c79b4c60b037026905f7a2c85e` |
