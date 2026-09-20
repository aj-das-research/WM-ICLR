# Adapted DINO-WM v1 diagnosis

This is a source- and history-only audit. No new inference, training, reserved/test access, or changes to the completed study were made. Raw records and exact source hashes are in `diagnosis_evidence.json`.

| Objective | Seed | Selected epoch | Train loss first → last | H10 validation first → selected | Training min |
|---|---:|---:|---:|---:|---:|
| matched_recursive_h10 | 0 | 27/30 | 0.6044 → 0.1552 | 0.4065 → 0.2381 | 16.23 |
| matched_recursive_h10 | 1 | 30/30 | 0.6048 → 0.1551 | 0.4102 → 0.2359 | 15.58 |
| matched_recursive_h10 | 2 | 29/30 | 0.6087 → 0.1556 | 0.4126 → 0.2382 | 15.28 |
| official_one_step_shifted | 0 | 30/30 | 0.2107 → 0.0677 | 0.3646 → 0.2293 | 4.71 |
| official_one_step_shifted | 1 | 30/30 | 0.2095 → 0.0675 | 0.3664 → 0.2291 | 4.34 |
| official_one_step_shifted | 2 | 30/30 | 0.2110 → 0.0676 | 0.3650 → 0.2307 | 4.45 |

All six runs completed 4,380 optimizer updates without restart/fallback; validation remained finite and improved. Native-objective checkpoints selected epoch 30 for all seeds; recursive checkpoints selected epochs 27/30/29. Epoch-25-to-best improvements are only 0.07–0.61%, but the learning rate had decayed near its floor. These histories show learning, not proof of adequate optimization. There is no strong late divergent-overfitting pattern. Native training uses three shifted slots and validation ten recursive slots: their loss gap is not a directly comparable generalization gap. Recursive train≈0.155 versus validation≈0.237 has a real population gap, with dropout/online-weight differences still relevant.

The completed development H10 native MSE is 0.355590 (native objective) and 0.331548 (recursive), versus persistence 0.226129 and ShiftWM 0.193141. Both external adaptations are worse than persistence at every measured horizon. Recursive training reduces external H10 error by 6.76% but worsens the early horizons. Strong internal autoregressive controls also outperform the external variants; the large gap therefore does not isolate transport or context contributions.

## Supported implementation facts

- Pinned official commit: `0a9492fa12044b852ae9e001cc74604b79c8bb0c` (`https://github.com/gaoyuezhou/dino_wm`). Its visual model consumes encoder patch embeddings directly; no fitted feature-channel whitening is applied. The transformer ends in learned-affine `LayerNorm(dim)` over visual and action/proprio channels, with no later output projection. V1 instead supplied channel-standardized pooled features. This is a material coordinate adaptation, not an established bug or a proof of a hard output constraint: learned affine parameters and unscored action coordinates matter.
- Pinned upstream defaults are 100 epochs, batch 32, predictor/action AdamW learning rate 5e-4, implicit AdamW weight decay 0.01, with no scheduler or gradient clipping in the training loop. V1 deliberately used the internal matched recipe: 30 epochs, batch 128, cosine 1e-4→1e-6, clip 1, BF16 training. Upstream `Accelerator(log_with="wandb")` does not pin precision; an exact original numerical precision cannot be inferred from this source alone.
- The checked adapter action alignment is correct: features 0/1/2 with outgoing blocks 0/1/2 predict shifted visual targets 1/2/3; recursive step k receives action blocks k:k+3. Predictions remain in the graph, action-output channels are excluded from loss, and targets never enter predictor input. Final selected-loss reconstruction agrees to roughly 1e-8 after FP32 denormalization/rescoring. No action-indexing, package-selection, population, or gross scoring-scale error was found.
- All external models have 19,412,420 parameters versus 1,026,305 for the internal models. Internal AR has an explicit last-feature residual and zero-initialized output; external DINO-WM predicts absolute features. These are architectural differences that may affect copying/optimization; this audit does not assign causality.

## Corrected follow-up, fixed before execution

New namespace `external_dinowm_raw_v2` preserves raw pooled DINO visual input/output coordinates, the exact official transformer and terminal LayerNorm, and raw visual training MSE. Six runs: three seeds for the official three-slot shifted objective and three seeds for an explicitly separate recursive-H10 control. Use upstream 100-epoch / batch-32 / constant AdamW-5e-4 defaults, weight decay 0.01, no clipping, and explicitly chosen FP32 with TF32 disabled. Train-only action normalization is retained for the same 35-D five-command blocks. Existing feature statistics are used only for common standardized H10 checkpoint selection and final scoring.

The same 18,660 H10-eligible training windows and 1,631 development windows are retained. This remains an adapted pooled-token, no-proprio predictor comparison, not reproduction of the original DINO-WM benchmark. Raw coordinates and recipe change together, so any improvement will establish a stronger adapted baseline rather than isolate which v1 choice caused weakness. No v1 checkpoint/result is overwritten; no reserved/test evaluation is proposed.

## Resource estimate, not a measurement of v2

Actual v1 training took 4.34–4.71 min/native seed and 15.28–16.23 min/recursive seed on RTX 5000 Ada. V2 has 58,400 updates per run (13.33× v1), 3.33× data exposures, a smaller batch and FP32 rather than BF16. Conservative planning range: 30–90 min/native seed and 1.5–4 h/recursive seed; approximately 6–16.5 aggregate GPU-hours, or roughly 2–5.5 h with three balanced GPU lanes, plus queue time. These deliberately broad bounds are not extrapolated benchmark timings; exact throughput will be recorded during the complete registered runs. Existing BF16/B128 capacity measurements are supporting evidence, not an exact FP32/B32 measurement. Two-hour allocations with epoch-boundary continuation at 90 min preserve the full budget without silently changing precision/batch/model.
