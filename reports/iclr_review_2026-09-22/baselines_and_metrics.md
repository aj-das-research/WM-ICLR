# Baseline and metric coverage — 22 September 2026

This is the current audit; earlier dated inventories remain historical records.
The paper's table-completeness checker checks active compiled inputs, not abandoned
tables. Its passing status does not establish external-baseline completeness.

| Evaluation | Included comparisons | Metrics and scope | Remaining gap |
|---|---|---|---|
| Main DROID table | Persistence, autoregression, additive and bounded-additive anchors, bounded mixing, no-tanh, no-context, no-actions; four adapted official DINO-WM recipes | Absolute native h10 MSE, MAE, raw L1, cosine; all 33 learned checkpoints, three seeds per arm; 141 development episodes | DINO-WM adaptations are worse than persistence. This is not a successful official benchmark reproduction. Untouched DROID confirmation is still needed. |
| IWS reserved feature tables | Persistence, autoregression, additive anchor, bounded and unbounded mixing | MSE, MAE, raw L1, cosine; all 36 learned checkpoints; 600 handles / 30 trajectories | RLA-WM remains pending gated backbone access and protocol reconciliation. |
| IWS RGB table | Same four learned arms, raw RGB persistence, decoded feature persistence; target-feature reconstruction separately identified | RGB MSE, PSNR, SSIM, LPIPS, UIQI; 45 complete rows / 9,000 endpoint scores | Offset-59 endpoint analysis differs from original IWS 192-step video evaluation. No FID/FVD. Target reconstruction is not a forecasting competitor. |
| Current spatial-model simulator tables | Autoregression, bounded additive, bounded and unbounded mixing | Native success, task-specific physical errors, support-inclusive interactions, complete planning time, memory; 24 models / 2,400 trials | No matched external published predictor in this current-model control campaign. Historical LeWM/AdaJEPA results cannot fill these cells. |
| Capacity-control table | Original bounded additive, exact active-parameter-matched additive, bounded mixing | Native and original pooled MSE at h5/h10, paired session–seed inference; nine revalidated models | Targeted development ablation, not a new benchmark leaderboard. Parameter matching does not match every optimization property. |
| Action-ranking diagnostic | All four current simulator arms and three seeds | Forecast MSE, physical and feature-cost Spearman, normalized physical regret; 528 executed development candidates | Eight development cases per task, random candidates rather than CEM; post-hoc diagnostic, not held-out control success. |
| Historical simulator/extension tables | Existing reference predictors and all registered arms retained | Original task-specific outcomes under the historical model | Separate architecture and populations. No claim of current-model transfer or robotics validation. |
| Paired-effect, ablation, resource and protocol tables | Named reference for each relevant effect; original comparison population retained | Units, direction, horizon, aggregation and uncertainty follow the source records | Not every provenance/protocol table is a performance leaderboard; unrelated model rows would be misleading. |

## Primary-source metric crosswalk

- [DINO-WM](https://arxiv.org/html/2411.04983v2): task success and particle-set
  Chamfer distance; auxiliary decoded LPIPS/SSIM. Our simulator tables use
  task-native success and physical units. Recorded IWS Rope provides no particle
  states, so its feature error is not Chamfer distance.
- [RLA-WM](https://arxiv.org/html/2605.07079v1): DINO-token L1, LPIPS, SSIM and
  inference FLOPs. Our raw DINOv2 L1 cannot be numerically equated with its masked
  DINOv3-Large L1; a shared RGB protocol is needed for a fair image comparison.
- [IWS](https://arxiv.org/html/2603.08546): RGB MSE, LPIPS, FID, PSNR, SSIM,
  UIQI and FVD under its video protocol. Five metric families are now measured
  on our fixed endpoint protocol. We do not label these published-protocol scores.

Exact source-code definitions and pinned implementations are documented in
`reports/metric_literature_audit_2026-09-21.md`; its old completion inventory is
superseded by this dated audit and the current manuscript ledger.

## Interpretation rules

Keep baseline wins visible. Unbounded mixing improves reserved feature errors,
but raw RGB persistence has the lowest endpoint LPIPS among forecasting methods.
The capacity-matched comparison supports native-grid mixing, but the pooled
contrast is inconclusive. Reacher control is substantially worse than AR.
Additional metrics are correlated diagnostics, not independent confirmations.
Published numbers with different encoders, horizons or populations are never
inserted into matched-score columns.
