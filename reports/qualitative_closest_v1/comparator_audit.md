# Closest-comparator qualitative audit

Audit: 2026-09-20T17:13:14.490145+00:00

The most defensible qualitative story is stronger feature forecasting against the fixed autoregressive baseline, with the closest ShiftWM component shown alongside it. The current completed evidence does **not** show a large DROID margin over the closest component ablation. No new inference or uncertainty calculation was performed.

## DROID: separate primary method, runner-up, and baseline

| Endpoint | Best method | Actual runner-up | Primary vs closest alternative | Primary vs fixed matched AR baseline |
|---|---|---|---:|---:|
| h5 | unbounded_transport (0.145037458) | transport (0.145074184) | -0.0253% | +3.01% vs AR |
| h10 | transport (0.193141489) | unbounded_transport (0.193274115) | +0.0686% | +5.30% vs AR |

The h5 and h10 bounded-minus-no-tanh paired absolute-MSE intervals both cross zero. They are, respectively, [-0.00030891143895655116, 0.0003870395414842035] and [-0.000606475686880044, 0.0002625520371859183]. These paired analyses do not establish a reliable difference between the two variants; they do not establish statistical equivalence either. AR is the fixed matched autoregressive baseline, not the lowest-error additive control. The best explicit nonmixing internal control is bounded additive (h5 MSE 0.149347244, h10 MSE 0.200009747), giving primary bounded spatial mixing relative gains of 2.8612% and 3.4340%. Its existing paired absolute-difference intervals favor spatial mixing at both endpoints; the complete values remain in the JSON audit. The 30-epoch adapted DINO-WM runs are substantially weaker and should not anchor the qualitative superiority headline.

## Reserved IWS: no-tanh secondary variant versus the actual runner-up

The no-tanh **ablation** has the lowest mean in all 12 task-by-metric endpoint cells. PushT runner-up is bounded ShiftWM; Box and Rope runner-up is AR. AR is the fixed matched autoregressive baseline for the qualitative analysis. It has lower error than persistence and additive anchoring in all 12 IWS cells, but this observation is not generalized to the DROID control roster. These are related feature metrics on the same 30 trajectories, not 12 independent demonstrations.

| Task | Metric | Actual runner-up | No-tanh relative reduction | Existing paired 95% interval |
|---|---|---|---:|---|
| pusht | standardized_mse | bounded_spatial_mix | 7.16% | [5.53, 8.60]% |
| pusht | standardized_mae | bounded_spatial_mix | 3.39% | [2.57, 4.11]% |
| pusht | raw_dinov2_l1 | bounded_spatial_mix | 3.50% | [2.67, 4.21]% |
| pusht | feature_cosine_distance | bounded_spatial_mix | 7.99% | [6.27, 9.42]% |
| bimanual_box | standardized_mse | autoregressive | 4.84% | [1.40, 8.00]% |
| bimanual_box | standardized_mae | autoregressive | 2.72% | [1.29, 3.95]% |
| bimanual_box | raw_dinov2_l1 | autoregressive | 3.10% | [1.64, 4.34]% |
| bimanual_box | feature_cosine_distance | autoregressive | 5.51% | [1.92, 8.79]% |
| bimanual_rope | standardized_mse | autoregressive | 6.80% | [5.72, 7.88]% |
| bimanual_rope | standardized_mae | autoregressive | 3.93% | [3.43, 4.44]% |
| bimanual_rope | raw_dinov2_l1 | autoregressive | 4.40% | [3.90, 4.92]% |
| bimanual_rope | feature_cosine_distance | autoregressive | 7.97% | [6.80, 9.19]% |

Intervals above are the existing paired seed-by-trajectory percentile intervals, unadjusted for multiple comparisons. Gains are **relative error reductions**, not accuracy percentage points. The primary bounded model is not uniformly superior to AR; preserve that distinction in captions and main text.

## Concrete matched cases from complete ledgers

Fix AR globally. Display the same input, target, AR error, bounded error, and no-tanh error for each case. Rank by the declared endpoint MSE and retain largest/median/smallest gain; never switch the comparator or choose the nicest window after viewing images. H60 means stored offset 59, not a 59-offset average. The IWS selection is explicitly post-hoc descriptive.

| Study | Selection | Episode | First window start | Ranked episode gain vs AR | Shown window gain vs AR |
|---|---|---|---:|---:|---:|
| DROID bounded | Largest gain | droid-4bdaae464e55875cda162aa9 | 0 | +44.92% | +49.26% |
| DROID bounded | Median gain | droid-2dce8777c34ed372dc9ff50c | 0 | +5.36% | -0.31% |
| DROID bounded | Largest regression | droid-419bfa9f829cb40102eae3ba | 0 | -9.62% | -14.52% |
| pusht no-tanh | Largest gain | 000000 | 2 | +14.89% | +18.60% |
| pusht no-tanh | Median gain | 000007 | 7 | +8.23% | +8.34% |
| pusht no-tanh | Smallest gain | 000004 | 2 | +0.67% | -11.78% |
| bimanual_box no-tanh | Largest gain | 000001 | 2 | +13.57% | +14.92% |
| bimanual_box no-tanh | Median gain | 000006 | 18 | +4.64% | +7.57% |
| bimanual_box no-tanh | Largest regression | 000008 | 9 | -7.56% | -17.84% |
| bimanual_rope no-tanh | Largest gain | 000009 | 1 | +9.06% | +7.88% |
| bimanual_rope no-tanh | Median gain | 000000 | 2 | +7.32% | +7.67% |
| bimanual_rope no-tanh | Smallest gain | 000004 | 2 | +4.38% | +4.44% |

Full population inventories and exact means are in `comparator_audit.json`. First-window outcomes can disagree with the ranking outcome; both must remain visible. Use endpoint feature-error maps, not invented RGB predictions. The model exposes transport mixing and gates; label them as learned feature mixture weights rather than attention-grounded physical correspondences or causal explanations.

## Evidence and pending external run

All 36 reserved evaluation JSON and NPZ hashes were verified against the completed finalizer. Primitive ledgers reproduce all per-episode metric curves and 144 task/method/seed/metric endpoint means. Nine DROID AR/bounded/no-tanh ledgers are hash-bound to completed finalizers. No checkpoint or RGB/feature payload was opened. No new confidence interval was calculated.

The stronger 100-epoch raw-feature DINO-WM study is excluded until all six runs and its finalizer complete and pass review. At audit time 3 completion markers existed; the dependent finalizer had not completed. Only metadata and queue status were inspected.

Machine-readable source inventory: `89` SHA-256 bindings in `comparator_audit.json`.

Comparator terminology was corrected after review: ownership-based strongest-baseline labels were removed. All numerical rankings, saved uncertainty intervals and selected cases are unchanged.
