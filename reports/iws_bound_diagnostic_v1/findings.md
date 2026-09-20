# Completed correction-range diagnostic

Job 202002 completed all 10,136 internal-development windows, 362 trajectories and 59 offsets in 183.60 seconds. No predictors were trained or run, and no reserved/test payloads were accessed. The completion and source-review receipts bind the exact registered code and inputs.

| Task | Endpoint coordinates outside range | Relaxed standardized-MSE floor | Saved bounded MSE | Saved no-tanh MSE |
|---|---:|---:|---:|---:|
| PushT | 0.935436% | 0.001897805 | 0.261116483 | 0.241617656 |
| Box | 1.355355% | 0.003434813 | 0.290926916 | 0.263284772 |
| Rope | 0.964014% | 0.001921505 | 0.222913001 | 0.197045831 |

The bounded decoder forms a convex mixture of observed patches per channel, plus a correction in [-1,1]. Target coordinates outside that expanded channel range cannot be represented by the real-arithmetic decoder. The box is a necessary relaxation: channel-shared mixing weights and gates impose additional constraints. Projection distance is therefore a relaxed floor, not a tight attainable error or a causal explained-gain fraction.

All three tasks contain out-of-range targets. The floors are small compared with the measured predictor errors, so this audit leaves channel coupling and optimization effects unresolved. The fixed roundoff sensitivity changes endpoint floors by at most 1.4779331e-7. Every recomputed persistence error matches the saved primitive ledger exactly.

The paper renderer independently reconstructs all trajectory and window-weighted analytical curves and all three-seed saved-model curves before displaying results. Main performance evidence remains the separate full training and reserved-evaluation studies. This audit supplies a mechanism diagnostic, not new accuracy observations.
