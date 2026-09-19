# Training-only decoder reachability diagnostic

The complete diagnostic passed for all 1,442 internal-training trajectories. It reads no development or reserved payloads and makes no model predictions. Inputs, normalization and strict H60 window eligibility remain frozen.

| Task | Training trajectories | Eligible windows | H60 necessary MSE floor (b=1) | Coordinates outside envelope |
|---|---:|---:|---:|---:|
| bimanual_box | 481 | 13468 | 0.003610 | 1.399% |
| bimanual_rope | 481 | 13469 | 0.002136 | 1.030% |
| pusht | 480 | 13440 | 0.001846 | 0.910% |

The unit correction bound excludes a small fraction of target coordinates even under the relaxed, per-channel source min/max envelope. Enlarging the bound lowers this necessary error floor. These are training-data lower bounds, not measured checkpoint errors or an attainable oracle; they cannot quantify how much of the separately measured development gap is caused by bounding. A zero or small floor also does not prove the true shared-vector convex mixture can reach the target.

The measured restriction warrants a clean bound-removal ablation, but does not establish that removal will improve accuracy. The nine-run follow-up isolates this change with matched initialization, inputs, optimizer, 30-epoch budget and seed grid. The fixed-source mechanism and all completed v1 outcomes remain preserved. Improving late training losses also motivates a possible later equal-budget optimization study across all compared arms.

Diagnostic source and output are immutable in their versioned paths. Computation uses float64 solely for this geometric diagnostic; it does not change the FP32 forecast evaluator.

Source SHA256: `ffc3fc441920d11e750ecc80ee3d14f452cbabe1b8f92f19c3002b37f1ca68f0`. Output SHA256: `36af1f84560bcedf8f339c1efc43f8053e241b0fe7c26d8d4bd544ab225ed5de`. Scheduler job: `201217`.
