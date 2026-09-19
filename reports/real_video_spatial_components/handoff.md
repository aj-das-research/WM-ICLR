# Spatial component follow-up handoff

The reviewed six-run study is complete: all six new models trained30 epochs and were evaluated; finalizer200449 completed all20 effects and six exact relocated offline predictors. The original startup graph below is retained as execution provenance. This is original train/validation development after the existing controls were revealed; no fresh-test reuse. All original scientific sources remain frozen.

- Registration: `configs/real_video_spatial_components/v1/registration.json`; SHA `4a626e92e6776da80f36b5d7ed11b291bb35e58c50e357dcb9779496385d0ab2`.
- New model/trainer/evaluator/analysis/export implementation: `src/shiftwm/real_video_spatial_components/`, `scripts/real_video_spatial_components/`.
- Protocol: `reports/real_video_development/spatial_components_protocol.md`.
- Independent source/mechanism/statistical review: `reports/real_video_development/spatial_components_independent_review.json`.
- Combined tests:83passed in23.56s. Exact full30-epoch CPU-fixture continuation, relocated offline predictor parity, learned-like causal/gradient tests, decoder bounds and an unbounded counterexample, strict recipe and population tamper tests passed.

| Lane | New arm | Seed0 | Seed1 | Seed2 | Node |
|---|---|---:|---:|---:|---|
| A | bounded_additive |200443|200444|200445|ws-l1-006|
| B | unbounded_transport |200446|200447|200448|ws-l5-004|

Jobs are sequential within each lane. CPU-only finalizer200449 waits afterok200445+200448. Maximum2 GPUs and16trainingCPUs. `submission.json` preserves the held/released graph; `scheduling_amendment_ws_l1_006.json` records the pending-lane relocation from occupied ws-l1-002. Source, configuration, seeds and full30epoch recipe did not change. Each allocated GPU records its actual visible-device free memory before training; both initial allocations passed with about31GiB free.

Finalization will strictly validate six new runs plus six revealed controls, recompute every window/episode aggregate, report16component contrasts+4difference-of-differences interactions with paired session/seed intervals, and export six inference-only predictors with new relocated CPU parity proofs. Outputs: `reports/real_video_spatial_components/finalization.json`, `results.md`, `artifacts/releases/spatial_components_v1/`. No public release or manuscript edits are part of this namespace. All outcomes, including negative/inconclusive, must remain visible. Mixing retains its gate/identity-bias and~1.84% active-capacity differences.

`handoff.json` is a timestamped startup snapshot, not a live status or completed-result claim.
