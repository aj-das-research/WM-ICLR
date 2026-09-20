# Fixed development geometry and tissue audit — 20 September 2026

Job **201909** completed successfully on ws-l3-009 with **2 CPUs, 8 GiB, no GPU**: 20 seconds scheduler wall time, 16.27 seconds inside the diagnostic. All 374 registered input hashes were unchanged before and after execution. No model was trained, selected again, or retuned; no simulator or final-test payload was opened.

## Drone: widening the gain did not recover the compressed goal geometry

The twelve already-selected transformer checkpoints cover both context variants and seeds 0/1/2. Each uses exactly the original 16 development goals and 120 unordered goal pairs, repeated under the same 16 first-window support contexts. A context is fixed across all goals within each comparison. Support comes from three observed states and two preceding action blocks; physical XY coordinates enter only the diagnostic correlation.

Canonical goal features have mean pairwise MSE **0.01020543** and Spearman correlation **0.61374** with squared physical XY distance. Uncorrected warm features have **0.00177574**, **17.40%** of canonical separation, and correlation **0.48985**. The original seed-0 probe values were reproduced.

| Context variant | Seed | Narrow / canonical | Wide / canonical | Narrow ρ | Wide ρ | Δρ |
|---|---:|---:|---:|---:|---:|---:|
| Constant dynamics | 0 | 16.185% | 16.039% | 0.48975 | 0.49432 | +0.00457 |
| Constant dynamics | 1 | 16.333% | 16.230% | 0.48965 | 0.49420 | +0.00455 |
| Constant dynamics | 2 | 16.318% | 16.250% | 0.49055 | 0.49437 | +0.00383 |
| Factorized (ours) | 0 | 16.221% | 16.070% | 0.49063 | 0.49417 | +0.00354 |
| Factorized (ours) | 1 | 16.316% | 16.194% | 0.49107 | 0.49420 | +0.00313 |
| Factorized (ours) | 2 | 16.282% | 16.210% | 0.49108 | 0.49507 | +0.00399 |

Widening reduces mean pairwise separation slightly in every matched comparison, while increasing the descriptive correlation by only **0.00313–0.00457**. Calibration MSE against canonical goals decreases by **0.00001331–0.00001900**. Attained wide gains span **0.80289–1.22894**, versus **0.90705–1.09067** for narrow gains; the wider adapter does not approach its permitted 0.25–4 extremes on these supports. Larger latent distances alone would not establish better geometry, and the small rank improvements do not establish a planning benefit. The support contexts and goal pairs are dependent; no confidence interval or significance claim is made.

This closes the missing follow-up requested by the geometry protocol. It weakens a gain-capacity-only explanation of the failure: the completed wider adapter still leaves roughly 84% of canonical pairwise separation absent. Its earlier development successes remain a distinct outcome measurement; these diagnostics neither change those scores nor justify final-test promotion.

## Tissue: target approach and invalid actions remain the observed failures

The audit retains all **18** original development policy records: two predictor families, three methods, three seeds, and the same eight goals. These are **144 method/seed/task instances**, not 144 independent goals.

| Saved-trace descriptor | Count |
|---|---:|
| Successful instances | 12 / 144 |
| Failed instances never entering 2 mm | 132 / 132 |
| Failure stopped at budget | 91 |
| Failure stopped on invalid action | 41 |
| Native steps / invalid steps / unstable-deformation steps | 24,369 / 41 / 0 |
| Complete five-command decisions | 4,563 |
| Decisions reducing target distance | 1,816 (39.80%) |

All failure instances remain in the denominator, including invalid-action stops. Next-step alignment uses only complete five-command blocks; partial terminal blocks are omitted from that descriptor, not from outcome counts. Success is recomputed using the unchanged evaluator rule. This audit describes executed actions, not counterfactual candidate ranking, and does not identify a unique causal defect. Zero recorded instability does not establish safety outside this fixed development set.

## Evidence and reproduction

- `registration.json` fixes all selected package generations, development inputs and original reports before new measurements.
- `source_review.json` records independent pre-execution review; all nine focused tests passed.
- `results.json` preserves every one of the 12 × 16 × 120 latent pair values, all context summaries, matched deltas and all 4,563 aligned tissue decisions.
- `completion.json` binds the results and the unchanged-source check.
- Reproduction commands and scope are in `scripts/extensions_diagnostics_20260920/README.md`. The runner intentionally refuses to overwrite this completed run.
- Scheduler accounting service (`sacct`) was unavailable; `scontrol` directly reported COMPLETED, exit 0:0, and 20 seconds. No measured peak-RSS claim is made.
