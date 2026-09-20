# Development-only correction-bound diagnostic

This is a new, post-development analytical audit. It will not train, run a predictor,
select checkpoints, change an evaluation, or access reserved/test data. Execution is
blocked until a separate `source_review.json` has `status: passed` and binds the
unchanged registration SHA. The registration command reads source/config/receipt
metadata and frozen statistics only; feature arrays and primitive error NPZs are
listed by their existing hashes without opening them.

The actual frozen bounded model normalizes an initial 6144-vector in FP32, reshapes
channel-major `(384,16)` to `(16,384)`, then computes
`(1-g_j) Z0_j + g_j sum_l T_jl Z0_l + tanh(residual_j)`.
`T` is a row softmax, `g` is a sigmoid, and all nine selected bounded configs use
innovation bound 1. No learned projection follows this sum. Hence in real arithmetic
each channel belongs to `[min_l Z0_lc - 1, max_l Z0_lc + 1]`. The no-tanh model removes
this correction constraint; all 18 selected configs must bind the same task-specific
training-only, shared-channel normalization.

This coordinate box is a **relaxed necessary envelope**, not a tight attainable set:
mixture weights and gates couple channels. Squared distance of a target to this box,
averaged over all 6144 feature coordinates, is a relaxed standardized-MSE lower
bound. It is not a fraction of gains causally explained. Nonzero violations establish
an envelope limitation; negligible violations do not rule out channel coupling,
optimization, or other effects of removing tanh. Report a null answer honestly.

Every existing eligible internal-development window is retained, in the frozen
order: strict `start+60 < N`, stride 5, offsets 1–59. No patch/channel mask, sample
selection, offset selection, or seed replication of the analytic descriptor.
Report each trajectory and all 59 offsets; primary curves average windows within
trajectory and then trajectories equally. Separate pooled-window curves remain
available. Three seeds are used only for the existing bounded/no-tanh error curves.
Native command-row horizon H corresponds to target offset H−1; no physical-time claim.

Initial normalization uses exactly the frozen FP32 subtraction/division. Target
coordinates are computed in float64 from original FP32 values and FP32 statistics,
so the algebraic envelope can be checked independently in raw coordinates and
compared to the standardized raw-error metric. This is an exact real-arithmetic
relaxation of that normalized anchor, **not a certified floating-point output bound**.
Report exact violations and a separately labeled roundoff sensitivity: expand each
endpoint by `32*eps32*(1+max(abs(Z0))+abs(mean/std))`. This fixed screen is deliberately
not advertised as a rounding theorem. Raw/normalized distance equivalence is checked.
All-window persistence errors are recomputed with the unchanged FP32 metric helper
and compared with the frozen CPU ledger; any identity or metric mismatch stops.

```bash
.venv/bin/python -B -m pytest -q scripts/iws_bound_diagnostic_v1/test_diagnose.py
.venv/bin/python -B scripts/iws_bound_diagnostic_v1/diagnose.py register
# Independent review writes reports/iws_bound_diagnostic_v1/source_review.json.
.venv/bin/python -B scripts/iws_bound_diagnostic_v1/diagnose.py check
sbatch scripts/iws_bound_diagnostic_v1/run.slurm
```

The planned job uses 2 CPU threads, 8 GiB, no GPU, and a 30-minute allocation cap.
No measured runtime or numerical outcome is available before execution. Results
and completion receipts are written only in this diagnostic report namespace and
never ingested automatically into paper/site claims. Original studies stay frozen.
