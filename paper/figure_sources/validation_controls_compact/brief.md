# Compact historical DROID validation controls

Private candidate; no manuscript includes or accepted figures have been edited.
The reader should distinguish ordinary optimization/capacity controls from the
separate training-horizon control, while seeing every previously plotted paired
effect and the unfavorable prespecified secondary result together.

## Scientific contract

- Historical two-context model is abbreviated **Context** and identified as
  ours; these results are not results of the later spatial anchored decoder.
- Panel a: all six context-minus-Framewise endpoint effects at h5/h10 in the
  three registered Slow, Decay and Compact arms. Thirty-six full 30-epoch runs;
  common five-query selection. Compact jointly changes predictor/context size.
- Panel b: all four h10-minus-h5 trained mean-h10 effects on identical query
  starts; plus the registered adverse Context matched-h5 mean secondary effect.
  Training and checkpoint-selection horizons change together.
- Panel c: Context-minus-Framewise after both methods train at h10, for standard
  h5/h10 populations and both query-mean and endpoint error.
- All fifteen marks are signed first-minus-second differences in standardized
  feature MSE, multiplied by 1000 for display. Separate axis limits are necessary
  to show each study's uncertainty; no pooled ranking or effect is computed.
- Open markers mean the paired 95% CI includes zero; point sign uses teal/amber.
  All three unfavorable points remain visible. Percentages are point reductions,
  never transformed CI endpoints. Each bootstrap jointly resamples sessions and
  seeds (10,000 draws), without multiple-comparison adjustment.
- All four methods' absolute means and seed SDs remain in the original tables
  and the staged complete ledgers. All twenty registered horizon contrasts are
  copied unchanged. No source model, evaluation, or registered file is modified.

## Design exploration

1. **A — three adjacent forests (selected):** same reading order and unit,
   separate comparison headers and meaningful row labels. Fits 5.5 × 2.5 inches
   with all regular labels at least 8 pt. Comparison a has the widest CI domain;
   b/c have independently labeled scales. The adverse h5 secondary gets its own
   row and a compact footer; no effect is silently dropped.
2. **B — two receipt groups, horizon subplots stacked:** gives the optimization
   effects more width but leaves too little vertical room for the two horizon
   comparisons at this height. Rejected for row-label collisions at 8 pt.
3. **C — vertical effect columns:** uses the same exact values but rotated
   category labels crowd the bottom and make comparison identity harder to read.
   Rejected at the requested compact physical height.

Exact plots are vector-authored with Matplotlib. Illustrative images or generated
assets would not clarify these comparisons and are not used. Liberation Sans
regular, STIX math, slate/teal/amber are consistent with the current manuscript.

## Evidence and reproducibility

`prepare_data.py` is a read-only extraction and arithmetic verification step. It
checks all 215 source hashes and recomputes all 26 paired intervals from saved
episode values using the already reviewed original arithmetic routines. It never
loads a checkpoint or performs inference. `data.json` preserves the complete old
ledgers, means, seed SDs, source hashes and exact plotted values.

`render.py` consumes only local `data.json`, NumPy/Matplotlib/Pillow and installed
fonts. It does not require cluster data or any private checkpoint. PDF/SVG/PNG
exports are deterministic; all original experiment data remain immutable.

Absolute-error pointers for integration: `tab:generalization-development`,
`tab:h10-native-endpoint`, `tab:h10-native-mean`. Complete horizon effect tables:
`tab:h10-matched-mean`, `tab:h10-matched-endpoint`, `tab:h10-ours-intervals`.
