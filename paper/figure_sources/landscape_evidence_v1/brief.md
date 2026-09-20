# Two separate landscape evidence figures

Use `paper-visual-design` and `paper-figure-creation` evidence, styling, layout,
and pixel-review guidance. These are quantitative result displays: extra images,
logos, or generated illustrations would not explain the comparison and are not
used. The actual manuscript width is 5.5 inches; all labels remain at least 8 pt.
No scientific source, model, result, or registered renderer is changed.

## Spatial ablations (formerly Figure 5)

Question: which matched controls change the current spatial model's native DROID
forecast error? Preserve six comparators and both horizons: 12 point estimates,
12 paired 95% session-by-seed bootstrap intervals, and 12 relative-gain labels.
Preserve the small negative h5 comparison and every interval crossing zero.

Three composition sketches:

1. **Selected: shared labels plus two aligned forests.**
   `Comparator | h5 interval · gain | h10 interval · gain`
   Six compact rows make horizon and comparator identity recoverable, while
   removing the oversized headline and repeated explanatory footer.
2. **Overlaid horizon forest.**
   `Comparator | h5 circle / h10 diamond on one axis`
   Saves width but intervals near zero and per-row gain labels collide; rejected.
3. **Six comparison mini-panels.**
   `[AR][additive][bounded additive] / [unbounded][context-off][action-free]`
   Each has h5/h10 marks. This needs two rows of axes and repeated labels, which
   works against the requested short landscape slot; rejected.

Chosen slot: 5.5 × 2.0 inches. Left-to-right native h5/h10, with matched row
positions. Circle/diamond distinguish horizons without relying on color; open
marks mean an interval includes zero. Shortened labels are explicitly expanded
in the caption. Point-gain labels are not significance marks.

## Historical simulation forecast (formerly Figure 12)

Question: how do the historical context model and four aligned comparators
forecast held-out composition and extrapolation in each simulator? Preserve all
20 means, all 16 three-training-seed sample SDs, the four frozen points without
invented SD, and the four signed changes relative to Framewise. These are not
the current spatial architecture and not planning-success results.

Three composition sketches:

1. **Selected: four aligned forests.**
   `Method | PushT held-out | PushT extra | Reacher held-out | Reacher extra`
   One shared method list eliminates duplicate labels. Four panel-specific log
   axes preserve the original geometries; a separate signed-change row keeps
   annotations away from data and retains the Reacher extrapolation regression.
2. **Two environment panels with both splits.**
   `Method | PushT [held-out / extra] | Reacher [held-out / extra]`
   The Reacher split scales are very different, compressing all held-out marks
   when both are placed on one scale; rejected.
3. **Mean ± SD matrix.**
   `Method × environment/split` with printed values and SD.
   This is a valid table but removes the error-bar comparison requested here and
   duplicates the detailed table; rejected.

Chosen slot: 5.5 × 2.0 inches. Method identity uses direct labels and distinct
markers. The native latent coordinates differ between environments, and each
panel has its own log limits. Signed relative MSE change is retained: negative
is favorable, positive unfavorable. No cross-panel magnitude ranking is implied.

## Evidence and review contract

`spatial.json` is the exact existing portable spatial ledger, retaining all 36
contrasts and the source IDs for the displayed 12. `forecast.json` is the exact
source-validated forecast ledger: its 52 result JSON hashes were checked during
extraction, seed means and sample SD were independently recomputed, and held-out
means matched the primary table ledger. Historical private paths remain
provenance; rendering requires only this portable pack.

Review the PDF-rendered 5.5-inch proof, enlarged proof and grayscale. Check all
12/20 point identities, CI versus SD distinction, axis limits/ticks, label
clearance and preserved negative evidence. Root integrates and reviews final
manuscript pages; standalone review does not claim integrated-page approval.

## Main-result integration: chosen composite

The selected spatial evidence is also delivered as a 5.5 × 2.25-inch main-results
composite: `50-point forecast profile | comparator labels | h5 forest | h10 forest`.
This replaces the previous main figure's six-interval subset and the standalone
appendix ablation. The alternate candidates were (1) stacked curves over forests,
which adds height, and (2) an overlaid h5/h10 forest, which crowds the near-zero
ablations. The chosen two-forest composition retains 12 separate intervals, all
50 original means, zero on both error and gain axes, and an 8-pt shared legend.
Relative percentages remain in the complete tables and the standalone alternate;
they are omitted from the main composite to keep readable axis widths.
`curves.json` is the exact previous main-figure ledger; all 50 means were checked
against the same spatial ledger's absolute errors, including persistence.
