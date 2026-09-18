# Training-curve visual brief

Slot: ICLR appendix, exactly 5.5 × 3.2 inches. These are observed validation
prediction losses, not performance on test tasks. Show five model variants in
two environment panels with separate vertical scales. Every curve uses all
three configured seeds, at the intersection of their recorded epochs. Shade
mean ± sample standard deviation; do not label it a confidence interval.
The source is training_ledger.json with hashes of all 30 metrics/config files;
training_plot.json records the exact plotted epoch means and deviations.

The previous 15 lines per panel crowded the model comparison and its six-point
legend used implementation names. Selected revision: model-level mean curves
with light seed-variation bands, readable names, five distinct line styles, and
a matching shared legend. Other compositions considered: fifteen seed curves
retain individuality but overlap; five small multiples require excessive paper
space. Per-seed records remain available in the ledger.

No smoothing, interpolated epochs, dropped modes, or data edits. Mean/SD are
calculated from actual records. If a configured seed has no data, that mode is
not drawn. Figure source: paper/scripts/render_training.py; PDF/SVG/PNG outputs
are regenerated together. Standalone preview inspected; manuscript inspection
is recorded in visual_refresh_review.md.
