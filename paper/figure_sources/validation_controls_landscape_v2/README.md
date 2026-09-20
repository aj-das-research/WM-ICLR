# Compact historical validation controls, landscape v2

Reproduce with:

```bash
python paper/scripts/render_validation_controls_landscape_v2.py
```

This presentation-only revision reduces the original 5.5 × 2.5 inch plot to
5.5 × 2.0 inches, keeping all labels at least 8 pt. It retains the established
three-panel forest-plot layout and every one of the 15 signed effects and paired
95% intervals. The source data and caption are byte-identical to
`paper/figure_sources/validation_controls_compact/`; the older figure is preserved.

Runtime requires only this directory's `data.json`, `caption.tex`, `manifest.json`
and the new renderer. Recorded historical paths/hashes inside these JSON files
are provenance, not runtime dependencies. Matplotlib, NumPy and Pillow suffice;
no checkpoints, private data, model inference or experiment execution is used.

Signed differences remain first minus second standardized MSE, multiplied by
1,000 for plotting. Negative values favor the first named method; open marks
indicate paired intervals including zero. Limits differ across panels, so visual
lengths must be read against each axis. The unfavorable Context h5 mean gain
of −0.116% and its zero-crossing interval remain explicit. All underlying
four-method means and twenty horizon contrasts remain in the unchanged ledger.

The PDF and SVG are editable vectors. Actual-PDF color/grayscale proofs, numeric
parity checks and the design review are in
`paper/design/validation_controls_landscape_v2/`. Root owns manuscript integration;
this renderer does not edit the main paper or any original figure/source.
