# Editorial quantitative-plot review

2026-09-18. Scope: the forecasting, training, teaser, and goal-calibration plots at the manuscript's **5.5-inch width**. The installed `paper-figure-creation` skill's evidence, design-system, review, and art-direction guidance was applied. Actual original PNGs were opened before editing; no numerical content or illustration asset was generated for this revision.

## Brief and representation

The main forecasting plot should let a reader compare ShiftWM with every aligned control, distinguish held-out composition from extrapolation, and notice that Reacher extrapolation is unfavorable. The appendix training plot should expose real optimization behavior and seed spread, including cases where simpler or diagnostic models have lower loss. The development diagnostic should keep goal error and prediction error visibly distinct. The teaser should identify its two methods without making the illustration look like experimental evidence.

Three plotting compositions were considered: retaining source-scaled small-multiple point ranges, replacing them with rank-ordered bars, or replacing them with normalized changes. The first is retained: bars are inappropriate for the existing logarithmic scales, and a difference-only plot would conceal raw error and the frozen baseline's favorable Reacher extrapolation. This is a local editorial revision, not a new statistical visualization or comparison scope.

The shared names are **ShiftWM (ours)**, **Unpaired contexts**, **Shared context**, **Framewise calibration**, **Frozen LeWM**, and **Unaligned predictor (diagnostic)**. Internal experiment IDs remain unchanged. The proposal is identified by its name, bold label, orange mark, and a lightly highlighted row where applicable; that emphasis applies in unfavorable panels too and is not a winner or significance annotation.

## Observed changes

- **Forecasting:** grouped condition headings replace repeated condition prose inside every panel title. A single metric label serves all four panels, and compact footnotes state three-seed mean ± sample SD and the frozen checkpoint's absence of a seed SD. Method labels have more left clearance. Darker green/purple controls and distinct marker shapes retain readable control identities; the proposal's highlighted row remains visible when it performs poorly. Every axis limit, logarithmic scale, point, and interval is unchanged.
- **Training:** a shared vertical metric label creates room between panels. The legend uses canonical names, a bold proposal entry, and matching shape/line encodings. Sparse markers lie on actual observed epoch means, at staggered epoch indices to distinguish naturally overlapping curves; no curve or point is jittered, smoothed, or extrapolated. The explanatory header states mean ± SD over three training seeds. The unaligned predictor's lower PushT curve and the framewise/diagnostic lower Reacher curves remain visible.
- **Teaser:** the evidence title identifies `ShiftWM (ours)` and `Framewise calibration` explicitly. Existing comparison marks, their labels and collision guard, axis bounds, callouts, and the RGBA concept illustration are preserved. The positive Reacher extrapolation change remains displayed.
- **Goal calibration:** rows use the same control ordering and names as the main comparison, with ShiftWM last. The two metric encodings retain their meanings: blue circles for goal calibration and orange diamonds for recorded-action prediction. A compact development/95% bootstrap header and a bottom legend separate uncertainty context from the panels. The log scale and every asymmetric interval are unchanged. The goal plot's orange color denotes the prediction metric, not method identity; its legend makes this explicit.

The training run appendix labels were also canonicalized in the same owned renderer at the table agent's request. No experiment identifier or completion state changed.

## Evidence verification

The before/after machine comparison preserves exactly:

- all 20 forecast means, sample SDs, per-seed values, source identities, fixed units, and extrapolation component values;
- all 10 training curves and their sample SD arrays;
- all eight development diagnostic records, including all metric means and bootstrap intervals;
- all four teaser comparisons and the unchanged concept-asset checksum.

Each renderer ran its existing source-validation path. No validator, aggregation formula, source completion gate, checkpoint selection, model, evaluator, or runtime configuration was changed. Means and SDs are not described as confidence intervals. The goal-calibration intervals remain source-provided 95% trajectory-seed bootstrap intervals at one trained seed.

Snapshots and the exact comparison audit are under `paper/build/editorial_plot_review/evidence_before.json` and `evidence_audit.json`. Canonical editable sources remain the four Matplotlib renderers in `paper/scripts/`; PDF/SVG/PNG exports remain in `paper/generated/`. SVG text is editable. The teaser is a hybrid vector/raster export; other quantitative plots are vector figures.

## Pixel review and limits

Fresh proofs were rendered from all four final PDFs with the installed `inspect_figure.py`, with full-size and targeted crop reports under `paper/build/editorial_plot_review/{forecast,training,teaser,goal}/`. Each PDF is exactly 396 points wide. Heights are 3.8 inches for forecasting, 3.35 for training, 2.18 for the teaser, and 3.15 for calibration.

All four 550-pixel paper-width proofs and enlarged exported PNGs were opened. Enlarged crops checked extrapolation marks/ranges, the training legend, and the teaser comparison labels. No clipped labels, ambiguous legend attachments, or material text overlaps were observed. The PDF-word check found no intersections exceeding 0.3 points in both axes, and the conservative page-bound check found no out-of-bounds words. These automated checks do not establish absence of every possible graphical defect.

An independent table agent inspected the forecasting and training paper-width proofs. It found canonical labels readable without overlap or clipping, the SD/frozen distinction visible, the unfavorable Reacher extrapolation retained, and the lower diagnostic/framewise training curves still visible. It reported no defect requiring edits. The assembled manuscript placement is the parent writer's next check; this review does not certify a newly compiled manuscript or confer statistical significance.

## Caption reconciliation

Use **ShiftWM (ours)** for the full paired-context model and the canonical control names above. Forecast and training error ranges remain **sample standard deviations over three training seeds**, not confidence intervals; frozen LeWM has one checkpoint. Training marker shapes identify methods at measured epochs, not individual seeds. The calibration figure remains a post-hoc development diagnostic with **95% trajectory-seed bootstrap intervals**, and its rows are now ordered frozen, framewise, shared, ShiftWM. Teaser improvements remain ratios of three-seed means without uncertainty estimates. No other scientific caption change is required.
