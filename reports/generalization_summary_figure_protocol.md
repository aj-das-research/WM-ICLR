# Generalization development figure protocol

This protocol fixes the figure's scope and representation before the complete
36-run / 72-evaluation campaign is available. It does not change the registered
training, checkpoint selection, evaluator, or finalizer.

## Scientific question and slot

At the ICLR manuscript's 5.5-inch text width, compare the three registered
optimization/capacity interventions and all four matched model variants, then
show how precisely each ShiftWM-versus-Framewise difference is estimated.
These are **original-validation development results**, not another test set and
not new confirmatory evidence. All three seeds, three arms, two horizons, four
methods and favorable/unfavorable effects remain visible.

The source contracts are `reports/real_droid_generalization_results.json`,
`reports/real_droid_generalization_finalization.json`, the registered 36 configs,
and their 72 original-validation result files. No image, action, feature or
prediction payload from either test set is read by this renderer.

## Compositions considered before completion

1. A heatmap of arm/method error with a separate forest plot: compact, but color
   differences would obscure the small absolute MSE differences.
2. Aligned six-row dot/error-bar and paired-difference panels: absolute errors
   retain all four methods, while a dedicated zero-centered difference panel
   makes the uncertainty directly interpretable. **Selected.**
3. An epoch/model-size heatmap plus gain chart: useful for diagnosis, but would
   replace the primary matched forecasting comparison with auxiliary metadata.
   Epochs and parameter counts remain in the companion finalized tables.

## Representation contract

- Six aligned rows: Slow / 5, Slow / 10, Decay / 5, Decay / 10, Compact / 5,
  Compact / 10. The row labels identify action-block forecast horizons.
- Left panel: absolute standardized endpoint feature MSE for Framewise, Constant
  dynamics, ShiftWM (ours), and Action-free. Marks are three-seed equal-episode
  means; horizontal whiskers show sample standard deviation across seeds.
  Method identity has both color and distinct marker shape. Small vertical
  offsets separate methods and do not encode a numerical variable.
- Right panel: ShiftWM minus Framewise MSE and the registered paired 95%
  session/seed bootstrap interval (10,000 draws). The visible zero line denotes
  no difference. A green mark denotes a favorable point difference; rust denotes
  an unfavorable point difference. An open mark means the interval includes
  zero; a filled mark means it excludes zero. These exploratory intervals are
  unadjusted for multiple comparisons.
- A separate aligned text column gives the source-derived relative reduction
  `100 * (Framewise - ShiftWM) / Framewise`. It is a point estimate, not the
  interval's unit and not a percentage-point change in a success rate.
- All intervals are fully contained in their axes. Axis limits follow all data;
  no outlier or unfavorable result is clipped. No pictorial assets or generated
  images are used because the numerical comparison is the focal relationship.

## Completion and publication gates

The renderer rejects absent/partial reports, any missing arm/method/seed,
non-completed 30-epoch training, missing 72 validation evaluations, missing 36
independent offline reload passes, changed registered config/evaluation hashes,
unmatched episode populations, and disagreement between raw equal-episode
means, sample seed SDs, registered aggregates and independently finalized
aggregates. It independently reproduces the registered bootstrap intervals.

Default rendering creates a **review candidate only** under
`artifacts/publishing/generalization_summary_review`, outside manuscript assets.
The candidate includes PDF/SVG/PNG, exact-data ledger, geometry report and a
standalone 5.5-inch LaTeX proof. It writes no manuscript include. A completed
candidate still requires actual pixel inspection at print size and an enlarged
view, including the zero line, all uncertainty bars, legend, text and caption.

Only `--publish-reviewed REVIEW.json` can copy the reviewed files into
`paper/generated/real_video/generalization_summary*`; the review must explicitly
bind the exact candidate PDF and ledger SHA256 and record visual checks. The
figure TeX include is written last. Automatic jobs may prepare a candidate after
successful finalizer job 200177, but cannot certify its aesthetics or publish an
unreviewed figure.

The canonical source is `paper/scripts/render_generalization_summary.py`.
