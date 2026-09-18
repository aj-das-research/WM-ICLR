# Goal-calibration diagnostic plot

Reader/slot: ICLR manuscript appendix, verified 5.5-inch full text width
(`iclr2027_conference.sty`, textwidth), 3.15-inch height, 8-point plot type.
Mode: ordinary experimental plot; neither a new architecture diagram nor a teaser.
Focal relationship: compare the error of the available corrected goal with the
error of the predicted terminal feature against the same canonical goal.
Small prediction error need not imply small goal-calibration error.

Two environment panels share logarithmic horizontal axes and four model rows.
Blue circles denote visual-goal error; orange diamonds denote future-prediction
error. Distinct marker shapes preserve identity without color. The point-range
layout exposes small Reacher errors that were nearly invisible as linear bars.
Marks derive exclusively from completed development diagnostics. The ledger
`paper/generated/goal_calibration.json` pins all eight sources and metric keys.
The renderer checks protocol/provenance, exact case pairing and recomputed means.
Error bars preserve source 95% trajectory-seed bootstrap intervals. Trained models
use seed0; intervals do not measure training-seed variability.

Selection is all32 prespecified development tasks, excluding1 PushT and2 Reacher
support-only successes identically for every method. Recorded future actions
are diagnostic inputs, not information supplied to the deployed controller.
Canonical goal renders are evaluator-only targets. No efficiency or causal
claim follows from this plot. The frozen baseline has no new training seed.

Editable source: `paper/scripts/render_goal_calibration.py`; vector PDF/SVG and
PNG preview are generated together. Final pixel review is recorded after rendering.

Root visual review, 18 September 2026: the first 5.5-inch export joined the
Factorized/Framewise tick labels. Rotating ticks 25 degrees and increasing bottom
space corrected the overlap. The final standalone PNG and compiled manuscript
page11 were inspected; axes, legend, labels, bars and asymmetric intervals are
visible without cropping. The hatch pattern preserves metric identity without
depending only on color. Numerical means were recomputed directly from measured
records before plotting. This is an evidence-backed diagnostic, not a claim of
publication readiness or a factorization benefit.

Independent agent review: `reports/goal_calibration_figure_review.md` agrees with
the diagnostic interpretation and finds no overlap/clipping. The almost-zero
Reacher prediction bars are a documented linear-scale limitation; the caption's
adjacent numerical prose supplies their values. Cross-environment error magnitude
is not a shared physical scale because each environment uses its own encoder.

Current visual revision: 18 September 2026. Replaced bars with horizontal point-range
marks on explicitly labeled logarithmic axes, retaining every original mean and
asymmetric 95% interval. Recomputed eight validated sources. Alternative layouts
considered: grouped bars (small prediction errors disappear), vertical log-scale
points (long model labels remain crowded), and the selected horizontal model
rows (readable labels and interval comparison). Standalone PNG inspected at full
resolution; final PDF/page review is recorded in visual_refresh_review.md.
