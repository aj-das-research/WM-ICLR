> Historical review. Superseded after user feedback that the figures remained too text-heavy. See visual_refresh_review.md for the structural redesign.

# Visual refresh review — 18 September 2026

This revision applies the locally customized paper-figure-creation skill. Its
GitHub source, verified revision, local file hashes and portable patch are in
paper/evidence/figure_skill_provenance.json. The skill's frontmatter validator,
local reference resolution, and patch applicability checks passed. The update
adds visual objects/icons, analogy mappings and limits, reference inspection,
and generated-image selection/integration; this paper uses actual simulator
thumbnails and original vector schematics rather than generated experiment images.

## Delivered figures

| Figure | Concrete change | Authoritative editable source |
| --- | --- | --- |
| Method | Actual history/goal thumbnails; explicit shared goal calibration; separate context, imagination, control and training paths | world_method.tex; assets/method_assets.json |
| Setup | 4×4 factor matrix with three extrapolation pairs, task glyphs, observation/action filmstrip, shared planning budget and evaluator boundary | factor_split.tex |
| Forecasts | Replaced empty result panels with four matched point-range panels, retaining unfavorable extrapolation and all five aligned comparators | ../scripts/render_forecast.py; ../generated/forecast_comparison.json |
| Training | Replaced 15 overlapping seed traces with five mean curves and sample-SD bands; readable names and redundant line styles | ../scripts/render_training.py; ../generated/training_plot.json |
| Goal calibration | Replaced linear bars with log-axis point ranges so small Reacher prediction errors remain visible | ../scripts/render_goal_calibration.py; ../generated/goal_calibration.json |

All figures export PDF, SVG and PNG. The method PDF/SVG contain four embedded
raster input thumbnails; vector labels/geometry remain precise and TikZ is the
editable master. Setup and statistical charts have vector marks. Matplotlib SVG
text is live; TikZ-exported SVG glyphs are outlined. No experiment result was
edited to change the figures.

## Evidence and semantics

- Forecast renderer verified 52 completed source files, checkpoint/config/data/
  evaluator identities and matched record keys. Its 20 plotted points retain
  exact means and sample SD over three training seeds; frozen has one checkpoint.
  Unaligned plain remains in the primary table. Held-out gains do not hide the
  Reacher extrapolation regression or frozen baseline advantage there.
- Training chart's 10 curves, 30 epochs each, were independently recomputed from
  all 30 original run logs with exact agreement for every mean and sample SD.
  No smoothing, interpolation, or test-performance interpretation is applied.
- All eight goal-calibration diagnostics were revalidated; numerical inputs and
  original asymmetric 95% trajectory-seed bootstrap intervals are unchanged.
- Observed support and goal images are model inputs. Canonical training targets,
  unobserved future states and factor IDs do not enter deployment context.
  Setup image labels use o, matching the manuscript's observation notation.

## Observed defects and repairs

1. Old calibration bars hid nearly-zero Reacher prediction errors. Horizontal
   log-scale point ranges reveal their scale, with explicit log labels and a
   caption distinguishing trajectory intervals from training-seed variation.
2. Original seed-trace training plots used crowded lines and a small legend.
   Mean/SD summaries and readable names improve method comparison while the
   ledger retains every seed.
3. Revised method's long stage titles crossed vertical context/goal routes.
   Shortened to “Imagine futures” and “Score and act”; root and independent
   reviewer rechecked the repaired geometry.
4. Setup first export extended beyond the canvas. The author repaired the
   layout and confirmed exact 5.5-inch width before integration.

## Pixel and manuscript review

Root inspected standalone full-resolution exports and 550-pixel width proofs.
Independent agents inspected every figure for actual rendered readability and
scientific interpretation, including grayscale diagram/forecast proofs. They
recovered the goal-calibration path, training-only target access, factor split,
planning budget, forecast limitation and correct uncertainty meanings.

The 14-page compiled research draft was checked on pages 3, 5, 9, 11 and 12,
which contain the five revised figures. No clipped labels, crossed captions,
overflow or unreadable legends were observed. LaTeX reports no overfull boxes,
undefined references or warnings. The main text ends on page 9; final venue
compliance and scientific review remain separate from this visual check.

## Limits

No empirical claim follows from an icon, generated illustration, analogy, or
computation schematic. Tiny forecast SD ranges can lie inside the point marker;
exact values remain in the ledger. Separate pretrained coordinates prevent
interpreting cross-environment MSE as one shared physical error scale. Closed-
loop multi-seed comparisons remain incomplete. This record documents an internal
source/pixel review, not a user study or a guarantee of publication readiness.
