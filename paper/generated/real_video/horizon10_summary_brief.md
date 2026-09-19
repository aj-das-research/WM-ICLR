# Figure brief and representation contract

Reader/slot: ICLR development manuscript, official 5.5-inch text width
(`paper/template/official/iclr2027/iclr2027_conference.sty`, line49), standalone
result figure outside the manuscript. Use 8-point minimum type at actual width.

Visual thesis: matching training and selection to a ten-step rollout reduces
mean long-horizon error for all four unchanged modes; the separately matched
ShiftWM-versus-Framewise advantage is smaller and must not be conflated with
the horizon-control gain. This figure does not claim a new architecture.

Objects are measured paired differences and uncertainty bars. Absolute signed
standardized feature MSE is common to both panels; zero means no difference.
Green diamonds and direct labels identify ShiftWM (ours); dark circles identify
the other unchanged modes. Green bold percentages encode favorable relative
point changes only, not statistical significance. No generated images, task
screenshots, causal arrows or unmeasured outputs are needed for this question.

Three genuinely different compositions are rendered with actual numbers:

1. Stacked paired forests on a common effect axis: selected, because it gives
   long method labels and modest effects enough room at 5.5 inches while
   separating the two scientific questions.
2. Side-by-side forests: rejected because method labels and gains squeeze the
   interval plots into narrow columns; different axes could overstate small
   cross-method advantages.
3. Raw-MSE dumbbells plus an all-contrast strip: rejected because it combines
   two measurement grammars and makes paired confidence intervals secondary.

Reading path: matched training-horizon comparison (four modes, mean1–10) →
equally h10-trained method comparison (h5/h10, means/endpoints) → explicit
unfavorable matched h5-prefix secondary. Invariants: three model seeds, original
validation only, original architecture and train-only normalization, paired
episode/session/window populations within each declared contrast.

Source: all20 comparisons in `reports/real_droid_horizon10_results.json`, fixed
SHA256 `38ebb812819d4326c568aeb1d6fdf4c5422856f9e3c8e683de2f5a25bc84fe91`.
The rendering script reloads every source evaluation ledger, verifies its hash,
and invokes the independent arithmetic/bootstrap part of the existing paper
postprocessor without its model-inference collector. All20 effect sizes and
CIs must agree before rendering. No raw features/images or model predictions
are read or generated. Native metric precision is kept in JSON/CSV; graphical
MSE differences are multiplied by1000 solely for legible axis labels.

Limitations: training and checkpoint-selection horizons change together;
validation was used for selection/development; comparisons are exploratory
without multiple-testing adjustment. All12 runs completed30epochs but every
selected checkpoint is epoch1. The h5-prefix secondary factorized mean is
slightly unfavorable and remains explicit. The caption retains populations,
uncertainty method, precise unfavorable value and all20-contrast context.
