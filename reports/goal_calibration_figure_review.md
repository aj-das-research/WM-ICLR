# Independent goal-calibration figure review

Reviewed 2026-09-18 using `view_image` on `paper/generated/goal_calibration.png`, with its source ledger, editable renderer, figure brief, and current `paper/main.tex` caption. No figure or manuscript source was changed during this review.

## Scientific takeaway and scope

The figure supports the distinction between error in the available corrected goal and prediction error against an immutable canonical goal. In Reacher, factorized seed 0 has approximately 0.0117 recorded-action prediction MSE versus 0.7967 corrected-goal MSE; framewise has approximately 0.00855 versus 0.2712. Small recorded-action prediction error therefore coexists with substantial corrected-goal error in these observed development cases. This motivates, but does not establish the outcome of, a controlled goal-correction intervention.

The caption and surrounding text correctly constrain this to posthoc development diagnostics, one training seed per trained method, five recorded future action blocks, and privileged evaluator-only canonical targets/future actions. There are 31 eligible PushT and 30 eligible Reacher trajectories after the shared support-success exclusion. The intervals are 95% trajectory-seed bootstrap intervals, not training-seed variation. The text explicitly declines causal attribution and improved-planning claims. The plotted means/intervals match the eight ledger entries.

PushT adds a useful counterpoint: lower prediction or goal-calibration errors occur without a demonstrated difference in the accompanying raw development success counts. These bars should not be described as planning success, action-ranking accuracy, or evidence that the calibration error causes the deficit. Reacher's factorized and framewise prediction intervals overlap; the point estimates alone do not establish a significant difference between those methods. The two environment panels use separate pretrained models, so cross-environment latent-error magnitude is not a common physical-distance scale.

## Visual assessment

The current plot is clear and suitable for the appendix. No overlap, clipped legend, or clipped interval is visible. All bars start at zero, and shared limits support honest visual comparison. Blue and orange are distinct; orange hatching additionally separates the metrics in grayscale. Method order and labels are consistent across panels. The editable source specifies the actual 5.5-inch manuscript width with 8-point text; the layout should remain readable at its intended size.

The main visual limitation is the near-zero Reacher prediction bars (approximately 0.009 and 0.012): on the 0–1.5 scale their heights and intervals are almost invisible. This is not misleading—the scale is appropriate for displaying the large calibration mismatch—and the adjacent prose gives the values. If the figure must stand alone, small numeric labels on those two bars would help. A log scale or cropped y-axis is unnecessary and would complicate the main comparison.

Optional caption polish for a later revision: explicitly name the development appearance/dynamics pair `(v1,p1)` and identify blue as corrected-goal error, just as orange is identified as recorded-action prediction. These are self-containment improvements, not blocking corrections. The current legend already identifies both metrics.

## Recommendation

Accept the current figure and caption for this development appendix, subject to the parent's final manuscript-page raster inspection. Preserve the pending-intervention wording until results are available. No visual or scientific issue found here requires changing the primary evaluation protocol or rerunning experiments.

Reviewed artifact hashes (SHA256):

- `paper/generated/goal_calibration.png`: `1877f722826b605d30427d942ebf980262d294f46b9b6a7c9beb67198ff7bb51`.
- `paper/generated/goal_calibration.json`: `eeef9425b86c1079f5071f2ff01059c45efcb8851b015c4f78120664ef8d2ae4`.
- `paper/scripts/render_goal_calibration.py`: `9a49c05b6d10e3db0d4a5d33b21a81d3066210335743e34b5ade0f120bdf4bd6`.
