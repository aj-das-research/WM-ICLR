# Paired planning comparison

This appendix figure makes the primary control question visible: does ShiftWM
improve closed-loop success over each matched control? It complements the main
success table and the forecasting plot rather than treating forecast error as
evidence of better decisions.

- **Format:** 5.5-inch-wide vector PDF/SVG and 300-dpi PNG; a 2-by-2 point-and-interval
  layout separates environments and held-out composition from extrapolation.
- **Marks:** orange diamonds are the mean raw-success difference, in percentage
  points, for ShiftWM (ours) minus the named control. Continuous horizontal
  intervals are the existing conditional 95% paired task-cluster bootstrap.
  A vertical dashed zero line means no difference. No connector encodes a model
  operation; no illustration or generated experimental image is used.
- **Missingness:** all three training seeds for both methods must be complete.
  Pending comparisons have axes-relative text and no data-coordinate mark.
- **Evidence:** `reports/evidence/paired_planning_results.json`, produced by the
  existing paired reporter. The renderer checks current reporter/validator and
  evaluation/data hashes, the three-seed gate, units, paired wins/losses, and mean
  arithmetic. It does not replace the upstream checkpoint/protocol validator.
- **Scope:** raw success includes the paid support period. Eligible success and
  training-seed SD remain in the appendix table. These intervals condition on the
  three observed trained models; they do not quantify a population of training
  runs and have no multiplicity adjustment. Crossing zero is visible.
- **Composition choice:** a shared-axis forest layout permits direct comparison
  with zero. Bars would overemphasize area; a connected curve would imply an order
  among methods. Repeated comparator rows make pending evidence easy to locate.

Canonical editable source: `paper/scripts/render_planning_comparison.py`.
The matching `.json` stores the exact input hash, rows, renderer hash, and units.
Pixel review is recorded in `paper/editorial_revision_review.md` after integration.
