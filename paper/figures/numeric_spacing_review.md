# Numeric-figure spacing review

2026-09-18. Scope: `world_teaser`, `forecast_comparison`, `training_curves`, and `goal_calibration`. All four PDFs are exactly 396 points (5.5 inches) wide. Reviewed both the actual PDF paper-width renders and enlarged PNGs; the review did not substitute source-code inspection for rendered pixels.

| Figure | Observed defect | Action and result |
|---|---|---|
| Teaser | The `−26.8%` value label overlapped the `Reacher` row label. Actual PDF word boxes intersected by 7.35 horizontal × 2.40 vertical points. | Only the colliding numeric label moves above/right of its unchanged marker. Its final PDF box is separated from `Reacher` by 31.98 horizontal and 7.60 vertical points. A renderer guard checks value labels against row-label boxes with a 2-point margin before export. |
| Forecast comparison | No material label, axis, title, interval or footer collision observed. | Renderer and plot geometry left unchanged. Panel-specific log axes and sample-SD intervals remain intact. |
| Training curves | No material axes/legend collision observed. Some curves overlap because their measured values are close. | Renderer and evidence left unchanged; no jitter or other artificial separation introduced. |
| Goal calibration | No material axes/title/legend/interval collision observed. | Renderer and evidence left unchanged; existing log scale and 95% bootstrap intervals preserved. |

After the repair, none of the four PDFs has an extracted-word intersection greater than 0.3 points in both axes. This limited geometric check is **not** a general detector of all overlaps, connector continuity or scientific correctness. Paper-width and enlarged rendered views were inspected separately. The teaser's appearance/dynamics callouts remain unobstructed; the moved blue annotation still belongs visibly to the Reacher held-out marker.

Before/after numeric equality was checked for all four teaser rows, forecast result values/uncertainties, training curves/SDs, and goal-calibration diagnostic runs/intervals. The RGBA concept artwork checksum is unchanged. All source-validation code remains in place. Changes are confined to `render_teaser.py`, its regenerated PDF/SVG/PNG/source-linked metadata, and this review.

Reproducible paper-width/detail proofs and a machine-readable PDF-box/source-hash audit are under `paper/build/numeric_layout_review/`. Those are temporary inspection artifacts; the numeric figure sources and ledgers remain authoritative. Root owns the full-manuscript rebuild and placement review.

An independent reviewer subsequently inspected the updated teaser at 550px, full size, and a targeted crop: the moved annotation clearly belongs to Reacher's held-out result, with no new overlap observed.
