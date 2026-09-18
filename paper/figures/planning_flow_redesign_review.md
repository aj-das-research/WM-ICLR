# Figure 2 planning-flow repair review

Date: 2026-09-18. Canonical editable source: `world_method.tex`.
Scope: the conventional planner strip between LeWM P and action execution. A/B context inference and C paired supervision retain their scientific meaning and assets.

## Observed defects and repairs

| Version / defect | Repair | Pixel evidence |
| --- | --- | --- |
| Prior: an unlabeled three-branch fan had no interpretable object or comparison operation. | Replace it with three schematic feature-state sequences, outlined terminal tokens, an explicit MSE comparison of gray predicted and blue goal features, and per-candidate cost J_i into CEM. | Final full-width proof and `terminal-mse.png`. |
| Prior: goal wire entered CEM directly, hiding what compared the prediction and goal. | Goal now terminates at the MSE block. | Final planning-strip and goal-input crop. |
| Prior: sampled candidates appeared disconnected from CEM refinement. | Draw a closed, labeled resampling route from CEM back to candidate action plans, separate from the final-mean-plan execution edge. | Final planning-strip and goal/resample crossing crop. |
| Draft 1: resampling label overlapped C-panel boundary; mean-plan text crowded CEM. | Move the label between the two routing lanes, shift C down by 0.7 cm, and move mean-plan label left. | Draft 2 and final full-figure proofs. |
| Draft 1: first latent row nearly touched its header; all per-step glyphs looked identical. | Increase header-to-row clearance and vary schematic feature heights over steps. | Final enlarged latent panel. |
| Draft 2: feedback stem crossed the `elite refit` label. | Move the entire label left of the vertical stem, with no opaque patch. | Final enlarged planning-strip. |
| Draft 2: terminal-latent notation crowded the adjacent box edges. | Widen the MSE-to-futures gap and center the notation in that gap. | Final terminal/MSE crop. |

## Semantics and evidence

- Exact operation: mean squared terminal feature error against the common calibrated goal, not an error over every intermediate prediction.
- CEM retains low-cost elites, refits both action-distribution mean and population standard deviation, and resamples. The returned plan is its final updated mean; the diagram labels that output `mean plan` and executes only the `first block`.
- The repeated state glyphs, their heights, and the three displayed candidate rows are explanatory vector schematics. They are not measured scores, plotted trajectories, generated pixel frames, or a claim of three actual candidates.
- No particular candidate is marked as a winner. The outlined tokens identify terminal time, not selected candidates.
- The blue goal route uses a visible continuous overpass at the single crossing with the gray resampling route. There is no junction dot; the wires carry distinct information.
- No inference-time parameter update is shown. Existing real input/training images and their manifests are unchanged.

## Inspected outputs

Standalone final PDF: `paper/build/planning_flow_redesign/final/method_standalone.pdf`, 396.020 × 337.329 pt (approximately 5.5 × 4.69 in). Source font remains 8.5 pt for primary labels, scaled by approximately 1 at insertion width.

Actual pixels inspected by the implementing agent:

- `final/paper-width.png`: 100 dpi, whole figure at intended width.
- `final/grayscale.png`: 100 dpi, whole figure; terminal outlines and flow remain distinguishable without color.
- `final/detail.png`: 300 dpi export, with the following inspected crops of its 1651 × 1406 px canvas:
  - `planning-strip.png`: (0, 626, 1651, 907).
  - `terminal-mse.png`: (479, 654, 1024, 837).
  - `goal-resample-crossing.png`: (1420, 745, 1634, 914).

No remaining material overlap, false junction, hidden connector shaft, detached arrowhead, or crop was found in these views. The overpass has an intentional local white under-stroke; this is a crossing convention, not an accidental break in either source edge.

## Independent feedback and integration boundary

Root independently inspected drafts 1 and 2. A second pixel reviewer correctly reconstructed predicted futures → terminal goal MSE → CEM elite/refit → resampling and final-mean-plan first-block execution. Both root and the second reviewer identified the feedback-stem/label collision; the final source repairs it. Root also requested and confirmed the representation of varying schematic feature states and additional header clearance.

Final manuscript-page review, caption matching, canonical PDF/SVG/PNG exports, and source-hash provenance update are performed by root during integration. This focused review does not certify unrelated figures or claim publication readiness.

## Completed manuscript integration

Root rebuilt `paper/world_model_draft.pdf` and the canonical method PDF/SVG/PNG
exports. Figure 2 remains on page 4; the manuscript has 22 pages and the appendix
starts on page 9. Root inspected `integrated/page-04.png` at 135 dpi, the final
planning-strip and terminal/MSE crops, the goal/resampling overpass, and the
whole-figure paper-width and grayscale proofs. The caption describes terminal
feature MSE, low-cost elite refitting, final-mean-plan first-block execution,
and the schematic status of the feature glyphs. No material clipping, label
collision, ambiguous junction, or detached tip was found in the inspected
regions. The build has no overfull boxes or unresolved references.

An independent pixel reviewer rechecked the final paper-width proof and
planning-strip/terminal-MSE crops: both reported spacing defects are resolved,
and the predicted-futures, scoring, mean-plan execution, and resampling paths
remain understandable without the caption. The goal wire is a scoring input,
not a predictor input or a training signal.

The reviewed geometry source SHA256 is
`1c03c1c5208cba0ed1c6592352ef8219080a9e02f7e2f67e4aa2a915eb56e8ec`.
The integrated PDF snapshot and machine-readable source record are retained
under `paper/build/planning_flow_redesign/integrated/`; the live build ledger
pins the current canonical artifacts. No model, scoring, or experiment code
changed as part of this repair.
