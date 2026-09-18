# Method figure redesign review

Date: 2026-09-18. Deliverable: `world_method.tex` and hybrid PDF/SVG/PNG exports. Skill references applied: visual-story, method, design-system, art-direction, review, hybrid-authoring, semantic-primitives, and the newly installed visual-elements guidance.

## Before / after defects addressed

- The previous image had no visible goal-calibration or terminal-cost branch; these now have explicit nodes and edges.
- The earlier generic input boxes now contain actual support and goal images with a reproducible selection and asset ledger.
- Observation context now exposes mean/variance pooling. Adapters visibly transform raw features or action embeddings using scale and shift; the dynamics branch exposes the transition/action tuple and GRU.
- The planner's input is a future **latent**, not a future image. A note prevents interpreting the schematic rollout as generated video.
- Training-only reference targets occupy a separated strip, with no edge into online contexts or goals.
- Stage numbering and Helvetica headings make the support/inference/planning sequence easier to scan. A blue goal-feature path visibly joins the neutral forecast path at the cost node.
- The first draft render showed title wrapping, labels colliding with lane headings, a candidate-actions label near the execution box, and dense footer text. Labels were shortened and lane spacing increased; fonts were not reduced to hide these issues.
- Root noted the manuscript's `x` notation denotes simulator state, and the first draft's future-image wording could exclude the authorized goal input. Both were corrected before export.

## Actual checks completed

- Inspected the previous `method.png` before changing its composition.
- Compiled the final standalone using `paper/build/method_redesign/`, without touching main build products.
- Inspected the final enlarged rendered pixels, the exact 5.5-inch proof at 96 dpi, and grayscale at 144 dpi. Main labels remain legible; arrows end at intended nodes; no clipped or overlapping text was observed.
- Verified the goal branch uses the same `Ao` and `co` as history, corrected history feeds both dynamics inference and prediction, executed/candidate actions are distinct, and canonical reference targets cannot reach deployment nodes.
- The smallest prose style is 8.2 pt before slightly upward manuscript scaling. Math subscripts remain normal typesetting subscripts, rather than tiny prose labels.
- All four SVG image elements contain embedded base64 PNGs. Asset hashes and source dimensions are recorded in `assets/method_assets.json`.
- Actual-size PDF width is 396.019 points, matching 5.5 inches to TeX rounding. The proof's typography and positions use the same source as the manuscript.
- No experiment code, evaluator, training configuration or GPU job was changed.

## Independent review

Completed by `/root/world_model` after the final Helvetica export and heading repair. The reviewer inspected the actual 5.5-inch proof and confirmed the shared observed-goal/history correction, dynamics GRU from corrected support/actions, conditioned candidate actions, terminal MSE, and offline-only canonical targets. Four embedded SVG images and absence of external image references were confirmed. The reviewer reported no remaining paper-width readability blocker. The root agent owns final manuscript caption integration and the whole-page raster inspection.

## Export / evidence boundaries

The figure is hybrid: editable TikZ geometry/type, actual raster observations, outlined SVG glyphs. No generated illustration was needed. Thumbnails are unmodified canonical-appearance input examples from one bundled PushT episode; the source indices are 0, 1, 2 and 7. They are not a prediction, a measured calibration effect, or selected successful behavior. All latent symbols are schematic. Context-consistency training terms are deliberately left to the text rather than represented by ambiguous crossing arrows.

### Review repair

Root and the independent model reviewer both found that the first exported stage-2 heading crossed the vertical observation-context route; stage 3 was also too close to the goal-feature route. This was a material defect missed in the first self-review. The headings were shortened to “2 Imagine futures” and “3 Score and act” while keeping type size, scientific labels and computation unchanged. PDF, SVG, PNG, exact-size and grayscale proofs were regenerated. Both headings now end well before the routes at x=4.90 cm. The standalone log has no overfull or underfull boxes, and the final SVG still has exactly four embedded PNG image elements.
