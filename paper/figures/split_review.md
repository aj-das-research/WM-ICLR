# Split figure: design and evidence review

Reviewed on 2026-09-18 using the installed paper-figure-creation skill. The brief and three distinct composition sketches are in `split_brief.md`. The canonical source is `factor_split.tex`; `split_standalone.tex` supplies a matching Helvetica/Times standalone render. The coordinating agent owns manuscript placement and caption integration.

## Scientific checks

- The existing PushT and Reacher tasks are recognizable through original vector glyphs. The physics labels refer to actual damping and density changes in the generator; no new task or camera/viewpoint shift is implied.
- The matrix has exactly seven training cells, development `(v1,p1)`, held-out `(v2,p2)`, and extrapolation `(v3,p0)`, `(v0,p3)`, `(v3,p3)`. Four additional intersections are marked unevaluated. The footer explicitly states that independent test trajectories cover all nine in-range combinations.
- The held-out composition is inside the familiar-factor grid. Dashed guides separate the new visual/physical factor values. This is a configuration matrix, never a performance heat map.
- The filmstrip and goal are original vector schematics, labelled as schematic PushT observations. They are not simulator screenshots, observed trajectories, success examples, or generated evidence. Observed frames use `o`, consistent with the manuscript; hidden physical state does not use this notation.
- The model-visible input panel shows observed history, executed controls and a goal image. State, factor IDs and task-success checking remain in an unconnected evaluator box. No score or simulator-state feedback into the predictor is drawn.
- The planner budget matches the main protocol: 300 candidates, 30 iterations, 30 elites, five-group horizon, five native controls per execution, 50 total including ten support controls. Termination at task success or the budget limit is stated. The caption/manuscript retain details of support-only success accounting.
- The figure makes no result, superiority, generality, causal-identification or identical-supervision claim.

## Pixel and export checks

The first render exposed excess whitespace from TeX interword glue around local color definitions and overflowing long annotations. Both were corrected in the canonical source. Main text is 9 pt, panel headings 10 pt and supporting labels 8 pt at the final 5.5-inch width; no rasterized typography is used in the PDF.

Inspected the actual color rendering enlarged at 990 pixels, at 550 pixels (5.5 inches at 100 dpi), and in grayscale at the same 550-pixel width. Then inspected the final 300-dpi PNG. Text, arrows, task glyphs and all matrix cells are readable without clipping or collisions. Direct labels, filled held-out cell, amber development outline and dashed extrapolation outlines retain distinctions in grayscale. Arrow endpoints and evaluator separation remain clear at paper width.

`pdflatex` completed with no warning, overfull or underfull entries in the standalone log. `pdffonts` reports all eight fonts embedded. Export sizes:

| Artifact | Geometry / role |
|---|---|
| `split.pdf` | 396 × 240.945 pt; exactly 5.5 inches wide; vector export |
| `split.svg` | Vector export from the same PDF; text represented as vector glyphs |
| `split.png` | 1650 × 1004 pixels, 300 dpi; shareable preview |
| `factor_split.tex` | Canonical editable TikZ geometry and labels |

Build products and review previews were rendered in the distinct directory `paper/build/split_figure_review`; manuscript builds, evaluation code and experiment artifacts were not changed.

## Independent comprehension review

The world-model agent inspected `split-paper-width.png` before reading the brief and correctly recovered all seven training cells, the development and held-out cells, all three extrapolation cells, the independent all-nine in-range test population, model-visible inputs and evaluator-only information. It also correctly read CEM 300/30/30, the five-group horizon, five-control execution and paid support within the 50-control total. It found no misleading geometry, clipping or readability defect at 550 pixels.

The coordinating agent separately checked notation and requested observation labels `o_0,o_1,o_2`; that correction is included. Final manuscript placement and surrounding caption remain subject to its full-page review. No claim is made here that this standalone check substitutes for full-paper inspection.

## Updated visual-elements skill: scenario audit

- **Compact world-model diagram:** the guidance encourages a persistent observed scene, filmstrip and controlled action/response comparison with schematic qualification. Exact learned operations and information boundaries stay in deterministic vector layers; it does not require a decorative robot or generated architecture.
- **Clinical qualitative example:** observed dataset evidence is required when an image establishes what was seen or predicted, with selection/source/crop tracked. A generated medical scene cannot substitute for results or imply an unperformed experiment; illustrative status and provenance are explicit.
- **Scalar benchmark chart:** the reference explicitly allows a restrained numerical plot without illustrations and imposes no icon or image quota. Preserving unfavorable outcomes, uncertainty and material conditions takes precedence over visual decoration.

These scenarios support richer explanatory visuals without forcing generated evidence or icons onto quantitative plots. No blocking ambiguity found in `references/visual-elements.md`.
