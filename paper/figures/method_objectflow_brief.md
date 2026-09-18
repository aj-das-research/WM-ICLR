# Sparse object-flow method figure

This brief describes the first sparse layout. The current geometry was regenerated after additional user-reported defects; see `arrow_regeneration_review.md` for the corrected group ports, explicit arrow sizes and wider routing gaps.

2026-09-18. This replaces the main text-heavy method figure; the root agent preserved that version as the archival editable source `world_method_detailed.tex`; it is not currently included in the appendix. This brief supersedes the earlier main-figure composition, not its source semantics or asset provenance.

## Design constraints

Actual paper width 5.5 inches, height at most 3.5 inches. Large real observations (~1.1 cm per frame), no paragraph boxes, no pooling/GRU/loss formulas, and approximately 55 prose words or fewer. Scientific objects and transformations should explain the diagram before its caption is read. All feature strips, action glyphs, and latent branches are schematics, not measured features, reconstructed pixels, physical trajectories, or successful robot behavior.

## Three compositions before rendering

1. **Object flow with a return action loop (selected).** Actual history/goal objects on the left, a shared blue affine-calibration rail in the center, compact dynamics/predictor operators and branching latent candidates on the right. The selected first action closes a low route back to real observations. This gives calibration a visible inside operation and makes shared goal correction the focal relation.

   ```text
   history filmstrip -> frozen E -> feature strips -> shared affine rail -> support latents -> dynamics
   executed arrows ----------------------------------------------------------^
   goal image       -> frozen E -> feature strip  -> same affine rail  -> goal latent
                                                                        predictor -> latent branches
                                                                        goal <---- compare/select
   real observation <-------------------------- highlighted first action -----------
   ```

2. **Paired correction close-up.** Two matched observations flank a large before/after latent-coordinate inset, followed by a small planner. This foregrounds the representation mechanism, but corrected-coordinate plots could be mistaken for measured alignment and the two-context dependencies would need a second diagram.

3. **Two-lane main figure.** Upper lane: history to rollout, lower lane: goal to goal comparison. Observation context bridges both lanes and dynamics conditioning enters above prediction. Clear data flow, but another row of rectangular modules would repeat the rejected visual grammar rather than showing scientific objects.

## Representation contract

- Four unmodified canonical-appearance PushT frames remain the real input objects; provenance is in `assets/method_assets.json`. Indices 0/1/2 are chronological history and index 7 is the illustrative same-episode goal. No new test evaluation is read and no input is presented as a prediction.
- Raw history and goal enter the same frozen encoder, depicted as two small `E0` chips with lock glyphs.
- Observation context is inferred from **raw history**, then conditions one visibly shared `Ao` calibration rail. Both history and goal feature strips cross that rail. The repeated multiplication/addition glyphs stand for context-dependent feature-wise affine calibration; there is no image-space recoloring.
- Corrected observed history plus executed actions enter the dynamics context. Candidate action embeddings receive that context through `Ad`; corrected history and conditioned actions enter the LeWM predictor.
- Candidate branches are explicitly labeled schematic latent futures. They cannot feed context inference. Terminal goal comparison and CEM select an action sequence; only its first action block is highlighted for execution. The feedback route denotes real action/observation, not a generated-image feedback loop.
- The main view omits the training graph. The caption and method body must state frozen reference coordinates, offline paired canonical supervision and consistency losses, factor IDs unavailable to deployment, contexts fixed within candidate rollouts, and model weights fixed online.

## Asset / analogy decisions

Actual observed frames establish the objects the method sees. Feature strips, the shared affine operator, action arrows, and branching latent graph are original editable vector geometry. The two frozen-encoder lock marks now use an unchanged transparent PNG from Google's official Material Design Icons repository; source URL, Apache-2.0 license, hash, display size, and alpha checks are in `assets/material_lock.json`. No background removal was needed. No decorative robot/camera icon or generated illustration is needed in the computation panel. The latent-branch analogy maps shared prediction prefixes and candidate continuations to paths; it does **not** imply physical trajectories, a two-dimensional measured embedding, actual candidate scores, or a planning gain.

## Final delivery and checks

The exact-width proof is 396.019×240.119 PDF points: **5.5×3.34 inches**, below the requested 3.5-inch height. Main labels are 8.7 pt, secondary text 8.2 pt, and stage labels 10.3 pt before the slight upward manuscript scaling. All four input images now display at 1.10 cm (roughly 510 effective dpi after scaling), compared with 0.59/0.62 cm in the rejected diagram. Source images and hashes are unchanged.

There are approximately **24 prose words**: Observe; Calibrate; Imagine + act; Executed; History; Goal; shared; Candidate actions; Schematic latents; LeWM; Goal distance; CEM; First block; act → observe; Weights fixed online. Mathematical variable labels and numerals are not counted as prose. There are no sentence paragraphs, loss formulas, pooling formulas, or GRU descriptions in this main view.

An explicit `z_g` bullseye is placed near the schematic selected terminal latent, using the same identity as the goal-comparison glyph. That illustrative geometry explains selection; it is not an embedding measurement or a claim that the method reaches the physical goal. Candidate-action/branch drawings abbreviate the actual five-transition planning horizon. `First block` means the first five native controls before replanning.

The original observed PNGs remain embedded in the PDF/SVG. TikZ is the canonical editable geometry and typography source; SVG glyphs are outlined, so the export is not advertised as live-text SVG. No generated raster content was needed in the computation panel.

Isolated standalone build uses `paper/build/method_objectflow/`. Final exports are `method.pdf`, `method.svg`, and 300-dpi `method.png`; the exact-width and grayscale proofs are temporary files in that build directory. Main-manuscript integration, caption and full-page review remain with the root agent.

The subsequent connector review in `geometry_repair_review.md` supersedes this first sparse-layout proof. It moves labels away from wires, connects collection buses, uses named endpoints, and directs execution to the highlighted first block. Final standalone dimensions/proof hashes are recorded by the installed skill's inspection helper, under `paper/build/precision_review/`.
