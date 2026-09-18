# Visual setup redesign: brief, provenance and review

2026-09-18. This replaces the text-heavy main setup figure; the previous source is preserved separately as `factor_split_detailed.tex`. The scope is deliberately narrower: explain appearance versus physical dynamics and the held-out factor crossing. Planner settings, visibility boundaries, success definitions and trajectory splitting belong in the caption/protocol and detailed appendix, not repeated inside this figure.

## Representation contract

**Visual thesis:** the same physical scene can look different, while the same action can produce different physical responses; a held-out combination crosses two factor values that were each observed separately.

Existing PushT and Reacher are shown as two large, actual canonical training frames. The appearance row headers repeat precisely the same PushT frame under the four pointwise RGB-affine transformations used in `src/shiftwm/data.py`. Pixel locations, object identities and poses do not change. The dynamics column headers are original vector **schematics**, never measured responses. A repeated blue action arrow and starting T/pusher are followed by different orange response poses. These lengths/rotations are not measured or ordered physical effects. The implementation changes PushT damping and Reacher arm/finger density; it does not change camera viewpoint, invent physical effects by editing images, or identify those parameters from the sketches.

The crossing map encodes exactly seven training combinations with dots, `(v1,p1)` development with a diamond, `(v2,p2)` held-out composition with a star, and `(v3,p0)`, `(v0,p3)`, `(v3,p3)` extrapolation with triangles. Empty intersections are unevaluated. Dashed boundaries distinguish unseen factor values. Symbols, geometry and a single legend carry the split; color is redundant.

Setup status: adapted crossed-shift evaluation on **existing** environments, sourced from `paper/main.tex` Experimental protocol, `src/shiftwm/data.py` split/appearance definitions and `src/shiftwm/generate.py` dynamics definitions. No numerical result is encoded.

## Three composition sketches

1. **Two aligned image columns for every appearance row.** Four PushT and four Reacher renders directly beside the crossing map. Rejected: fitting eight complete frames and four response headers within three inches makes every task scene small again.
2. **Large task scenes + one matched appearance strip + crossing map — selected.** Two large observed environments establish the tasks; one persistent PushT example teaches the appearance transformation; response schematics teach the separate physical factor above the map. This gives each visual object a clear role and keeps the split exact.
3. **Large before/after analogy above a miniature map.** A wide two-branch physical/visual comparison supplies the focal story, with a small factor lattice below. Rejected: the prespecified held-out composition becomes secondary and requires duplicated labels to recover its exact row/column meaning.

## Observed assets and rebuild

Asset authority is `split_assets/manifest.json`, produced by `split_assets/build_assets.py`. It records source archive hashes, exact frame/episode identity, no-crop status, transformations, quantization and final PNG hashes. No image-generation model was used.

- PushT: frame 0 of `train-s31001-d0`; selected as the first lexicographic canonical-dynamics training episode with its entire T block at least eight pixels inside the image boundary. The preceding episode has a partly off-frame block. No rollout outcome or score enters selection.
- Reacher: frame 0 of the first lexicographic canonical-dynamics training episode, `train-s31000-d0`.
- Each native image is 224 × 224. Large task panels display it at 2.55 × 2.55 cm without cropping. Appearance thumbnails display the same PushT frame at 1.08 × 1.08 cm.
- Transforms call the authoritative `appearance_transform` on float RGB; results are rounded to nearest uint8 only for artifact display. They are paired render examples, not independent physical states.

Rebuild assets from the repository root with `.venv/bin/python paper/figures/split_assets/build_assets.py`. From `paper`, compile `figures/split_standalone.tex` into `build/split_visual_review`. `factor_split.tex` is canonical editable geometry. The PDF/SVG contain observed raster images plus editable-source vector geometry; these are **hybrid exports**, not fully vector evidence images.

## Caption suggestion

**Crossing appearance and dynamics in existing control tasks.** Canonical PushT and Reacher frames illustrate the environments. The row strip applies the exact RGB transformations to one PushT state; response glyphs are schematic, with unmeasured poses that do not rank physical effects. Dots mark seven training pairs, the diamond development `(v1,p1)`, the star held-out composition `(v2,p2)`, and triangles three extrapolation pairs; blank intersections are unevaluated. Both factors of the held-out pair are seen separately in training. Initial-state seeds are split before paired rendering; independent test trajectories cover all nine in-range pairs. The same factor scheme changes PushT damping or Reacher density, although response glyphs depict pushing. The protocol section specifies common model-visible inputs and planning budgets.

## Review record

- Current physical canvas after geometry repair: exactly 396 × 212.598 pt, **5.5 × 2.953 inches**. The redesigned canvas has 19 prose words, excluding factor indices and the numeric training count; no repeated cell labels or planner paragraphs remain.
- The first visual redesign was inspected at 1100 px, 550 px and in grayscale. That review was insufficient: a later fresh inspection found short-arrow and endpoint defects despite the earlier pass. The repair and its focused pixel checks are recorded below.
- Grayscale preserves split membership and the new-factor boundaries. Warm/cool hue differences are necessarily reduced, while the repeated geometry and exposure levels remain visible; the caption specifies the color transformation.
- Standalone LaTeX compiles without warnings, overfull or underfull boxes; all four fonts are embedded. The exported PDF is exactly the stated width.
- The main figure no longer includes the full planner/information-access protocol. This is an intentional scope change, not a claim that visual restyling alone preserved every old on-canvas detail. Those details remain in the existing manuscript and preserved detailed source.
- Independent first-view review by the world-model agent recovered both tasks, unchanged scene geometry under appearance transforms, the fixed blue action versus changed orange responses, all seven train dots, development `(v1,p1)`, held-out `(v2,p2)`, the three extrapolation cells and unseen-factor boundaries. It found no paper-width readability or clipping blocker and judged the object-based explanation substantially clearer than the previous repeated-word matrix. Its caption qualifications about illustrative responses, the shared Reacher factor scheme and independent test trajectories are retained above.
- Current exports: `split.pdf` and `split.svg` preserve vector typography/geometry around the original raster examples; `split.png` is 1650 × 886 pixels at 300 dpi. The final PNG was separately inspected after export.

## Fresh arrow and spacing repair

The user reported broken arrows and spacing after the first redesign. A new inspection of the original exported pixels, without relying on the prior review, identified four concrete issues: default arrowheads consumed much of each short orange connector; the rotated response T in column `p2` intersected its connector endpoint; blue action heads were oversized for their shafts; and the legend's diamond had almost no clearance from the bottom canvas. Task subtitles and the `p3` header were also crowded against nearby image/marker boundaries.

The repair uses explicit 2.4 pt orange response heads and 2.7 pt blue action heads on .60/.65 pt shafts. Response connectors are straight, with visibly exposed shafts; initial and response poses are separated before drawing the connectors. Analytical transformed-vertex bounds give 3.79–5.96 pt horizontal tip-to-response-outline clearance, including outline half-width. These are drawing clearances, not simulated physical measurements. All initial states and blue action arrows remain identical; the orange response poses remain explicitly schematic.

Header/glyph baselines and the two task panels were repositioned to open space around captions and factor labels. An extra .15 cm bottom margin prevents legend tips from touching the crop. The canvas remains below three inches tall. The exact factor membership, all six source image hashes, image display sizes, wording and caption contract were preserved.

Built in the isolated directory `paper/build/split_geometry_repair`, then inspected the actual 1650-pixel rendering, the 550-pixel paper-width proof and grayscale proof. The revised arrowheads have visible shafts and no object intersections; caption and `p3` clearances and the bottom legend margin are visible in the final-size proof. The old PNG is retained for direct comparison at `/tmp/split-before-arrow-repair.png`. No experiment/evaluator source or manuscript build script was changed.
