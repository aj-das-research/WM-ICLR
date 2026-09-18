# Structural visual redesign — 18 September 2026

The final connector/asset pass is recorded in `geometry_repair_review.md`. It found and repaired label-masked wires, feature-label crossings, ambiguous execution endpoints, terminal-marker overpainting, and a downsampled raster layer. That targeted review supersedes the initial sparse-layout geometry assessment below.

The previous revision was legible but still depended on text inside boxes. User feedback rejected that aesthetic. Its historical review is preserved in `visual_refresh_review_text_heavy.md`; earlier acceptance does not negate the feedback. This revision changes the visual representation itself.

## Current figures

| Figure | What now carries the explanation | Canonical source |
| --- | --- | --- |
| Introduction | Generated pushing/camera illustration with exact vector leaders; four source-derived forecast comparisons | `../scripts/render_teaser.py`, `teaser_brief.md`, `assets/concept_asset.json` |
| Method | Larger observed frames, raw/corrected feature strips, shared calibration rail, candidate latent branches, visible goal, and executed-block feedback | `world_method.tex`, `method_objectflow_brief.md`, `assets/method_assets.json` |
| Setup | Large real task scenes, same-state RGB variants, same-action response schematics, symbol-based crossing map | `factor_split.tex`, `split_visual_brief.md`, `split_assets/manifest.json` |
| Forecast/training/calibration | Existing source-validated quantitative plots with exact labels and uncertainty conventions | `../scripts/render_forecast.py`, `render_training.py`, `render_goal_calibration.py` |

Method and setup contain roughly two dozen and 19 prose words respectively, excluding mathematical indices; neither contains paragraph modules or repeated prose in cells. Important method frames are 1.1cm wide and setup task images 2.55cm wide. Implementation details remain in the manuscript/captions. The old detailed diagram sources are archived, not inserted as additional main-paper figures.

All figures have PDF/SVG/PNG exports. TikZ is the editable geometry/text master for method/setup; exported SVG glyphs are outlined. The teaser retains editable SVG text/marks and reproducible Python geometry. These are hybrid figures: raster assets are embedded separately from vector diagrams/plots, not entirely vector drawings.

## Evidence and semantics

- Generated concept art is explicitly illustrative, not a simulator frame, measured result, predicted video, or camera-viewpoint experiment. Its prompt, original file, dimensions, tool information, and checksum are retained. The unchanged source is composited over white; rebuilds never invoke image generation.
- Independent recomputation from 24 completed raw forecast files recovered teaser changes −3.6762%, −4.2022%, −26.7835%, and +4.1518%. These are ratios of three-training-seed mean MSE@5, with no estimated uncertainty for the ratios. Unfavorable Reacher extrapolation is visible. The full forecast figure retains all aligned comparators and raw errors with seed variability. No planning-success claim follows from forecasts.
- Shared observation context calibrates real history and available goal. Dynamics context uses corrected real transitions and executed actions, then conditions action embeddings. Contexts stay fixed during candidate search. Branches are schematic latent geometry, not measured spatial paths. The first block means five native controls followed by real observation and replanning. Offline targets/losses remain in the text.
- Setup uses exact implemented RGB transformations on one observed state. Response glyphs have no measured or monotonic physical ranking. Damping applies to PushT; density to Reacher. Seven training cells, development/held-out cells, three extrapolation cells, and unseen-factor boundaries match the protocol. Independent trajectories cover all nine in-range test pairs.
- No raw experiment, checkpoint or quantitative outcome was changed to improve appearance.

## Actual visual review

Root inspected enlarged pixels, paper-width proofs, and integrated manuscript pages containing the introduction, method, setup, full forecast and calibration figures. Agents independently checked method/setup at final size and in grayscale. The method reviewer recovered computation from the objects without paragraph parsing; the setup reviewer recovered task identity, same-state appearance changes, fixed action versus response, and split membership. The independent teaser reviewer checked actual 550px pixels and raw numerical provenance.

The first method proof lacked a goal beside the selected terminal state and labeled execution as one action. Both were repaired: a blue goal marker makes selection visible and the label is now “First block.” Scalable Computer Modern support removed font-substitution warnings. Hyperlink borders were hidden to remove colored boxes from text. The official ICLR style file was not edited.

The integrated draft compiles successfully with no overfull boxes, undefined references, or font-substitution warnings. Underfull warnings remain in ordinary draft paragraphs/page spacing; inspected figure pages have no observed clipping or label overlap. The current PDF is 15 pages, with new figures on pages 2, 4, and 6, and the full forecast figure on page 9. This is a visual-development check, not submission-readiness certification.

## Skill and refresh

The installed skill explicitly rejects paragraph-box redesigns, asks for a context-appropriate short-label budget, prioritizes recognizable scientific objects, and requires a label-hidden comprehension check. Frontmatter validation, local links and exact application of the portable upstream patch passed. Hashes are in `../evidence/figure_skill_provenance.json`.

The paper watcher fingerprints saved visual assets and renders the teaser after validated forecast evidence. GPU evaluations were not interrupted by figure work. All changes remain local; nothing was pushed or published.
