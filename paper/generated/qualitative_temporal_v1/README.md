# Measured temporal qualitative figures

`temporal_outcomes.pdf` (Figure 7) aligns one recorded observation and four
withheld future frames with measured feature-error differences and all 59
forecast offsets. `temporal_decoder.pdf` (Figure 8) follows actual observed-source
mixing weights and whole-grid gate/correction traces on the same three cases.

The cases are the existing middle-ranked reserved IWS trajectories, using their
first registered windows. No new checkpoint selection, training or inference
was performed. The full largest/middle/smallest gallery, including regressions,
remains in the manuscript. The bounded model is primary; no-tanh is an ablation.
These diagnostics describe feature forecasting, not RGB video generation,
object saliency, physical correspondence or a causal explanation by themselves.

## Editable sources and verification

From the project root:

```bash
.venv/bin/python -B paper/scripts/render_qualitative_temporal_v1.py
.venv/bin/python -B paper/scripts/render_temporal_companion_v1.py
PATH="$PWD/.venv/bin:$PATH" bash paper/build.sh
```

The reviewed derived numerical pack is in
`paper/figure_sources/qualitative_closest_v1/`. Original recorded frames remain
local under `artifacts/qualitative/closest_v1/`; their expected file and pixel
hashes are in the pack. Rebuilding the photographic composites requires those
registered local frames. No raw frame archive, commands, features or full
prediction arrays are included in this presentation folder.

Python is the canonical geometry source. PDF and SVG retain editable numerical
marks and labels; the unchanged, attributed photographic excerpts are raster.
PNG previews, manuscript-width proofs and grayscale proofs accompany them.
The diagrams.net workflow is not used for these measured plots.

The static figures use the personal `paper-visual-design` skill, the companion
`paper-figure-creation` evidence/layout/review workflow and the paper's 5.5-inch
width. Sources, scales, frame identities and numerical checks are bound in
`figure_evidence.json`; independent reviews are in
`reports/qualitative_temporal_v1/`. Negative difference-map cells have diagonal
marks so their sign remains visible without color. Every curve retains the
full measured range. Task-specific normalization prevents interpreting shared
display scales as a comparison of task difficulty.

## Animated companion

Four annotated composites show stored offsets 14, 29, 44 and 59. Each holds for
two seconds: the eight-second playback is presentation time, not physical time.
The MP4 duplicates measured composites without temporal interpolation. The GIF
uses one fixed color palette. Neither is model-generated video. The project page
uses native playback controls and supplies both static figure PDFs.

The GIF is a local convenience export; the public snapshot includes its renderer
and annotated PNGs, plus the MP4 under `site/assets/temporal-companion.mp4`.
`companion_evidence.json` records exact image bindings, three-seed scores,
diagnostic arrays, encoding and output hashes.

Attribution: recorded RLA-WM/IWS dataset excerpts, with dataset license
unspecified in the existing source record. Only reduced, intact-field-of-view
research excerpts (at most 256x192 each) are embedded in annotated figures.
No generated illustrations or externally fetched decorative assets are used.
