# Retained feature-history teaser

Run from the repository root:

```bash
python paper/scripts/render_teaser_visual_design.py
```

The driver emits final `paper/generated/editorial/teaser_visual_design` PDF,
SVG, PNG and native draw.io, with caption/include/evidence JSON. Three design
alternatives and their contact sheet remain under `paper/design` only. Geometry
is shared between the editable native document and publication exports.

The driver imports SHA-pinned existing `render_teaser_sources.py` and its pinned
`render_teaser_benchmarks.py` evidence/geometry helper. These are public source
dependencies, never modified by this figure. Inputs are public RGB PNGs,
manifests and original DROID per-window validation JSON ledgers. No feature-cache
NPZ, raw videos, checkpoints, reserved IWS data, training or inference is used.

The new h=2 illustration is a source-identity example, not a reported h=2 result.
AR's predicted output updates its next whole feature window. ShiftWM retains
its original observed feature history and fixed last-observation decoder source.
AR's initial inferred transition context remains fixed too. The gold ports mean
the same available causal commands, not identical GRU-state slicing. All tile
textures are symbolic; they are neither measured feature values nor RGB output.

The footer remains the complete measured DROID development comparison at h=10:
141 episodes, 136 positive, five negative,5.30% reduction in the ratio of mean
native standardized-feature MSE. This is not an average of episode percentages.
All source-window rows are recomputed at render time. Other gallery entries have
their separate existing scopes and do not inherit the DROID score.

All nine original RGB arrays are preserved. See benchmark_gallery, original
spatial_qualitative and teaser_cinematic source manifests/attribution files for
source selection and licenses. DROID/Open-H:CC BY4.0. IWS dataset terms are not
inferred from its model-code license. The camera is an original vector viewfinder
around the DROID photo; no data pixels are replaced or recolored.

Native draw.io preview is checked separately in the actual application. Primary
PDF/PNG and editable geometry are deterministic; SVG has an export timestamp.
