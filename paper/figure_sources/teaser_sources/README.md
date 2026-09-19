# Forecast-source teaser

Regenerate the reviewed candidate from the repository root:

```bash
python paper/scripts/render_teaser_sources.py
```

This writes `paper/generated/editorial/teaser_sources.{pdf,svg,png,drawio}`,
the caption/include/evidence JSON, and this directory's editable primitive
snapshot. `--drafts` additionally recreates the three composition alternatives
under `paper/design/teaser_sources/`; alternative C is deliberately retained as
a rejected sketch because its shared return path could conflate the methods.

The public driver reuses the SHA-pinned primitive and evidence helper in
`render_teaser_benchmarks.py`. It needs only public PNGs, their manifests and the
public validation JSON ledgers. It does not read raw video, feature-cache NPZ,
checkpoints, optimizer states, or any reserved IWS payload. NumPy, Pillow and
Matplotlib are required. No model inference or new image selection occurs.

The rendered comparison consists of two disconnected computations. The AR
control owns its predictor, output and rolling-history feedback. ShiftWM keeps
its observed reference fixed, applies its gated row-convex mixture, and adds a
projected tanh-bounded innovation. The colored palette/weights and update curve
are conceptual, not measured features, correspondence, or error. Blue and gold
ports abbreviate conditioning; Figure2 supplies the full backbone. The camera
is a disclosed generated illustration and does not cover any recorded pixels.

The top row preserves all nine source examples and separates their status:
historical simulated tasks use different context models; IWS examples come from
internal training and carry no completed score; Open-H is a physical-phantom
input audit only; DROID supplies the completed spatial-development evidence.
Image attribution and exact selection rules remain in the adjacent benchmark,
teaser_cinematic and original spatial_qualitative manifests and attribution
files. DROID/Open-H are CC BY4.0; the IWS dataset's license must not be inferred
from a separately licensed model repository.

All 141 DROID episode marks, including five regressions, remain. The 5.30% is the
relative reduction of aggregate native h10 standardized-feature MSE, not the
mean of per-episode percentages. Raw window rows are re-aggregated every render.
The figure supplies no all-task win, clinical, closed-loop deployment or SOTA
claim.

The native draw.io document and PDF/SVG share the exact primitive geometry.
Native-app preview is a separate review step; successful XML export alone is
not evidence of an application rendering. SVG export timestamps may differ;
PDF/PNG/draw.io geometry and pixels are otherwise reproducible.
