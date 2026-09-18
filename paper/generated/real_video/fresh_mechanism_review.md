# Fresh mechanism figure review

The exact PDF/SVG/PNG snapshot is pinned in `fresh_mechanism_review.json`.
The standalone PDF was rasterized and inspected at the intended 5.5-inch
manuscript width, including enlarged mechanism and median-case crops and grayscale.

Repairs addressed a long title collision, a clipped distribution y-label,
a floating schematic arrow and a median-case note crowding the next row.
The resulting layout audit reports zero text/image/canvas collisions. The arrow
is continuous from the base forecast square to the contracted forecast diamond.

All 65 episode gains were recalculated independently from the six saved primary
result files; all displayed first-window errors were recalculated in NumPy float64
from stored predictions and targets. Six image exports are pixel-identical to
their registered original frames. CPU diagnostic replay matched saved GPU episode
scores within 1.4249235374963476e-8. The median episode's contradictory first
window and the largest regression are preserved. No case/window was replaced.

The figure has editable vector labels, curves and geometry over six observed
RGB assets. Attribution and CC-BY-4.0 source hashes are in
`fresh_mechanism_asset_provenance.json`. The numeric predictions are latent
features; no image is presented as a generated future.

The caption and `fresh_mechanism_figure.tex` are supplied for later integration.
No main manuscript file was edited and no paper build was run. A different
reader's comprehension review and inspection of the integrated manuscript page
and caption remain pending; this record does not claim final submission readiness.
