# Spatial source retention and the historical simulator study

Rebuild the figure with `python paper/scripts/render_decoder_evidence.py` from
the repository root. It uses only this folder's hash-bound JSON/PNG pack, NumPy,
Pillow and Matplotlib; no GPU, raw data, checkpoint, network or private design
folder is needed. Output PDF/SVG/PNG and an evidence receipt are generated under
`paper/generated/editorial/decoder_evidence.*`.

## Two explicitly different studies

Panel a uses the current spatial model's existing DROID development case and
fixed IWS training illustrations. Every cell is a diagonal element of T or
M, meaning the weight on that target patch's own observed source. The maps are
row-major 4×4 grids of these sixteen elements, NOT a row of the 16×16 matrix,
optical flow, object motion, an RGB prediction, or a forecast-error map. The same
linear 0–1 palette applies to all eight maps. Means use the actual seed population,
with M formed per patch and seed before averaging. The gate identity increases
own-source weight by construction; it does not show improved accuracy. These
weights exclude the bounded additive correction.

DROID retains the existing median-episode first-window case, forecast step 10,
seeds 0/1/2. IWS uses episode 000011/frame 0, stored offset 59: PushT seeds 0/1, Box
0/1/2 and Rope0/2. The two unavailable seeds are not imputed. The original full
horizon curves, checkpoint identities, data scope, and raw-array hashes remain
in `mixing.json`. New endpoint diagonals match the original curve endpoints
exactly. Full private T/g/M arrays back this small derivative pack; they are not
required by the public renderer. No new model inference was needed for this
redesign, and the complete IWS campaign gate remains unchanged.

Panel b is the separate HISTORICAL CONTEXT model. `simulation.json` retains the
complete matched fixed-reference MSE@5 inputs, including all underlying seed
values and population records. PushT/Reacher show held-out-composition and
three-condition extrapolation-aggregate results. Drone/tissue show both
Transformer and GRU on their prespecified development population. Neither
architecture is chosen by its result. All eight reported comparisons, including
negative results, remain visible. These are not evidence that spatial-model
checkpoints transfer to the simulators and are not planning-success rates.

Each numeric label and dark marker is 100×(1−mean(MSEours)/mean(MSEFramewise)).
Pale dots show the three per-seed ratios. Every small stem has the same −10…30%
axis extent and a zero tick. Positive means lower error. No confidence interval,
significance, cross-domain pooled average, or common-checkpoint claim is made.
Source-coordinate audits verified fixed encoder/projector identity across all 36
selected comparison checkpoints; exact matched raw window records reproduce
the aggregates. Simulator images are training task illustrations, not the scored
forecast windows. Both inputs and outcomes retain their distinct source scopes.

## Assets and editable boundaries

All eight input PNGs are unchanged full images. DROID attribution is CC BY 4.0,
with original provenance in `paper/figure_sources/spatial_qualitative/`.
IWS images are the already attributed RLA-WM/IWS training inputs documented in
`paper/figure_sources/teaser_gallery/asset_manifest.json`; an upstream blanket
dataset license is not specified, and the code license is not applied to data.
Simulator selections and rendered assets are documented in
`paper/figures/split_assets/manifest.json` and
`paper/figure_sources/benchmark_gallery/asset_manifest.json`.

The renderer is the canonical geometry source. PDF/SVG contain editable vector
maps, plots and typography around original raster photos. No generated reference
image enters the figure. A quantitative figure does not require a native draw.io
conversion; no such export is claimed. The previous gate-trajectory figure
remains archived at `mixing_multibenchmark.*` with its independent source pack.
