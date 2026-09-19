# Decoder mechanism illustrations

Reproduce the figure from the repository root:

```bash
python paper/scripts/render_mixing_multibenchmark.py
```

The portable runtime reads only `manifest.json`, `metrics.json` and the four PNGs in this folder. It requires Matplotlib, NumPy and Pillow. It does not open videos, checkpoints, private NPZ arrays, cluster directories or `paper/design`, and runs without a GPU. PDF/SVG/PNG and a source-bound evidence JSON are written to `paper/generated/editorial/mixing_multibenchmark.*`. Scientific-source hashes in the JSON are provenance, not runtime file dependencies.

## What is measured

For each seed and forecast offset, the recorded decoder matrices define raw own-location weight as the mean of16 diagonal entries of T, effective retention as the mean of16 diagonal entries of M, and gate as the mean of16 gate values. The gate is applied within each seed and patch before any averaging: M=(1−g)I+gT. Float64 reductions of the original saved FP32 arrays are retained per seed. The figure shows means across the explicitly available seeds; endpoint whiskers and time-curve bands are seed min–max ranges, not confidence intervals. Every plotted scale covers0–1.

The identity M[j,j]−T[j,j]=(1−g[j])(1−T[j,j])≥0 makes the increase in own-location retention a construction property. These decoder descriptors do not establish accuracy improvement, physical correspondence, causal benefit or test generalization. No new model accuracy/error/gain is computed or plotted by this diagnostic. Historical DROID case-selection gains remain archived as provenance.

## Deliberately different scopes

- **DROID:** existing median episode-gain development case, first eligible window; three observed frames and inferred context. The displayed image is its last observation, native frame10. Seeds0/1/2; steps1–10, with five native command rows per step.
- **IWS PushT:** separately trained single-observation adapter, fixed previously disclosed internal-training episode000011/frame0; available seeds0/1, with seed2 absent from this snapshot.
- **IWS Box:** the same fixed input-selection policy, available seeds0/1/2.
- **IWS Rope:** the same fixed input-selection policy, available seeds0/2, with seed1 absent.

Each included IWS checkpoint passed its full30-epoch completion and identity checks. IWS curves cover stored offsets1–59, using the supplied native command rows0:60; there is no fabricated offset0 gate or weight. These indices are not matched physical durations across domains. This is neither a matched held-out comparison nor a common checkpoint across tasks. It does not bypass the separate27-run aggregate-results completion gate.

## Provenance and images

`metrics.json` includes all per-seed curves, endpoint identities, checkpoint hashes/selected epochs, source replay/NPZ hashes, the original three DROID diagnostic cases, and the private reconstruction receipt. The full raw T/g/M replays remain the backing evidence; they are unnecessary for rendering and are not redistributed by this figure bundle. The public renderer validates seed populations, array dimensions/ranges, source hashes and exact decoded RGB hashes.

All four photographs are unchanged full input frames. DROID is CC BY4.0; its attribution and case-selection details are in `paper/figure_sources/spatial_qualitative/ATTRIBUTION.md` and `replay.json`. IWS is attributed to Zhang et al., RLA-WM/IWS, using the already included internal-training illustrations recorded by `paper/figure_sources/teaser_gallery/asset_manifest.json`. The upstream dataset card has no explicit blanket redistribution license; the model-code license is not applied to the dataset here. No generated photos, crops, recoloring, future RGB targets or model-generated RGB are included.
