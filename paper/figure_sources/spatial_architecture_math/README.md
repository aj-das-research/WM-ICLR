# Spatial architecture and mathematical contract

Reproduce with:

```sh
python paper/scripts/render_spatial_architecture_math.py
```

The canonical editable source is the Matplotlib renderer; SVG retains live text
and vectors around original raster observations. Output width is5.5in with8pt
minimum labels. The renderer verifies its source/model/image hashes and does not
load raw datasets or modify scientific sources.

Runtime assets are shared with existing public figure sources:

- `paper/figure_sources/spatial_method/canonical_graph.json` preserves the previous28-dependency graph and frozen source identities.
- `paper/figure_sources/spatial_qualitative/case1_recorded_frame_{0,5,10}.png` are unchanged DROID observations; see that directory's CC BY4.0 attribution.
- `paper/figure_sources/teaser_gallery/asset_manifest.json` records the unchanged IWS internal-training example, source URL, native index and dataset-license status. It is attributed to Zhang etal., RLA-WM, arXiv:2605.07079; no model-repository license is assigned to the dataset.
- `paper/figures/split_assets/reacher.png` is the existing training-frame illustration for the separately trained historical simulation study.
- `paper/scripts/teaser_story_glyphs.py` supplies unchanged symbolic feature objects.

The equation panel belongs to the current bounded spatial decoder. The historical
two-context/CEM algorithm and its paired losses are not used. The task strip
locates distinct studies; it contains no decoder-to-planner dataflow or claim of
one checkpoint across all tasks. All colored feature objects, mixing weights and
gates are schematic. The model predicts visual features, not RGB.
