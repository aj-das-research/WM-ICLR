# Editable observed-memory architecture

Reproduce from the repository root with:

```bash
python paper/scripts/render_architecture_visual_design.py
```

The renderer creates one point-coordinate scene, records it in `geometry.json`, and exports native editable `architecture-visual-design.drawio` plus `paper/generated/editorial/architecture_visual_design.{pdf,svg,png}`. PDF/SVG are deterministic vector-renderer exports, not claimed to be native-app exports; the `.drawio` has separate native-app review. Labels and equations remain editable text, connectors are native edges, and only the four unchanged real photographs are raster. Export timestamps are suppressed.

Scientific model files are read only for identity checks; no training, model execution or raw dataset access is needed. `paper/design` contains generated review outputs and is not a runtime input. Public inputs are the canonical graph, two pinned model source files and four existing DROID PNGs. `semantic_contract.json` explains all aliases and collapsed operations; `brief.md` records representation choices. The generated reference is optional art-direction context and not a runtime dependency.

The source/model and photograph hashes are recorded in the generated evidence sidecar. DROID photographs are CC BY4.0; use the original dataset attribution in `paper/figure_sources/spatial_qualitative/ATTRIBUTION.md`. The depicted case is the previously selected median-by-episode-gain first window; it was not selected again for this drawing. No result numbers or model-predicted RGB are depicted.

This new candidate does not integrate or publish itself. The project owner controls the canonical manuscript include and final compiled-page review.
