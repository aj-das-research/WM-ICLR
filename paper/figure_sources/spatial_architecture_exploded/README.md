# Editable observation-anchored decoder

Open `spatial-architecture-exploded.drawio` in diagrams.net. Every label, equation,
connector, matrix cell and feature piece is a native editable object; the three
recorded photographs are separately embedded images. Equations retain MathJax
source as editable text. The scene uses no flattened generated diagram.

From the repository root:

```bash
python paper/scripts/render_spatial_architecture_exploded.py
```

The command validates pinned scientific/image inputs and writes this folder's
`geometry.json`, `semantic_contract.json`, native XML and the candidate
`paper/generated/editorial/spatial_architecture_exploded.*` exports. It does not
change the manuscript's canonical figure include or any experiment source.
Required packages are NumPy, Matplotlib and Pillow. Runtime inputs are public
repository paths; no raw dataset or API key is required.

The Python scene is the canonical geometry source. If editing the native diagram
manually, preserve that copy and migrate edits back to the renderer before
regenerating; a rerun otherwise replaces generated XML. PDF/SVG are rendered from
the same scene by Matplotlib. Native application exports are separately identified
in the review evidence and must not be confused with these deterministic exports.

See `brief.md` for source/image attribution, generated-reference limits and the
exact model contract. The recorded images are unchanged; all feature objects,
matrix weights and activation bars are illustrative.
