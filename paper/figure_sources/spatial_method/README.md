# Native editable spatial architecture

Open **`spatial-transport-architecture.drawio`** with draw.io Desktop or diagrams.net.
The file contains 447 unique native cells, including 41 editable text labels,
36 connectors and editable patch/matrix primitives. There is no embedded raster.

`brief.md` documents the exact scientific scope, source citations, skill revision,
and symbolic composition-reference provenance. `canonical-graph.json` records
all nodes, required semantic edges, cross-panel aliases and equations.
`validation.json` records the actual checks and their limits.

To regenerate and validate from this repository:

```bash
python paper/figure_sources/spatial_method/build_drawio.py
python paper/figure_sources/spatial_method/validate_drawio.py
```

The generator requires the unchanged reviewed model and manuscript-figure hashes.
The validator uses Python's XML parser plus Pillow for conservative text bounds.

**Native preview export was skipped:** no draw.io CLI was found on this server.
No PNG/SVG/PDF is presented as an export of this XML. The existing reviewed Figure24
PDF/SVG/PNG remain unchanged. Once draw.io CLI is available, the skill's export
command is:

```bash
drawio -x -f png -e -b 10 -o spatial-transport-architecture.drawio.png spatial-transport-architecture.drawio
```

Inspect that actual export for app-specific typography, clipping and connector
routing before substituting it into a manuscript. The current XML/semantic/
geometry checks passed; an actual draw.io GUI/export visual review is outstanding.
