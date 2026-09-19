# Historical composition protocol

Rebuild from the repository root:

```sh
python paper/scripts/render_composition_landscape.py
```

Dependencies: Python, NumPy, Pillow, Matplotlib, and Liberation Sans. Runtime
inputs are this renderer, `protocol.json`, `manifest.json`, and six PNG files in
`assets/`. No raw trajectory, model checkpoint, simulator, network, or private
design directory is needed. The renderer checks every input hash and the exact
split populations before exporting a 5.5 × 1.73684 inch (19:6) figure.

The PNGs are full, unchanged recorded simulator frames and their prescribed RGB
variants. `protocol.json` retains exact original archive/asset-generator hashes,
selection rules, physical parameter values, and scientific source identities.
The appearance variants change RGB at fixed pixel locations; they are not new
viewpoints. Reacher changes arm and finger density together; parameter IDs do
not imply an increasing physical ladder. No response trajectory or performance
value is drawn.

Seven pairs are used for training and ordinary validation, (v1,p1) for
development, and (v2,p2) for the held-out composition. Independent test seeds
cover all nine in-range pairs; their aggregate is not wholly unseen composition.
Three extrapolation pairs use the test trajectory family. Blank cells are
unevaluated. Factor IDs are not deployed-model inputs. This is the separate
historical context-model study, not a claim about current spatial checkpoints.

PDF/SVG contain editable vector typography and split geometry around unchanged
raster images. `composition_landscape.drawio` is generated from the same geometry;
native application rendering is a separate review, not implied by XML validity.
The previous tall figure remains archived without changes.
