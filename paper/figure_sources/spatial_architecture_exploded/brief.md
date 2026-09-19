# Expanded spatial decoder: figure brief and representation contract

The reader should see one forecast backbone, then understand the two consequential
operations inside its decoder: reusing fixed observed feature pieces through a
learned spatial mixture, and adding an independently predicted bounded correction.

The final candidate is **5.5 × 3.097 inches**, with regular **8 pt** body labels and
8.5–8.6 pt panel labels. The user's final landscape preference superseded an earlier
request for mandatory zoom leaders. Matching B/C names now establish the expansion
without extra leader lines. Three structurally distinct composition sketches are
retained in `paper/design/spatial_architecture_exploded/composition_drafts.*`:
overview with two enlarged regions; radial decoder workbench; aligned operator
rails. The selected first grammar was compressed into a narrow A backbone, large
side-by-side B/C regions and a one-line D contract.

## Sources and authority

The generated composition reference was actually inspected:
`paper/figure_sources/visual_story_references_v1/architecture_composition_reference.png`,
SHA256 `d435eb3f6b803c751abecda051d4d1276e83f5dc64d283ac03c902a5e2e5d9dc`.
Its useful grammar was the overview/expanded-module hierarchy, retained patch
identities and shallow feature depth. Its generated road images and incorrect
wiring were rejected. No generated pixel is in the final architecture.

The three unchanged photographs are recorded DROID frames 0, 5 and 10, with
attribution in `paper/figure_sources/spatial_qualitative/ATTRIBUTION.md`. They are
the only raster content; all text, feature pieces, matrix cells, gates, equations
and connectors are deterministic editable vector objects. Feature colors, matrix
weights, activations and depth are schematic, not model predictions or measured
coefficients. Depth denotes channels qualitatively, not channel count.

The implemented model is pinned to
`src/shiftwm/real_video_spatial/model.py`, SHA256
`054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2`.
`semantic_contract.json` maps all 28 original dependencies to explicit visual
edges or precise collapsed causal-module internals. The original graph's old
left-multiplication shorthand is not adopted: the figure uses functional
row-wise projections Q(H), K(E), G(H).

## Reading path and invariants

* **A:** observed RGB passes through frozen DINOv2, fixed training normalization
  and the trainable causal predictor. The past-context port sees two past command
  blocks; the separate GRU sees the chronological past and supplied query prefix.
  Fixed Z0/E0 are aliases for observed evidence, not future inputs.
* **B:** Q(Hh) and K(E0) define scaled compatibility with +4I outside the scale;
  row-softmax forms the actual-shape 16×16 schematic matrix. One row is emphasized.
  Its weights govern a visible weighted-sum junction over observed source pieces.
  Three representative pieces do not imply sparse mixing. Z0 is the normalized
  observation; E0 is the last spatial encoding before FiLM.
* **C:** the affine sigmoid is per target patch. Complementary coefficients
  multiply the direct and mixed sources before their sum. A separate same-Hh
  branch passes through LayerNorm/affine projection and tanh, then adds after
  blending. There is no blend-to-innovation edge.
* **D:** W is row-convex and the channel envelope is in training-standardized
  coordinates. It is a representational bound, not a guarantee about prediction
  error, physical motion, stability or downstream planning.

Only DINOv2 weights are frozen. The current diagram contains no historical CEM,
paired-context loss, future RGB input, query-feedback recurrence, physical warp or
RGB generator. Fine-grained temporal/context definitions are retained in the
caption and ledger rather than squeezed into the conventional backbone.

## Authoring and checks

Applied the installed `paper-figure-creation` skill (method, art direction,
geometry/inspection and hybrid authoring references) and
`codex-paper-figure-skill` reference-to-native-XML workflow. The latter is pinned
to revision `5538ad98a8724ecb4ed70102514cd9e51c00c0ee` of
`https://github.com/pengqianhan/codex-paper-figure-skill`.

The renderer generates one point-coordinate geometry ledger, native draw.io XML,
and PDF/SVG/PNG. The PDF is a deterministic-geometry Matplotlib render, **not** an
unperformed draw.io CLI export. Parent-agent native-app browser export and review
are recorded separately; inspect their exact source hash when comparing exports.
Automatic checks cover unique XML cells, edge geometry, all 28 dependencies,
minimum physical font size, text intersections and clipping. Actual PDF color,
grayscale, enlarged B/C crops and the ICLR-width proof are reviewed separately.
All three embedded RGB arrays are recovered exactly from the PDF, allowing only
the PDF storage-row orientation reversal.
