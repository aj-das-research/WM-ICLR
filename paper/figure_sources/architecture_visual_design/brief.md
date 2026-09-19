# ShiftWM: observed memory to one composed future patch

## Figure question and slot

Given three recorded observations and supplied commands, how does ShiftWM construct a future feature grid while preserving observed evidence? Target:5.5×3.667in in the ICLR single-column manuscript, Liberation Sans8pt minimum with STIX math. This is the current DROID spatial instantiation. No historical context calibration, paired loss, goal image, CEM or simulated domain is imported.

## Composition selection

Three distinct thumbnail layouts are preserved in `paper/design/architecture_visual_design/composition_drafts.{png,pdf}`: persistent-memory spine and patch workbench; aligned processing lanes; one-target-patch horizontal pipeline. The first gives fixed evidence a persistent place, keeps conventional conditioning quiet, and makes the novel mixture/correction operation the largest explanatory region. A new built-in generated reference supplies only the visual hierarchy; all real photographs, math, labels and connections were rebuilt from sources. Its errors and exact prompt are in `reference/generation_manifest.json`.

## Representation contract

A shows actual recorded DROID frames0/5/10 and a frozen DINOv2 encoder. Three symbolic backplanes retain the complete observed support. A learned spatial encoder processes all three grids; E_-2:0 in A/B is the same tensor family, and only its last pre-FiLM E0 supplies the fixed keys. Normalized Z0 is a distinct feature tensor taken before learned spatial encoding. Fixed evidence values do not imply frozen predictor weights.

B retains two separate input routes: past-only patch-mean transition context and the chronological command-prefix GRU. The two gold past blocks enter both; ten supplied future blocks enter only the action GRU, whose prefix throughh determines H_h. All three FiLM-conditioned support encodings stay fixed during this forecast. H_h is the same changing-state alias in C/D.

C follows target patch6 (one-based row2,column2). It displays a complete16×16 schematic row-softmax matrix with destination row6 highlighted. Observed source IDs1,6,11 and their colors persist from A to the displayed pieces. The omitted thirteen contributors remain in the exact sum; the ellipsis and caption prevent a sparsity claim. D reuses sourcepatch6 and exactly the mixed-piece alias m_i. Complementary coefficients multiply separate contributions before summation. LayerNorm+affine projection→tanh is an independent H_h branch, added afterward. The unit bound is in standardized feature coordinates.

Repeat the per-patch computation over all16patches to form the lower lane's forecast grid. Across a clear boundary, actual recorded frame60 enters only the target encoder and feature-MSE comparator. It never enters the predictor. No heatmap, result value or accuracy claim is added. Feature objects, activation bars and weights are schematic, never generated RGB or physical patch motion.

## Semantic and image evidence

`semantic_contract.json` binds all28 scientific dependencies to native edges or explicit faithful aliases/collapsed operators. It pins the exact frozen implementation and coordinate conventions. Four embedded RGB arrays are byte-identical to original decoded PNG pixels after accounting for PDF row-storage orientation; the extracted-array audit is in the proof folder. DROID source images are CC BY4.0 and preserve the prior case/first-window selection.

## Review and scope

Author inspected actual-PDF paper-width color/grayscale, enlarged conditioning/mixing/gate/target crops, and the official ICLR-width proof. Labels have no measured text collisions/clipping. An isolated selected-copy render without paper/design, raw datasets, pretrained weights or model inference reproduces PDF/SVG/PNG/nativeXML/geometry bytes exactly. Native application inspection and final integrated-page review are separately recorded by their actual reviewers; no successful export is treated as visual approval. Existing manuscript includes, old figures, shared glyphs and registered experiment code remain untouched.
