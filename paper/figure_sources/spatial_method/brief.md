# Editable spatial architecture counterpart

This is a native draw.io companion to the reviewed spatial method architecture
(current Figure 24). It does not replace the manuscript PDF, SVG, PNG, equations,
training pipeline or measurements. Its scientific scope is the normalized
`transport` branch of `_predict_normalized`, with one fixed support context.
It is distinct from the original two-context ShiftWM architecture.

The message is: three observed patch grids and a causal action prefix produce a
horizon state; a fixed last-observation anchor is mixed with row-stochastic latent
weights, blended by a per-patch gate, and corrected by a bounded innovation.

## Scientific graph and source contract

1. The three **observed**, training-standardized 4×4 feature grids feed the shared
   spatial encoder. Its output `E_-2:0` is explicitly before context FiLM.
2. Patch means of those encoded observations, together with the two normalized
   past action blocks, feed `TransitionContext`. This context is support-only
   and stays fixed for every query horizon. It conditions the observed encoded
   support through FiLM.
3. A separate unidirectional prefix GRU encodes the two past action blocks followed
   by the given causal query-action prefix. The temporal predictor uses its two
   past states and current horizon state, together with the FiLM-conditioned
   observed support. This produces one horizon state `H_h` per spatial patch.
   The action-prefix GRU does **not** feed the FiLM context network.
4. The normalized last observed grid `Z_0` is the fixed anchor. Keys use `E_0`,
   the last observed grid's spatial encoding **before** FiLM; queries use `H_h`.
   Both query and key projections are bias-free. The mixing matrix is
   `T_h = softmax_row(Q_h K_0^T / sqrt(96) + 4 I)`.
5. The learned gate is `g_h = sigmoid(W_g H_h + b_g)`; `b_g` starts at −3. The
   diagram explicitly multiplies the identity branch by `1-g_h` and the mixed
   branch by `g_h`, then sums the two contributions.
6. `R_h` is LayerNorm followed by an affine projection, including bias.
   `Delta_h = tanh(R_h)` is bounded componentwise by 1 for this frozen transport
   arm. The exact normalized forecast is
   `Zhat_h = (1-g_h)*Z_0 + g_h*(T_h @ Z_0) + Delta_h`.

Frozen image encoding and final shared-channel unstandardization are outside this
normalized-predictor panel scope, just as in the reviewed Figure 24. No generated
RGB image, pixel warp, optical flow, selected robot action, performance improvement
or causal attribution is claimed by the diagram.

Cross-panel repeated `H_h`, `E_0`, and `Z_0` objects are declared aliases in
`canonical-graph.json`, not independent inputs. The validator checks their exact
provenance and the incoming source set for every operation. Node labels, grouped
patch cells, matrix cells and all 36 connectors remain native editable objects.

Source references:

- [Frozen spatial model](../../../src/shiftwm/real_video_spatial/model.py), SHA256
  `054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2`.
  `_predict_normalized` supplies all depicted inputs, branches, heads and merges.
- [Support context and ResidualFiLM](../../../src/shiftwm/model.py), definitions
  of `TransitionContext` and `ResidualFiLM`; exact file hash is in the graph ledger.
- [Reviewed manuscript figure](../../generated/real_video/spatial_method.pdf),
  SHA256 `a877aafc7dd3aab6a3f9640158a7ebfa389f5970ebeb845aba0f963725bae190`.
- [Existing vector renderer](../../scripts/render_spatial_method.py), which
  establishes the anchor/mixing/innovation palette and the symbolic grid grammar.

## Composition and native representation

Two portrait panels separate causal state construction from the normalized
spatial decoder. A broad upper identity bypass stays outside the gate module;
short orthogonal branches distinguish the two weighted contributions and the
innovation addition. The canonical computation is expanded compared with the
compressed reviewed PDF so the context inputs and gate complement are editable
and explicit.

All colored patch cells and matrix shades are symbolic objects. For continuity
with the reviewed schematic, the illustrative matrix uses
`0.50 I + 0.35 roll(I,1) + 0.15 ones(16,16)/16`, never checkpoint weights.
The mixed colors follow that illustrative convex combination; output colors use
an illustrative blend and bounded color perturbation. These colors are not
384-channel measurements or generated image pixels. No external icons, logos,
stock photographs or experimental frames are embedded.

The canvas is 1520×1540 draw.io units. Native export and final application font
layout have not been visually verified because no draw.io CLI is installed.
Conservative text-fit and connector geometry checks are recorded, but they do
not replace inspection of an actual draw.io export.

## Skill and composition reference

Applied [Codex Paper Figure Skill by Pengqian Han](https://github.com/pengqianhan/codex-paper-figure-skill/blob/5538ad98a8724ecb4ed70102514cd9e51c00c0ee/codex-paper-figure-skill/SKILL.md),
revision `5538ad98a8724ecb4ed70102514cd9e51c00c0ee`; installed SKILL.md SHA256
`e7cefd7bae7151952a19fc3296ad845e70012e4caee83c7961c7944a64ee9df6`.
The existing evidence-grounded paper-figure skill remains installed separately.

The skill's image-generation composition stage produced the local symbolic study
`artifacts/qualitative/spatial_method_editable_reference/composition-reference.png`,
SHA256 `53de4f7b176639cb56e71a6ea7367fba8a7896414d43a6030bac9886d40b3d65`.
That raster was actually inspected. Its two-panel hierarchy and identity-bypass
idea were usable, but its dark background and simplified GRU-to-FiLM/context
connections were rejected. The final native XML uses the source-grounded graph
above, not generated text or inferred raster equations. The composition reference
is not included in the manuscript or presented as a measured result.
