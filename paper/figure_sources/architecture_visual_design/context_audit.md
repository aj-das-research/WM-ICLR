# Context audit for the next current-method architecture

Independent read-only science and actual-pixel audit. Prepared UTC 2026-09-19T13:33:11.486271+00:00. Applies the personal `paper-visual-design` skill and the evidence/access/semantic-trace requirements of `paper-figure-creation`. No renderer, manuscript, model, data or registered experiment changed.

## Which figures the numbers currently identify

The reviewed manuscript has **70 pages**. Figure numbers are publication-state dependent; use the identities below, not a remembered number.

| Identity | Current location and actual content | Relevance to the new Figure 2 |
| --- | --- | --- |
| Current Figure 2 | Page 3, expanded spatial decoder: causal observed/action backbone; fixed-source patch mixture; complementary gate and separate projected tanh correction; row-convex output envelope. | The correct current method. Most algebra is already present; preserve it rather than adding historical equations. |
| Current Figure 6 | Page 16, “Recorded inputs, predicted features, withheld evaluation.” Three DROID observations, supplied actions, frozen encoder and feature prediction; a visibly separate evaluator encodes frame 60 and computes patch MSE. | Best source of useful missing information: explicit input/output and target-access boundary. |
| Current Figure 13 | Page 34, “Composition protocol.” Historical simulated PushT/Reacher appearance × dynamics train/development/held-out/extrapolation grid. | Historical dataset design, not a current spatial module or DROID split. Keep in its appendix; it does not belong inside the current forward graph. |
| Earlier architecture called Figure 13 | Surviving `paper/figures/world_method.tex` and `method.pdf/png`; visual panels A shared calibration, B dynamics inference, C paired context supervision, plus calibrated goals and CEM. | Different original context model (Ours-1). Its readable separation of inputs/learned operations/supervision is reusable as design grammar; its modules and losses are not current-method operations. |

Actual pixels inspected: full current PDF pages 3, 16 and 34 rendered at 120–135 dpi, plus the full historical `method.png`. Temporary page renders are in `/tmp/shiftwm-architecture-context-audit/`. Captions, current method and historical appendix were read alongside the implementation. Earlier-number provenance is also recorded in the prior mathematical audit; the current figure identification above was independently extracted from this PDF.

## Strongest legitimate integration

Add a compact **prediction versus target-scoring boundary** to the current architecture, using the same real DROID example already present in the current Figure 6:

1. Three observed RGB frames **0, 5, 10** → frozen DINOv2 → pooled **4×4×384** features and fixed training-channel standardization. Explicit supplied actions comprise **2 past + 10 future blocks**; each block is five native recorded commands, not five seconds. An output at horizon h uses only its action prefix.
2. Preserve the current causal conditioning: spatial encoding → patch-mean observed transitions plus past actions → **one** context → FiLM. A separate chronological command-prefix GRU and the temporal predictor produce H_h. The context is fixed from observed support; the future-command prefix changes with h.
3. Preserve the decisive decoder paths: normalized last-observed Z_0 feeds direct and mixed branches; last-observed spatial encoding E_0 **before FiLM** supplies keys; H_h supplies queries, the per-patch affine gate and the independent LN/affine/tanh correction. The correction is added **after** complementary blending.
4. Label the output **predicted feature grid**, not predicted RGB or actions. At h=10 it corresponds to source frame 60.
5. Across a clear **evaluation-only** boundary, withheld frame 60 → the **same frozen encoder and same fixed normalization** → target feature grid. Prediction and target meet only at feature MSE. No target arrow reaches context, action GRU, spatial encoder or decoder inputs. If displaying the offline loss instead, label that branch **training targets / training loss**, since training images are legitimate targets but still never forward conditioning inputs.

The full Figure 6 heatmap need not be copied into the architecture: a target feature object and one MSE join explain the task without repeating the qualitative results. If the measured heatmap is retained, preserve its exact case, three-seed aggregation, linear 0–0.787 scale, standardized h10 coordinate meaning, mean 0.1727 and **−0.305% shown-window gain**. It is the prespecified median-by-episode-gain example's first window, not an example selected for success. No synthetic image may replace its target or error map.

## What the historical figure must not introduce

- No separate observation context c_o and dynamics context c_d in the current graph. `SpatialWorldModel` has a single support transition context. Reusing `TransitionContext` code does not make the current model a two-factor decomposition.
- No shared goal/support calibration A_o, canonical paired render target, appearance/dynamics pairing labels, alignment loss, L_o or L_d. Current spatial `forward()` uses MSE on all ten normalized future feature grids; no paired-consistency objective exists there.
- No goal image, candidate-action optimizer, CEM/MPC, action execution, replanning or online gradient loop. Current DROID forecasting consumes recorded/supplied commands. Historical closed-loop planning belongs to the separate context-model study.
- No 7-pair appearance×dynamics split, held-out composition star, damping/density factor or historical planning gain attached to DROID spatial prediction. The current study is original-validation development evidence.
- Beware notation collision: historical E_0 denotes a frozen visual encoder. In the current decoder E_0 denotes a **tensor** from the last observed trainable spatial encoding before FiLM. Its values stay fixed within a forecast; spatial-encoder weights are not frozen during training.

## Compact visual contract

Use one broad observed-to-feature-prediction path and a visibly secondary target-scoring branch, with the mixer/gate/correction occupying the explanatory focus. Reuse the same example and source identities across panels. The two past action blocks should visibly join context and chronological action processing; future blocks must not enter the past-only context. Aliases are acceptable when explicitly defined, but omitted paths must be named in the caption rather than invented.

Keep standardization visible once or explicit in the caption. The envelope is min/max observed standardized channel values ±1, **not** an error, raw-coordinate, stability or safety bound. Keep the row-softmax, identity bias outside the sqrt(96) division, destination-wise gate broadcast across384channels and separate projected innovation accurate. Frozen encoder versus learned-offline/fixed-at-evaluation modules must remain distinct.

A useful extra training equation is the existing all-ten-step standardized feature MSE, not the historical multi-loss objective. Do not label its window-weighted checkpoint selector equal-episode: reported comparison aggregates windows within episodes and then episodes/seeds equally. Different metric and aggregation roles should not become one unlabeled “loss” arrow.

No new experiments or results are needed. Dataset breadth belongs in the teaser/gallery; a current decoder diagram must not suggest one checkpoint has demonstrated success on historical simulators, pending IWS training and the Open-H input audit.

## Source bindings

These hashes describe the reviewed state, not an approval of an as-yet unrendered replacement.

- `paper/world_model_draft.pdf` — `1786ceb95a7126db3b5c47c9cb354d59034c75c94cdd0f846de21342bf1b8351`
- `paper/sections/main_method.tex` — `b89d802cdf0d82a269b56d75ced3ba6efcc6ada473f0c231105dccc5003a386b`
- `paper/sections/appendix/original_context_study.tex` — `c19b05fa35183732334d7dc39663c4a12351a9942456772dc67d021500529e33`
- `paper/generated/editorial/spatial_architecture_figure.tex` — `1d460be7c575101b4b7e2360a745399d87c2197233897950f8fa7c8315568923`
- `paper/generated/editorial/task_story_caption.tex` — `8d887670fa55fae88e6f261deda1e9dff1fe5056979f03bfff0dd505cb7aa35f`
- `paper/generated/editorial/task_story.pdf` — `0dec654179d938561764a7bc88da9b57ed1e31c3623068e8065746fec063c03e`
- `paper/figures/factor_split.tex` — `679c960c2fb6387850be1e54f3798568486fcde5227174a0f085f2d1b0f4be40`
- `paper/figures/world_method.tex` — `1c03c1c5208cba0ed1c6592352ef8219080a9e02f7e2f67e4aa2a915eb56e8ec`
- `paper/figures/method.pdf` — `7a0c990713ae697916e6bafbef207fe226674e148c36bed2b3ebc07bece6f7b3`
- `paper/figures/method.png` — `593eb6df2fd818597c74c6a9901987c97686c70a4a49dfba3fb3388c48a7c7e7`
- `src/shiftwm/real_video_spatial/model.py` — `054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2`
- `src/shiftwm/real_video_spatial/data.py` — `5a502e9acea1ba0da4d5953198185a29a79854df7f7232faf8e0b212646103d4`
- `scripts/real_video_spatial/evaluate.py` — `dad9bd2df511c87fc2dfc7dbbd48d413327885a111cfc8211c8d7169571cdfbb`
- `src/shiftwm/model.py` — `2f783caf8eed699a8d024f689aa956efd126a2c8b6c67aea23e4ff5dfb8536f8`
- `paper/design/spatial_architecture_math/math_review.md` — `32c83b6cd0a40177e18be874776d3f868f3365d0ae6a2534266b52332ebeffbc`
