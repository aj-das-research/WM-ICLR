# Method figure redesign brief

Date: 2026-09-18. Scope: `world_method.tex`, `method_standalone.tex`, `method.pdf`, `method.svg`, `method.png`, and `assets/method*`. The root agent owns the manuscript, its caption and full build. This is an implementation diagram, not an empirical result or a claim of benefit.

## Reader and one-sentence story

For a world-model reader at the ICLR manuscript's actual 5.5-inch width: real history first determines an observation correction and a dynamics-conditioned action representation; the planner compares predicted terminal latents with a goal corrected using the same observation context.

The important visual relationship is shared observation calibration for both history and goal. The earlier diagram relegated goal scoring to a small footer. This revision puts its actual connection on the page and prevents the privileged reference targets from appearing to be deployment inputs.

## Three compositions considered

1. **Parallel training / deployment panels.** Repeat the encoder–adapter–predictor chain in two panels and put targets only in training. This would make the access boundary obvious but duplicate conventional machinery and consume space needed for readable goal wiring.

   ```text
   training:    images -> correction -> predictor -> losses <- canonical targets
   deployment: history -> correction -> predictor -> planner <- corrected goal
   ```

2. **A central shared-calibration hub.** Feed support and goal into a central adapter, with context inference above and prediction/planning to the right. This puts the central operation in focus, but the history-to-context-to-history dependency creates crossings or visually ambiguous fan-in at this physical width.

   ```text
                 support -> contexts
                      \      |
   support + goal -> shared adapter -> rollout + goal score -> action
   ```

3. **Selected: three numbered stages with a separate supervision strip.** The first row shows the full support/context chain; the second shows the actual goal and autoregressive rollout branches; the third joins their outputs at the terminal cost and action execution. A compact fourth strip isolates the reference targets.

   ```text
   history -> observation context -> calibrated history -> dynamics context
   goal    -> same calibration        rollout          <- action adapter
                      \                 |                     ^ candidates
                       -----------> terminal goal cost -> execute / replan
   training only: canonical targets --> alignment and next-step losses
   ```

This preserves conventional branch routing, uses physical space for the shared goal correction, and avoids a decorative scene or misleading generated future video.

## Representation contract and implementation sources

- `src/shiftwm/model.py::encode_images`: frozen default visual encoder/projector `E0`, ImageNet normalization and bilinear antialiased resize. Encoding is condensed into the real-input nodes.
- `infer_context`: observation context uses the **mean and variance** of raw support features, not their standard deviation. Corrected chronological support and normalized executed action blocks feed the transition GRU. The GRU's transition object is `(z_t, z_{t+1}-z_t, a_t)`.
- `ResidualFiLM` / `correct_observations`: `Ao` uses a context-conditioned feature-wise scale and shift. The same learned adapter and inferred context apply to history and goal. The scale parameterization and dimensions remain in the method text.
- `predict_features`: candidate native actions are normalized and encoded; `Ad` applies a dynamics-context-dependent feature-wise affine transform to the action embedding.
- `rollout_features`: the upstream LeWM predictor rolls out future latents autoregressively using the observed/predicted temporal window. The two inferred contexts stay fixed within a candidate rollout.
- `goal_embedding`: the actual goal image goes through `E0` and the **same** observation adapter/context. No canonical goal is supplied during planning.
- `src/shiftwm/evaluate.py::LatentGoalCost`: terminal mean squared latent distance, followed by the existing CEM solver, execution of the first action block, and replanning from new real observations. CEM itself is established code, not a proposed novel solver.
- `model.py::forward`: canonical alignment and teacher-forced one-step query prediction use frozen reference targets. Paired context-consistency terms remain in the manuscript rather than adding a dense second training graph. The drawing makes no claim of causal identification or online gradient adaptation.

Solid edges denote forward computation or conditioning. Dashed edges denote training-only targets. Blue is observation calibration and orange is action/dynamics conditioning; names and formulas retain meaning without color. `E0` is labeled frozen. Other weights are trained offline and stay fixed online; this is stated in the proposed caption. The final manuscript uses `o` for images and `x` for simulator state; the figure avoids conflicting `x` image notation.

## Visual objects and asset authority

Four actual PushT frames form a support filmstrip and goal-image example. All are losslessly extracted from `demo/samples/pusht_test-s3031000-d2.npz`, using the first three recorded frames and future frame index 7 of that same episode. The source file contains unmodified canonical-appearance RGB frames. **No color transform, crop, synthetic scene, learned prediction or success-based selection was used.** The displayed goal is an illustrative input object, not a declaration that this exact horizon is the evaluation protocol.

The input images teach what a history and goal are. The diagram's equations expose the mean/variance pooling, feature-wise affine correction, transition/action GRU, recursive latent sequence and cost comparison. They are scientific structure, not measured latent values. The images do not demonstrate learned calibration or establish performance.

`assets/method_assets.json` records source and output SHA-256 hashes, frame indices, dimensions, selection rule and operations. Each PNG is 224×224; support images display at 0.59 cm and the goal at 0.62 cm before manuscript scaling (roughly 900+ effective pixels/inch). TikZ's geometry and typography remain editable. PDF/SVG are **hybrid** exports containing embedded raster observations; SVG glyphs are outlines from `pdftocairo`, not live editable text. TikZ is the canonical editable text source. A rebuild requires no image-generation service or API key.

## Physical design and build

The canvas is approximately 13.7 cm wide. At the manuscript's `resizebox{5.5in}{!}`, body labels are nominally 8.6 pt, secondary labels 8.2 pt, and stage headings 10 pt; the final scaling is slightly upward. Font family is Helvetica for prose and the existing Times-compatible math. The palette preserves the existing manuscript's blue `#0072B2`, orange `#B75B16`, and ink `#243447`.

Standalone build is isolated from the watcher and main manuscript:

```bash
cd paper
mkdir -p build/method_redesign
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=build/method_redesign figures/method_standalone.tex
cp build/method_redesign/method_standalone.pdf figures/method.pdf
pdftocairo -svg figures/method.pdf figures/method.svg
pdftoppm -png -singlefile -r 300 figures/method.pdf figures/method
```

The separate local proof `build/method_redesign/method_actual_size.tex` wraps the same source in an exact 5.5-inch resize. The PDF measures 396.019 PDF points wide (5.5003 inches, the small discrepancy is TeX rounding); it was inspected at 96 dpi and as a 144 dpi grayscale proof, as well as enlarged. Proof files are temporary build products, not manuscript dependencies.

## Proposed caption

Factorized context inference and goal-conditioned planning. Real observation/action history determines observation context `co` and dynamics context `cd`. The same observation adapter calibrates history and goal features; dynamics context modulates action embeddings for autoregressive LeWM rollouts. CEM scores terminal latents against the corrected goal, executes the first action block, then replans from real observations. Encoders are frozen; context/adaptation modules and predictor are trained offline and fixed online. Dashed arrows show privileged canonical training targets; context-consistency losses are described in the text. Thumbnails are unmodified observed PushT frames from one episode, illustrating inputs rather than predictions or success.

## Limits

This diagram depicts the factorized implementation, not every ablation or the post-hoc goal-only intervention. It does not portray canonical images as online inputs, decoder-generated video, tuned physics estimates, learned causal disentanglement, or observed task improvements. Exact action normalization, temporal windows, training losses and experimental configurations belong to the code and method text. The canonical observation examples do not themselves illustrate an appearance shift.
