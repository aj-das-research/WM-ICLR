# Fresh residual-calibration figure: selection and measurement protocol

This is a post-hoc explanatory figure of a completed, separately frozen real-DROID
comparison. It changes no model, scalar, population, primary analysis or result.
The selection JSON was saved before decoding any new illustration or computing
additional qualitative diagnostics. Its source hashes bind the six primary
calibrated result files and complete three-seed episode scores.

## Fixed cases and measurements

- Primary camera 1, horizon five only. For each of all 65 eligible episodes,
  average saved h5 standardized MSE across its three seeds separately for
  calibrated ShiftWM and calibrated Framewise. Gain is Framewise minus ours.
- Sort by descending absolute MSE advantage (signed difference, not relative
  percentage), breaking ties by episode ID. Select ranks 1, 33 and 65: largest
  improvement, median and largest regression. Do not replace a case after
  inspecting images or new model outputs.
- For each case, choose the first eligible existing window. Show its last
  observed support frame and its actual recorded h5 target, preserving complete
  RGB pixels and the publisher's face blurring. No synthesized prediction image
  or reconstructed scene is permitted. Native indices are ordinal, not seconds.
- Recompute predictions for all eligible windows of those same three episodes,
  using unchanged recorded support, commands, checkpoint weights and original
  normalization. Reproduce saved h1/h3/h5 episode means for each of three seeds
  before trusting extra per-step scores. Inference is float32 with autocast and
  TF32 off. CPU replay may differ from the original GPU results by floating-point
  roundoff; require absolute agreement at most 2e-6 or relative at most 1e-5.
- The displayed curves are the first window's three-seed mean errors at every
  predicted step 1 through 5, for original ShiftWM, calibrated ShiftWM and
  calibrated Framewise. Preserve all values, including crossings and regressions.
  State the episode-average selection gain separately. If a selected episode's
  first-window gain has the opposite sign, display that contradiction explicitly.
- Preserve raw replayed feature predictions, actions and targets in a numeric
  artifact; bind model/configuration/payload/source hashes in a ledger. Future
  observations are scoring targets only, never inputs to the prediction call.
- Show the complete 65-episode gain distribution, not only selected cases. The
  population mean and paired session/seed CI must reproduce the frozen report;
  convert its ours-minus-Framewise interval to gain units by reversing/negating
  the bounds. Do not mistake per-episode variation for this population CI.

## Visual and scientific contract

Reader/slot: an ICLR appendix or supporting figure, 5.5-inch manuscript width.
Visual thesis: a training-fitted, fixed contraction of predicted feature
displacement changes matched forecast errors; the held-out effect is small and
varies by recording and window. The operation is a standard calibration control,
not a new architecture, causal attribution, novel method or pixel generator.

Use exact vector geometry for z0 + alpha (z_hat_h - z0), with all rays anchored
at the same last observed feature and the contraction point on the original
forecast ray. Annotate the schematic nature of this feature-space diagram and
the train-only scalar fit. Use recognizable actual recorded scenes for cases;
never invent imagery to suggest improved robot behavior.

Composition sketches considered before rendering:

1. Two full method lanes above a large winning filmstrip. Rejected: duplicates
   unchanged machinery and crowds out the median, worst and full distribution.
2. Three columns, each a case with a local mechanism inset. Rejected: repeating
   the same global operation suggests case-specific fitting and shrinks photos.
3. One shared operation strip; three aligned observed-frame/error rows; a final
   all-episode distribution plus population interval. Selected: preserves one
   fixed global operation and separates illustrative cases from aggregate proof.

The figure uses editable vector labels/arrows/curves over untouched observed RGB.
No external/generated images are needed because actual recorded evidence is the
appropriate pictorial content. Use a shared method palette and redundant line
styles. The caption carries full selection and uncertainty details; on-canvas
labels are short. Inspect intended-width PDF proof, enlarged mechanism/curve
crops and grayscale; independently recalculate plotted numeric arrays. Root
will integrate and review the full manuscript later; this task must not modify
paper sections, main.tex or rebuild the manuscript.
