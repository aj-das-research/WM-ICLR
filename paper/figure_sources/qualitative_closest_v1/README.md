# Matched qualitative comparisons: derived numerical sources

`derived.json` is produced only after the independently reviewed replay in
`scripts/qualitative_closest_v1/replay.py` passes its completed-result review.
`manifest.json` binds it to that review and the frozen case/implementation
registration. The renderer is
`paper/scripts/render_qualitative_closest_v1.py`. The main-paper DROID figure and
attached IWS/decoder galleries share this numerical source and geometry code.

The pack contains per-seed and mean feature errors, source-mixing weights,
effective mixture matrices, gate values, correction magnitudes and exact case
identities. It contains no model weights, original feature vectors, commands,
predicted feature vectors or raw image arrays. Image paths and hashes identify
local sources; they do not imply those inputs are redistributed here. A public
checkout retains the exported figures; reproducing their photographs requires
the original dataset access and local replay described in the protocol.

DROID cases preserve the original development selection (largest, median and
smallest episode-level gain), with the same first eligible window and a new
matched no-tanh component replay. IWS cases are explicitly posthoc ranks over
the completed reserved study, with the smallest original registered handle in
each trajectory. No case is reselected after image inspection. These examples
are gain-conditioned illustrations, not a random sample or new independent
performance estimate. AR is a fixed matched baseline, not the strongest of
every DROID control. The actual closest component is displayed alongside it.

`W` in the manuscript is `M16x16` in the data: each seed's effective matrix
`(1-g)I + gT`, averaged only after its computation. Its row at index5 contains
all source weights for the fixed destination patch (row2,column2). Correction
maps average per-seed RMS over384 channels; they are not pooled RMS. No map is
semantic attention, physical correspondence or pixel-level saliency.

## Image provenance

- DROID: Khazatsky et al., *DROID: A Large-Scale In-The-Wild Robot Manipulation
  Dataset*, [dataset project](https://droid-dataset.github.io/),
  [paper](https://arxiv.org/abs/2403.12945), CC BY4.0. Recorded frames are shown
  with labels and, in the decoder figure, an outlined feature-grid coordinate.
  The original field of view is retained. These images are not model outputs.
- Recorded PushT, bimanual Box and Rope: the
  [RLA-WM/IWS dataset](https://huggingface.co/datasets/xyzhang368/RLA-WM).
  Its dataset license is unspecified in the reviewed source metadata. A small
  number of reduced-size (at most256x192 pixels, Lanczos interpolation with
  the full field of view), attributed research excerpts appears embedded in
  the annotated manuscript figures. No standalone genuine frame pack or
  trajectory data is included in this new public source pack or model release.
  The project's source-code license does not relicense upstream imagery.

PDF and SVG preserve editable vector labels and scientific marks, with recorded
frames embedded as raster content. PNG and grayscale exports are previews.
No generated illustration is used as experimental evidence. There is no draw.io
export for these numerical plots.
