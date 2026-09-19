# Compact recorded-forecast comparison

This figure merges the display of two separate **historical context-model** DROID
studies. It does not pool populations, compare calibration as a causal treatment
across those populations, or describe current spatial-model checkpoints.

- Original test: 132 episodes / 58 sessions, uncalibrated;73 positive / 59 negative
  episode gains. Curves average windows within an episode and then three seeds,
  at the saved horizons 1, 3, 5.
- Fresh sessions: 65 episodes / 52 sessions, matched train-fitted calibration;
  46 positive / 19 negative episode gains. Curves are the fixed first window's five
  saved horizon means over three seeds. The median episode improves overall,
  while its displayed window regresses. The calibration formula and complete
  matched outcomes remain in the manuscript.

Every case was already selected in the earlier source figures. The display order
is now largest gain / median / largest regression in each study; no case was
reselected. All 12 full original input/target photographs remain unchanged.
They are native frames 10 and 35 from the first eligible window, not RGB model
predictions. Each forecast block consumes five native recorded commands; no
physical elapsed time is inferred.

Gain means Framewise error minus matched context-variant error, positive favoring
the context variant. Curves, ranked marks, E/W annotations and population
intervals are multiplied by 1000. E is the episode h5 gain; W is the shown first
window h5 gain. Case curves and episode-rank strips share the same symmetric
±15 plotted range. The latter retain every signed gain, ordered largest to
smallest. Case curves are point means, not confidence intervals; lines between
measured horizons guide the eye. Population intervals are the original 95%
paired session/seed bootstrap, with both signs retained. Relative percentages
compare population mean MSE, not the mean of episode percentages.

The data pack preserves all 197 signed episode gains; all original four-method
absolute episode curves (including persistence); the fresh three-variant case
curves (original context, calibrated context, calibrated Framewise); both fresh
original/calibrated four-method population results; calibration scalars; exact
case IDs; and source hashes. Original full figures remain archived unchanged:
`paper/generated/real_video/comparison_recorded_droid.pdf` and
`paper/generated/real_video/fresh_mechanism.pdf`. Full population tables remain
`tab:real-video-primary` and `tab:fresh-real-video-complete`.

Rebuild from repository root:

```sh
python paper/scripts/render_recorded_forecast_compact.py
```

Runtime requirements: Python, NumPy, Pillow, Matplotlib and Liberation Sans.
The renderer reads only itself, `manifest.json`, `data.json`, and the 12 PNGs in
`assets/`. It does not load private design files, checkpoints, raw recordings,
or network resources. The public data was prepared by independently reopening
the completed raw evaluation JSONs and checking the saved fresh replay arrays;
no model inference or additional evaluation was run. The PDF/SVG contain vector
text/axes/marks around unchanged raster photos; editable code is canonical.

Images are from DROID (CC BY 4.0), with publisher blurring and complete field of
view preserved. Original asset paths, byte and pixel hashes are in `data.json`.
