# Positive qualitative figures — review record

Reviewed locally on 19 September 2026, Dubai time, using the
`paper-figure-creation` skill. Editable source:
`paper/scripts/render_positive_qualitative.py`. The brief is
`positive_qualitative_brief.md`; the source ledger is
`paper/generated/qualitative/positive_evidence.json`.

## Evidence and scientific scope

- The two figures include all four eligible ShiftWM-success / Framewise-failure
  cases in the saved 64-task, seed-zero development gallery. Shared context
  also fails on all four. This is an explicitly outcome-selected subset.
- The middle frames are actually recorded at native action 20 for both
  methods. Each endpoint keeps its real stopping time: 21 or 23 for ShiftWM,
  50 for Framewise. Rulers include the ten common support actions.
- Goal contours are annotations extracted from observed goal pixels, with
  original image coordinates preserved. They are not predictions, inferred
  physical trajectories, or inputs to evaluation.
- PushT's gray block and blue pusher define the scored goal; the persistent
  green T is excluded from the contour. Physical errors are read from records,
  in native units, rather than estimated from displayed pixels.
- Reacher uses one common square crop per case for all displayed methods,
  times, and goal images. The union of displayed arm pixels has twelve pixels
  of padding. Overview boxes locate each crop. Original RGB values are retained.
- Reacher success is evaluated per joint; the displayed L2 distance of 0.060
  radians can therefore accompany a success. The caption states this explicitly.
- The complete gallery and balanced examples retain failures and counterexamples.
  These panels explain observed behavior; attribution to a model component
  remains unestablished and requires the controlled study.

## Repairs and actual visual inspection

Initial review identified clipped ruler end markers, tick-label registration,
and Reacher details that were too small at paper width. The editable source
now includes ruler headroom and correctly transformed ticks. Reacher details
use the shared padded crops and full-scene overviews described above.

Inspected both final high-resolution PNGs, a 5.5-inch paper-width proof,
grayscale proof, and the actual integrated manuscript pages 16 and 17
(Figures 6 and 7). Proofs are under
`paper/build/positive_qualitative_review/`. The gallery was also inspected
in a 1400-pixel-wide browser rendering. Positive cases appear first, with all
64 tasks still accessible. The final gallery wording separately describes
the balanced selection rule and the exhaustive four-case positive selection.

No material clipping, text overlap, broken connector, contour misregistration,
or ruler misalignment remains in the inspected views. Success/failure words
and different end-marker shapes preserve meaning in grayscale. Large views
make the small Reacher orientation differences visible. PushT retains the full
scene because the baseline's departure from the view is part of the observation.

An independent agent inspected both final PNGs and integrated pages 16/17 and
reported no material issue. Its review confirmed contour registration, padded
crops, real endpoint times, correct captions, and the absence of a causal claim.
The targeted positive-figure checks passed 26 tests, including all three
recorded Reacher crops. Tests supplement, rather than replace, pixel inspection.

Remaining scientific limits: four selected development cases at one training
seed cannot establish an overall planning advantage or explain its mechanism.
The aggregate comparison remains mixed and is reported separately.
