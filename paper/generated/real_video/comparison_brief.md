# Recorded DROID comparison figure brief

This is an appendix qualitative comparison at 5.5-inch manuscript width. The
visual claim is limited: actual held-out recordings accompany matched, measured
feature-forecast errors, including the largest favorable difference, largest
unfavorable difference and a median case. It does not claim decoded video
prediction, physical robot success, or a causal explanation.

The first reading path is full-test context, then three equal-height case rows.
Each row aligns two unaltered camera frames with three measured horizon errors.
Recorded support and target frames show the same episode and earliest evaluated
window. Curves average all evaluated windows in that episode and three training
seeds. No image is a model prediction or localized feature-error explanation.

Composition options considered: (1) three aligned rows of frame pairs and
curves, (2) a large scene with inset method curves, or (3) a grid of outcomes
with one pooled chart. The existing aligned-row composition is retained because
it gives favorable, unfavorable and typical cases equal space, retains the
entire field of view, and makes the episode-to-curve correspondence direct.

Selection is fixed by `reports/real_video_comparison_protocol.md`; adding a
checkpoint-independent persistence curve does not change case identities or
selection. Framewise, Constant dynamics and ShiftWM (ours) use distinct markers
and colors. Persistence uses a dashed line and x markers. Action-free and
constant-feature-velocity controls remain in the full comparison tables; this
figure is the matched framewise/context comparison, not a complete method
ranking. No curve receives a synthetic confidence interval.

Figure sources must pass the finalized report, 12 checkpoints, 48 evaluation
and verified offline-release gate. The renderer checks raw evaluation hashes,
common episode/window identities, source RGB hashes and agreement with the
full-test aggregate. The ledger records the complete case inventory, plotted
values, counts, selected episode identities, RGB files, figure hashes and
selection scope.

The editable source uses exact vector text/axes and original raster camera
images. All labels are at least 8 points at the declared manuscript width.
There are no generated pixels, invented object annotations, inferred action
arrows, crops, or color adjustments. Review includes paper-width and enlarged
PDF rasterizations, grayscale, a compiled figure/caption proof, and source
arithmetic checks. Main-manuscript inclusion is delegated to the root agent.
