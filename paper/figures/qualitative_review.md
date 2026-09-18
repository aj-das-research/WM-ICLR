# Qualitative figure review — 2026-09-18

Reviewed the four paired paper exports at 5.5-inch width, enlarged PNGs, the
Reacher control panel in grayscale, the Chrome-rendered local gallery, and
integrated manuscript pages 16–19 (Figures 6–9). Also checked the new pending
rollout table on page 28. Exact reviewed manuscript and renderer hashes are
saved in `paper/build/qualitative_review/review.json`.

Data audit: 42 focused tests pass. All 64 development tasks, 192 saved method
rollouts and 1,565 observed frames validate against the existing checkpoint,
task, support and protocol checks. All paired initial and goal pixels match
exactly. The ledger pins 241 source dependencies. Time labels derive from
executed action lengths, including short final blocks and support-only success.

The independent reviewer inspected all four initial exported PNGs and supplied
case-specific captions from source records. The root review then increased
labels/metric annotations for paper-size readability and increased angle
precision so a small nonzero error is not rounded to an apparent exact zero.
Final exports and integrated pages were reinspected. No visible text overlap,
clipping, broken connectors or image rescaling mismatch was identified. The
figures use no schematic motion arrows: columns show recorded states, with
their own time labels, rather than implying an unmeasured physical path.

Success/failure is expressed in words as well as color, and remains readable
in grayscale. Both favorable and unfavorable examples use identical full-frame
display scales. Repeated early-terminal frames are explicitly explained; no
extra interactions or decoded predictions are implied. The gallery shows every
recorded frame, retaining empty space after termination at a fixed cell scale.

The captions clarify two otherwise misleading interpretations: PushT's green
renderer marker is not the scored episode goal, and Reacher's displayed L2
distance is not the per-joint success threshold. The PushT failure despite
small block error is explained by the recorded pusher-position error.

Compilation completed without overfull boxes, oversized floats, undefined
references or LaTeX errors. The new table preserves all eight pending arms
without numerical placeholders. The gallery was opened in headless Chrome;
its initial counter shows all 64 tasks, and the first expanded case displays
the three matched observed timelines without missing assets.

Scope: selected paper examples are outcome-stratified development illustrations
at one training seed, not an estimate of success frequency, main-test evidence,
proof of context identification, or an effect attributable to the new queued
revision. No new planning measurement was produced by this figure build.
