# IWS forecast-transfer figure: gated design brief

The figure answers: after the complete registered single-observation campaign,
how does error evolve across all59 native future offsets on PushT, Box and Rope,
and what is ShiftWM's signed H60 improvement against the learned additive anchor?
No result exists at design time. This document is not a result figure or visual
approval of a future rendering.

Three compositions considered before implementation:

- **A — three aligned small multiples (selected):** PushT | Box | Rope; four
  distinct method curves in each; a shared external legend; one H60 signed gain
  and its paired95% interval below each panel. Equal widths expose every task
  without ranking by outcome. Separate zero-based y scales retain the native
  metric and are explicitly disclosed.
- **B — three stacked full-width traces:** more room per temporal curve, but
  greater appendix height and harder simultaneous cross-task scanning.
- **C — curves plus a separate endpoint forest:** useful when endpoint evidence
  dominates, but duplicates the endpoint table and reduces space for all59
  measured offsets. No task is dropped to make a headline.

Selected physical target:5.5×2.75in, Liberation Sans regular8pt minimum, dark
slate conventional AR, cobalt additive anchor, amber dotted persistence, green
solid ShiftWM (ours). Markers and dashes preserve method identity in grayscale.
Green denotes the proposed method; gain text uses its actual sign and explicitly
marks intervals that include zero. A negative outcome stays in the same panel.
The primary comparison is the learned additive anchor, not persistence.

Curves use means already reconstructed by the frozen reporting helper: equal
windows within trajectory, then equal trajectories and seeds. X is H=2…60,
corresponding to stored future offsetH−1; these are not physical seconds. There
are no invented curve error bands. Only the existing paired H60 relative-gain
interval is shown; it is a10,000-draw seed×trajectory exploratory95% interval.
The method, aggregation, bootstrap and checkpoint selection are unchanged.

The renderer calls frozen `completed_development(...)` after cache-identity
validation. Missing/incomplete finalizers produce no output files. A complete
render has708 exact plotted means and three primary gain/interval annotations.
The JSON sidecar retains the entire validated numerical payload, all comparator
contrasts, original source SHA ledger and selected checkpoint identities.

## Future review after real completion

The renderer makes PDF/SVG/PNG plus a source-bound sidecar and emits its TeX
include last, after numerical and automatic clipping/legend geometry checks.
It never asserts that future actual pixels have been reviewed. Its status is
`numerically_validated_geometry_checked_visual_review_pending` until a separate
human/agent receipt names the exact exported hashes.

After completion, inspect the actual PDF at5.5in width, grayscale and enlarged
panels; compile the actual manuscript page and neighbors. Check every curve
endpoint against the JSON, all signs/intervals and the correct H60 comparator.
Check for tick/annotation collisions under the realized numeric ranges. Preserve
all four methods if curves coincide or a method regresses; change display only
through a new reviewed rendering revision, never the frozen scientific sources.

Typical read-only proof commands (run only once actual outputs exist):

```sh
pdftoppm -png -r 120 paper/generated/experiment_alignment/forecast_transfer.pdf /tmp/iws-forecast-paper-width
pdftoppm -png -gray -r 120 paper/generated/experiment_alignment/forecast_transfer.pdf /tmp/iws-forecast-gray
pdftoppm -png -r 300 paper/generated/experiment_alignment/forecast_transfer.pdf /tmp/iws-forecast-zoom
```

Use the existing official ICLR manuscript build to verify actual inclusion and
caption, without changing its font size, margins or line spacing. Automated
polling is idempotent: unchanged finalizer/source/runtime fingerprints and
verified export bytes cause no rewrite. A conflicting prior output fails closed.
