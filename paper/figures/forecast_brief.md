# Forecast comparison brief

Reader/slot: ICLR-style research reader, quantitative results figure at the actual
5.5-inch manuscript width. This is a measured experimental plot, not an
architecture, teaser, environment illustration, or generated image.

Visual thesis: Given the same observed support and recorded future actions,
separate observation/dynamics contexts improve several held-out latent forecast
comparisons, while unseen appearance extrapolation exposes clear limitations.
The figure must preserve favorable and unfavorable comparisons, including the
frozen baseline's lower Reacher extrapolation error.

## Composition decision

Three sketches considered before drawing:

1. **Selected: four aligned point-range panels.** Rows are environments; columns
   are held-out composition and extrapolation. Every panel retains the same five
   method rows. Horizontal position encodes error; horizontal ranges show seed
   standard deviation. This preserves exact raw metrics and includes every aligned
   control without relying on an arbitrary normalization.
2. **Connected held-out/extrapolation slope plots.** Each method would have a
   left/right endpoint per environment. Rejected because connections could suggest
   a continuous shift-strength trajectory; extrapolation averages three distinct
   conditions and is not the next value of one scalar independent variable.
3. **Method-versus-framewise ratio scatter.** A unity line would make gain/loss
   visible across environments. Rejected because normalized ratios hide the very
   different absolute error magnitudes and require an additional uncertainty
   calculation not supplied by the current primary table.

## Representation contract

- Evidence objects: five completed model variants, two simulator environments,
  and two explicitly defined evaluation conditions.
- Encoding: distinct marker shapes and method-row positions supplement color;
  paired factorization uses orange `#B75B16`, framewise green `#009E73`, shared
  context blue `#0072B2`, unpaired factorization purple `#CC79A7`, frozen gray.
- Focal relationship: corresponding method rows across held-out and extrapolation
  columns. The paired row has the same pale emphasis in all four panels, including
  unfavorable ones; emphasis identifies the proposed method, not significance.
- Reading path: top-left PushT held-out → top-right PushT extrapolation → bottom
  row Reacher. All axes explicitly label the logarithmic scale and lower direction.
- Invariants: exact matched trajectory/window populations, frozen visual reference
  coordinates, action units, forecasting horizon, and completed validation-best
  checkpoint selection for trained methods.
- Detail boundary: the figure excludes planning outcomes, hardware/timing claims,
  and mechanistic explanations. The plain unaligned diagnostic remains in the
  primary table; caption explicitly discloses its lower PushT held-out forecast
  error. Component-wise extrapolation means remain in the JSON ledger.

## Evidence and uncertainty

Canonical editable source: `paper/scripts/render_forecast.py`. The output JSON
contains the exact 52 source-file hashes, source locations, per-seed values, 20
plotted means, sample standard deviations, comparison protocol, and caption.
Held-out values must also agree with `paper/generated/primary_results.json`.
Every run is checked against the actual checkpoint, action statistics, dataset
manifest, evaluator version, exact expected windows and row-derived means.
Trained checkpoints must come from completed training and the validation-best
epoch. No evaluation is launched and no result artifact is changed.

MSE@5 is terminal mean squared error in 192 fixed-reference latent dimensions
after five autoregressive transitions, using recorded future actions. Each
transition is five native controls; the three-observation support contains ten
native controls. This is not a planning success metric or a pixel reconstruction
score. Per condition there are 64 initial-state seeds and 768 overlapping windows.
The held-out appearance2/dynamics2 combination is excluded from train/validation.
Extrapolation averages the three prespecified combinations `(3,0)`, `(0,3)`,
`(3,3)` with matched equal window counts.

Points show mean across training seeds 0, 1 and 2; ranges show sample standard
deviation with `ddof=1`. Frozen LeWM has one checkpoint and no estimated seed
spread. These are not confidence intervals. The three-seed comparison does not
establish statistical significance or broad task/model generalization.

## Physical styling and review record

The vector canvas is exactly 5.5 × 3.8 inches, normal/tick/support text 8 pt,
panel titles 9 pt, white background, editable SVG text and embedded PDF fonts.
Panel-specific log-axis limits are declared in the caption and crop checks reject
any mean or uncertainty endpoint outside the plotted range. Five distinct marker
shapes and consistent method rows preserve meaning in grayscale.

Review completed 2026-09-18 for the current standalone exports:

| Gate | Evidence |
| --- | --- |
| Traceable results | All 52 completed forecast files validated against actual checkpoint/data/evaluator hashes and exact expected trajectory-window populations; all 20 plotted means/SDs recomputed, held-out aggregates checked against the primary ledger. |
| Guard behavior | A real valid record passed; five independently mutated records were rejected for changed summary, duplicate window, changed stride, changed checkpoint hash, and interrupted status. |
| Honest geometry | Log scales declared on each axis and in the caption; fixed limits checked against every mean ± SD endpoint. Frozen has no fabricated error bar. |
| Physical export | PDF is 396 × 273.6 pt, exactly 5.5 × 3.8 in; PNG is 1650 × 1140 px at 300 dpi. SVG text stays editable; PDF embeds TrueType text. |
| Pixel inspection | Inspected enlarged 300-dpi PNG and a 550-pixel-wide rasterization of the PDF (5.5 in at 100 dpi), then the same grayscale proof. Labels and method identities remained readable; no crop or collision remains. |
| Independent comprehension | `world_data` reviewed the 550-pixel proof without the brief: identified paired factorization's lowest plotted MSE in three panels and all adapted methods' worse Reacher extrapolation versus frozen; correctly identified SD over training seeds and no frozen seed spread. Reported no clipping/legibility defect. |

Observed initial defects were repaired once: the long lower-right panel title
extended toward the page edge and the two-line note collided with lower x-axis
labels. The final version abbreviates panel titles to “extrap.” and reserves more
bottom margin. The enlarged, paper-width and grayscale proofs were inspected
again after that repair.

Small training-seed SDs can be shorter than the plotted marker diameter. The exact
values remain in the JSON rather than being enlarged for appearance. Grayscale
review is not a claim of comprehensive color-vision accessibility testing.
Final manuscript integration is owned by the parent agent and requires a separate
check after insertion; standalone exports do not establish final-page layout.
