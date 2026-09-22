# Historical physical-outcome completion

The same model can change success and final physical error in different directions.
This is a secondary analysis of saved, completed historical experiments; it is
not a new run and not a control result for the current spatial decoder.

## Three compositions and selected artifact

- **A, selected:** four task columns; final-error empirical CDF above success
  reached by native call. Fixed Framewise and ShiftWM comparisons; separate
  Transformer/GRU line styles on extensions. This preserves units, physical
  tails and the actual paid interaction budget in one 5.5 × 3.3 inch figure.
- **B:** paired error-gain CDF above the same executed-success curves. Useful
  for matched differences, but the positive/negative convention takes more
  explanation and absolute physical scale is less immediate.
- **C:** two-by-two task cards, each juxtaposing final errors and success.
  Preserves the scientific quantities but repeats axis prose and consumes more
  height. All three are rendered as real PDF/SVG/PNG, not wireframes.

Exact signed gains with paired intervals are tabulated, not converted into
unmeasured control-benefit claims. All 578 condition/metric/comparator contrasts
remain in CSV/JSON, including unfavorable ones. Complete method distributions
are in a nine-page supplement. No backbone or physical unit is pooled.

## Inputs and aggregation

The extractor reads only finalized JSON and saved planning JSON: 64 core
evaluation files (32 packages × test/extrapolation), 36 original extension runs,
six wide-gain development runs, and four untrained extension reference policies.
There are 24,944 task–training-seed–condition rows. Finalized-report bindings
are checked before extracting measurements; exact source hashes are archived.

Core test contains all nine in-range appearance/dynamics conditions, each with
64 shared physical initial-state seeds. Extrapolation contains the separate
three designated conditions. All six methods remain in tables and source data;
the frozen reference has one checkpoint, five trained arms have three seeds.
The figure's comparison is fixed matched Framewise, not an outcome-selected
runner-up. Extensions retain original versus wide-gain studies and both
backbones separately, on the same eight development tasks.

Core intervals preserve the established fixed-training-seed, 20,000-draw paired
initial-state bootstrap (RNG1701). Extension intervals preserve crossed matched
training/task resampling, 5,000 draws (RNG417). The new continuous-error intervals
are post-hoc, unadjusted descriptions. The validator matches all 21 existing
comparable success intervals and 62,224 physical scalar values directly to their
source evidence. Ten synthetic gate tests cover incomplete grids, changed
success criteria, paired identities, early failure censoring and distinct seed
uncertainty conventions.

## Interpretation and missing measurements

Success means the recorded valid criterion was reached at any native call;
final physical error is measured at termination/budget and can disagree with
success. Every native command, including ten context/support commands, counts
toward the total. Tables report actual attempted calls, never disguise early
failure as fast success. The source additionally provides success-only call
distributions and an explicitly budget-penalized command score that assigns
the full budget to failures. Success curves retain all failures in the
denominator and are not Kaplan–Meier estimates. Gray bands mark paid support;
15/576 PushT test and 5/192 extrapolation conditions per checkpoint are already
solved during support. Reacher and these extensions have no support successes.

PushT positional units are native simulator pixels; its complete criterion also
uses angle. Reacher wrapped joint L2 is not its native per-joint success test.
Drone distance, speed and altitude criteria retain millimetre/mm-per-second
units; tissue reports target-point error in mm and validity/instability flags.
Failure flags may overlap. There is no clinical claim.

Unmeasured here: current spatial-model closed-loop outcomes, real-robot success,
IoU, decoded RGB/video quality, candidate-plan outcome rankings and dense core
physical state paths. No missing quantity is filled from feature MSE.

## Reproduction and integration

```
.venv/bin/python scripts/metrics_completion_v1/physical_extract.py
.venv/bin/python -m pytest -q scripts/metrics_completion_v1/test_physical_extract.py
.venv/bin/python scripts/metrics_completion_v1/physical_validate.py
.venv/bin/python paper/scripts/render_metric_completion_physical.py
```

All work is CPU JSON arithmetic, with no model loading, simulator execution or
GPU use. Extractor, tests and validator use the `physical_*` namespace only.
Main output directory: `paper/generated/metric_completion_physical/`.
Use `figure.tex` plus the four `*_outcomes.tex` tables if desired; manuscript
integration remains the root agent's responsibility. The full CSVs and source
JSON preserve every condition and complementary error. No old figure/table or
registered source has been changed. Minimum chart text is 8 pt; tables are 9 pt
without resizebox. Color/grayscale and standalone 5.5-inch table proofs are
archived under this directory's `proof/` folder.

Author QA is distinguished from independent review; no unavailable independent
review is claimed. Root receives all sources and actual pixels for review.
