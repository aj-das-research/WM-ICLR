# Compact IWS figure and paired table

From the repository root:

```bash
.venv/bin/python paper/scripts/render_iws_compact_evidence.py
.venv/bin/python paper/scripts/render_iws_compact_evidence.py --output-dir /tmp/iws-compact-replay
.venv/bin/python -m pytest -q paper/tests/test_iws_compact_evidence.py
```

This portable pack contains completed aggregate results only. Rendering does
not load models, video, reserved data or checkpoints. The script is the canonical
geometry source for PDF, editable SVG, PNG and the paired-contrast TeX table.
Native draw.io is not provided for this quantitative plot.

Source mapping:

- All five methods' 59-offset MSE means and populations: `iws.tasks` in
  `paper/figure_sources/current_real_scorecards/data.json` (885 values).
- Original bounded-vs-additive and bounded-vs-AR paired intervals:
  `bound_payload.full_validated_development.bootstrap.task_intervals` in
  `paper/generated/experiment_alignment/forecast_transfer.json`.
- Follow-up no-tanh-vs-bounded and no-tanh-vs-AR point effects and intervals:
  `results.task_results.<task>.standardized_mse.h60_comparisons` in
  `reports/real_video_iws_unbounded/development_finalization.json`.
- Original macro and interval: `macro_primary_relative_gain_percent` and
  `bootstrap.macro_gain_interval` in the original portable figure payload.
- Follow-up macro and interval: `results.macro_h60.standardized_mse.bounded_spatial_mix`.

The manifest binds all three source files and the exact packed data. Endpoint
differences and relative gains are independently checked against the plotted
means. Every original/follow-up contrast keeps its identity. Standardized MSE
is measured in task-specific training coordinates; similar axis ranges do not
make the tasks interchangeable. The unchanged absolute scorecard retains all
four metrics. Original displays remain archived for reproduction.
