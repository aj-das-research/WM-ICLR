# Gated five-predictor IWS forecast plot

This is a new presentation renderer; existing four-predictor v1/v2 plots and
every scientific registration remain unchanged. Production command:

```bash
python paper/scripts/render_iws_variant_forecasts.py --if-ready
```

Run it after `render_current_real_scorecards.py --if-ready`. It consumes only the
validated portable `current_real_scorecards/{data,manifest}.json` pack and its
reviewed validator. It opens no model, cache, raw data, reserved payload, or
private finalizer. A present invalid pack fails. Until the pack contains the
completed nine-new-plus-27-original study, it returns `pending` and creates no
production directory or files. The preceding live scorecard refresh rejects any
present partial/corrupt finalizer before this renderer is invoked.

Once complete, `paper/generated/iws_variant_forecasts/` receives:

- `forecast_transfer_all_variants.pdf`, `.svg`, `.png`;
- `forecast_transfer_all_variants.json`: all 885 plotted means, provenance,
  geometry checks, runtime/font identity, output hashes;
- `forecast_transfer_all_variants_figure.tex`: caption and the existing
  `fig:iws-forecast-transfer` label.

The caller can select the new include if it exists and retain the current v2
include otherwise. The generated caption references the separately gated
`tab:iws-unbounded-component` table for paired endpoint inference. All five
methods, three tasks and 59 offsets are shown. H2–60 targets stored offsets1–59;
no physical-time claim is implied. Means weight windows within trajectory,
trajectories and three seeds equally. Each task has a zero-based local vertical
scale. No bands or RGB predictions are invented. The unbounded variant is a
distinct post-development ablation, not a replacement checkpoint selected for
each task. Colors, line styles and markers distinguish every method in print.

The figure is 5.5 ×2.4 inches with text at least8pt. Actual-font clipping,
adjacent tick clearance, legend/data/title clearance, zero origins and every
plotted x/y value are checked before export. Identical retries write nothing;
changed or damaged existing outputs fail rather than silently replacing a
reviewed snapshot.

The completed four-method data can exercise the layout without producing a
manuscript figure:

```bash
python paper/scripts/render_iws_variant_forecasts.py --proof-v1 \
  --output-dir paper/design/iws_variant_forecasts/proof_v1_final
```

Those files are named `FOUR_METHOD_LAYOUT_PROOF.*`, marked in the image, and
have no LaTeX include. Proof mode is forbidden under `paper/generated`.
Five-method synthetic fixtures exist only in tests and temporary directories.
Actual five-method pixel review remains pending the completed ablation.
