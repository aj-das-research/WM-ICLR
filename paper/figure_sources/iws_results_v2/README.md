# IWS forecast display revision 2

This revision fixes adjacent-panel baseline-tick clearance. It reuses the exact
completed public v1 numerical payload and plotting helper; each axis width
changes from 0.257 to 0.247 of the figure canvas. Panel origins, heights, data,
domains, caption, legend, markers and font sizes are unchanged. The figure stays
5.5 × 2.75 inches with an 8 pt minimum. Both measured clearances are 3.128 pt.
The v1 source, exports, scientific registration and previous review remain intact.

The new renderer is `paper/scripts/render_iws_results_v2.py`. Its only inputs
are the immutable `paper/scripts/render_iws_results.py` and completed public
`paper/generated/experiment_alignment/forecast_transfer.json` plus its four
authoritative PDF/SVG/PNG/TeX siblings. These exact v1 hashes are validated before
rendering. No reporter, private cache, checkpoint, raw dataset or model inference
is opened. All 708 displayed values are compared directly with the bound payload.

From a public snapshot with that repository layout:

```sh
python paper/scripts/render_iws_results_v2.py --output-dir /tmp/iws-display-v2
```

To compare a clean replay with the authoritative v2 outputs, also keep the five
`forecast_transfer_v2` exports and run:

```sh
python paper/scripts/render_iws_results_v2.py \
  --output-dir /tmp/iws-display-v2-replay \
  --verify-against paper/generated/experiment_alignment
```

The recorded NumPy/Matplotlib versions and Liberation Sans font bytes are required;
Pillow and Poppler `pdftoppm` support pixel verification. Network access is blocked.
The verification compares complete PNG pixels and first-page PDF pixels at
144 dpi before committing outputs. PDF dates are omitted and SVG IDs use a fixed
salt. Existing outputs are never silently replaced. The source-bound successful
isolated replay is recorded in `replay_validation.json`; it reconstructs the
published display and does not rerun the scientific experiment.

The active include may use `forecast_transfer_v2_figure.tex`; its caption and
figure label are identical to v1, with only the PDF filename changed. Manuscript
integration and publication are handled separately.
