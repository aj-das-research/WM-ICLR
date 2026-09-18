# ShiftWM project page

A static research page built around interactive PushT / Reacher executions, DROID robot footage, complete forecast comparisons, the current paper, reusable checkpoint downloads and a separately hosted inference app. It extends the established white / ink / blue / rust design system with restrained motion, keyboard-accessible tabs, native video controls and reduced-motion support. There is no framework, analytics, font CDN or credential dependency.

## Refresh and preview

```bash
python site/publish.py
python -m http.server 8008 --bind 127.0.0.1 --directory site/export
```

`publish.py` verifies the completed 12-run / 48-evaluation DROID study and generates the simulation showcase with `prepare_showcase.py`. The latter requires NumPy and Pillow, supplied by the research `.venv` when the launching interpreter does not have them. Refresh copies the latest paper, source-derived results, source footage, protocol, interpretation and publisher license. The original source experiment files remain unchanged.

`site/export` is generated output. Publication uses only the explicit export lists and source-validated showcase media; private paths, source datasets, checkpoint weights and server credentials are excluded. Model downloads use the public `real-droid-v1` GitHub release. Older `build.py`, `status.json` and source-ZIP artifacts belong to a previous local dashboard and are excluded.

## What the demos show

- **PushT and Reacher:** six selected development examples, each with original ShiftWM / Framewise observations, shared goal, actual native-call timeline and measured endpoints. Select an outcome, play the saved action-block observations, scrub the sequence or jump to endpoints. When one method stops early, its last image stays visible with its actual stop call; no intermediate frames are invented.
- **DROID:** recorded three-camera video with play / seek / replay / speed controls, download, dataset attribution and clip provenance. The deterministic example was selected before model-result review. Frames are displayed at 6 fps for viewing, not as a physical-time calibration.
- **Results:** five simulation methods across held-out composition and shift extrapolation for both tasks; six DROID methods across both cameras and horizons. Exact mean MSE, seed standard deviation and available paired confidence intervals accompany gains and regressions. Forecasting and planning are distinct metrics.
- **Inference app:** actual checkpoint inference lives in `demo/live/` and can be run locally. An externally verified HTTPS deployment may be enabled with `demo-config.json`; the page checks `/health` before displaying its iframe. The model predicts visual features, while the image strip displays recorded observations. The download-and-run path remains usable when temporary hosting is unavailable.

The headline gains are explicitly labeled by comparator: Reacher 26.78% and PushT 3.68% versus Framewise calibration on held-out-composition forecasting; DROID 2.94% versus persistence on primary camera-1 horizon-5 forecasting. The 0.20% DROID comparison against Framewise has an interval including zero; all extrapolation results remain visible. A separate additional session-disjoint evaluation is highlighted alongside its own protocol: 0.74% lower error for calibrated ShiftWM versus equally calibrated Framewise on 65 new episodes / 52 sessions, primary 95% difference interval below zero. Its source-derived summary is `fresh-results.json`; the original study tables remain unchanged. Selected planning examples are illustrations, not estimates of success frequency.

## Scientific assets and provenance

`prepare_showcase.py` uses the previously audited qualitative inventory and forecast ledger. It checks NPZ SHA256, decoded pixel SHA256, frame counts/times, and exact equality of the first observation and goal across methods before exporting 103 unaltered lossless PNG files for six examples. The selection includes ShiftWM-only success, Framewise-only success and joint failure in each task. `showcase.json` preserves source IDs, source hashes, exact call times, checkpoint identities and outcome metrics.

`publication-manifest.json` hashes all scientific assets before deployment, including every simulation frame. Snapshot packaging rejects mismatches instead of publishing stale or partially refreshed results.

## GitHub Pages

Pages deploys the root of the `gh-pages` branch. That branch contains only the verified static export and `.nojekyll`; source code remains on `main`. All local links and fetches work beneath `/WM-ICLR/`.

```bash
python site/publish.py --snapshot-only --output /tmp/shiftwm-pages/WM-ICLR
python -m http.server 8009 --bind 127.0.0.1 --directory /tmp/shiftwm-pages
```

The snapshot-only command uses the Python standard library and committed assets; no cluster dataset or checkpoint access is required. `deployment/github-pages.yml` is an optional Actions workflow template. The normal publishing synchronization uses branch deployment and keeps credentials outside Git.

## Design and browser verification

The existing Image Gen concepts in `design/` establish the visual system. `design/benchmark-explorer-concept.png` supplies the new section layout; neutral media placeholders in that design concept are replaced by pixel-verified research images in the implementation. No generated scientific observations or predictions appear on the site.

Browser QA uses the existing Playwright environment (`demo/.venv/bin/python`) because the Browser plugin is unavailable. Desktop and mobile verification covers all benchmark and outcome selectors, playback, timeline scrubbing, endpoint views, eight forecast-result states, DROID playback, copying, keyboard tabs, reduced motion, local downloads and overflow checks. Temporary screenshots and the runner stay outside the repository in `/tmp/shiftwm-interactive-qa` and `/tmp/shiftwm_interactive_browser.py`.
