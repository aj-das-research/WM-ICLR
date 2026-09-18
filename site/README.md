# ShiftWM project page

A static research page with actual DROID recording playback, an interactive four-population result table, the current paper, architecture and qualitative figures, source links, and dataset attribution. It preserves the existing white / ink / blue / rust visual system and has no framework, analytics, font CDN or credential dependency.

## Refresh and preview on the cluster

```bash
python site/publish.py
python -m http.server 8008 --bind 127.0.0.1 --directory site/export
```

`publish.py` requires the completed 12-run / 48-evaluation result report and the local source assets. It refreshes the current PDF, source-derived results, real video, first-video-frame poster, protocol, interpretation and publisher license. The published video is 24 seconds of actual recorded frames. Playback is deliberately 6 fps, not a claim about calibrated physical time.

`site/export` is generated output. Public deployment uses only the explicit file list in `publish.py`; private paths, source data, weights and server credentials are not included. The source ZIP and old `status.json` from the previous local dashboard are not part of this deployment.

## GitHub Pages

The deployed site is published from the root of the `gh-pages` branch. Repository Pages settings use **Deploy from a branch → gh-pages → / (root)**. That branch contains only the verified static export and `.nojekyll`. Relative assets and fetches work at the project subpath `/WM-ICLR/`.

`deployment/github-pages.yml` preserves an optional GitHub Actions workflow template. It is not activated: the current Git credential has repository access without the separate workflow-management scope. Branch deployment needs no workflow token or secrets in the repository. A synchronization tool may refresh the verified export and fast-forward `gh-pages`; source code remains on `main`.

```bash
python site/publish.py --snapshot-only --output /tmp/shiftwm-pages/WM-ICLR
python -m http.server 8009 --bind 127.0.0.1 --directory /tmp/shiftwm-pages
```

The snapshot-only command needs Python's standard library and committed assets; it does not read cluster datasets or checkpoints. `publication-manifest.json` verifies each scientific asset by SHA256 before deployment. Refresh the snapshot after a paper/figure change, then commit it alongside the code.

## What the demo does

- Plays an actual held-out DROID episode, with two external cameras and a wrist view.
- Provides play / seek / restart / speed controls, a download and a provenance record.
- Switches both camera and horizon to display the corresponding measured MSE, seed variability, paired intervals, gains and regressions.
- Links the current manuscript, detailed results, interpretation, protocol and source repository.

This is browser-side recorded-video playback and result exploration. It does not run online inference, generate future RGB frames or connect to a robot. Real-video checkpoints have been packaged locally; the page deliberately does not invent public download links. Session disjointness is established, but scene/object disjointness is not.

## Browser verification

Verified with Playwright Chromium 145 at 1440×1080 and 390×844 under `/WM-ICLR/`: all four result states, video decoding/playback, speed change, restart, clipboard feedback, disclosure, all local downloads, no horizontal overflow, no console errors or failed requests. Browser plugin was unavailable. Screenshots and the temporary QA runner are stored outside the source repository, under `/tmp/shiftwm-pages-qa` for this session.

The saved Image Gen concepts in `design/` establish the existing visual system. The refreshed page uses actual research artifacts; no generated scientific evidence or imagery was introduced. Older `build.py` and `validate_browser.py` are retained as the previous local dashboard workflow, not the Pages publication path.
