# Local validation record

Validated on 2026-09-18. This verifies the demonstrator and checkpoint interface; it does not establish aggregate model quality or planning success.

- `validation/inference.json`: four real CPU inference cases, two environments × canonical/cool appearance, executed outside the project working directory using the bundled loader. Checked sample hashes, eight actual RGB frames, five finite latent forecast rows, and nonzero recorded-versus-zero action response.
- `validation/trained_release_inference.json`: real five-step forecast from the completed 30-epoch PushT factorized run, using its validation-selected epoch-four package. Checkpoint and sample hashes are retained.
- `validation/browser.json`, `desktop.png`, `mobile.png`: Firefox through official Mozilla GeckoDriver; actual Gradio button click returned real numerical results. Desktop viewport was 1366 pixels wide. Firefox enforced a 500-pixel minimum narrow viewport despite a 390-pixel request; the narrow screenshot is therefore 500 pixels, not a claimed 390-pixel mobile test. No page-level horizontal overflow occurred. The numerical table scrolls within its own region.
- `validation/api.json`: environment-switch callback and a Reacher dim-appearance forecast through the running Gradio HTTP API; checkpoint weights and configuration download endpoints return HTTP200 and the expected byte counts. UI table values are rounded to five decimal places; downloaded JSON retains full precision.
- Plots were visually inspected and labels repaired before delivery. The browser shows saved frames as observations, not generated video; every curve comes from an executed checkpoint.

The first browser launch preceded app readiness; it was rerun after confirming the app served HTTP200. A cluster-wide `/tmp/gradio` permission conflict was fixed by setting the app cache to `demo/outputs/gradio_cache`. Subsequent browser checks passed.

The isolated `demo/.venv` adds Gradio and UI dependencies while reading the project's existing model libraries. Training packages were not upgraded. This is a verified local runtime and a portable source/model bundle; a fresh hosted Space build has not been executed or published.

Reproduce from the project root:

```bash
demo/.venv/bin/python demo/app.py
# In another terminal:
demo/.venv/bin/python demo/validate_inference.py
demo/.venv/bin/python demo/validate_api.py
demo/.venv/bin/python demo/validate_browser.py
```

The browser validator additionally requires system Firefox and the official GeckoDriver executable at `demo/tools/geckodriver`; that testing-only tool is excluded from a Space bundle. Server PID is in `demo/app.pid`; the local default address is `http://127.0.0.1:7860`, with public sharing disabled.
