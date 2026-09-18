# Live checkpoint explorer

The project includes a working CPU inference application in `demo/live/`. Every
click executes the released ShiftWM and matched Framewise predictors on the same
recorded input. The browser shows the three support observations, per-horizon
prediction errors, and spatially pooled feature-error maps alongside the actual
future reference image. A downloadable JSON record includes checkpoint hashes,
selected epochs, input hashes, actual output hashes and execution time.

The initial public deployment is a temporary Cloudflare Quick Tunnel. Its current
URL is recorded in `reports/live_demo_status.json` and the project page checks
`/health` before displaying it. The address can change when the tunnel restarts;
availability is not guaranteed. The repository and released models support
independent hosting. [Cloudflare documents Quick Tunnel limitations](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).

## Run locally

Download and verify the model archive from the
[real-droid-v1 release](https://github.com/aj-das-research/WM-ICLR/releases/tag/real-droid-v1).
Set `SHIFTWM_LIVE_RELEASE` to its extracted base bundle directory: the directory
containing `models/droid_factorized_s0`, `source/src` and `source/scripts`.
On the original cluster this defaults to `artifacts/releases/real_droid_v1`.

```bash
python -m venv /tmp/shiftwm-live-env
/tmp/shiftwm-live-env/bin/pip install -r demo/live/requirements.txt
SHIFTWM_LIVE_RELEASE=/path/to/extracted/base_bundle /tmp/shiftwm-live-env/bin/python demo/live/app.py
```

The application binds to `127.0.0.1:7861`. It needs no GPU, login token, DINO
download, or dataset download for the included examples. Encoder outputs are
the exact original cached features; the forecasting networks run afresh on CPU.
The existing cluster environment can launch it with
`demo/.venv/bin/python demo/live/app.py`.

## Inputs and comparison scope

- Three previously published DROID examples: the largest reduction, median, and
  largest regression by episode-mean, three-seed horizon-five comparison.
- The live calculation uses the first window at one selected training seed,
  camera and forecast horizon. These window scores differ from aggregate paper
  results and do not constitute a new benchmark or model-selection experiment.
- Select camera 1 or camera 2, seed 0–2, and horizon 5 or 10. Horizon 10 tests
  extrapolation beyond the five-step training objective.
- Each checkpoint is the validation-selected epoch after its complete 30-epoch
  training run. Baseline inputs, normalization and selected seed match.
- Models predict latent DINOv2 feature states, not RGB video. Future images enter
  scoring only. The 2×2 error cells correspond to channel-major pooled feature
  positions, not object masks, attention maps or generated images.
- Both favorable and unfavorable outcomes appear. The baseline is the matched
  Framewise model; it is not labeled a reproduced external state of the art.

The 1.99 MB sample bundle contains six numerical NPZ payloads (read with
`allow_pickle=False`) and 78 compressed recorded frames. `samples/manifest.json`
records all hashes, original feature/recording hashes, exact native-frame
indices, original selection-ledger hash and DROID CC BY 4.0 attribution. It
contains no checkpoint weights or fresh-holdout data. Rebuilding the samples
requires the original frozen dataset and `prepare_samples.py`.

## API

`GET /health` returns a small public readiness document, with cross-origin access
allowed so a static project page can detect availability.

`GET /api/catalog` lists the fixed examples and their frame metadata.

`POST /api/forecast` accepts exactly:

```json
{"sample_id":"droid-3ba4bffb7d639b9cbc1bc715","camera":1,"seed":0,"horizon":5}
```

It returns per-horizon standardized MSE, four spatial error cells per horizon,
and computation provenance. All requests are limited to one simultaneous
forecast and thirty forecast attempts per minute globally. Concurrent calls
receive HTTP 429. The server exposes only allowlisted JPEGs and fixed frontend
files. It accepts no uploaded images, arbitrary filesystem paths, user models,
shell commands, or arbitrary input arrays.

The UI supports direct visits and an iframe on
`https://aj-das-research.github.io`. The application has a restrictive content
security policy and does not use third-party JavaScript or browser analytics.

## Cluster services and validation

User systemd units `shiftwm-live-demo.service` and
`shiftwm-live-tunnel.service` manage the application and temporary HTTPS tunnel.
Their configuration and logs live outside the repository. The model process is
limited to two CPU cores and 2 GB memory, and the tunnel to half a CPU and
256 MB. Both use `NoNewPrivileges`, read-only home and filesystem protections.
The tunnel continues running across application restarts.

```bash
systemctl --user status shiftwm-live-demo.service shiftwm-live-tunnel.service
demo/.venv/bin/python -m pytest demo/live/test_demo.py -q
```

Validation checks independent predictor parity, correct spatial feature layout,
future-target exclusion, bundled asset hashes, request bounds, repeat-request
determinism, rate limiting, CORS and content security policy. Browser inspection
covers desktop/mobile layouts, both cameras, seed selection, extrapolation,
the horizon slider and the observed outputs.
