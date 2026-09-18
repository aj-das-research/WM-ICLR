---
title: What Changed? World Model Explorer
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
suggested_hardware: cpu-basic
pinned: false
short_description: Inspect real action-conditioned latent forecasts and reusable checkpoints
---

# World Model Explorer

A runnable local/Space bundle for the compact world-model project. It executes real weights on held-out simulator clips and reports measured latent prediction errors. It does not fabricate future videos or benchmark results. Publication has not been performed.

## Run in this cluster

The isolated demo environment reuses the project's installed model dependencies without changing the training environment:

```bash
demo/.venv/bin/python demo/app.py
```

The app binds to `127.0.0.1:7860`, with public Gradio sharing disabled. Reach it through your existing remote port-forwarding workflow. Set `PORT` for another local port. CPU is the default; the demo does not request cluster GPUs.

The catalog prefers hash-verified `artifacts/releases/*`, then bundled packages, and discovers `runs/world/*/best` only when `training_summary.json` verifies the whole training run completed. Refresh/restart the app to discover newly completed runs. Incomplete fine-tuning runs are excluded. The original absolute-control PushT corpus is rejected.

## Standalone or Hugging Face Space

From a clean Python3.11 environment:

```bash
python -m pip install -r requirements.txt
GRADIO_SERVER_NAME=0.0.0.0 python app.py
```

The README front matter is the Space manifest. A portable checkout needs `app.py`, `backend.py`, `MODEL_CARD.md`, `requirements.txt`, `runtime/`, `samples/`, `checkpoints/`, and `bundle_manifest.json`. The bundle contains real frozen weights and any verified seed-zero factorized release available at preparation time. Do not copy `.venv`, outputs, logs, or browser caches into a Space. Uploading/publishing is a separate action and has not been performed here. On Spaces, set `GRADIO_SERVER_NAME=0.0.0.0` in the Space environment.

## Rebuild data/model bundle

```bash
.venv/bin/python demo/prepare_bundle.py
```

This copies the portable source loader, three small test clips, available frozen model exports, and completed seed-zero factorized releases into `demo/`. It verifies original episode checksums, retains manifest hashes and selection rules, and rejects the obsolete PushT action interface. It does not run training. Canonical future frames are used only for offline scoring.

## What the app demonstrates

Choose environment, eligible checkpoint, held-out clip, and appearance. The support gallery contains actual input frames; the endpoint and future galleries are actual recorded observations. The button runs two action-conditioned latent rollouts: recorded future actions and all-zero future actions. Downloadable JSON includes hashes, action inputs, measured errors, and scope. Downloadable weights and `config.json` form the actual selected checkpoint; keep both files together for the bundled loader.

Zero-action mismatch against a recorded-action path is a diagnostic, not counterfactual ground truth. Success/failure on these three clips is not a benchmark conclusion. See `MODEL_CARD.md` and `VALIDATION.md` for limitations and local verification.

Deployment metadata follows the [official Hugging Face Space configuration](https://huggingface.co/docs/hub/spaces-config-reference).
