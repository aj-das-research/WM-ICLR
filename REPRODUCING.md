# Reproducing ShiftWM

Run commands from the repository root. Python 3.11, CUDA 12.4-compatible PyTorch
and an allocated GPU are used for model work. Training need not use Slurm;
the included Slurm files record the original cluster and require adapting their
partition, memory and working-directory headers on another cluster.

## Environment

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
bash scripts/setup_environment.sh
source .venv/bin/activate
export PYTHONPATH=src
```

`requirements.lock.txt` and `references/world_upstream_revisions.json` pin the
training dependencies and upstream source. Vendored LeWM modules retain their
MIT notice. Simulator-specific installation instructions are in `environments/`.

## Real DROID videos: download through complete evaluation

The registered study uses 24 preselected official shards: 1,126 real robot
recordings, 851 training, 143 validation and 132 test episodes. This is a fixed
subset study, not a reproduction of the full published DROID policy benchmark.
Reserve at least 60 GB for the 22 GB raw download, extracted camera streams,
features and training artifacts. Keep additional space for checkpoints.

```bash
python scripts/real_video/fetch_droid.py \
  --inventory reports/evidence/real_video/droid_selected_inventory.json \
  --prefix robotics/droid/ --output data/real_video/droid_selected/raw
python scripts/real_video/download_encoder.py

python3.11 -m venv environments/real_video/.venv
environments/real_video/.venv/bin/python -m pip install \
  -r environments/real_video/requirements.lock.txt
environments/real_video/.venv/bin/python scripts/real_video/prepare_droid.py \
  --raw data/real_video/droid_selected/raw \
  --output data/real_video/droid_selected/processed \
  --inventory reports/evidence/real_video/droid_selected_inventory.json \
  --expected-episodes 1126 --workers 4

python -m shiftwm.real_video.features \
  --data data/real_video/droid_selected/processed \
  --output data/features/droid_selected_v1
```

Download verifies pinned object generations, published checksums and local
SHA256 values. Extraction verifies episode boundaries, ordinal frame/action
alignment, camera streams and session-disjoint splits. The frozen DINO encoder
has a pinned Hugging Face revision and per-file SHA256 checksums.

Run each seed in its own GPU allocation, or sequentially on one GPU:

```bash
python scripts/real_video/run_campaign.py --seed 0
python scripts/real_video/run_campaign.py --seed 1
python scripts/real_video/run_campaign.py --seed 2
python scripts/real_video/finalize_campaign.py \
  --campaign configs/real_video/campaign.json
```

Each seed trains four predictors for 30 epochs and evaluates both external
cameras at five- and ten-block horizons. The finalizer requires all 12 complete
training runs and 48 evaluations, recomputes paired uncertainty, writes paper
tables, and verifies portable checkpoint reloads. Models predict frozen visual
features, not RGB pixels. Camera two is a second view of the same held-out
recordings, not a second independent dataset. The clocks in the downloaded
RLDS schema do not support physical latency claims.

The original completed campaign is documented in
[`reports/real_droid_results.md`](reports/real_droid_results.md) and independently
interpreted in [`reports/real_droid_interpretation.md`](reports/real_droid_interpretation.md).
Existing test results must not be reused for tuning a claimed confirmatory study.
Changing code, runtime, data or configuration produces a new scientific identity;
the scripts intentionally reject resuming an incompatible original run.

## Simulation studies and baseline reuse

```bash
python scripts/download_artifacts.py
python scripts/make_training_configs.py
python scripts/make_control_configs.py
```

Follow [`reports/world_execution.md`](reports/world_execution.md) for collection
and feature caching, then use the campaign scripts under `scripts/` and
`configs/world/`. Drone and surgical simulator extensions have separate runtimes,
configuration families and collection scripts under `environments/`,
`configs/extensions/` and `scripts/extensions/`. Completed extension results and
all exploratory failures are summarized in `reports/completed_extension_results.md`.
AdaJEPA's official reproduction uses its own observation/control protocol and
is not a matched comparison with our RGB-only campaigns.

## Models, demo and paper

Trained checkpoints currently remain on the research server. They are not
silently embedded into source Git or advertised as downloadable releases.
The real-video finalizer produces `artifacts/releases/real_droid_v1`; simulator
exports use `artifacts/releases/extensions_v1`. These contain strict offline
loaders, provenance, model cards and the necessary encoder. The upstream
pretrained downloads are available now through the scripts above.

[`demo/README.md`](demo/README.md) describes local inference and bundle creation.
The public project page is in `site/`; recorded demonstrations must be labeled
separately from live model inference. Build the manuscript with:

```bash
bash paper/build.sh
```

A TeX installation with `pdflatex` and `bibtex` is required. The checked-in
generated figures/tables permit compiling the current paper without raw data;
regenerating them requires the corresponding experiment evidence. Some audit
records preserve original absolute paths as historical provenance, not portable
execution requirements.
