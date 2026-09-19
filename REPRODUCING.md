# Reproducing ShiftWM

The current main-paper method is the spatial `transport` predictor (historical
Ours-5), labeled **ShiftWM (ours)**. Other spatial configurations are component
ablations. The earlier two-context predictor and its simulation/original-DROID
campaigns remain separate historical studies; their commands and artifact IDs
below are preserved. See [paper organization](paper/README.md) for the mapping.

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

The [real-DROID release](https://github.com/aj-das-research/WM-ICLR/releases/tag/real-droid-v1)
contains all 12 validation-selected predictors, the shared DINO encoder and
12 training-fitted calibration wrappers. Download the archive, `SHA256SUMS`
and the release README from that page, verify the checksum, and follow its
offline loading example. The 245,700,575-byte archive SHA256 is
`f8f02886b9de56aa697509889fc50a26904a72f86a328e932690d612b4c4a946`.
Each original and calibrated model passed exact offline CPU prediction parity
after relocation. Checkpoints are release assets rather than ordinary Git files.

The [generalization development release](https://github.com/aj-das-research/WM-ICLR/releases/tag/generalization-v1)
contains a further 36 predictors: three training interventions, four matched
methods, and three seeds, each trained for all 30 epochs. All 72 validation
evaluations and all 36 isolated CPU reloads passed verification. The
443,302,514-byte archive SHA256 is
`8ee06a9ea5ea4b10d6c485a173ef2a6921fda350e57656295d3224f212e747e2`.
Use its own README/loading example and manifest; these development checkpoints
do not replace the frozen original or fresh-session test results.

The [simulation development release](https://github.com/aj-das-research/WM-ICLR/releases/tag/simulator-development-v1)
contains 42 predictors: 36 original drone/surgical-proxy extension models and six
drone geometry revisions. Each completed 30 epochs; every selected model passed
exact offline source-versus-release cached-feature and image-input prediction
checks. Its 1,321,091,128-byte archive SHA256 is
`528724d917442e6d8e5f02607c65de8aa6ac2665f16e3017701fd73f5a4dce5b`.
The package stores shared tensors once and reconstructs complete model states;
use its own loader and manifest. All six public assets passed anonymous checksum
verification. These are simulator development models with mixed outcomes.

The [ten-step development release](https://github.com/aj-das-research/WM-ICLR/releases/tag/horizon10-development-v1)
adds all 12 h10-trained predictors, the shared encoder, loader, source and all
60 evaluation ledgers / 20 comparisons. Each completed 30 epochs and selected
epoch 1. Physical relocation and isolated offline CPU prediction checks match
all 12 source models exactly. The 246,426,955-byte archive SHA256 is
`e4e6e03bdd6cc0d31420b48cb41677b3c162ec83e5e34ade4c4f674215479c5f`.
All six public assets passed anonymous byte/hash verification. Follow this
release's own loader because its package kind differs from the original study.
The [spatial model release](https://github.com/aj-das-research/WM-ICLR/releases/tag/spatial-world-models-v1)
adds 15 predictors, their encoder, all five comparison arms and three seeds.
Each completed 30 epochs and passed exact relocated CPU prediction parity.
The 151,557,890-byte archive SHA256 is
`d84a817a74b03168417e0174052132fbb21364d6768301b67868a5c4b07790a1`.
All six public assets and all 217 archive payload hashes were independently
verified. Use this release's loader for its spatial package format; it produces
4×4 visual feature grids. The [independent audit](reports/evidence/spatial_completed_independent_audit.md)
also checks the complete 15-model study and all 16 paired comparisons.
Across five releases, **117 trained predictors** are publicly available:
75 real-video predictors and 42 simulation predictors. Calibration wrappers
are counted separately from trained networks.

The real-video finalizer creates `artifacts/releases/real_droid_v1`; the original
local simulator packages remain in `artifacts/releases/extensions_v1`, with
the smaller inference-only public export in `artifacts/publishing/simulator-release`.
They include strict offline loaders, provenance, model cards and encoder weights. The upstream pretrained downloads are also available through the
scripts above. Calibration is fitted using training recordings and its reported
development comparisons use validation recordings; it does not change the
original completed test results.

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
