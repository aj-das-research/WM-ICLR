# Local no-tanh inference export

This separate release builder packages all nine completed IWS no-tanh component
ablation models. It calls the frozen full-30-epoch inference exporter unchanged.
It neither trains nor evaluates benchmark accuracy, and opens no reserved data.
The original 27-model bundle and registered experiment sources remain unchanged.

From the project root:

```bash
.venv/bin/python -B -m unittest discover -s scripts/real_video_iws_unbounded_release -p 'test_*.py'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B scripts/real_video_iws_unbounded_release/build.py
```

Default output is the ignored local directory
`artifacts/releases/iws_unbounded_single_observation_local_v1`. Existing bundles
and receipts are never overwritten. The builder checks the completed 9+27
finalization, all bound sources, exact task/seed checkpoint identities and all
four frozen normalization buffers before exporting selected weights and configs.

The package contains only inference weights/configs, frozen predictor/vendor
source and three target-free internal-training inputs (episode `000011`, frame
0, command rows 0–59), with nine prediction-reference arrays. Its distinct kind
is `shiftwm_iws_single_observation_unbounded_v1`. Neither optimizer/RNG state nor
an RGB encoder/decoder is included. Data-derived fixtures remain local.

The builder copies the entire bundle to a temporary directory and invokes an
isolated Python process with CPU FP32 and two threads. All nine arrays of shape
`[1,59,6144]` must agree exactly. Network and original-workspace access are denied
except reads of installed Python environment libraries. This is a portability
proof on fixed inputs, not a new accuracy or GPU-equivalence result.

Receipts are under `reports/real_video_iws_unbounded/release/`. To repeat offline
parity after copying the bundle elsewhere, keep the output outside it:

```bash
.venv/bin/python -I /path/to/copied/bundle/runtime.py --output /tmp/no-tanh-parity.json
```

For interactive API imports, start Python with `-B` before importing `runtime`.
The strict inventory rejects unlisted bytecode and any other extra bundle file.
No upload, public weight distribution, reserved evaluation or timer changes are
performed by this package.
