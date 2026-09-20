# IWS one-observation predictors: explicit row-wise CPU release

This new local bundle contains all 36 selected predictors: PushT, Box and Rope;
autoregressive, anchored additive, bounded spatial mixing and no-tanh spatial
mixing; three training seeds each. All completed 30 epochs. Model/config bytes
are identical to the selected packages used in the completed reserved-recovery
evaluation. The older 27-model and 9-model native-backend bundles remain intact.

The explicit backend is `one_command_row_gru_cpu_fp32_v1`: one command row per
GRU call, CPU FP32, eight intra-op threads and one inter-op thread. This changes
dispatch only, not weights, equations, module names or state keys. The original
prefix failure's cause remains unresolved; input-only probes did not reproduce
it. This release does not silently replace either older runtime or its proof.

## Load and forecast offline

Install the versions in `requirements.txt`, then use a fresh Python process:

```python
from pathlib import Path
import torch
from runtime import load_model, synthetic_inputs

predictor, metadata = load_model("pusht_unbounded_spatial_mix_s0", Path("."))
initial, commands = synthetic_inputs("pusht", 1)
forecast = predictor.predict(torch.from_numpy(initial), torch.from_numpy(commands))
print(metadata["backend"], forecast.shape)  # [1, 59, 6144]
```

Run `python -B example.py` for the same entirely synthetic example. Inputs are
one raw DINOv2 feature vector `[B,6144]` and native task commands `[B,H,A]`, all
CPU float32. Action widths are PushT 4, Box 14, and Rope 8. The output has shape
`[B,H-1,6144]`; offset k consumes command rows 0 through k, as in the registered
data alignment. H60 therefore returns 59 offsets, not 60. Three internal
temporal slots repeat the single observation; they are not observed history.
Frozen training-only feature/command normalization is internal. The exact image
coordinate and preprocessing specification is in `preprocessing.json`.

`load_model` checks the full inventory and every digest, selects the declared
package kind, verifies the selected epoch/training identity, reconstructs exact
normalization, and installs the explicitly named backend. Inference rejects GPU,
non-FP32, training mode, or changed thread settings. Use a fresh interpreter if
another installed `shiftwm` package was already imported. Use `python -B` so
the immutable bundle does not gain unlisted bytecode files.

## Evidence and portability scope

The bundle includes complete saved development and reserved-recovery summaries,
all methods and regressions, source hashes and individual model cards. These
saved results are not recomputed during export. Primary bounded-versus-additive
results are mixed; the later no-tanh comparisons are secondary. See
`MODEL_CARD.md` and the two evidence JSONs for scopes and intervals.

The portability proof uses deterministic synthetic uniform inputs generated
independently of all data and normalization statistics. No genuine image,
feature/command example, reserved payload or future target is redistributed.
For every model it compares all 59 output offsets at batches 1, 64 and 8 against
the selected original package with the same row-wise backend. The B1 numerical
reference is included; larger-batch reference bytes are represented by SHA256.
Separate H15/H30/H45 calls must satisfy the unchanged rtol 1e-5 / atol 2e-5
prefix gate. A relocated isolated process forbids network and original-workspace
reads. These finite synthetic checks establish the recorded environment's
portability, not arbitrary-platform/GPU equivalence or benchmark accuracy.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I runtime.py --output /tmp/parity.json
sha256sum -c SHA256SUMS
```

The exact Torch build and NumPy versions used are recorded in the manifest and
proof. A plain Torch version requirement does not promise byte equality across
other builds, CPUs or BLAS implementations. Native-versus-row-wise differences
on the fixed B1 synthetic input are recorded separately in the references; no
general equivalence between these backends is claimed.

These checkpoints forecast features. There is no RGB decoder, encoder weights,
policy, simulator, medical validation, or real-robot control API. Existing
predictor-cost measurements describe the older sequence-GRU backend and do not
measure this revised runtime's latency. No speedup follows from this release.

## Build provenance and publication status

The new builder is `scripts/real_video_iws_release_v2/build.py`. Its metadata-only
registration binds runtime/source/documents, exact old bundle manifests, selected
model/config/package bytes, and already completed result summaries. Independent
review is required before synthetic inference. It copies only weights/configs
and source; optimizer/RNG state and all older genuine fixtures are excluded.
An atomic new destination prevents overwriting older bundles. The final archive
is extracted and tested separately before the local readiness receipt is sealed.

All files remain local until the project owner publishes and verifies the public
download. Local readiness is not a public release URL. Licenses and third-party
attribution are in `MODEL_LICENSE.md`, `LICENSE`, and the vendored LeWM notices.
