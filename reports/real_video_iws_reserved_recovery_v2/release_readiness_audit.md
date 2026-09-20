# IWS 36-model release readiness audit

Snapshot: 2026-09-20T13:24:05.274153+00:00. Read-only inspection and file hashing; no model inference,
reserved payload access, upload, or mutation of either existing bundle.

**All 36 predictors are ready for local native-backend feature inference. None
of these 36 weights is publicly released. Reproducing the reserved recovery
backend requires a new explicit runtime/backend version and fresh parity proof.**

| Authoritative local bundle | Predictors | Files, including manifest | Exact bytes | MiB |
|---|---:|---:|---:|---:|
| `artifacts/releases/iws_single_observation_local_v1` | 27 | 124 | 223,069,665 | 212.74 |
| `artifacts/releases/iws_unbounded_single_observation_local_v1` | 9 | 52 | 74,780,289 | 71.32 |
| Total | 36 | 176 | 297,849,954 | 284.05 |

The first bundle contains three tasks × three methods (autoregressive, anchored
additive, bounded spatial mixing) × three seeds. The second contains the same
three tasks/seeds for the no-tanh ablation, with a distinct package kind. All runs
completed 30 epochs; the selected original checkpoints are at epochs 25, 26, 28,
29 or 30, and all no-tanh selected checkpoints are epoch 30. All 174 payload
hashes, exact inventory coverage, both manifest bindings and selected model/config
hashes passed this audit. No optimizer/RNG continuation state or symlink is present.

An unauthenticated GitHub API read during this audit returned exactly five public
release tags: `real-droid-v1`, `generalization-v1`, `simulator-development-v1`,
`horizon10-development-v1`, and `spatial-world-models-v1`; none contains an IWS
asset. The repository's documented 117 public predictors are separate. Both IWS
export receipts explicitly say `published: false`, and `artifacts/releases/` is
ignored by Git. [Public release inventory](https://github.com/aj-das-research/WM-ICLR/releases).

## Existing proof and supported contract

Each release has `local_inference_export.json`, `relocated_cpu_parity.json`,
`readiness_checks.json`, and `independent_source_review.json` under its respective
`reports/real_video_iws[/_unbounded]/release/` directory (the actual second path is
`reports/real_video_iws_unbounded/release/`). Existing isolated relocation receipts
contain 27/27 and 9/9 exact CPU-FP32 comparisons, maximum absolute error **0**,
all 59 offsets, two CPU threads, PyTorch 2.6.0+cu124, NumPy 2.2.6, and no network
attempts. The no-tanh proof additionally records zero original-workspace reads.
The independent source reviews reran 12 original boundary tests plus five path
subtests, and 26 no-tanh tests. This audit checked the receipts and hashes; it did
not rerun inference or infer new accuracy.

Inputs are one raw DINOv2 feature vector `[B,6144]` (channel-major 384×4×4) and
native commands `[B,H,A]`, with widths **4 / 14 / 8** for PushT / Box / Rope.
H60 returns `[B,59,6144]`: offset k uses command rows 0..k. Frozen training-only
normalization is internal. Outputs are features, not RGB, actions, or control
scores. Neither bundle includes encoder weights or an RGB decoder. Stored parity
inputs are internal-training episode `000011`, frame 0 and rows 0–59; reference
arrays are model predictions, not future targets. The proof is batch 1 on three
fixed inputs, not arbitrary-platform, GPU, batch-64, or reserved-set equivalence.

Both immutable runtimes call the original native sequence-GRU implementation.
They contain neither the new prefix-backend module nor a backend-selection API.
Their old CPU proof remains valid for that exact original runtime; it does not
certify the new row-wise runtime or resolve the earlier prefix failure.

## Recovery reproducibility requires a separate backend identity

The recovery registration binds
`src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py`, version
`one_command_row_gru_cpu_fp32_v1`. It changes only GRU dispatch to one command row
per call, retains weights/modules/state keys, and is installed on **all 36**
models. Recovery inference uses CPU FP32, eight threads, interop one, batches
64/64/64/8 per 200-handle task, H60 plus separate H15/H30/H45 calls, and the
unchanged prefix tolerances rtol=1e-5 / atol=2e-5.

Use a **new bundle version or explicitly named backend option**, recorded in the
manifest and returned inference metadata. Do not silently replace either v1
runtime, its fixture predictions, or its historical parity receipt. The selected
model/config bytes and model package kinds can remain unchanged; the enclosing
runtime manifest must change. Reuse the unchanged frozen inference exporter, or
copy the already validated inference packages, without exporting training state.

Required new files: the pinned backend module, explicit runtime dispatcher,
backend/provenance manifest, revised model card, backend-specific reference
predictions, pinned environment metadata, and relocated offline parity receipt.
The proof should compare the selected source package plus the same row-wise
backend against the relocated bundle on fixed **training/synthetic inputs only**:
all 36 models, all 59 outputs, separate H15/30/45 prefix calls, and representative
batch sizes 1/64/8 with the recorded eight-thread settings. Record native-versus-
row-wise differences separately; do not require or advertise equality between
numerical backends unless measured. Existing predictor-cost timings also describe
the original backend, so row-wise latency needs a separately labeled profile.

The independent recovery review states that the original failure's root cause
was **not established**; input-only probes did not reproduce it. Row-wise dispatch
is an execution-order stabilization attempt, not proof of a mathematical error
or a new learned model. Forthcoming accuracy claims require complete 36-run
recovery finalization and independent numerical review; this release audit makes
no pending recovery-score claim.

## Public packaging gaps

1. **Real fixtures lack an established redistribution basis in this record.**
   The pinned RLA-WM dataset card/API revision
   `34bd8a8cbf3fa68e09ebd69aa35cb673279f4fc2` has no license field or license file
   in the saved metadata. The separate RLA-WM *model* card's CC-BY-4.0 entry does
   not establish a dataset-fixture grant. Keep the genuine-input proofs local;
   use deterministic, clearly labeled synthetic feature/command fixtures and
   regenerate prediction references for a public portability package, unless an
   applicable dataset permission is documented. This is a fixture-permission
   gap, not a conclusion that publishing the trained tensors is prohibited.
2. **Explicit tensor notice and source attribution.** Both bundles carry project
   MIT software and pinned LeWM MIT notices. Add an explicit `MODEL_LICENSE.md`
   for the project's trained predictors, following the existing release pattern,
   without applying it to third-party dataset content. No RLA-WM or DINO encoder
   weights are bundled. Preserve the LeWM NOTICE and source hashes.
3. **Self-contained encoder sidecar.** Add a public preprocessing/provenance JSON:
   `facebook/dinov2-small` revision
   `ed25f3a31f01632728cabb09d1542f84ab7b0056`; 224×224 bilinear antialias resize,
   ImageNet normalization, remove CLS, FP32 pooling of 16×16 tokens to 4×4,
   channel-major flatten; registered extraction CUDA BF16, TF32 off. The no-tanh
   README already spells this out; the original README refers mainly to absent
   external provenance contents. DINOv2's recorded Apache-2.0 attribution should
   accompany any future encoder distribution. CPU re-encoding is not claimed
   cache-identical.
4. **Seal the actual public archive.** Add task/method/seed inventory, selector,
   complete development results and regressions, explicit backend/split labels,
   `MODEL_CARD.md`, `SHA256SUMS`, exact archive inventory, and offline verification
   of the final archive. Public model cards must distinguish original development
   results from later reserved recovery and the post-development no-tanh ablation.
   Upload and anonymous-download verification are subsequent work, not performed
   here. Do not count these 36 among public predictors before those checks.

## Concrete next commands and bounded work

The following existing commands verify the **original** local backend only;
outputs stay outside immutable bundles. They are listed, not executed here:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -I \
  artifacts/releases/iws_single_observation_local_v1/runtime.py \
  --output /tmp/iws27-native-parity.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -I \
  artifacts/releases/iws_unbounded_single_observation_local_v1/runtime.py \
  --output /tmp/iws9-native-parity.json
.venv/bin/python -B -m unittest discover \
  -s scripts/real_video_iws_release -p 'test_*.py'
.venv/bin/python -B -m unittest discover \
  -s scripts/real_video_iws_unbounded_release -p 'test_*.py'
```

For interactive imports use `python -B` before importing runtime; otherwise the
strict inventory can reject a newly created bytecode file. Use separate fresh
processes for the two bundles to avoid importing the wrong bundled model module.

The next implementation is a new versioned 36-model release builder/runtime
(e.g. `scripts/real_video_iws_release_v2/` and
`artifacts/releases/iws_single_observation_rowwise_local_v2/`), with the explicit
backend, synthetic public fixtures, sidecars, and fresh offline proof above.
**No current command builds that artifact.** Existing `build.py --output ...`
commands build their native-backend bundle only, refuse existing destinations,
and do not add the row-wise backend. Keep the current 27+9 bundles and all
registered study files unchanged. No new training, model selection, held-out
payload access, or change to evaluation tolerances is needed for release work.

## Source bindings

| File | SHA256 |
|---|---|
| `artifacts/releases/iws_single_observation_local_v1/manifest.json` | `37483a16d9aedfa8986af2899fafb215f9dcdae87ee3fa52cdb5f03e21579b5d` |
| `artifacts/releases/iws_unbounded_single_observation_local_v1/manifest.json` | `906912e301d6abd653f6eb941b23121f18295562cb1ddd4235e2dd6c021b1dec` |
| `reports/real_video_iws/release/relocated_cpu_parity.json` | `81c084cfa30aec54327d557fa449be165f709a48463583a66be5798b95f18db7` |
| `reports/real_video_iws_unbounded/release/relocated_cpu_parity.json` | `d384f460b2e112a7cf69370e177e2251363844e3960f8aeea9a2555f854496d3` |
| `reports/real_video_iws/release/independent_source_review.json` | `3fe7808ab7b3015382e0ce1f57f20dca621a7d246e9b87135a8f699976ad9bf6` |
| `reports/real_video_iws_unbounded/release/independent_source_review.json` | `e4599c5976a3669fb234759531432c8e70b06d5e777097bb884738f8036295d7` |
| `reports/real_video_iws/release/public_source_metadata.json` | `42c21104c40e4ac7b418bccd4417986ffa08efdfdf2ca2217d30132310942ff8` |
| `src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py` | `d0525e17a65b9b079ef26f2036e568050a63d99afdc899ccc46bd97d1f9c2d0d` |
| `configs/real_video_iws_reserved_recovery_v2/registration.json` | `891207e1c88f758c45750a56a59bd5fa6b793391026f75085cb8263c1d2c21d4` |
| `reports/real_video_iws_reserved_recovery_v2/source_review.json` | `94f874a4a8aebd1a4c3f26f085456902268add6f2c5a89a112af615841531eaf` |
