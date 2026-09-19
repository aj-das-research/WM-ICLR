# Local IWS inference packages

All **27 selected predictors** are exported and pass exact relocated CPU prediction parity at all 59 forecast offsets. The local bundle is `artifacts/releases/iws_single_observation_local_v1` (223,069,665 bytes). It includes all three tasks, all three arms and all three seeds, with byte-identical selected weights/configs and the frozen model/vendor code.

The bundle is feature-input inference only. It contains no optimizer/RNG state, RGB encoder weights, RGB decoder, reserved data or future targets. Its input is one raw 6,144-dimensional DINOv2 feature vector plus native recorded command rows (PushT:4, Box:14, Rope:8); H60 returns59 feature forecasts. See the bundle's `README.md` for the runnable API example and exact coordinate/indexing contract.

Validation used the already disclosed internal-training episode000011/frame0 for each task, with command rows0–59, CPU FP32 and two threads. A copied directory outside the workspace ran `python -I` with network connections blocked and no checkout imports. All27 original-versus-relocated output arrays were byte-equivalent. This is a portability check on three fixed inputs, not a new accuracy evaluation. The known CUDA AR prefix issue is not certified by CPU parity.

- `local_inference_export.json`: complete selected checkpoint provenance and export status.
- `relocated_cpu_parity.json`:27 exact output comparisons, imported source paths and zero network attempts.
- `independent_source_review.json`: independent loader/builder audit, including the corrected file-coverage and task/arm/seed binding checks.
- `readiness_checks.json`: final bindings,12 boundary tests with5 path subtests, and all 80 registered dependencies unchanged.

Manifest SHA256: `37483a16d9aedfa8986af2899fafb215f9dcdae87ee3fa52cdb5f03e21579b5d`.

For an imported API example, start Python with **`-B` before importing `runtime`**. A normal import can create a bytecode file before the module disables bytecode writing, and the strict bundle inventory correctly rejects that extra file. From the bundle directory, use an existing environment with its pinned requirements:

```bash
python -B -c "from pathlib import Path; from runtime import load_model; model, info = load_model('models/pusht_bounded_spatial_mix_s0', Path.cwd()); print(info)"
```

Keep example scripts and outputs outside the immutable bundle. The existing direct parity command, `python -I runtime.py --output /tmp/iws-parity.json`, remains valid. The audit removed only its own transient bytecode file and reverified all 123 original payload hashes; no bundle source, model, manifest, or parity receipt changed. See [publication_readiness.md](publication_readiness.md) for the separate conditions for a future public bundle.

No checkpoints were uploaded or published. The bundle remains under ignored `artifacts/releases/`. This local export is independent of the common-CPU development finalizer and does not authorize reserved evaluation. The earlier successful parity snapshot is retained as an explicitly named initial snapshot; the directory above and current receipts are authoritative.
