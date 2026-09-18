#!/usr/bin/env python3
"""Package the verified DROID inference release and development calibration."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / "artifacts/releases/real_droid_v1").resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "completed" or len(manifest["models"]) != 12:
        raise ValueError("Require the completed twelve-model release")
    for relative, record in manifest["files"].items():
        path = source / relative
        if path.is_symlink() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Original release changed: {relative}")
    if any(row["completed_epochs"] != 30 or row["checkpoint_epoch"] not in (1, 2)
           for row in manifest["models"]):
        raise ValueError("Unexpected trained/selected-epoch record")
    if args.output.exists():
        raise FileExistsError("Use a fresh packaging directory")
    bundle = args.output / "shiftwm-real-droid-v1"
    shutil.copytree(source, bundle / "real_droid_v1")
    extra = bundle / "calibration"
    shutil.copytree(source / "source", extra / "source")
    for relative in ("src/shiftwm/real_video_development.py",
                     "scripts/real_video_development/calibrate_residual.py",
                     "scripts/real_video_development/diagnose.py"):
        target = extra / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    shutil.copytree(root / "configs/real_video_development/calibrations", extra / "configs")
    shutil.copy2(root / "reports/model_cards/real_droid_residual_calibration.md", extra / "MODEL_CARD.md")
    for name in ("real_droid_residual_calibration_protocol.md", "real_droid_residual_calibration_results.md", "real_droid_residual_calibration_results.json"):
        shutil.copy2(root / "reports" / name, extra / name)
    licenses = bundle / "licenses"
    licenses.mkdir()
    shutil.copy2(root / "LICENSE", licenses / "PROJECT-MIT.txt")
    shutil.copy2(root / "site/assets/DROID-LICENSE.txt", licenses / "DROID-CC-BY-4.0.txt")
    (bundle / "MODEL_LICENSE.md").write_text("""# Model distribution licenses

The project-authored predictor weights and calibration wrapper code/configurations
are distributed under the project MIT license in `licenses/PROJECT-MIT.txt`.
The unchanged shared DINOv2 encoder retains Apache 2.0; vendored LeWM source
retains MIT. Their full notices are preserved in `real_droid_v1/licenses/`.
DROID-derived provenance and the single validation feature/action example retain
CC BY 4.0 and attribution to Khazatsky et al., DROID (2024),
https://droid-dataset.github.io/ . The full data license is included here.

This publication notice adds the explicit distribution terms for our learned
predictors. Original local-release files are preserved byte-for-byte, including
their earlier notice that did not yet make a separate learned-weight grant.
No source video archive or private patient data is included.
""")
    (bundle / "README.md").write_text("""# ShiftWM real-DROID inference checkpoints

This offline research release contains twelve validation-selected predictors,
one shared frozen DINOv2-small encoder, source, model cards, exact dependencies,
training journals, reported outcomes and provenance. All twelve runs completed
30 epochs; validation selected epoch 1 for ten models and epoch 2 for two.
It also includes twelve optional scalar calibration wrappers fitted exclusively
on training episodes. These wrappers add no newly trained neural weights.

## Scope and measured limitations

Outputs are 1,536-dimensional frozen image features, not RGB video. Inputs are
three observed feature vectors, two past 35-D command blocks and future blocks.
The original primary real-DROID result improves MSE by 2.94% over persistence
and 0.20% over Framewise; the latter paired interval includes zero. Ten-block
performance is worse than Framewise. No state-of-the-art, physical robot-control
or clinical claim is made. The fixed subset has 1,126 real recordings; it is not
the full published DROID policy benchmark.

Calibration improves validation h5 by 0.779% and h10 by 0.154% relative to the
also-calibrated Framewise control. These are development findings, not new test
results. The calibration model card includes all limits and provenance.

## Offline use

Install dependencies from `real_droid_v1/source/requirements.lock.txt`; the
original runtime is recorded in `real_droid_v1/provenance/runtime.json`.
From this directory, the base predictors load as follows:

```python
import sys, torch
sys.path[:0] = ["real_droid_v1/source/src", "real_droid_v1/source/scripts/real_video"]
from train import load_package
model, metadata = load_package("real_droid_v1/models/droid_factorized_s0", "cpu")
with torch.inference_mode():
    prediction = model.predict(features, past_actions, future_actions)
```

In a fresh Python process, calibrated inference is:

```python
import sys, torch
from pathlib import Path
sys.path[:0] = ["calibration/source/src", "calibration/source/scripts/real_video_development"]
from calibrate_residual import load_calibrated_package
model = load_calibrated_package(
    "calibration/configs/droid_factorized_s0.json", device="cpu",
    base_checkpoint=Path("real_droid_v1/models/droid_factorized_s0").resolve())
with torch.inference_mode():
    prediction = model.predict(features, past_actions, future_actions)
```

The base-checkpoint override supports relocation without changing checkpoint
bytes. For images, load `real_droid_v1/encoder/dinov2-small` locally and use the
exact resize, normalization and spatial pooling in the bundled feature code.
The recorded cache used BF16 encoding; CPU FP32 encoding need not match bitwise.
The bundled validation feature window verifies loading/parity and is not a new
benchmark. Optimizer/RNG states remain in the original training runs and are
not included in this inference release.

See `MODEL_LICENSE.md`, per-model cards and `PUBLICATION_MANIFEST.json`.
""")
    module_path = root / "scripts/publishing/prepare_public_snapshot.py"
    spec = importlib.util.spec_from_file_location("public_scan", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files, findings = {}, []
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError("Archive cannot contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        if any(part in {".git", ".env", ".ssh", "__pycache__"} for part in path.parts):
            raise ValueError(f"Forbidden release content: {relative}")
        data = path.read_bytes()
        for name, rule in module.TOKEN_RULES.items():
            for match in rule.finditer(data):
                findings.append({"path": relative, "line": data[:match.start()].count(b"\n") + 1, "rule": name})
        files[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if findings:
        print(json.dumps({"secret_findings": findings}))
        raise SystemExit(1)
    audit = {"status": "verified", "original_manifest_sha256": digest(source / "manifest.json"),
             "original_files_preserved": True, "predictors": 12, "encoders": 1,
             "calibration_wrappers": 12, "secret_findings": [], "files": files}
    (bundle / "PUBLICATION_MANIFEST.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"bundle": str(bundle), "files": len(files) + 1, "bytes": sum(r['bytes'] for r in files.values())}))


if __name__ == "__main__":
    main()
