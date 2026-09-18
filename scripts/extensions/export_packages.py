#!/usr/bin/env python3
"""Export only fully completed registered extension runs to a portable local release.

The resolved validation-best generation is copied in full, including its exact
package manifest and optimizer snapshot. No symlink points back into training
runs. An independent offline process checks actual bundled weights and inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

from shiftwm.data import TrajectoryDataset
from shiftwm.extensions.checkpoint import atomic_json, file_sha256, read_package
from shiftwm.extensions.train import scientific_config, validate_completed

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def copy_file(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and file_sha256(target) == file_sha256(source):
        return
    temporary = target.with_name(target.name + ".exporting")
    shutil.copy2(source, temporary, follow_symlinks=True)
    temporary.replace(target)


def tracked_files(root):
    return {str(path.relative_to(root)): file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts
            and not path.name.endswith(".pyc")}


def bundle_source(destination):
    source = destination / "source"
    for path in (ROOT / "src/shiftwm").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json"}:
            copy_file(path, source / path.relative_to(ROOT))
    for relative in ["LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements.lock.txt",
                     "src/shiftwm/vendor/lewm/LICENSE", "src/shiftwm/vendor/lewm/NOTICE.json"]:
        copy_file(ROOT / relative, source / relative)
    for path in (ROOT / "scripts/extensions").glob("*"):
        if path.suffix in {".py", ".slurm"}:
            copy_file(path, source / path.relative_to(ROOT))
    for path in (ROOT / "configs/extensions").glob("*.json"):
        copy_file(path, source / path.relative_to(ROOT))
    copy_file(ROOT / "reports/domain_extension_protocol.md", source / "reports/domain_extension_protocol.md")
    for domain in ("drone", "surgery"):
        for name in ("README.md", "requirements.lock.txt", "setup.sh", "run_python.sh", "provenance.json"):
            path = ROOT / "environments" / domain / name
            if path.exists():
                copy_file(path, source / path.relative_to(ROOT))
    for src, dst in [("external/gym-pybullet-drones/LICENSE", "licenses/gym-pybullet-drones-MIT.txt"),
                     ("external/sofa_env/LICENSE", "licenses/sofa_env-MIT.txt"),
                     ("environments/surgery/SOFA_v24.06.00_Linux/LICENSE-LGPL.md", "licenses/SOFA-LGPL.md")]:
        copy_file(ROOT / src, destination / dst)
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "transformers", "numpy", "einops", "safetensors", "huggingface-hub", "Pillow"]}
    atomic_json({"python": sys.version, "inference_dependencies": versions,
                 "full_training_runtime": "source/requirements.lock.txt",
                 "simulator_runtimes": ["source/environments/drone", "source/environments/surgery"]},
                destination / "runtime.json")
    (destination / "requirements-inference.txt").write_text(
        "# Exact currently tested direct inference dependencies; no simulator needed.\n" +
        "--extra-index-url https://download.pytorch.org/whl/cu124\n" +
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
    return source


def verify_training_sources(config):
    provenance = config["provenance"]
    for relative, expected in provenance["extension_source_hashes"].items():
        if file_sha256(ROOT / "src/shiftwm" / relative) != expected:
            raise ValueError(f"Current source differs from training provenance: {relative}")
    registered = config["metadata"]["config"]
    checks = [(Path(registered["data_root"]) / "manifest.json", "data_manifest_sha256"),
              (Path(registered["dataset_kwargs"]["feature_cache"]) / "manifest.json", "cache_manifest_sha256"),
              (Path(registered["action_stats"]), "action_stats_sha256"),
              (Path(registered["protocol_path"]), "extension_protocol_sha256")]
    for relative, key in checks:
        if file_sha256(ROOT / relative) != provenance[key]:
            raise ValueError(f"Training provenance mismatch: {relative}")


def example_for(domain, run_config, destination):
    output = destination / "examples" / f"{domain}.npz"
    metadata_path = output.with_suffix(".json")
    data_root = ROOT / run_config["data_root"]
    cache_root = ROOT / run_config["dataset_kwargs"]["feature_cache"]
    dataset = TrajectoryDataset(data_root, split="val", sequence_length=8, stride=4, feature_cache=cache_root)
    sample = dataset[0]
    episode = next(e for e in dataset.manifest["episodes"] if e["trajectory_id"] == sample["trajectory_id"])
    cache_file = cache_root / episode["file"]
    raw_file = data_root / episode["file"]
    with np.load(raw_file, allow_pickle=False) as raw:
        images = raw["images"][int(sample["start"]):int(sample["start"]) + 3].copy()
    arrays = {"features": sample["features"][:3].numpy()[None],
              "past_actions": sample["actions"][:2].numpy()[None],
              "future_actions": sample["actions"][2:7].numpy()[None],
              "support_rgb": images[None]}
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    metadata = {"domain": domain, "split": "val", "trajectory_id": episode["trajectory_id"],
                "start": int(sample["start"]), "appearance_id": int(sample["observation_id"]),
                "example_sha256": file_sha256(output), "raw_source_sha256": file_sha256(raw_file),
                "feature_source_sha256": file_sha256(cache_file),
                "source_dataset_manifest_sha256": file_sha256(data_root / "manifest.json"),
                "use": "actual observed validation support and recorded candidate actions; API verification only"}
    if metadata["appearance_id"] != 0:
        raise ValueError("Portable RGB example must use the canonical cached appearance")
    atomic_json(metadata, metadata_path)
    return str(output.relative_to(destination))


def model_card(row, summary, state, package_files):
    config = state["config"]
    ours = "ShiftWM (ours)" if row["mode"] == "factorized" else row["mode"]
    return f"""# {row['domain']} / {row['architecture']} / {ours} / seed {row['seed']}

This local package comes from a registered run that completed **30 full epochs**.
Its validation-best checkpoint is epoch **{state['epoch']}**, selected solely by
five-step recursive validation MSE ({summary['best_validation_prediction_loss']:.9g}).
Validation loss is not a planning-success result. Learned control evidence is
pending or mixed; this card makes no superiority, SOTA, real-flight, or clinical claim.

## Contents and inputs

`model.pt`, `config.json`, `training_state.pt`, and `package_manifest.json` are
byte-identical copies of the resolved best generation. The optimizer/RNG state
belongs to that selected epoch, not necessarily epoch30. Training completion
is established separately by the copied summary and complete 30-epoch history.
All files are real local files; no training-directory symlink is required.

Inference uses three observed RGB frames (float BTCHW in [0,1]) or their frozen
192-dimensional features, two past action blocks, and candidate future action
blocks. Each action block concatenates five chronological 2D commanded controls.
For batched cached inference: history[B,3,192], past[B,2,10], future[B,K,10];
`model.rollout_features` returns[B,K,192]. Call `model.rollout` for RGB histories.
The package contains normalization and frozen visual-encoder weights. Simulator
state, hidden gain, future images, and target coordinates are not model inputs.

## Architecture and scope

The transformer family reuses pinned LeWM initialization. The GRU family is an
**in-house recurrent architecture control, not Dreamer**. Both transfer the frozen
PushT visual encoder. Framewise and constant-dynamics models are matched in-house
controls; this package is not a reproduced AdaJEPA model. Comparisons are primarily
within predictor family because family parameter counts and initialization differ.

Drone scope: fixed external-camera planar CF2X control with a common upstream PID;
not onboard navigation or real flight. Surgery scope: simulated LapGym deformable
tissue positioning; no patient data, clinical efficacy, or autonomous-surgery safety
claim. See the bundled registered protocol and simulator runtime READMEs.

## Provenance and use

Training identity: `{summary['training_identity']}`.
Dataset manifest SHA256: `{config['provenance']['data_manifest_sha256']}`.
Best model SHA256: `{package_files['model.pt']}`.
Exact model configuration, training history, optimizer state, source hashes,
original dataset/cache identities and runtime requirements accompany the release.
Historical filesystem paths in metadata are provenance only; offline loading
constructs the model from bundled configuration and tensors.

See `../../README.md` for the standalone CPU loader and verified actual-data
examples, and `../../LICENSES.md` for code/data/weight license distinctions.
"""


def export_one(row, destination):
    output = ROOT / row["output"]
    run_id = output.name
    summary = read(output / "training_summary.json")
    identity = summary.get("training_identity")
    registered = read(ROOT / row["config"])
    if scientific_config(registered) != scientific_config(read(output / "run_config.json")):
        raise ValueError(f"Registered and executed configs differ: {run_id}")
    # Includes full 30-epoch checks, both complete package generations, optimizer
    # agreement, validation-only selection and strict CPU state-dictionary load.
    summary = validate_completed(output, identity)
    source, state = read_package(output / "best", require_training=True)
    verify_training_sources(state["config"])
    original_manifest = read(source / "package_manifest.json")
    target = destination / "checkpoints" / run_id
    target.mkdir(parents=True, exist_ok=True)
    for name in [*original_manifest["files"], "package_manifest.json"]:
        copy_file(source / name, target / name)
    for name in ("training_summary.json", "run_config.json", "metrics.jsonl"):
        copy_file(output / name, target / name)
    # Verify the independent copies, including original optimizer/RNG bytes.
    copied_path, copied = read_package(target, require_training=True)
    if copied_path != target.resolve() or target.is_symlink() or copied["config"] != state["config"]:
        raise ValueError("Export did not produce an independent exact package")
    (target / "MODEL_CARD.md").write_text(model_card(row, summary, state, original_manifest["files"]))
    record = {**row, "run_id": run_id, "status": "exported_completed_30_epoch_best",
              "release_path": str(target.relative_to(destination)), "training_identity": identity,
              "selected_epoch": state["epoch"], "completed_epochs": 30,
              "best_validation_recursive_mse": summary["best_validation_prediction_loss"],
              "source_generation": str(source.relative_to(ROOT)), "package_files": original_manifest["files"],
              "package_manifest_sha256": file_sha256(target / "package_manifest.json"),
              "data_manifest_sha256": state["config"]["provenance"]["data_manifest_sha256"],
              "cache_manifest_sha256": state["config"]["provenance"]["cache_manifest_sha256"],
              "upstream_initialization": state["config"]["provenance"].get("download", {}),
              "source_hashes": state["config"]["provenance"]["extension_source_hashes"],
              "performance_claim": "pending or mixed; validation-best selection does not establish better control"}
    atomic_json(record, target / "release_record.json")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/releases/extensions_v1")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.chdir(ROOT)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    registry = read(ROOT / "configs/extensions/campaign.json")
    source = bundle_source(destination)
    records, pending, rejected = [], [], []
    examples = {}
    for row in registry["runs"]:
        summary_path = ROOT / row["output"] / "training_summary.json"
        if not summary_path.exists() or read(summary_path).get("status") != "completed":
            pending.append(row)
            continue
        try:
            record = export_one(row, destination)
            if row["domain"] not in examples:
                examples[row["domain"]] = example_for(row["domain"], read(ROOT / row["config"]), destination)
            records.append(record)
            print(json.dumps({"exported": record["run_id"], "count": len(records)}), flush=True)
        except Exception as error:
            rejected.append({**row, "error": type(error).__name__, "message": str(error)})
    # Required provenance is copied even though full datasets/caches are not.
    for domain in examples:
        row = next(r for r in registry["runs"] if r["domain"] == domain)
        config = read(ROOT / row["config"])
        for origin, name in [(ROOT / config["data_root"] / "manifest.json", "dataset_manifest.json"),
                             (ROOT / config["dataset_kwargs"]["feature_cache"] / "manifest.json", "feature_manifest.json"),
                             (ROOT / config["action_stats"], "action_stats.json")]:
            copy_file(origin, destination / "provenance" / domain / name)
    selections = {}
    for record in sorted(records, key=lambda r: r["seed"]):
        key = f"{record['domain']}/{record['architecture']}/{record['mode']}"
        selections.setdefault(key, {"package": record["release_path"], "example": examples[record["domain"]],
                                    "key": key, "mode": record["mode"], "architecture": record["architecture"]})
    atomic_json(list(selections.values()), destination / "verification_cases.json")
    copy_file(ROOT / "scripts/extensions/verify_exported_packages.py", destination / "verify_packages.py")
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(source / "src"), "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with tempfile.TemporaryDirectory(prefix="extension-offline-check-") as isolated:
        result = subprocess.run([sys.executable, str(destination / "verify_packages.py"),
                                 "--release", str(destination)], cwd=isolated, env=environment,
                                text=True, capture_output=True)
    (destination / "offline_verification.log").write_text(result.stdout + result.stderr)
    verification = read(destination / "offline_verification.json") if result.returncode == 0 else {
        "status": "failed", "return_code": result.returncode}
    status = "complete" if len(records) == len(registry["runs"]) and not rejected else "partial_completed_runs_only"
    if rejected or verification["status"] != "passed":
        status = "validation_failed"
    catalog = {"schema_version": 1, "status": status, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "local_only": True, "registered_runs": len(registry["runs"]), "exported_runs": len(records),
               "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "registry_sha256": file_sha256(ROOT / "configs/extensions/campaign.json"),
               "source_bundle_files": tracked_files(source), "runs": records, "pending": pending,
               "rejected": rejected, "offline_verification": verification,
               "exporter_sha256": file_sha256(Path(__file__).resolve())}
    write_readme(destination, catalog)
    catalog["release_files"] = {path: digest for path, digest in tracked_files(destination).items()
                                if path not in {"manifest.json", "SHA256SUMS"}}
    atomic_json(catalog, destination / "manifest.json")
    (destination / "SHA256SUMS").write_text("".join(
        f"{digest}  {path}\n" for path, digest in tracked_files(destination).items()
        if path != "SHA256SUMS"))
    print(json.dumps({"status": status, "exported": len(records), "pending": len(pending),
                      "rejected": rejected, "verification": verification.get("status")}, indent=2))
    if status == "validation_failed" or (args.require_all and pending):
        raise SystemExit(1)


def write_readme(destination, catalog):
    table = "\n".join(f"| {r['run_id']} | {r['selected_epoch']} | [card]({r['release_path']}/MODEL_CARD.md) |"
                       for r in catalog["runs"])
    (destination / "README.md").write_text(f"""# ShiftWM local drone/surgery checkpoint release

**{catalog['exported_runs']}/{catalog['registered_runs']} registered checkpoints exported.**
Status: `{catalog['status']}`. Only runs that completed all 30 epochs and passed
the strict training/package checks appear here. Future exports may add newly
completed registered runs; they cannot substitute partial training checkpoints.
Nothing in this directory has been published externally.

These are usable model packages, not evidence that our learned policy wins.
Current performance is pending or mixed; final test results remain separate.
The GRU is an in-house architecture control, not Dreamer. Surgery is simulated,
non-clinical tissue positioning; drone control uses an external camera and PID.

## Offline loading

Use Python3.11 with `requirements-inference.txt`, then set
`PYTHONPATH="$PWD/source/src"` from this directory. No model download, original
upstream checkout, simulator, dataset, or training-directory symlink is required
for loading and cached/RGB inference. Dependency installation itself may need
internet or preinstalled wheels; the verified inference process blocks network.

```python
from shiftwm.extensions.checkpoint import load_package
import numpy as np
import torch

model, state = load_package("checkpoints/drone_transformer_factorized_s0", device="cpu")
with np.load("examples/drone.npz", allow_pickle=False) as example:
    history = torch.from_numpy(example["features"])
    past = torch.from_numpy(example["past_actions"])
    future = torch.from_numpy(example["future_actions"])
with torch.inference_mode():
    predicted_features = model.rollout_features(history, past, future)
print(predicted_features.shape)  # [1,5,192]
```

`python verify_packages.py --release .` repeats strict loading and actual-input
inference checks for one checkpoint per available domain/family/mode. It verifies
that imported model code came from the bundle and blocks outgoing connections.
See `offline_verification.json`; the check measures compatibility, not accuracy.

Each selected generation retains its original package manifest and complete
optimizer/RNG snapshot byte for byte. The optimizer snapshot is at the
validation-best epoch, while copied histories establish completion of30epochs.
Original paths inside metadata describe provenance; they are not loader inputs.
Full generated datasets and feature caches are not included. Their manifests,
payload hashes, training-only action statistics, registered configurations,
generation/evaluation source, and domain runtime instructions are included.

## Available packages

| Run | Selected epoch | Documentation |
|---|---:|---|
{table}

See `manifest.json` for hashes and pending/rejected runs, `source/` for the
bundled implementation and registered protocol, and `LICENSES.md` for notices.
""")
    (destination / "LICENSES.md").write_text("""# Licenses and scope

Project source is MIT (`source/LICENSE`). The unchanged bundled LeWorldModel
implementation retains its own MIT LICENSE and exact-revision NOTICE under
`source/src/shiftwm/vendor/lewm/`. Released initialization is pinned to
`quentinll/lewm-pusht`, revision `22b330c28c27ead4bfd1888615af1340e3fe9052`;
model provenance records the original weight hash. This local packaging does
not invent or override separate upstream model-distribution terms.

The generated drone and tissue-manipulation data are local synthetic simulation
records, not downloaded real-flight or patient datasets. Full datasets are not
distributed in this checkpoint bundle. The included validation snippets retain
their generated-data provenance. A simulator's code license is not automatically
a declaration of license for every generated dataset or model weight. A public
dataset/model release should carry its own explicit distribution notice.

gym-pybullet-drones and LapGym/sofa_env are MIT-licensed; their notices are under
`licenses/`. The SOFA24.06 runtime retains its LGPL license (also copied there)
and is not relicensed or bundled as a binary here. Simulator source revisions,
runtime requirements and renderer constraints are documented in
`source/environments/`. No clinical efficacy, medical-device authorization,
autonomous surgical safety, or real-flight capability is claimed.
""")


if __name__ == "__main__":
    main()
