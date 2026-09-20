#!/usr/bin/env python3
"""Build local inference-only IWS packages; no evaluation or publication.

Uses the unchanged frozen exporter after its full-30-epoch completion gate.
The selected packages and model source are copied exactly. Three authorized
training inputs support bounded CPU H60 relocation parity, not accuracy claims.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/real_video_iws_unbounded/release"
DEFAULT = ROOT / "artifacts/releases/iws_unbounded_single_observation_local_v1"
FINALIZATION = ROOT / "reports/real_video_iws_unbounded/development_finalization.json"
ORIGINAL_BUNDLE = ROOT / "artifacts/releases/iws_single_observation_local_v1"
COPIED_SOURCES = ["src/shiftwm/__init__.py", "src/shiftwm/upstream.py",
                  "src/shiftwm/real_video_iws_unbounded/__init__.py", "src/shiftwm/real_video_iws_unbounded/model.py",
                  "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/jepa.py",
                  "src/shiftwm/vendor/lewm/NOTICE.json", "src/shiftwm/vendor/lewm/LICENSE", "LICENSE"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Refusing to replace receipt: " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def binding(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def copy_exact(source, destination):
    before = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(source) != before or sha(destination) != before:
        raise ValueError("Copy source changed")
    return before


def completed_study(registration):
    """Bind the complete saved 9+27 result; never run an evaluator or load a split."""
    value = read(FINALIZATION)
    if (value.get("status") != "passed" or value.get("completed_new_runs") != 9
            or value.get("completed_v1_comparator_runs") != 27
            or value.get("official_validation_payloads_read") != 0
            or value.get("scope") != "exploratory_internal_development_after_v1"):
        raise ValueError("Complete internal-development 9+27 finalization required")
    if value.get("registration_sha256") != sha(ROOT / "configs/real_video_iws_unbounded/registration_v1.json"):
        raise ValueError("Finalized registration differs")
    rows = [r for r in value["per_run"] if r["mode"] == "unbounded_spatial_mix"]
    tasks = ("pusht", "bimanual_box", "bimanual_rope")
    expected_new = {f"{t}_unbounded_spatial_mix_s{s}" for t in tasks for s in range(3)}
    expected_all = expected_new | {f"{t}_{m}_s{s}" for t in tasks for s in range(3)
                                   for m in ("autoregressive", "anchored_additive", "bounded_spatial_mix")}
    if (len(value["per_run"]) != 36 or len(rows) != 9
            or {r["name"] for r in value["per_run"]} != expected_all
            or {r["name"] for r in rows} != expected_new
            or {r["name"] for r in registration["runs"]} != expected_new
            or any(r.get("completed_epochs") != 30 for r in value["per_run"])
            or not value.get("source_dependencies")):
        raise ValueError("Incomplete finalized task/seed grid")
    for path, expected in value["source_dependencies"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Finalized input changed: " + path)
    return {r["name"]: r for r in rows}


def inventory(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def release_destination(destination):
    path = Path(destination)
    if path.exists() or path.is_symlink():
        raise ValueError("Refusing to replace an existing inference bundle")
    destination = path.resolve()
    allowed = (ROOT / "artifacts/releases").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed:
        raise ValueError("Local checkpoint binaries must remain under ignored artifacts/releases")
    return destination


def build(destination):
    import numpy as np
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    campaign = module(ROOT / "scripts/real_video_iws_unbounded/campaign.py", "_release_iws_campaign")
    train = campaign.trainer()
    registration = campaign.check_registration()
    finalized = completed_study(registration)
    finalization_binding = binding(FINALIZATION)
    original_inventory = inventory(ORIGINAL_BUNDLE)
    if not original_inventory:
        raise ValueError("Expected original 27-model bundle is missing")
    config = read(campaign.CONFIG)
    destination = release_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = Path(tempfile.mkdtemp(prefix=".iws-bundle-pending-", dir=destination.parent))
        sources = {p: copy_exact(ROOT / p, temporary / p) for p in COPIED_SOURCES}
        sources["scripts/real_video_iws_unbounded_release/runtime.py"] = copy_exact(
            ROOT / "scripts/real_video_iws_unbounded_release/runtime.py", temporary / "runtime.py")
        sources["scripts/real_video_iws_unbounded_release/build.py"] = sha(Path(__file__))
        fixtures, fixture_sources, input_values, statistics = {}, {}, {}, {}
        for task in sorted(config["tasks"]):
            cache = train.open_cache(config, task)
            statistics[task] = cache.statistics
            episode_id = "000011"  # Already disclosed fixed training input; not selected by outcome.
            cache.inventory.authorize(episode_id, "internal_train")
            receipt, arrays = cache.episode(episode_id, "internal_train")
            if receipt["split"] != "internal_train" or len(arrays["commands"]) < 60:
                raise ValueError("Unauthorized or too-short parity input")
            initial = arrays["features"][0:1].copy()
            commands = arrays["commands"][None, :60].copy()
            if initial.shape != (1,6144) or commands.shape != (1,60,config["tasks"][task]["action_dim"]):
                raise ValueError("Fixed input alignment differs")
            filename = "fixtures/" + task + ".npz"
            (temporary / "fixtures").mkdir(exist_ok=True)
            np.savez_compressed(temporary / filename, initial_features=initial, commands=commands)
            fixtures[task] = filename; input_values[task] = (initial, commands)
            fixture_sources[task] = {
                "episode_id": episode_id, "split": "internal_train", "initial_frame": 0,
                "command_rows": [0,59], "forecast_offsets": [1,59], "targets_included": False,
                "selection": "fixed previously disclosed input, no outcome-dependent selection",
                "cache_receipt": binding(cache.output / "episodes" / episode_id / "receipt.json"),
                "cache_payload": binding(cache.output / "episodes" / episode_id / "arrays.npz"),
                "training_statistics": binding(cache.output / "training_statistics.json"),
                "cache_identity": binding(cache.output / "identity.json"),
                "initial_feature_values_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                "command_values_sha256": hashlib.sha256(commands.tobytes()).hexdigest()}
        rows = []
        for row in registration["runs"]:
            run = ROOT / row["output"]
            target = temporary / "models" / row["name"]
            package, state = train.read_package(run / "best")
            expected_recipe = {"task": row["task"], "mode": row["mode"], "seed": row["seed"],
                               "training": config["training"], "model": config["model"],
                               "task_config": config["tasks"][row["task"]],
                               "study_config_sha256": sha(campaign.CONFIG)}
            if state["config"]["metadata"]["identity"]["scientific_config"] != expected_recipe:
                raise ValueError("Selected package differs from its registered task/arm/seed/recipe")
            selected = finalized[row["name"]]
            if (selected["completed_epochs"] != 30 or selected["selected_epoch"] != state["epoch"]
                    or selected["selected_checkpoint_sha256"] != sha(package / "model.pt")
                    or state["config"]["metadata"]["identity"]["normalization"] != statistics[row["task"]]):
                raise ValueError("Selected checkpoint or training statistics differ from finalized study")
            exported = train.export_inference_package(run, target)
            summary = read(run / "training_summary.json")
            model, _ = train.load_package(run / "best", "cpu")
            for key in ("feature_mean", "feature_std", "command_mean", "command_std"):
                if not torch.equal(getattr(model, key), torch.tensor(statistics[row["task"]][key], dtype=torch.float32)):
                    raise ValueError("Loaded normalization buffer differs: " + key)
            initial, commands = input_values[row["task"]]
            with torch.inference_mode():
                expected = model.predict(torch.from_numpy(initial.copy()), torch.from_numpy(commands.copy())).numpy()
            if expected.shape != (1,59,6144) or not np.isfinite(expected).all():
                raise ValueError("Invalid original-model prediction")
            expected_path = "fixtures/" + row["name"] + "_prediction.npz"
            np.savez_compressed(temporary / expected_path, predictions=expected)
            records = {name: binding(package / name) for name in ("model.pt", "config.json", "package_manifest.json")}
            if any(sha(target / key) != records[key]["sha256"] for key in ("model.pt", "config.json")):
                raise ValueError("Export altered selected model/config bytes")
            rows.append({**{k: row[k] for k in ("name", "task", "mode", "seed")},
                         "directory": str(target.relative_to(temporary)), "selected_epoch": state["epoch"],
                         "training_identity": summary["training_identity"], "completed_epochs": 30,
                         "source_packages": records, "training_summary": binding(run / "training_summary.json"),
                         "export": exported, "fixture": fixtures[row["task"]],
                         "expected_prediction": expected_path})
            print(row["name"] + ": exported selected epoch " + str(state["epoch"]), flush=True)
        (temporary / "requirements.txt").write_text("torch==" + str(torch.__version__).split("+")[0] +
            "\nnumpy==" + np.__version__ + "\neinops==" + __import__("einops").__version__ + "\n")
        (temporary / "README.md").write_text(README)
        manifest = {"schema": "iws_unbounded_local_inference_bundle_v1", "package_kind": train.PACKAGE_KIND,
                    "status": "local_inference_prerelease", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "benchmark_finalization_claimed": False, "reserved_data_access_authorized": False,
                    "completed_development_source": finalization_binding,
                    "source_registration": binding(campaign.REGISTRATION), "scientific_config": binding(campaign.CONFIG),
                    "runtime_sources": sources, "training_inputs": fixture_sources, "models": rows,
                    "files": {str(p.relative_to(temporary)): sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()}}
        write(temporary / "manifest.json", manifest)
        campaign.check_registration()
        for p, h in sources.items():
            if sha(ROOT / p) != h:
                raise ValueError("Source changed during export")
        # Copy into a different absolute root and start an isolated interpreter.
        with tempfile.TemporaryDirectory(prefix="iws-relocated-offline-") as relocation:
            relocated = Path(relocation) / "bundle"
            shutil.copytree(temporary, relocated)
            proof = Path(relocation) / "parity.json"
            env = {k:v for k,v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            completed = subprocess.run([sys.executable, "-I", str(relocated / "runtime.py"),
                "--bundle", str(relocated), "--output", str(proof), "--forbid-root", str(ROOT)], cwd=relocation, env=env,
                timeout=180,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (REPORTS / "relocated_parity.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError("Relocated inference failed; pending bundle retained: " + str(temporary))
            result = read(proof)
            if (result["status"] != "passed" or len(result["models"]) != 9
                    or not result["isolated_python"] or result["network_attempts"] != 0
                    or not result["original_workspace_reads_denied"]
                    or result["original_workspace_read_attempts"] != 0):
                raise ValueError("Incomplete relocation proof")
            write(REPORTS / "relocated_cpu_parity.json", result)
        campaign.check_registration()
        completed_study(registration)
        if binding(FINALIZATION) != finalization_binding or inventory(ORIGINAL_BUNDLE) != original_inventory:
            raise ValueError("Completed study or original 27-model bundle changed")
        for row in rows:
            for item in row["source_packages"].values():
                if sha(ROOT / item["path"]) != item["sha256"]:
                    raise ValueError("Selected source changed during parity")
        os.rename(temporary, destination)
        outcome = {"status": "passed_local_inference_export", "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "bundle": str(destination.relative_to(ROOT)), "manifest_sha256": sha(destination / "manifest.json"),
                   "models": 9, "selected_epochs": {r["name"]: r["selected_epoch"] for r in rows},
                   "parity": binding(REPORTS / "relocated_cpu_parity.json"),
                   "completed_development_source": finalization_binding,
                   "original_27_bundle_unchanged": True,
                   "original_27_inventory_digest": hashlib.sha256(json.dumps(original_inventory, sort_keys=True).encode()).hexdigest(),
                   "source_dependencies": sources, "source_registration": binding(campaign.REGISTRATION),
                   "optimizer_or_rng_included": False, "published": False, "encoder_weights_included": False,
                   "future_target_inputs_included": False, "official_validation_payloads_read": 0,
                   "benchmark_finalization_claimed": False, "device": "cpu", "threads": 2,
                   "bundle_bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())}
        write(REPORTS / "local_inference_export.json", outcome)
        print(json.dumps(outcome, indent=2), flush=True)


README = """# IWS no-tanh component ablation — local inference bundle

Nine completed predictors: PushT (4 command coordinates), Bimanual Box (14),
Bimanual Rope (8), each at seeds 0/1/2. Every run completed 30 epochs. The selected
checkpoint is the earliest strict minimum of internal-development equal-trajectory
H60 standardized MSE. This exploratory post-development ablation removes only
`tanh` from the correction head of the frozen fixed-source spatial mixer. Its
package kind is `shiftwm_iws_single_observation_unbounded_v1`; it cannot be loaded
as the bounded v1 method. The original 27-model release is separate and unchanged.

The manifest binds the complete 9+27 development result, exact selected weights,
configurations, training normalization, model/vendor source and fixtures. Export
performs no accuracy evaluation or model selection. This local bundle is not a
public release and does not authorize reserved validation. Real dataset-derived
fixtures remain local; their redistribution permission is not established here.

Run Python with the versions in requirements.txt. Start with `-B` before importing
runtime so an immutable bundle does not acquire an unlisted bytecode file. Keep
scripts and outputs outside the bundle. Example (set ROOT to this bundle):

```python
from pathlib import Path
import numpy as np
import torch
from runtime import load_model
root = Path(ROOT)
model, info = load_model('models/pusht_unbounded_spatial_mix_s0', root)
with np.load(root / 'fixtures/pusht.npz', allow_pickle=False) as f:
    initial = torch.from_numpy(f['initial_features'])
    commands = torch.from_numpy(f['commands'])
with torch.inference_mode():
    future_features = model.predict(initial, commands)  # [1,59,6144]
```

Input is ONE raw channel-major 384x4x4 DINOv2-small feature grid and H native command
rows. Offset k consumes rows 0..k: offset 1 uses two rows; offset 59 uses all 60.
Three internal slots repeat the one observation, not three RGB observations.
Saved training-only normalization is applied internally; output is raw feature
coordinates. No future targets enter this API; it produces no RGB forecasts,
actions or control scores. No encoder weights, RGB decoder, optimizer or RNG
state are included.

Encoder input contract: facebook/dinov2-small revision
ed25f3a31f01632728cabb09d1542f84ab7b0056; RGB resized to 224x224 with bilinear
antialiasing and ImageNet normalization; remove CLS; pool 16x16 patch tokens in
FP32 to 4x4; channel-major flatten. Frozen caches used CUDA BF16 extraction,
TF32 off. CPU re-encoding is not asserted cache-identical. Encoder/code provenance
is retained in package metadata; arbitrary feature encodings are incompatible.

The relocation proof uses CPU FP32/two threads and all 59 outputs for the fixed
internal-training episode 000011/frame 0 with commands 0..59 in each task. Stored
reference predictions are model outputs for parity, not targets or accuracy
measurements. Run from the bundle:

    python -I runtime.py --output /tmp/iws-unbounded-parity.json

All nine predictions must match exactly. The builder additionally blocks network
connections and reads from the original workspace during its relocated proof
(except installed Python environment libraries). CPU parity is not GPU or
arbitrary-platform numerical certification. No new backend performance is claimed.

Weights/config are copied byte-for-byte from selected frozen packages. Archival
absolute paths inside metadata are identities, not runtime dependencies. The
runtime imports only bundled predictor/vendor code. MIT software license and
upstream attribution are included; no third-party dataset license is asserted.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    build(args.output)
