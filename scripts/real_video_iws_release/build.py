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
REPORTS = ROOT / "reports/real_video_iws/release"
DEFAULT = ROOT / "artifacts/releases/iws_single_observation_local_v1"
COPIED_SOURCES = ["src/shiftwm/__init__.py", "src/shiftwm/upstream.py",
                  "src/shiftwm/real_video_iws/__init__.py", "src/shiftwm/real_video_iws/model.py",
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


def build(destination):
    import numpy as np
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    campaign = module(ROOT / "scripts/real_video_iws/campaign.py", "_release_iws_campaign")
    train = campaign.trainer()
    registration = campaign.check_registration()
    config = read(campaign.CONFIG)
    destination = Path(destination).absolute()
    if not destination.is_relative_to(ROOT / "artifacts/releases"):
        raise ValueError("Local checkpoint binaries must remain under ignored artifacts/releases")
    if destination.exists():
        raise ValueError("Refusing to replace an existing inference bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = Path(tempfile.mkdtemp(prefix=".iws-bundle-pending-", dir=destination.parent))
        sources = {p: copy_exact(ROOT / p, temporary / p) for p in COPIED_SOURCES}
        sources["scripts/real_video_iws_release/runtime.py"] = copy_exact(
            ROOT / "scripts/real_video_iws_release/runtime.py", temporary / "runtime.py")
        sources["scripts/real_video_iws_release/build.py"] = sha(Path(__file__))
        fixtures, fixture_sources, input_values = {}, {}, {}
        for task in sorted(config["tasks"]):
            cache = train.open_cache(config, task)
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
            exported = train.export_inference_package(run, target)
            summary = read(run / "training_summary.json")
            model, _ = train.load_package(run / "best", "cpu")
            initial, commands = input_values[row["task"]]
            with torch.inference_mode():
                expected = model.predict(torch.from_numpy(initial.copy()), torch.from_numpy(commands.copy())).numpy()
            if expected.shape != (1,59,6144) or not np.isfinite(expected).all():
                raise ValueError("Invalid original-model prediction")
            expected_path = "fixtures/" + row["name"] + "_prediction.npz"
            np.savez_compressed(temporary / expected_path, predictions=expected)
            records = {name: binding(package / name) for name in ("model.pt", "config.json", "package_manifest.json")}
            if sha(target / "model.pt") != records["model.pt"]["sha256"]:
                raise ValueError("Export altered selected model bytes")
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
        manifest = {"schema": "iws_local_inference_bundle_v1", "package_kind": train.PACKAGE_KIND,
                    "status": "local_inference_prerelease", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "benchmark_finalization_claimed": False, "reserved_data_access_authorized": False,
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
                "--bundle", str(relocated), "--output", str(proof)], cwd=relocation, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (REPORTS / "relocated_parity.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError("Relocated inference failed; pending bundle retained: " + str(temporary))
            result = read(proof)
            if (result["status"] != "passed" or len(result["models"]) != 27
                    or not result["isolated_python"] or result["network_attempts"] != 0):
                raise ValueError("Incomplete relocation proof")
            write(REPORTS / "relocated_cpu_parity.json", result)
        campaign.check_registration()
        for row in rows:
            for item in row["source_packages"].values():
                if sha(ROOT / item["path"]) != item["sha256"]:
                    raise ValueError("Selected source changed during parity")
        os.rename(temporary, destination)
        outcome = {"status": "passed_local_inference_export", "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "bundle": str(destination.relative_to(ROOT)), "manifest_sha256": sha(destination / "manifest.json"),
                   "models": 27, "selected_epochs": {r["name"]: r["selected_epoch"] for r in rows},
                   "parity": binding(REPORTS / "relocated_cpu_parity.json"),
                   "source_dependencies": sources, "source_registration": binding(campaign.REGISTRATION),
                   "optimizer_or_rng_included": False, "published": False, "encoder_weights_included": False,
                   "future_target_inputs_included": False, "official_validation_payloads_read": 0,
                   "benchmark_finalization_claimed": False, "device": "cpu", "threads": 2,
                   "bundle_bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())}
        write(REPORTS / "local_inference_export.json", outcome)
        print(json.dumps(outcome, indent=2), flush=True)


README = """# IWS single-observation predictors — local inference prerelease

All 27 registered task × arm × seed selected checkpoints are included: PushT
(4 commands), Bimanual Box (14), Bimanual Rope (8); autoregressive, anchored
additive and bounded spatial mixing; seeds 0, 1, 2. Each run completed 30 epochs.
The best checkpoint is the earliest strict minimum of internal-development
equal-trajectory endpoint H60 standardized MSE. This local package is independent
of benchmark finalization and does not assert official-validation or SOTA results.

Use Python with the versions in requirements.txt. No network is needed. The
runtime verifies the complete manifest, package kind, weights and normalization.

```python
from pathlib import Path
import numpy as np
import torch
from runtime import load_model
root = Path(__file__).resolve().parent  # this bundle directory
model, info = load_model('models/pusht_bounded_spatial_mix_s0', root)
with np.load(root / 'fixtures/pusht.npz', allow_pickle=False) as f:
    initial = torch.from_numpy(f['initial_features'])  # [B,6144]
    commands = torch.from_numpy(f['commands'])         # [B,60,4]
with torch.inference_mode():
    raw_feature_forecasts = model.predict(initial, commands)  # [B,59,6144]
```

Inputs are ONE raw DINOv2-small feature grid, channel-major 384×4×4, and native
recorded command rows. At forecast offset k>=1 the prefix contains rows 0..k;
offset 1 uses two rows, offset 59 uses all 60. Three model slots repeat the one
observation; they do not imply three observed frames. The saved model performs
its own frozen training-only feature/command normalization. Outputs are raw
feature vectors, not RGB images, actions or control success predictions.

Encoder weights and an RGB decoder are not bundled. Features must use the
registered DINOv2-small revision/preprocessing recorded in config metadata;
arbitrary feature grids are not interchangeable. Cached training features used
for the included parity fixtures were extracted with the registered CUDA BF16
encoder recipe. CPU FP32 re-encoding is not claimed bitwise cache-equivalent.

The supported parity proof is CPU FP32 with two threads, all 59 offsets, one
fixed authorized training input (episode 000011/frame 0) per task. Expected
prediction fixtures support exact relocation checks only; they are not target
features or new accuracy measurements. Run `python -I runtime.py --output
/tmp/iws-parity.json` from this bundle. All 27 results must match exactly.
CUDA autoregressive prefix consistency previously failed its strict evaluator
gate, so this release does not certify GPU numerical equivalence. Do not relax
that gate or interpret CPU parity as a GPU correctness result.

model.pt/config.json are byte-identical to the frozen selected packages and
retain scientific provenance/history metadata. They contain no optimizer or RNG
continuation state. Source absolute paths in metadata are archival identities;
inference does not read them. The runtime imports only bundled model/vendor code.
The original last checkpoints remain in the training workspace for resumption.
No reserved payloads, training dataset, future targets or generated photos are
included. LICENSE and the pinned upstream vendor notice accompany the code.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    build(args.output)
