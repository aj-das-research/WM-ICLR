#!/usr/bin/env python3
"""Export all 42 completed simulator models losslessly; optionally publish.

Creates a separate inference-only format. Never changes trained models, frozen
experiments, existing releases, or source/publication Git branches.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlparse

TAG = "simulator-development-v1"
BUNDLE = "shiftwm-simulator-development-v1"
SCRIPT = "scripts/publishing/publish_simulator_release.py"
VERIFIER = "scripts/publishing/verify_simulator_release.py"
REPORT = "reports/completed_extension_results.json"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)
    if sha(source) != sha(destination):
        raise ValueError("Copy checksum mismatch")


def inventory_rows(root):
    rows = []
    for family, registry_path in (("original", "configs/extensions/campaign.json"),
                                   ("geometry", "configs/geometry_revision/campaign.json")):
        for row in read(root / registry_path)["runs"]:
            rows.append({**row, "family": family, "domain": row.get("domain", "drone"),
                "architecture": row.get("architecture", "transformer"),
                "id": family + "/" + Path(row["output"]).name, "registry": registry_path})
    expected = {("original", d, a, m, s) for d in ("drone", "surgery")
        for a in ("transformer", "gru") for m in ("framewise", "constant_dynamics", "factorized") for s in range(3)}
    expected |= {("geometry", "drone", "transformer", m, s)
        for m in ("constant_dynamics", "factorized") for s in range(3)}
    actual = {(r["family"], r["domain"], r["architecture"], r["mode"], r["seed"]) for r in rows}
    if len(rows) != 42 or actual != expected or len({r["id"] for r in rows}) != 42:
        raise ValueError("Require all 36 original and six geometry models exactly once")
    return rows


def build_sources(root, bundle):
    paths = set()
    paths.update(p.relative_to(root) for p in (root / "src/shiftwm").glob("*.py"))
    for directory in ("src/shiftwm/extensions", "src/shiftwm/vendor", "scripts/extensions",
                      "configs/extensions", "configs/geometry_revision"):
        paths.update(p.relative_to(root) for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and (p.suffix in {".py", ".json", ".slurm"} or p.name == "LICENSE"))
    paths.update(Path(name) for name in (SCRIPT, VERIFIER, "tests/test_simulator_release.py",
        "LICENSE", "THIRD_PARTY_NOTICES.md", "requirements.lock.txt", "pyproject.toml",
        "references/world_artifact_sources.json", "reports/domain_extension_protocol.md",
        "reports/geometry_revision_protocol.md", "reports/extension_release_independent_audit.md",
        "reports/extension_release_independent_audit.json"))
    for domain in ("drone", "surgery"):
        for name in ("README.md", "requirements.lock.txt", "setup.sh", "run_python.sh", "provenance.json"):
            relative = Path("environments") / domain / name
            if (root / relative).is_file():
                paths.add(relative)
    hashes = {}
    for relative in sorted(paths):
        copy(root / relative, bundle / "source" / relative)
        hashes[str(relative)] = sha(root / relative)
    for old, new in (("LICENSE", "SHIFTWM-MIT.txt"),
        ("src/shiftwm/vendor/lewm/LICENSE", "LeWM-MIT.txt"),
        ("external/gym-pybullet-drones/LICENSE", "gym-pybullet-drones-MIT.txt"),
        ("external/sofa_env/LICENSE", "sofa_env-MIT.txt"),
        ("environments/surgery/SOFA_v24.06.00_Linux/LICENSE-LGPL.md", "SOFA-LGPL.txt")):
        copy(root / old, bundle / "licenses" / new)
    copy(root / "artifacts/releases/extensions_v1/requirements-inference.txt", bundle / "requirements-inference.txt")
    copy(root / "artifacts/releases/extensions_v1/runtime.json", bundle / "runtime.json")
    copy(root / VERIFIER, bundle / "verify_packages.py")
    write(bundle / "provenance/source_hashes.json", hashes)
    return hashes


def documentation(records):
    epochs = dict(sorted(Counter(r["selected_epoch"] for r in records).items()))
    return f"""# Simulator development checkpoints: 42 portable feature world models

This research prerelease includes every model from two completed development studies:
36 original models (two simulated domains × two predictor families × three methods × three seeds)
and six separately labeled observation-gain revisions (drone transformer, two methods × three seeds).
Every run completed all 30 training epochs. Inference weights are the checkpoint selected by
minimum recursive five-step validation MSE, not necessarily the final epoch. Selected-epoch counts: {epochs}.

## What these models do

Forecast 192-dimensional frozen visual features from three observed frames and candidate commands;
they do **not generate RGB video**, output a robot policy, or demonstrate physical deployment.
The common frozen visual encoder is the pinned LeWorldModel PushT ViT-tiny encoder. The transformer
uses pinned LeWM initialization; GRU is our in-house recurrent control, not Dreamer. Compare methods
within a predictor family because cross-family initialization and parameter counts differ.
`factorized` denotes ShiftWM (ours); framewise and constant dynamics are matched in-house controls.
The six geometry models change the observation-gain range and use the version-2 constructor.

Drone: fixed external-camera planar CF2X simulation with a shared upstream PID controller.
Surgery: LapGym/SOFA simulated deformable tissue positioning, with no patient data or clinical result.

## Offline inference

Install the tested dependencies in `requirements-inference.txt`, then run from this extracted folder:

```python
import sys, numpy as np, torch
sys.path.insert(0, "source/src")
from shiftwm.simulator_release import load_package
model, state = load_package(".", "original/drone_transformer_factorized_s0", "cpu")
with np.load("examples/drone.npz", allow_pickle=False) as sample:
    history = torch.from_numpy(sample["features"].copy()).float()
    past = torch.from_numpy(sample["past_actions"].copy()).float()
    future = torch.from_numpy(sample["future_actions"].copy()).float()
with torch.inference_mode():
    prediction = model.rollout_features(history, past, future)
print(prediction.shape)  # [1, 5, 192]
```

Inputs are history[B,3,192], past actions[B,2,10], future candidate actions[B,K,10]. Each action
block contains five chronological two-dimensional commands; normalization is bundled. RGB input
to `model.rollout(images, past, future)` is float[B,3,3,H,W] in[0,1]. No historical absolute path,
external checkout, simulator installation or model download is needed for inference.

List all model IDs in `models.json`. For example the revision ID is
`geometry/drone_transformer_factorized_s0`. The strict loader selects the appropriate constructor.
Run `python -I -B verify_packages.py --release . --output local-verification.json` for all-model verification.

## Lossless portable format and evidence

`shared/common.pt` holds tensors verified exactly identical across **all 42** source checkpoints;
each `models/.../delta.pt` contains the remaining tensors and original configuration/selected-epoch metadata.
This is lossless storage sharing, not pruning, quantization, distillation or additional training.
Hash checks, nonoverlapping state keys, strict model loading and complete source tensor fingerprints protect reconstruction.
Optimizer/RNG state remains in the original training workspace; it is intentionally absent here.
All 42 models are checked in a separate relocated CPU process with network access disabled, against original
source predictions from both cached observations and RGB observations. The two genuine validation support
snippets serve engineering parity only. They do not add benchmark observations or establish model quality.
RGB encoding versus historical cached features is checked with rtol0.005/atol0.002; same-input source/export
CPU predictions and all reconstructed tensors must match exactly. This tests the recorded Python3.11/Linux
environment, not every platform or backend.

## Complete results, including regressions

Read `reports/completed_extension_results.md` and its machine-readable companion. These are **development**
tasks, with eight common simulated tasks per domain and three model seeds (24 task–seed instances,
not 24 independent tasks). Original drone-transformer and surgery-GRU point gains are +4.17 percentage
points; drone-GRU and surgery-transformer comparisons favor baselines. No original paired interval has
a strictly positive lower bound. The wide-gain revision is +12.50 points against its matched constant-dynamics
control, with interval[0,37.50]; it is a follow-up on the same development tasks and remains exploratory.
No SOTA, real-flight, real-surgical, safety, universal transfer, or independent-test advantage is established.
The historical completed-results report accurately describes publication state at its generation time;
this release is a later inference-only publication and does not include the separate AdaJEPA weights.

## License and attribution

Project-authored source, newly trained predictor additions and these two generated simulator verification
snippets are explicitly distributed under MIT; see `MODEL_LICENSE.md` and `licenses/SHIFTWM-MIT.txt`.
Unchanged upstream LeWM source/model components retain their original MIT notices. The simulator source
notices are preserved; no simulator executable, installed environment, complete raw dataset, credentials,
or optimizer state is included. SOFA's LGPL notice is attribution for the separately installed simulator,
not a relicensing of its binary. Full pinned upstream model/source provenance is in
`source/references/world_artifact_sources.json`, `source/THIRD_PARTY_NOTICES.md`, and `source/src/shiftwm/vendor/lewm/NOTICE.json`.
"""


def export(root, destination):
    import numpy as np
    import torch
    sys.path.insert(0, str(root / "src"))
    from shiftwm.extensions.checkpoint import read_package, load_package
    from shiftwm.extensions.geometry_revision import load_geometry_package
    from shiftwm.simulator_release import KIND
    old = module(root / "scripts/extensions/export_packages.py", "sim_original_exporter")
    reporting = module(root / "scripts/extensions/report_completed_results.py", "sim_completion_gate")
    verifier = module(root / VERIFIER, "sim_portability_verifier")
    scanner = module(root / "scripts/publishing/prepare_public_snapshot.py", "sim_secret_scanner")
    rows = inventory_rows(root)
    torch.set_num_threads(1)
    os.chdir(root)
    bundle = destination / BUNDLE
    if bundle.exists():
        manifest = read(bundle / "manifest.json")
        if manifest.get("status") != "verified" or manifest.get("model_count") != 42:
            raise ValueError("Existing artifact has not passed the complete gate")
        for name, row in manifest["files"].items():
            if sha(bundle / name) != row["sha256"]:
                raise ValueError("Previously verified artifact changed")
        return bundle, manifest
    report = read(root / REPORT)
    if report["status"] != "verified_complete" or report["completion"]["original_full_30_epoch_runs"] != 36 or report["completion"]["geometry_full_30_epoch_runs"] != 6:
        raise ValueError("Completed simulator evidence is required")
    # Recheck the entire published evidence ledger, including adverse outcomes;
    # this reads existing artifacts but performs no new benchmark/selection.
    for name, expected in report["source_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Completed evidence changed: " + name)
    report_hash = sha(root / REPORT)
    finalized = {(family, r["name"]): r for family, key in (("original", "original_runs"), ("geometry", "geometry_runs")) for r in report[key]}
    common = None
    for row in rows:
        gate = reporting.verify_training(row, geometry=row["family"] == "geometry")
        expected = finalized[(row["family"], Path(row["output"]).name)]
        if gate["training_summary"] != expected["training_summary"] or gate["checkpoint_sha256"] != expected["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from the completed result evidence")
        _, state = read_package(root / row["output"] / "best", require_training=True)
        tensors = state["state_dict"]
        if common is None:
            common = {k: v.detach().clone() for k, v in tensors.items()}
        else:
            for key in list(common):
                if key not in tensors or common[key].dtype != tensors[key].dtype or common[key].shape != tensors[key].shape or not torch.equal(common[key], tensors[key]):
                    del common[key]
        row["completion"] = gate["training_summary"]
        row["source_model_sha256"] = gate["checkpoint_sha256"]
        print(json.dumps({"completion_verified": row["id"], "common_tensors_remaining": len(common)}), flush=True)
        del state, tensors
    if not common:
        raise ValueError("No exact common tensor bank")
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as temporary:
        stage = Path(temporary) / BUNDLE
        stage.mkdir()
        source_hashes = build_sources(root, stage)
        (stage / "shared").mkdir()
        torch.save(common, stage / "shared/common.pt")
        for domain in ("drone", "surgery"):
            run = next(r for r in rows if r["domain"] == domain)
            old.example_for(domain, read(root / run["config"]), stage)
        (stage / "verification/references").mkdir(parents=True)
        records = []
        for row in rows:
            run = root / row["output"]
            source, state = read_package(run / "best", require_training=False)
            if sha(source / "model.pt") != row["source_model_sha256"]:
                raise ValueError("Source checkpoint changed during export")
            target = stage / "models" / row["id"]
            target.mkdir(parents=True)
            for key, value in common.items():
                if not torch.equal(state["state_dict"][key], value):
                    raise ValueError("Shared bank is not exact")
            payload = {key: state[key] for key in ("config", "epoch", "step", "best_metric")}
            payload["state_dict"] = {key: value.detach().cpu().clone() for key, value in state["state_dict"].items() if key not in common}
            torch.save(payload, target / "delta.pt")
            write(target / "config.json", state["config"])
            for name in ("training_summary.json", "run_config.json", "metrics.jsonl"):
                copy(run / name, target / name)
            loader = load_geometry_package if row["family"] == "geometry" else load_package
            model, _ = loader(source, "cpu")
            example = stage / "examples" / (row["domain"] + ".npz")
            with np.load(example, allow_pickle=False) as values:
                features = torch.from_numpy(values["features"].copy()).float()
                past = torch.from_numpy(values["past_actions"].copy()).float()
                future = torch.from_numpy(values["future_actions"].copy()).float()
                images = torch.from_numpy(values["support_rgb"].copy()).permute(0, 1, 4, 2, 3).float() / 255
            reference = stage / "verification/references" / (row["id"].replace("/", "--") + ".npz")
            with torch.inference_mode():
                outputs = {"cached": model.rollout_features(features, past, future).numpy(),
                    "rgb": model.rollout(images, past, future).numpy(), "encoded": model.encode_images(images).numpy()}
            np.savez_compressed(reference, **outputs)
            record = {key: row[key] for key in ("id", "family", "domain", "architecture", "mode", "seed", "source_model_sha256")}
            record.update({"selected_epoch": state["epoch"], "completed_epochs": 30,
                "training_identity": row["completion"]["training_identity"],
                "best_validation_recursive_mse": row["completion"]["best_validation_prediction_loss"],
                "source_tensor_sha256": verifier.tensor_digest(state["state_dict"]),
                "state_tensor_count": len(state["state_dict"])})
            for key, path in (("delta", target / "delta.pt"), ("config", target / "config.json"),
                              ("example", example), ("reference", reference)):
                record[key] = {"path": str(path.relative_to(stage)), "sha256": sha(path), "bytes": path.stat().st_size}
            write(target / "release_record.json", record)
            (target / "MODEL_CARD.md").write_text(f"# {row['id']}\n\n"
                f"Completed all 30 epochs; validation-selected epoch {state['epoch']}. "
                f"Recursive validation MSE {record['best_validation_recursive_mse']:.9g}.\n\n"
                f"Method: {'ShiftWM (ours)' if row['mode'] == 'factorized' else row['mode']}; "
                f"predictor: {row['architecture']}; seed {row['seed']}; scope: {row['family']} development study.\n\n"
                "This predicts 192-dimensional visual features, not RGB video or policy actions. "
                "Simulation only, with mixed development evidence; no clinical, real-flight or SOTA claim. "
                "The GRU is an in-house control, not Dreamer. Geometry packages use the version-2 observation-gain constructor.\n\n"
                "See the release root README.md, MODEL_LICENSE.md, models.json, full completed-results reports and "
                "offline_verification.json for inputs, limitations, provenance and exact source/export parity. "
                "This lossless inference delta requires the release's hash-bound shared/common.pt. "
                "Optimizer/RNG state is intentionally excluded.\n")
            records.append(record)
            print(json.dumps({"inference_exported": row["id"], "selected_epoch": state["epoch"]}), flush=True)
            del model, state, payload
        write(stage / "models.json", {"kind": KIND, "format_version": 1, "models": records,
            "shared": {"path": "shared/common.pt", "sha256": sha(stage / "shared/common.pt"),
                       "tensor_count": len(common), "bytes": (stage / "shared/common.pt").stat().st_size}})
        (stage / "README.md").write_text(documentation(records))
        (stage / "MODEL_LICENSE.md").write_text("# Model and verification snippet distribution notice\n\n"
            "Copyright (c) 2026 ShiftWM contributors. The project's newly trained predictor additions, "
            "project source, and the two locally generated simulator verification snippets included here "
            "are distributed under the full MIT terms in licenses/SHIFTWM-MIT.txt.\n\n"
            "Unchanged pretrained LeWorldModel components and vendored source retain their original MIT "
            "license/attribution (licenses/LeWM-MIT.txt and source/references/world_artifact_sources.json). "
            "The exact initialization is quentinll/lewm-pusht revision 22b330c28c27ead4bfd1888615af1340e3fe9052.\n\n"
            "MIT simulator notices for gym-pybullet-drones and sofa_env and the separate SOFA LGPL notice "
            "are preserved under licenses/. No simulator binary is bundled or relicensed. "
            "The project snippet license is an explicit distribution notice; it is not inferred from the simulator code license. "
            "No complete raw dataset, patient information, or real flight recordings are distributed.\n")
        for name in ("completed_extension_results.json", "completed_extension_results.md"):
            copy(root / "reports" / name, stage / "reports" / name)
        # Retain every development result and identity; exclude videos/raw NPZ traces.
        for relative, expected in report["source_sha256"].items():
            path = Path(relative)
            if path.parts[0] == "results" and path.suffix == ".json":
                copy(root / path, stage / "evidence" / path)
        run = subprocess.run([sys.executable, "-I", "-B", str(stage / "verify_packages.py"),
            "--release", str(stage), "--output", str(stage / "offline_verification.json")],
            cwd=temporary, env={**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
        (destination / "offline_verification.log").write_text(run.stdout)
        if run.returncode != 0:
            raise RuntimeError("Independent relocated verification failed; inspect offline_verification.log")
        verification = read(stage / "offline_verification.json")
        if verification["status"] != "passed" or verification["verified_models"] != 42:
            raise ValueError("All-model portability gate failed")
        files, findings = {}, []
        for path in sorted(stage.rglob("*")):
            if path.is_symlink():
                raise ValueError("Unexpected symlink")
            if not path.is_file():
                continue
            relative = str(path.relative_to(stage))
            if "__pycache__" in path.parts or path.name.endswith(".pyc") or path.name == "training_state.pt":
                raise ValueError("Forbidden inference release file: " + relative)
            content = path.read_bytes()
            for label, expression in scanner.TOKEN_RULES.items():
                if expression.search(content):
                    findings.append({"path": relative, "rule": label})
            files[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        if findings:
            raise ValueError("Secret scan findings; values redacted: " + json.dumps(findings))
        for relative, expected in source_hashes.items():
            if sha(root / relative) != expected:
                raise ValueError("Source changed while packaging: " + relative)
        if sha(root / REPORT) != report_hash:
            raise ValueError("Completed evidence report changed")
        manifest = {"status": "verified", "kind": KIND, "model_count": 42, "original_models": 36,
            "geometry_models": 6, "epochs_each": 30, "selected_epoch_counts": dict(Counter(r["selected_epoch"] for r in records)),
            "models": records, "shared_tensor_count": len(common),
            "completion_report_sha256": report_hash, "source_hashes": source_hashes,
            "files": files, "total_bytes": sum(x["bytes"] for x in files.values()),
            "offline_verification": {"status": "passed", "verified_models": 42,
                "exact_tensor_parity": True, "exact_cached_and_rgb_cpu_prediction_parity": True},
            "secret_findings": [], "scope": "Simulated development tasks; mixed results; no SOTA or real deployment claim"}
        write(stage / "manifest.json", manifest)
        stage.rename(bundle)
    return bundle, manifest


def archive(bundle, manifest, destination):
    path = destination / (BUNDLE + ".tar.gz")
    if not path.exists():
        partial = path.with_suffix(".partial")
        with partial.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=4) as zipped, tarfile.open(fileobj=zipped, mode="w|") as tar:
            for member in sorted(bundle.rglob("*")):
                if not member.is_file():
                    continue
                info = tar.gettarinfo(str(member), arcname=bundle.name + "/" + str(member.relative_to(bundle)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode, info.pax_headers = 0o644, {}
                with member.open("rb") as stream:
                    tar.addfile(info, stream)
        partial.replace(path)
    if path.stat().st_size >= 1_900_000_000:
        raise ValueError("Archive exceeds conservative 1.9GB cap; split by domain before publishing")
    seen = set()
    with tarfile.open(path, "r|gz") as tar:
        for item in tar:
            parts = Path(item.name).parts
            if not item.isfile() or not parts or parts[0] != bundle.name or ".." in parts or Path(item.name).is_absolute():
                raise ValueError("Unsafe archive member")
            name = "/".join(parts[1:])
            expected = sha(bundle / "manifest.json") if name == "manifest.json" else manifest["files"][name]["sha256"]
            digest = hashlib.sha256()
            with tar.extractfile(item) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            if name in seen or digest.hexdigest() != expected:
                raise ValueError("Archive checksum or duplicate-entry failure")
            seen.add(name)
    if seen != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Archive inventory mismatch")
    names = [path.name, "README.md", "MODEL_LICENSE.md", "manifest.json", "offline_verification.json"]
    for name in names[1:]:
        copy(bundle / name, destination / name)
    (destination / "SHA256SUMS").write_text("".join(f"{sha(destination / name)}  {name}\n" for name in names))
    return names + ["SHA256SUMS"]


def publish(destination, names, credential_file):
    import requests
    if credential_file.stat().st_mode & 0o077:
        raise ValueError("Credential store must be mode 0600")
    credential = next(urlparse(line) for line in credential_file.read_text().splitlines() if urlparse(line).hostname == "github.com")
    client = requests.Session()
    client.headers.update({"Authorization": "Bearer " + unquote(credential.password),
        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ShiftWM-simulator-research-release"})
    api = "https://api.github.com/repos/aj-das-research/WM-ICLR"
    response = client.get(api + "/releases/tags/" + TAG, timeout=30)
    if response.status_code == 404:
        commit = client.get(api + "/commits/main", timeout=30)
        commit.raise_for_status()
        body = "\n\n".join(["42 completed simulator-development feature world models: 36 original (two domains × two predictor families × three methods × three seeds) and six separately labeled drone observation-gain revisions.",
            "Every run completed 30 epochs; checkpoints were selected by recursive validation MSE. Includes a losslessly shared frozen LeWM visual encoder, complete inference source, model cards, licenses, registrations, all development gains/regressions, checksums, and genuine validation snippets for engineering parity.",
            "All 42 reconstructed state dictionaries and both cached-input and RGB-input CPU forecasts match their original checkpoints exactly in an independent relocated process with network disabled. No optimizer/RNG state, complete raw dataset, simulator executable, or credentials is included.",
            "These models predict 192-dimensional visual features, not RGB video or robot actions. Drone and deformable tissue tasks are simulations, not real flight, patient data or clinical experiments. GRU is an in-house recurrent control, not Dreamer.",
            "Results are mixed on eight development tasks per domain: original drone-GRU and surgery-transformer comparisons favor baselines, and no original paired interval has a strictly positive lower bound. The wide-gain drone follow-up is exploratory (+12.50pp versus its matched constant-dynamics control, interval [0,37.50]); it is not new held-out or SOTA evidence.",
            "Project-authored predictor additions and the two generated verification snippets use MIT. Unchanged LeWM components and simulator source retain their upstream notices. Read README, MODEL_LICENSE and the complete results before reuse. This is a research prerelease."])
        response = client.post(api + "/releases", json={"tag_name": TAG,
            "target_commitish": commit.json()["sha"], "name": "Simulator development v1: all 42 portable checkpoints",
            "body": body, "draft": True, "prerelease": True}, timeout=30)
    response.raise_for_status()
    release = response.json()
    known = {x["name"]: x for x in release["assets"]}
    assets = []
    for name in names:
        path, expected = destination / name, sha(destination / name)
        if name in known:
            asset = known[name]
        else:
            if not release["draft"]:
                raise ValueError("Published releases are immutable")
            url = release["upload_url"].split("{", 1)[0]
            if urlparse(url).hostname != "uploads.github.com":
                raise ValueError("Unexpected release upload host")
            with path.open("rb") as stream:
                response = client.post(url, params={"name": name}, data=stream,
                    headers={"Content-Type": "application/octet-stream"}, timeout=(30, 1800))
            response.raise_for_status()
            asset = response.json()
        if asset["size"] != path.stat().st_size or asset.get("digest") != "sha256:" + expected:
            raise ValueError("Remote asset digest/size mismatch")
        assets.append({"id": asset["id"], "name": name, "bytes": asset["size"], "sha256": expected,
                       "server_digest_verified": True})
        print(json.dumps({"uploaded_verified": name, "bytes": asset["size"]}), flush=True)
    if release["draft"]:
        response = client.patch(api + "/releases/" + str(release["id"]), json={"draft": False, "prerelease": True}, timeout=30)
        response.raise_for_status()
        release = response.json()
    published = {x["id"]: x for x in release["assets"]}
    for asset in assets:
        asset["url"] = published[asset["id"]]["browser_download_url"]
        digest, count = hashlib.sha256(), 0
        with requests.get(asset["url"], stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(4 << 20):
                digest.update(chunk)
                count += len(chunk)
        if count != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
            raise ValueError("Anonymous public download verification failed")
        asset["public_download_verified"] = True
    return {"status": "published_verified", "tag": TAG, "prerelease": True,
        "release_url": release["html_url"], "release_id": release["id"],
        "target_commit": release["target_commitish"], "assets": assets}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--directory", type=Path, default=Path("artifacts/publishing/simulator-release"))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--credential-file", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.directory if args.directory.is_absolute() else root / args.directory
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle, manifest = export(root, destination)
        names = archive(bundle, manifest, destination)
        result = {"status": "verified_local", "model_count": 42, "original_models": 36, "geometry_models": 6,
            "offline_parity_passes": 42, "epochs_each": 30, "archive_sha256": sha(destination / names[0]),
            "archive_bytes": (destination / names[0]).stat().st_size, "shared_tensor_count": manifest["shared_tensor_count"],
            "secret_findings": 0, "completion_report_sha256": manifest["completion_report_sha256"],
            "scope": manifest["scope"]}
        if args.publish:
            if not args.credential_file:
                raise ValueError("Publishing requires a private credential-file path")
            result.update(publish(destination, names, args.credential_file))
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(destination / "publication_receipt.json", result)
        write(root / "reports/simulator_checkpoint_publication_status.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
