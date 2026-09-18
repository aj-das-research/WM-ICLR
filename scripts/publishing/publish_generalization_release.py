#!/usr/bin/env python3
"""Gate, export and publish all 36 completed validation-development predictors.

Never trains, changes registered scientific files, reads test payloads, chooses
winning arms, pushes source branches, or includes optimizer/credential files.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlparse

ARMS = ("slow", "decay", "compact")
MODES = ("framewise", "constant_dynamics", "factorized", "action_free")
REGISTRY = "configs/real_video_development/generalization_v1/registration.json"
FINAL = "reports/real_droid_generalization_finalization.json"
SCRIPT = "scripts/publishing/publish_generalization_release.py"
TAG = "generalization-v1"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def validate_completion_gate(report, registry, registry_hash):
    expected = {(a, m, s) for a in ARMS for m in MODES for s in range(3)}
    if (report.get("status") != "completed" or report.get("completed_training_runs") != 36
            or report.get("completed_validation_evaluations") != 72
            or report.get("registered_checkpoint_parity_passes") != 36
            or report.get("independent_cpu_reload_parity_passes") != 36
            or report.get("registration_sha256") != registry_hash
            or report.get("scope") != "original validation development only; no original or fresh test payloads read"
            or report.get("review", {}).get("status") != "passed"
            or report.get("paper_build", {}).get("status") != "passed"):
        raise ValueError("Completion gate: require 36 runs, 72 evaluations, independent parity and passed finalizer/build")
    for label, rows in (("registry", registry.get("runs", [])), ("report", report.get("runs", []))):
        if (len(rows) != 36 or {(r["arm"], r["mode"], r["seed"]) for r in rows} != expected
                or len({r["name"] for r in rows}) != 36):
            raise ValueError("Completion gate: incomplete or duplicated " + label)
    if {(r["name"], r["arm"], r["mode"], r["seed"]) for r in registry["runs"]} != {
            (r["name"], r["arm"], r["mode"], r["seed"]) for r in report["runs"]}:
        raise ValueError("Completion gate: registry/report identity mismatch")
    for row in report["runs"]:
        training = row["training"]
        if (training.get("status") != "completed" or training.get("completed_epochs") != 30
                or type(training.get("best_epoch")) is not int or not 1 <= training["best_epoch"] <= 30):
            raise ValueError("Completion gate: incomplete training or invalid selected epoch")
    aggregates = report.get("aggregate", [])
    if len(aggregates) != 6 or {(r["arm"], r["horizon"]) for r in aggregates} != {(a, h) for a in ARMS for h in (5, 10)}:
        raise ValueError("Completion gate: all six arm/horizon populations are required")
    for population in aggregates:
        if set(population["methods"]) != set(MODES):
            raise ValueError("Completion gate: a matched method is absent")
        for result in population["methods"].values():
            if not math.isfinite(result["mean"]) or result["mean"] < 0:
                raise ValueError("Completion gate: nonfinite/negative error")


def frozen_sources(root, destination):
    registry = read(root / REGISTRY)
    paths = set(registry["dependencies"]) | {REGISTRY, SCRIPT,
        "scripts/real_video/finalize_campaign.py", "scripts/real_video_development/finalize_generalization.py",
        "scripts/publishing/prepare_public_snapshot.py", "site/assets/DROID-LICENSE.txt",
        "pyproject.toml", "requirements.lock.txt", "LICENSE"}
    paths.update(row["config"] for row in registry["runs"])
    # Preserve every source file copied by the established exporter, even where
    # it is auxiliary rather than a registered scientific dependency.
    paths.update(["src/shiftwm/__init__.py", "src/shiftwm/checkpoint.py",
        "src/shiftwm/real_video/__init__.py", "src/shiftwm/real_video/features.py",
        "src/shiftwm/vendor/lewm/__init__.py", "src/shiftwm/vendor/lewm/jepa.py",
        "src/shiftwm/vendor/lewm/LICENSE", "scripts/real_video/run_campaign.py",
        "scripts/real_video/prepare_droid.py", "scripts/real_video/fetch_droid.py"])
    value = {"status": "frozen_before_execution", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "dependencies": {name: sha(root / name) for name in sorted(paths)},
             "requires": {"runs": 36, "epochs_each": 30, "validation_evaluations": 72,
                          "independent_relocated_models": 36, "finalizer_job": 200177},
             "scope": "Original validation development only; no fresh-test or SOTA claim."}
    target = destination / "export_registration.json"
    if target.exists():
        raise FileExistsError("Exporter registration already exists")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / SCRIPT, destination / "frozen_exporter.py")
    value["frozen_exporter_sha256"] = sha(destination / "frozen_exporter.py")
    write_json(target, value)
    return value


def verify_sources(root, sources):
    for name, expected in sources.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError("Frozen source changed: " + name)


def readme(report):
    lines = ["# Real-DROID generalization development checkpoints", "",
        "Thirty-six validation-selected predictors: three fixed development arms × four methods × three seeds.",
        "Every run completed 30 epochs. Checkpoint selection uses the common five-step validation objective.",
        "The slow arm changes learning rate, decay changes weight decay, and compact jointly changes width/depth/context capacity.",
        "These are ordinary matched optimization/capacity controls, not a new algorithm or a state-of-the-art claim.", "",
        "All 72 evaluations use the original validation recordings only. Neither original test nor the separate fresh holdout selects these models.",
        "Compare methods within an arm: settings and training budgets are matched there. Cross-arm comparisons alter the intervention/budget family.",
        "All positive, negative and inconclusive outcomes remain in the complete reports and table below.", "",
        "| Arm | Horizon | Framewise | Constant dynamics | ShiftWM (ours) | Action-free | Ours gain vs matched Framewise |", 
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregate"]:
        values = row["methods"]
        lines.append(f"| {row['arm']} | {row['horizon']} | " + " | ".join(f"{values[m]['mean']:.6f}" for m in MODES)
                     + f" | {values['factorized']['gain_percent']:+.3f}% |")
    lines += ["", "Lower training-standardized frozen-feature MSE is better. Exploratory session/seed intervals and all methods' differences appear in `reports/finalization.json`; no multiplicity adjustment is implied.", "",
        "## Offline inference", "", "Install the dependencies in `source/requirements.lock.txt`. From this extracted directory:", "", "```python",
        "import sys, torch", 'sys.path[:0] = ["source/src", "source/scripts/real_video"]', "from train import load_package",
        'model, state = load_package("models/compact_factorized_s0", "cpu")', "with torch.inference_mode():",
        "    prediction = model.predict(support_features, past_actions, future_actions)", "```", "",
        "Inputs: support [B,3,1536], past [B,2,35], future [B,K,35]. Outputs are frozen DINO features, not RGB video or physical state.",
        "Use the bundled shared `encoder/dinov2-small` and exact resize/ImageNet normalization/2×2 pooling in `source/src/shiftwm/real_video/features.py` for images.",
        "The original cache used BF16 encoding; CPU FP32 encoding is not guaranteed bitwise equal. No external model checkout/download is required for inference.",
        "The single original-validation feature window and reference predictions verify package parity; they are not additional benchmark outcomes.",
        "Optimizer/RNG states remain in the original training directories and are omitted from this inference-only release.", "",
        "## Limits and licenses", "", "No physical robot control, patient use, fresh-test improvement, universal applicability or SOTA is established by these validation experiments.",
        "Project-authored predictor weights/code use MIT (`licenses/SHIFTWM-MIT.txt`). Unchanged DINOv2 uses Apache 2.0, LeWM source MIT, and DROID-derived data/provenance CC BY 4.0.",
        "All upstream attribution and full licenses are included. Source videos, raw datasets, private credentials and installed environments are excluded.", ""]
    return "\n".join(lines)


def export(root, destination):
    import numpy as np
    import torch
    sys.path[:0] = [str(root / "src"), str(root / "scripts/real_video")]
    old = module(root / "scripts/real_video/finalize_campaign.py", "original_exporter")
    finalizer = module(root / "scripts/real_video_development/finalize_generalization.py", "checked_finalizer")
    scanner = module(root / "scripts/publishing/prepare_public_snapshot.py", "release_secret_scan")
    frozen = read(destination / "export_registration.json")
    verify_sources(root, frozen["dependencies"])
    if sha(__file__) != frozen["frozen_exporter_sha256"]:
        raise ValueError("Running exporter differs from frozen exporter")
    report, registry = read(root / FINAL), read(root / REGISTRY)
    validate_completion_gate(report, registry, sha(root / REGISTRY))
    final_hash = sha(root / FINAL)
    verify_sources(root, report["sources"])
    configs = finalizer.check_registration(registry, root)
    finalizer.require_complete_files(registry, configs, root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    finalized = {row["name"]: row for row in report["runs"]}
    for row in registry["runs"]:
        config = configs[row["name"]]
        run = root / config["output_dir"]
        summary = old.train.validate_completed(run)
        model_config = read(run / "best/config.json")
        if summary != finalized[row["name"]]["training"] or sha(run / "best/model.pt") != finalized[row["name"]]["checkpoint_sha256"]:
            raise ValueError("Selected model differs from passed finalization")
        for horizon in (5, 10):
            path = run / f"validation_h{horizon}.json"
            if sha(path) != report["sources"][str(path.relative_to(root))]:
                raise ValueError("Validation evidence changed")
        records.append({"row": row, "config_path": root / row["config"], "config": config,
                        "run": run, "summary": summary, "identity": model_config["metadata"]["identity"]})
    bundle = destination / "shiftwm-generalization-v1"
    if bundle.exists():
        manifest = read(bundle / "manifest.json")
        if manifest["finalization_sha256"] != final_hash:
            raise ValueError("Existing release was built from another finalization")
        for name, item in manifest["files"].items():
            if sha(bundle / name) != item["sha256"]:
                raise ValueError("Existing release bytes changed")
        return bundle, manifest
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        old.source_bundle(stage, root / REGISTRY, records)
        cache = root / records[0]["config"]["cache_root"]
        cache_manifest = read(cache / "manifest.json")
        old.encoder_bundle(stage, root / "data/pretrained/dinov2-small", cache_manifest["identity"]["encoder"])
        old.checked_copy(root / "site/assets/DROID-LICENSE.txt", stage / "licenses/DROID-CC-BY-4.0.txt")
        for name in (SCRIPT, "scripts/real_video_development/generalization_campaign.py", "scripts/real_video_development/finalize_generalization.py"):
            old.checked_copy(root / name, stage / "source" / name)
        old.checked_copy(destination / "export_registration.json", stage / "provenance/export_registration.json")
        old.checked_copy(root / FINAL, stage / "reports/finalization.json", final_hash)
        old.checked_copy(root / "reports/real_droid_generalization_results.json", stage / "reports/registered_results.json")
        for name in ("comparison.tex", "intervals.tex", "checkpoints.tex", "proof.pdf", "review.json", "sources.json"):
            old.checked_copy(root / "paper/generated/real_video/generalization" / name, stage / "reports/tables" / name)
        data = old.RealVideoDataset(cache, "val", horizon=10, stride=5, verify=True)
        sample = data[0]
        episode = data.episodes[sample["episode_index"]]
        (stage / "verification").mkdir()
        np.savez_compressed(stage / "verification/sample.npz", features=sample["features"].numpy(), actions=sample["actions"].numpy())
        old.train.atomic_json({"split": "val", "episode_id": episode["episode_id"], "session_id": episode["session_id"],
            "window_start": sample["window_start"], "horizon": 10, "source_payload": episode["cameras"]["exterior_image_1_left"],
            "purpose": "offline package parity only; no benchmark selection"}, stage / "verification/sample_provenance.json")
        models = []
        for record in records:
            exported = old.export_model(record, stage, sample)
            exported["arm"] = record["row"]["arm"]
            card = f"""# {exported['run']}

Arm: {exported['arm']}; method: {exported['mode']}; seed: {exported['seed']}.
All 30 training epochs completed. Validation selected epoch {exported['checkpoint_epoch']} with all-five-query MSE {exported['validation_mse']:.10g}.
This is a validation selection statistic, not an untouched-test result.

Frozen-DINO feature forecasting only: support [B,3,1536], past commands [B,2,35], future commands [B,K,35]; output [B,K,1536].
The commands concatenate five chronological recorded Cartesian/gripper command vectors. No RGB decoder, clinical or physical-control validation is included.

Total parameters: {exported['parameter_counts']['total']}; trainable: {exported['parameter_counts']['trainable']}.
Compare with the other three methods in this same arm and their matched settings/seeds. All 36 models and all 72 validation results are retained.
This ordinary optimization/capacity development study does not establish new algorithmic novelty or SOTA; it did not evaluate these checkpoints on original or fresh test data.

Project predictor weights use MIT; upstream encoder, source and derived-data licenses remain unchanged. The shared encoder is at ../../encoder/dinov2-small.
Strict offline loading and relocation checks are recorded in ../../verification/offline_packages.json. Optimizer/RNG files remain in original training runs.
"""
            (stage / exported["path"] / "MODEL_CARD.md").write_text(card)
            for horizon in (5, 10):
                old.checked_copy(record["run"] / f"validation_h{horizon}.json",
                                 stage / "reports/evaluations" / exported["run"] / f"h{horizon}.json")
            models.append(exported)
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2"}
        result = subprocess.run([sys.executable, "-I", "-c", old.OFFLINE_CHECK, str(stage.resolve())],
            cwd=stage, env=environment, text=True, capture_output=True, check=True, timeout=1200)
        verification = json.loads(result.stdout.strip().splitlines()[-1])
        if verification["status"] != "passed" or len(verification["models"]) != 36 or {r["run"] for r in verification["models"]} != {r["run"] for r in models}:
            raise ValueError("Not all 36 relocated packages passed offline parity")
        old.train.atomic_json(verification, stage / "verification/offline_packages.json")
        (stage / "README.md").write_text(readme(report))
        (stage / "MODEL_CARD.md").write_text(readme(report))
        notice = stage / "licenses/THIRD_PARTY_NOTICES.md"
        notice.write_text(notice.read_text().replace("This local bundle does not assert a separate license grant for learned weights.",
            "Project-authored predictor weights are distributed under the included project MIT license."))
        inventory = old.file_inventory(stage)
        findings = []
        for name in inventory:
            path = stage / name
            if path.is_symlink() or path.name == "training_state.pt" or any(p in {".git", ".ssh", "__pycache__", ".venv"} for p in path.parts):
                raise ValueError("Forbidden inference release content: " + name)
            content = path.read_bytes()
            for rule_name, rule in scanner.TOKEN_RULES.items():
                if rule.search(content):
                    findings.append({"path": name, "rule": rule_name})
        if findings:
            raise ValueError("Secret scan findings (values redacted): " + json.dumps(findings))
        verify_sources(root, frozen["dependencies"])
        verify_sources(root, report["sources"])
        if sha(root / FINAL) != final_hash:
            raise ValueError("Finalization report changed during export")
        manifest = {"status": "completed", "kind": "real_droid_generalization_development_inference",
            "finalization_sha256": final_hash, "registration_sha256": sha(root / REGISTRY),
            "exporter_registration_sha256": sha(destination / "export_registration.json"),
            "models": models, "validation_evaluations": 72, "offline_verification": verification,
            "secret_findings": [], "scope": "Original validation development; no fresh-test/SOTA claim",
            "files": inventory, "total_bytes": sum(r["bytes"] for r in inventory.values())}
        old.train.atomic_json(manifest, stage / "manifest.json")
        stage.rename(bundle)
    return bundle, manifest


def archive_bundle(bundle, manifest, destination):
    archive = destination / "shiftwm-generalization-v1.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=4) as compressed, tarfile.open(fileobj=compressed, mode="w|") as tar:
            for path in sorted(bundle.rglob("*")):
                if not path.is_file():
                    continue
                info = tar.gettarinfo(str(path), arcname=bundle.name + "/" + path.relative_to(bundle).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                info.pax_headers = {}
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
        temporary.replace(archive)
    seen = set()
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            parts = Path(item.name).parts
            if not item.isfile() or not parts or parts[0] != bundle.name or ".." in parts or Path(item.name).is_absolute():
                raise ValueError("Unsafe archive member")
            relative = "/".join(parts[1:])
            digest = hashlib.sha256()
            with tar.extractfile(item) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            expected = sha(bundle / "manifest.json") if relative == "manifest.json" else manifest["files"][relative]["sha256"]
            if relative in seen or digest.hexdigest() != expected:
                raise ValueError("Archive member differs from verified inference export")
            seen.add(relative)
    if seen != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Archive file inventory differs")
    for name in ("README.md", "MODEL_CARD.md", "manifest.json"):
        shutil.copy2(bundle / name, destination / name)
    shutil.copy2(bundle / "verification/offline_packages.json", destination / "offline_packages.json")
    names = [archive.name, "README.md", "MODEL_CARD.md", "manifest.json", "offline_packages.json"]
    (destination / "SHA256SUMS").write_text("".join(f"{sha(destination / n)}  {n}\n" for n in names))
    return names + ["SHA256SUMS"]


def publish(destination, names, credential_file):
    import requests
    if credential_file.stat().st_mode & 0o077:
        raise ValueError("Git credential store must be mode 0600")
    credential = next(urlparse(line) for line in credential_file.read_text().splitlines() if urlparse(line).hostname == "github.com")
    client = requests.Session()
    client.headers.update({"Authorization": "Bearer " + unquote(credential.password),
        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "ShiftWM-generalization-release"})
    api = "https://api.github.com/repos/aj-das-research/WM-ICLR"
    response = client.get(api + "/releases/tags/" + TAG, timeout=30)
    if response.status_code == 404:
        commit_response = client.get(api + "/commits/main", timeout=30)
        commit_response.raise_for_status()
        body = "\n".join(["Completed, independently gated original-validation development study.",
            "36 validation-selected predictors (slow/decay/compact × four methods × three seeds), each trained for all 30 epochs; 72 validation evaluations.",
            "Includes the shared frozen DINO encoder, exact offline source, full model cards, licenses, registration, all gains/regressions/uncertainties, and checksums.",
            "Every portable checkpoint passed an independent relocated CPU forecast-parity check before publication. Optimizer states remain in the original training runs.",
            "Compare methods within each matched arm. Optimization/capacity controls are ordinary development interventions, not a new algorithm or SOTA claim.",
            "These checkpoints predict latent image features, not RGB video. This is neither an original-test nor a fresh-holdout evaluation and establishes no physical-robot or clinical performance.",
            "Read the attached model card, complete comparison table and provenance before reuse. Project predictor weights/code use MIT; DINO/LeWM/DROID retain their respective Apache2.0/MIT/CC-BY4.0 terms.",
            "This is a research prerelease. The tagged source commit anchors publication; the archive includes exact inference source pinned in its manifest."])
        response = client.post(api + "/releases", json={"tag_name": TAG, "target_commitish": commit_response.json()["sha"],
            "name": "Generalization v1: all 36 validation-development checkpoints", "body": body,
            "draft": True, "prerelease": True}, timeout=30)
    response.raise_for_status()
    release = response.json()
    known = {asset["name"]: asset for asset in release["assets"]}
    uploaded = []
    for name in names:
        path = destination / name
        expected = sha(path)
        if name in known:
            asset = known[name]
        else:
            if not release["draft"]:
                raise ValueError("Published releases are immutable; refusing to add/replace assets")
            url = release["upload_url"].split("{", 1)[0]
            if urlparse(url).hostname != "uploads.github.com":
                raise ValueError("Unexpected upload host")
            with path.open("rb") as stream:
                response = client.post(url, params={"name": name}, data=stream,
                    headers={"Content-Type": "application/octet-stream"}, timeout=(30, 1200))
            response.raise_for_status()
            asset = response.json()
        if asset["size"] != path.stat().st_size or asset.get("digest") != "sha256:" + expected:
            raise ValueError("Uploaded asset size/digest mismatch")
        uploaded.append({"name": name, "id": asset["id"], "bytes": asset["size"], "sha256": expected,
                         "server_digest_verified": True})
        print(json.dumps({"asset_verified": name, "bytes": asset["size"]}), flush=True)
    if release["draft"]:
        response = client.patch(api + "/releases/" + str(release["id"]), json={"draft": False, "prerelease": True}, timeout=30)
        response.raise_for_status()
        release = response.json()
    assets = {asset["id"]: asset for asset in release["assets"]}
    for asset in uploaded:
        asset["url"] = assets[asset["id"]]["browser_download_url"]
        digest, size = hashlib.sha256(), 0
        with requests.get(asset["url"], stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(4 << 20):
                digest.update(chunk)
                size += len(chunk)
        if size != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
            raise ValueError("Anonymous public download integrity mismatch")
        asset["public_download_verified"] = True
    return {"status": "published_verified", "release_url": release["html_url"], "release_id": release["id"],
            "tag": TAG, "target_commit": release["target_commitish"], "assets": uploaded, "prerelease": True}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--directory", type=Path, default=Path("artifacts/publishing/generalization-release"))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--credential-file", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.directory if args.directory.is_absolute() else root / args.directory
    if args.freeze:
        frozen_sources(root, destination)
        print(json.dumps({"status": "exporter_frozen", "directory": str(destination)}))
        return
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle, manifest = export(root, destination)
        names = archive_bundle(bundle, manifest, destination)
        record = {"status": "verified_local", "models": 36, "validation_evaluations": 72,
                  "offline_parity_passes": len(manifest["offline_verification"]["models"]),
                  "archive_sha256": sha(destination / names[0]), "archive_bytes": (destination / names[0]).stat().st_size,
                  "export_registration_sha256": sha(destination / "export_registration.json"),
                  "finalization_sha256": manifest["finalization_sha256"], "secret_findings": 0,
                  "scope": "All matched original-validation development outcomes; no fresh-test or SOTA claim"}
        if args.publish:
            if args.credential_file is None:
                raise ValueError("Publishing requires the private credential-store path")
            record.update(publish(destination, names, args.credential_file))
        record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(destination / "publication_receipt.json", record)
        write_json(root / "reports/generalization_checkpoint_publication_status.json", record)
        print(json.dumps({"status": record["status"], "models": 36, "archive_bytes": record["archive_bytes"],
                          "release_url": record.get("release_url")}), flush=True)


if __name__ == "__main__":
    main()
