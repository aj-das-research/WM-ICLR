#!/usr/bin/env python3
"""Stage and optionally publish the independently audited 36-model IWS release."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("reports/real_video_iws_release_v2")
BUNDLE = Path("artifacts/releases/iws_single_observation_rowwise_local_v2")
STAGE = Path("artifacts/publishing/iws_rowwise_v2")
TAG = "iws-rowwise-v2"
REPOSITORY = "aj-das-research/WM-ICLR"
REGISTRATION_SHA = "aea28e75dee31876b76ba590ed698c7b9db0d8ddfab93d80d8724dff62dbe41a"
SOURCE_REVIEW_SHA = "9c58f3d6485bfb63434a4518726e3ef70a82d9782ad55988a8f33ef4dab20853"
BACKEND = "one_command_row_gru_cpu_fp32_v1"
ARCHIVE_NAME = BUNDLE.name + ".tar.gz"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected independent regular release file")
    return path


def scan_tree(root, directory):
    scanner = module(root / "scripts/publishing/prepare_public_snapshot.py", "iws_publisher_scanner")
    transport = module(root / "scripts/publishing/spatial_release.py", "iws_publisher_inventory")
    return transport.inventory(directory, scanner)


def validate_proof(proof, manifest):
    require(proof.get("schema") == "iws_rowwise_relocation_proof_v2"
            and proof.get("status") == "passed" and proof.get("backend") == BACKEND
            and proof.get("device") == "cpu" and proof.get("precision") == "float32"
            and proof.get("threads") == 8 and proof.get("interop_threads") == 1,
            "Wrong completed numerical runtime proof")
    require(proof.get("isolated_python") is True
            and proof.get("original_workspace_reads_denied") is True
            and proof.get("network_attempts") == 0
            and proof.get("original_workspace_read_attempts") == 0
            and proof.get("dataset_payloads_read") == 0
            and proof.get("accuracy_evaluation_performed") is False,
            "Offline synthetic-only proof is incomplete")
    rows = {row["name"]: row for row in manifest["models"]}
    models = proof.get("models", [])
    require(len(rows) == len(models) == 36 and {row["name"] for row in models} == set(rows),
            "Exact 36-model proof required")
    for model in models:
        row = rows[model["name"]]
        require(model.get("status") == "passed" and model.get("backend") == BACKEND
                and model.get("epoch") == row["selected_epoch"]
                and model.get("checkpoint_sha256") == row["checkpoint_sha256"]
                and model.get("selected_package_kind") == row["package_kind"],
                "Proof selected-model identity differs")
        cases = model.get("cases", [])
        require(len(cases) == 3 and {c["batch_size"] for c in cases} == {1, 64, 8},
                "Missing or duplicate synthetic batch proof")
        for case in cases:
            require(case.get("shape") == [case["batch_size"], 59, 6144]
                    and case.get("byte_identity_verified") is True
                    and case.get("maximum_absolute_difference") == 0
                    and re.fullmatch(r"[0-9a-f]{64}", case.get("prediction_sha256", "")),
                    "Full-forecast parity failed")
            checks = case.get("prefix_checks", [])
            require(len(checks) == 3 and {p["horizon"] for p in checks} == {15, 30, 45},
                    "Missing or duplicate prefix invocation")
            for prefix in checks:
                ratio = prefix.get("maximum_tolerance_ratio", float("inf"))
                gap = prefix.get("maximum_absolute_difference", float("inf"))
                require(prefix.get("allclose") is True and math.isfinite(ratio)
                        and 0 <= ratio <= 1 and math.isfinite(gap) and gap >= 0,
                        "Unchanged prefix tolerance was not met")


def validate_independent_review(review, bindings):
    require(review.get("status") == "passed" and review.get("models") == 36,
            "Independent completed-artifact review required")
    for key in ("readiness_sha256", "manifest_sha256", "archive_sha256", "proof_sha256",
                "registration_sha256", "source_review_sha256"):
        require(review.get(key) == bindings[key], "Independent artifact review is stale: " + key)


def verify_ready(root=ROOT):
    root = Path(root).resolve()
    report = root / REPORT
    paths = {name: regular(report / (name + ".json")) for name in
             ("registration", "source_review", "readiness", "relocated_cpu_parity", "independent_execution_review")}
    require(sha(paths["registration"]) == REGISTRATION_SHA
            and sha(paths["source_review"]) == SOURCE_REVIEW_SHA, "Wrong reviewed release source")
    registration, source_review = read(paths["registration"]), read(paths["source_review"])
    require(source_review.get("status") == "passed"
            and source_review.get("registration_sha256") == REGISTRATION_SHA
            and source_review.get("source_sha256") == registration["dependencies"], "Source review binding differs")
    for name, digest in registration["dependencies"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Unsafe registered source path")
        require(sha(regular(root / relative)) == digest, "Registered release source changed: " + name)
    bundle = root / BUNDLE
    archive = regular(bundle.with_suffix(".tar.gz"))
    require(not bundle.is_symlink() and archive.stat().st_size < 1_900_000_000,
            "Unsafe bundle or oversized GitHub asset")
    runtime = module(root / "scripts/real_video_iws_release_v2/runtime.py", "iws_publisher_runtime")
    manifest = runtime.verify_bundle(bundle)
    require(manifest["models"] == registration["models"], "Selected bundle roster changed")
    readiness, proof = read(paths["readiness"]), read(paths["relocated_cpu_parity"])
    bindings = {"readiness_sha256": sha(paths["readiness"]), "manifest_sha256": sha(bundle / "manifest.json"),
                "archive_sha256": sha(archive), "proof_sha256": sha(paths["relocated_cpu_parity"]),
                "registration_sha256": REGISTRATION_SHA, "source_review_sha256": SOURCE_REVIEW_SHA}
    require(readiness.get("status") == "passed_local_release" and readiness.get("published") is False
            and readiness.get("models") == 36 and readiness.get("backend") == BACKEND
            and readiness.get("bundle") == str(BUNDLE)
            and readiness.get("archive") == str(BUNDLE.with_suffix(".tar.gz"))
            and readiness.get("archive_bytes") == archive.stat().st_size
            and readiness.get("dataset_payloads_read") == 0
            and readiness.get("accuracy_evaluation_performed") is False
            and readiness.get("old_bundle_manifests_unchanged") is True,
            "Local release readiness is incomplete")
    for key in ("manifest_sha256", "archive_sha256", "proof_sha256", "registration_sha256", "source_review_sha256"):
        require(readiness.get(key) == bindings[key], "Readiness digest differs: " + key)
    require(proof.get("manifest_sha256") == bindings["manifest_sha256"], "Relocation proof binds another bundle")
    validate_proof(proof, manifest)
    validate_independent_review(read(paths["independent_execution_review"]), bindings)
    inventory = scan_tree(root, bundle)
    require(set(inventory) == set(manifest["files"]) | {"manifest.json", "SHA256SUMS"},
            "Secret-audited inventory differs")
    return {"root": root, "bundle": bundle, "archive": archive, "paths": paths,
            "bindings": bindings, "independent_execution_review_sha256": sha(paths["independent_execution_review"]),
            "secret_findings": 0, "payload_files": len(inventory)}


def stage_release(root=ROOT, directory=None):
    checked = verify_ready(root)
    root = checked["root"]
    destination = Path(directory) if directory else root / STAGE
    require(not destination.is_symlink() and destination.resolve() == (root / STAGE).resolve(),
            "Use the separate IWS publication staging directory")
    destination = destination.resolve()
    sources = {ARCHIVE_NAME: checked["archive"]}
    sources.update({name: checked["bundle"] / name for name in
                    ("README.md", "MODEL_CARD.md", "MODEL_LICENSE.md", "manifest.json")})
    sources.update({name + ".json": path for name, path in checked["paths"].items()})
    source_files = {str(path.relative_to(root)): sha(path) for path in checked["paths"].values()}
    source_files[str(Path(__file__).resolve().relative_to(root))] = sha(__file__)
    for name in ("scripts/publishing/spatial_release.py", "scripts/publishing/prepare_public_snapshot.py"):
        source_files[name] = sha(root / name)
    publication = {"status": "verified_local", "tag": TAG, "model_count": 36, "backend": BACKEND,
                   **checked["bindings"], "independent_execution_review_sha256": checked["independent_execution_review_sha256"],
                   "source_files_sha256": source_files, "secret_findings": 0,
                   "assets_sha256": {name: sha(path) for name, path in sources.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ".iws_rowwise_v2.stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = Path(tempfile.mkdtemp(prefix=".iws-stage-", dir=destination.parent))
        try:
            for name, path in sources.items():
                shutil.copyfile(regular(path), temporary / name)
                require(sha(temporary / name) == publication["assets_sha256"][name], "Staging copy changed")
            (temporary / "PUBLICATION_MANIFEST.json").write_text(encoded(publication))
            names = sorted([*sources, "PUBLICATION_MANIFEST.json"])
            (temporary / "SHA256SUMS").write_text("".join(f"{sha(temporary / name)}  {name}\n" for name in names))
            names.append("SHA256SUMS")
            require(set(scan_tree(root, temporary)) == set(names), "Staged secret-audit inventory differs")
            if destination.exists():
                require(destination.is_dir() and {p.name for p in destination.iterdir()} == set(names),
                        "Existing staging inventory differs")
                for name in names:
                    require(sha(regular(destination / name)) == sha(temporary / name), "Existing staged asset differs")
            else:
                temporary.rename(destination)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
    return destination, names, publication


def validate_remote_release(release, names, target_commit):
    require(release.get("tag_name") == TAG and release.get("prerelease") is True
            and type(release.get("draft")) is bool and release.get("target_commitish") == target_commit,
            "Existing release has another tag/target/scope")
    assets = release.get("assets", [])
    require(len({a["name"] for a in assets}) == len(assets)
            and {a["name"] for a in assets} <= set(names), "Unexpected or duplicate remote assets")
    link = urlparse(release.get("html_url", ""))
    require(link.scheme == "https" and link.netloc == "github.com"
            and link.path.startswith(f"/{REPOSITORY}/releases/") and not link.query,
            "Unexpected release URL")
    if not release["draft"]:
        require(release["html_url"] == f"https://github.com/{REPOSITORY}/releases/tag/{TAG}",
                "Unexpected public release URL")


def validate_staged_assets(destination, names, publication):
    require(len(names) == len(set(names)) and set(names) == set(publication["assets_sha256"]) |
            {"PUBLICATION_MANIFEST.json", "SHA256SUMS"}, "Staged asset list differs")
    require({p.name for p in destination.iterdir()} == set(names), "Unexpected staged file")
    require(read(regular(destination / "PUBLICATION_MANIFEST.json")) == publication,
            "Staged publication contract differs")
    for name, digest in publication["assets_sha256"].items():
        require(Path(name).name == name and sha(regular(destination / name)) == digest,
                "Staged asset changed: " + name)
    checksums = "".join(f"{sha(regular(destination / name))}  {name}\n" for name in
                        sorted(set(names) - {"SHA256SUMS"}))
    require(regular(destination / "SHA256SUMS").read_text() == checksums, "Staged checksums differ")
    require(publication["assets_sha256"].get(ARCHIVE_NAME) == publication["archive_sha256"],
            "Staged archive differs from reviewed readiness")


def verify_public_assets(assets, requests_module):
    for name, asset in assets.items():
        parsed = urlparse(asset["url"])
        require(parsed.scheme == "https" and parsed.netloc == "github.com" and not parsed.query
                and unquote(parsed.path) == f"/{REPOSITORY}/releases/download/{TAG}/{name}",
                "Unexpected public download URL")
        digest, count = hashlib.sha256(), 0
        with requests_module.get(asset["url"], stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(4 << 20):
                digest.update(chunk); count += len(chunk)
        require(count == asset["bytes"] and digest.hexdigest() == asset["sha256"],
                "Anonymous public download differs: " + name)
        asset["public_download_verified"] = True


def publish(destination, names, publication, credential_file, target_commit=None):
    import requests
    validate_staged_assets(destination, names, publication)
    credential_file = regular(credential_file)
    require(credential_file.stat().st_mode & 0o077 == 0, "Credential store must be private (0600)")
    credentials = [urlparse(line) for line in credential_file.read_text().splitlines()
                   if urlparse(line).hostname == "github.com"]
    require(len(credentials) == 1 and credentials[0].password, "Expected one private GitHub credential")
    client = requests.Session()
    client.headers.update({"Authorization": "Bearer " + unquote(credentials[0].password),
                           "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                           "User-Agent": "ShiftWM-IWS-rowwise-release"})
    api = f"https://api.github.com/repos/{REPOSITORY}"
    require(target_commit is None or re.fullmatch(r"[0-9a-f]{40}", target_commit), "Expected full target commit SHA")
    response = client.get(api + "/commits/" + (target_commit or "main"), timeout=30)
    response.raise_for_status(); commit = response.json()["sha"]
    require(re.fullmatch(r"[0-9a-f]{40}", commit) and (target_commit is None or commit == target_commit),
            "Remote target commit differs")
    response = client.get(api + "/releases/tags/" + TAG, timeout=30)
    if response.status_code == 404:
        body = "\n\n".join([
            "All 36 completed IWS predictors: three manipulation tasks × four learned methods × three seeds. Every run completed 30 epochs; the unchanged selected weights and model cards identify the selected development epoch.",
            "This is the explicit one_command_row_gru_cpu_fp32_v1 runtime: CPU FP32, eight intra-op threads, one inter-op thread. It preserves trained weights and source equations. The earlier prefix failure cause remains unresolved; this release does not silently replace the original native-backend bundles.",
            "All 36 models passed archive-extracted, isolated offline parity on independent synthetic inputs at batches 1/64/8 and all 59 forecast offsets, plus separate H15/H30/H45 checks with unchanged tolerances. No genuine dataset fixture, future target, encoder weights, optimizer state or credential is included.",
            "Complete development and reserved-recovery summaries retain every method, mixed primary bounded-versus-additive results and secondary no-tanh comparisons. These are visual-feature forecasts, not RGB video, policies, physical-control success or external-SOTA results. Original runtime cost measurements do not measure this revised backend.",
            "Project predictor weights/wrappers use MIT; unchanged LeWM notices remain included. Dataset examples are omitted. Read preprocessing, model cards, all limitations, checksums and the independent execution review. Research prerelease; synthetic parity is specific to the documented environment."])
        response = client.post(api + "/releases", json={"tag_name": TAG, "target_commitish": commit,
            "name": "IWS row-wise v2: 36 selected feature predictors", "body": body,
            "draft": True, "prerelease": True}, timeout=30)
    response.raise_for_status(); release = response.json()
    validate_remote_release(release, names, commit)
    known = {a["name"]: a for a in release["assets"]}; uploaded = {}
    for name in names:
        path = regular(destination / name); expected = sha(path)
        if name in known:
            asset = known[name]
        else:
            require(release["draft"], "Published releases are immutable")
            url = release["upload_url"].split("{", 1)[0]; parsed = urlparse(url)
            require(parsed.scheme == "https" and parsed.netloc == "uploads.github.com"
                    and parsed.path.startswith(f"/repos/{REPOSITORY}/releases/"), "Unexpected upload host/path")
            with path.open("rb") as stream:
                response = client.post(url, params={"name": name}, data=stream,
                    headers={"Content-Type": "application/octet-stream"}, timeout=(30, 1200))
            response.raise_for_status(); asset = response.json()
        require(asset["name"] == name and asset["size"] == path.stat().st_size
                and asset.get("digest") == "sha256:" + expected, "Remote asset size/digest differs")
        uploaded[name] = {"id": asset["id"], "sha256": expected, "bytes": asset["size"],
                          "server_digest_verified": True}
    if release["draft"]:
        validate_staged_assets(destination, names, publication)
        response = client.patch(api + "/releases/" + str(release["id"]),
                                 json={"draft": False, "prerelease": True}, timeout=30)
        response.raise_for_status(); release = response.json()
    validate_remote_release(release, names, commit)
    require(release["draft"] is False and {a["name"] for a in release["assets"]} == set(names),
            "Published inventory is incomplete")
    remote = {a["name"]: a for a in release["assets"]}
    for name, row in uploaded.items():
        require(remote[name]["id"] == row["id"], "Published asset identity changed")
        row["url"] = remote[name]["browser_download_url"]
    verify_public_assets(uploaded, requests)
    return {"status": "published_verified", "tag": TAG, "release_url": release["html_url"],
            "model_count": 36, "archive_sha256": publication["archive_sha256"],
            "manifest_sha256": publication["manifest_sha256"], "proof_sha256": publication["proof_sha256"],
            "readiness_sha256": publication["readiness_sha256"], "backend": BACKEND,
            "source_review_sha256": publication["source_review_sha256"],
            "independent_execution_review_sha256": publication["independent_execution_review_sha256"],
            "source_files_sha256": publication["source_files_sha256"], "assets": uploaded,
            "anonymous_download_verified": True, "secret_findings": 0, "prerelease": True,
            "target_commit": commit, "verified_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--target-commit")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.publish:
        require(args.credential_file is not None, "Publishing requires a private credential-file path")
    directory, names, publication = stage_release(root, args.directory)
    if not args.publish:
        print(json.dumps({"status": "verified_local", "model_count": 36, "assets": len(names),
                          "archive_sha256": publication["archive_sha256"]})); return
    with (directory.parent / ".iws_rowwise_v2.publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = publish(directory, names, publication, args.credential_file, args.target_commit)
        # Only sanitized, fully verified public state is written to these paths.
        for path in (root / "reports/iws_rowwise_checkpoint_publication_status.json", root / REPORT / "public_release.json"):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            temporary.write_text(encoded(result)); os.replace(temporary, path)
        print(json.dumps({"status": result["status"], "release_url": result["release_url"], "model_count": 36}))


if __name__ == "__main__":
    main()
