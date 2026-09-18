#!/usr/bin/env python3
"""Fetch pinned model artifacts; add --datasets for the large official archives.

Only checksum-verified files are promoted from .partial paths. Existing final
files with unexpected content are preserved and reported as conflicts. Downloads
resume only when their sidecar identifies the same pinned artifact. The default
--verify-only checks models and small metadata, not the 36.9 GB dataset archives.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "references" / "world_artifact_sources.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def artifact_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Artifact destination escapes root: {relative}")
    return path


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact manifest schema")
    seen = set()
    for kind in ("models", "datasets"):
        for source in manifest[kind]:
            revision = source["revision"]
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Every artifact source requires a complete pinned commit")
            prefix = "datasets/" if kind == "datasets" else ""
            for item in source["files"]:
                expected_url = f"https://huggingface.co/{prefix}{source['repo']}/resolve/{revision}/{item['name']}"
                if item["url"] != expected_url:
                    raise ValueError("Artifact URL does not match its pinned repository revision")
                if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or int(item["bytes"]) <= 0:
                    raise ValueError("Every artifact needs SHA256 and positive byte count")
                if item["path"] in seen:
                    raise ValueError("Duplicate artifact destination")
                seen.add(item["path"])
    for item in manifest["metadata_files"]:
        content = item["content_utf8"].encode("utf-8")
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Embedded metadata checksum differs: {item['path']}")
        if item["path"] in seen:
            raise ValueError("Duplicate metadata destination")
        seen.add(item["path"])


def verify_file(path, item):
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != item["bytes"]:
        raise ValueError(f"Preserved existing file with unexpected size: {path}; expected {item['bytes']}, got {actual_bytes}")
    digest = sha256(path)
    if digest != item["sha256"]:
        raise ValueError(f"Preserved existing file with unexpected SHA256: {path}; expected {item['sha256']}, got {digest}")
    return {"path": item["path"], "sha256": digest, "bytes": actual_bytes, "status": "verified"}


@contextmanager
def destination_lock(path):
    lock = path.with_name(path.name + ".download.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def atomic_write(path, content, replace=True):
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(name, path)
        else:
            # Atomic creation with no replacement, including writers that do
            # not participate in our advisory destination lock.
            os.link(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def fetch_artifact(root, item, verify_only=False, runner=subprocess.run):
    path = artifact_path(root, item["path"])
    if verify_only:
        return verify_file(path, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with destination_lock(path):
        if path.exists():
            return verify_file(path, item)
        partial = path.with_name(path.name + ".partial")
        sidecar = path.with_name(path.name + ".partial.source.json")
        identity = {key: item[key] for key in ("url", "sha256", "bytes")}
        if partial.exists() or sidecar.exists():
            if not sidecar.is_file() or json.loads(sidecar.read_text()) != identity:
                raise ValueError(f"Preserved unrecognized partial download: {partial}")
            if partial.exists() and partial.stat().st_size > item["bytes"]:
                raise ValueError(f"Preserved oversized partial download: {partial}")
        else:
            atomic_write(sidecar, (json.dumps(identity, indent=2) + "\n").encode(), replace=False)
        if not partial.exists() or partial.stat().st_size < item["bytes"]:
            print(json.dumps({"event": "download", "path": item["path"],
                              "resume_bytes": partial.stat().st_size if partial.exists() else 0,
                              "expected_bytes": item["bytes"]}), flush=True)
            runner(["curl", "--fail", "--location", "--silent", "--show-error",
                    "--proto", "=https", "--proto-redir", "=https",
                    "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
                    "--continue-at", "-", "--output", str(partial), item["url"]], check=True)
        result = verify_file(partial, item)
        os.link(partial, path)  # atomic publication; never replace a concurrent final file
        partial.unlink()
        sidecar.unlink()
        result["status"] = "downloaded_and_verified"
        return result


def restore_metadata(root, item, verify_only=False):
    path = artifact_path(root, item["path"])
    if verify_only:
        return verify_file(path, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with destination_lock(path):
        if path.exists():
            return verify_file(path, item)
        atomic_write(path, item["content_utf8"].encode("utf-8"), replace=False)
        result = verify_file(path, item)
        result["status"] = "restored_from_versioned_metadata"
        return result


def run(manifest_path=DEFAULT_MANIFEST, root=ROOT, datasets=False, verify_only=False):
    manifest_path, root = Path(manifest_path), Path(root)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    selected = [item for source in manifest["models"] for item in source["files"]]
    if datasets:
        selected += [item for source in manifest["datasets"] for item in source["files"]]
    # Validate destination boundaries and any existing final files before downloads.
    for item in selected + manifest["metadata_files"]:
        path = artifact_path(root, item["path"])
        if path.exists() and not verify_only:
            verify_file(path, item)
    results = [fetch_artifact(root, item, verify_only) for item in selected]
    metadata = [restore_metadata(root, item, verify_only) for item in manifest["metadata_files"]]
    return {"status": "verified", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
            "root": str(root.resolve()), "verify_only": verify_only,
            "dataset_archives_included": datasets, "artifacts": results, "metadata": metadata,
            "note": "Dataset archive hashes were not recomputed by this invocation" if not datasets else
                    "Selected official dataset archives were checksum verified"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--datasets", action="store_true", help="Also fetch/verify both large official source archives")
    parser.add_argument("--verify-only", action="store_true", help="Read-only validation; never download/restore artifacts")
    parser.add_argument("--report", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    result = run(args.manifest, args.root, args.datasets, args.verify_only)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(args.report, (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
