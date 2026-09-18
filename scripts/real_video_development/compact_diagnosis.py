#!/usr/bin/env python3
"""Keep a small Git-friendly diagnostic report and verified local raw archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact(path, archive):
    path, archive = Path(path), Path(archive)
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("status") != "completed" or "raw_archive" in data:
        raise ValueError("Require complete, not previously compacted diagnostic")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    zipped = gzip.compress(raw, compresslevel=6, mtime=0)
    if digest(gzip.decompress(zipped)) != digest(raw):
        raise ValueError("Raw diagnostic archive roundtrip failed")
    archive.write_bytes(zipped)
    for run in data["runs"]:
        for split in ("validation", "train_subset"):
            if split not in run:
                continue
            population = run[split]
            for key in ("window_arrays", "rows"):
                del population[key]
    data["raw_archive"] = {"path": str(archive), "sha256": digest(zipped), "uncompressed_sha256": digest(raw),
                           "description": "All per-window metrics and exact recording/window identities; local compressed original output."}
    temporary = path.with_suffix(".compact.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    partial = path.with_suffix(".partial.json")
    if partial.exists():
        partial.unlink()  # Owned transient progress output; complete raw archive retained.
    return data["raw_archive"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", default="reports/real_droid_development_diagnosis.json")
    parser.add_argument("--archive", default="artifacts/development/real_droid_development_diagnosis.raw.json.gz")
    args = parser.parse_args()
    print(json.dumps(compact(args.input, args.archive), indent=2))
