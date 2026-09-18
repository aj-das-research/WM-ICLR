#!/usr/bin/env python3
"""Recover released-model action statistics from the pinned official archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import h5py
import hdf5plugin  # noqa: F401 - register upstream compression filters
import numpy as np
import zstandard


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    with archive.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        if archive.name.endswith(".tar.zst"):
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith((".h5", ".hdf5")):
                        continue
                    target = output / Path(member.name).name
                    if not target.exists() or target.stat().st_size != member.size:
                        tmp = target.with_suffix(target.suffix + ".partial")
                        with tar.extractfile(member) as source, tmp.open("wb") as destination:
                            shutil.copyfileobj(source, destination, 8 << 20)
                        tmp.replace(target)
                    sources.append(target)
                    print(json.dumps({"extracted": str(target), "bytes": target.stat().st_size}), flush=True)
        else:
            target = output / archive.name.removesuffix(".zst")
            if not target.exists():
                tmp = target.with_suffix(target.suffix + ".partial")
                with tmp.open("wb") as destination:
                    shutil.copyfileobj(reader, destination, 8 << 20)
                tmp.replace(target)
            sources.append(target)
    # Prefer the training partition if the archive also contains test data.
    selected = [p for p in sources if "train" in p.stem.lower()] or sources
    total, total_sq, count = None, None, 0
    columns = []
    for path in selected:
        with h5py.File(path, "r") as handle:
            keys = []
            handle.visititems(lambda name, value: keys.append(name) if isinstance(value, h5py.Dataset) and name.split("/")[-1] == "action" else None)
            if not keys:
                raise ValueError(f"No action dataset in {path}; keys={list(handle)}")
            for key in keys:
                array = np.asarray(handle[key][:], dtype=np.float64)
                array = array.reshape(-1, array.shape[-1])
                array = array[np.isfinite(array).all(axis=1)]
                total = array.sum(0) if total is None else total + array.sum(0)
                total_sq = (array * array).sum(0) if total_sq is None else total_sq + (array * array).sum(0)
                count += len(array)
                columns.append({"file": str(path), "column": key, "rows": len(array),
                                "minimum": array.min(0).tolist(), "maximum": array.max(0).tolist()})
    if count < 2:
        raise ValueError("No usable actions")
    mean = total / count
    centered = np.maximum(total_sq - count * mean * mean, 0)
    result = {"action": {"mean": mean.tolist(), "std": np.sqrt(centered / (count - 1)).tolist()},
              "population_std": np.sqrt(centered / count).tolist(), "count": count,
              "normalization": "sample_std_ddof1_matching_pinned_lewm_training_utils",
              "archive": str(archive), "archive_bytes": archive.stat().st_size,
              "sources": columns}
    # Archive SHA256 lets downstream checkpoints record the exact source data.
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    result["archive_sha256"] = digest.hexdigest()
    (output / "action_stats.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output)
