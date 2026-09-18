#!/usr/bin/env python3
"""Verify complete extension data and fit action statistics on training only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shiftwm.data import validate_manifest
from shiftwm.evaluate import atomic_json


def prepare(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest.get("action_block") != 5:
        raise ValueError("This prespecified extension uses five two-dimensional calls")
    totals, squared, count = np.zeros(10), np.zeros(10), 0
    counts, inputs = {}, {}
    for ep in manifest["episodes"]:
        path = root / ep["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ep["sha256"]:
            raise ValueError(f"Episode digest changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            images, actions = data["images"], data["actions"]
            if images.dtype != np.uint8 or images.shape[0] != ep["steps"] + 1 or images.shape[-1] != 3:
                raise ValueError(f"Invalid RGB sequence in {path}")
            if actions.shape != (ep["steps"], 10) or not np.isfinite(actions).all() or np.abs(actions).max() > 1.000001:
                raise ValueError(f"Invalid command sequence in {path}")
            if ep["steps"] < 7:
                raise ValueError(f"No complete eight-image window in {path}")
            if ep["split"] == "train":
                values = actions.astype(np.float64)
                totals += values.sum(0)
                squared += (values * values).sum(0)
                count += len(values)
        counts[ep["split"]] = counts.get(ep["split"], 0) + 1
        inputs[ep["file"]] = digest
    if set(counts) != {"train", "val", "development", "test"} or count < 2:
        raise ValueError("All four complete trajectory splits are required")
    mean = totals / count
    std = np.sqrt(np.maximum(squared - count * mean * mean, 0) / (count - 1))
    if (std < 1e-5).any():
        raise ValueError("Training actions do not excite every control dimension")
    stats = {"action": {"mean": mean.tolist(), "std": std.tolist()},
             "count": count, "fit_split": "train", "normalization": "sample_std_ddof1",
             "dataset_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
    atomic_json(stats, root / "action_stats.json")
    report = {"status": "verified", "episodes": sum(counts.values()), "splits": counts,
              "manifest_sha256": stats["dataset_manifest_sha256"], "episode_hashes": inputs,
              "action_stats_sha256": hashlib.sha256((root / "action_stats.json").read_bytes()).hexdigest()}
    atomic_json(report, root / "training_data_validation.json")
    return {k: v for k, v in report.items() if k != "episode_hashes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", required=True)
    print(json.dumps(prepare(parser.parse_args().data)), flush=True)
