#!/usr/bin/env python3
"""Build only the separately reviewed reserved cache; metadata mode opens no payload."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws.data import require
from shiftwm.real_video_iws.features import DinoSpatialEncoder
from shiftwm.real_video_iws_reserved.data import ReservedInventory, TASK_WIDTHS, safe_path
from shiftwm.real_video_iws_reserved.cache import ReservedFeatureCache, build_cache, extraction_contract


def runtime_identity():
    import cv2
    import torch
    import transformers
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Allocated BF16-capable CUDA device required; no CPU fallback")
    free, total = torch.cuda.mem_get_info()
    require(free >= 4 * 1024**3, "Allocated GPU has less than 4 GiB free memory")
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": importlib.metadata.version("numpy"), "h5py": importlib.metadata.version("h5py"),
            "opencv": cv2.__version__, "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            "cuda_version": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()), "cuda_total_bytes": total}


def run(command, tasks, root=ROOT, registration_path=None):
    root = Path(root).resolve()
    require(command in ("metadata", "build", "validate"), "Unknown cache operation")
    require(tasks and len(set(tasks)) == len(tasks) and all(t in TASK_WIDTHS for t in tasks), "Invalid task selection")
    if command == "metadata":
        return {t: ReservedInventory.metadata_only(root, t, registration_path).audit for t in tasks}
    # Establish the entire requested authorization before CUDA setup or a raw open.
    inventories = [ReservedInventory(root, t, registration_path) for t in tasks]
    result = {}
    encoder = None
    for inventory in inventories:
        if command == "validate":
            cache = ReservedFeatureCache(root, inventory.task, registration_path)
            for eid in cache.episode_ids:
                cache.episode(eid)
            result[inventory.task] = {"status": "complete_all_packages_verified", **cache.audit}
            continue
        contract = extraction_contract(inventory)
        runtime = runtime_identity()
        if encoder is None:
            encoder_root = safe_path(root, contract["encoder_root"])
            encoder = DinoSpatialEncoder(root, encoder_root, "cuda")
        result[inventory.task] = build_cache(inventory, encoder, runtime)
        print(json.dumps({"event": "reserved_task_cache_complete", "task": inventory.task}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=("metadata", "build", "validate"))
    p.add_argument("--task", choices=("all", *TASK_WIDTHS), default="all")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--registration", type=Path)
    args = p.parse_args()
    tasks = list(TASK_WIDTHS) if args.task == "all" else [args.task]
    print(json.dumps(run(args.command, tasks, args.root, args.registration), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
