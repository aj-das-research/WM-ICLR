#!/usr/bin/env python3
"""Pin AdaJEPA's auxiliary Torch Hub import to official Python-3.9-compatible DINOv2."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[2]
REVISION = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
CACHE = ROOT / "environments/adajepa/torch_cache/hub"
TARGET = CACHE / "facebookresearch_dinov2_main"
MARKER = CACHE / "dinov2_pinned_source.json"


def tree_hash(path):
    files = {}
    for item in sorted(path.rglob("*.py")):
        files[str(item.relative_to(path))] = hashlib.sha256(item.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    if MARKER.exists():
        record = json.loads(MARKER.read_text())
        if record["revision"] != REVISION or record["python_source_tree_sha256"] != tree_hash(TARGET):
            raise RuntimeError("Pinned auxiliary source changed")
        print(json.dumps(record, indent=2))
    else:
        url = f"https://codeload.github.com/facebookresearch/dinov2/zip/{REVISION}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        if len(response.content) > 100_000_000:
            raise RuntimeError("Auxiliary source archive exceeds expected bound")
        CACHE.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for item in archive.infolist():
                if Path(item.filename).is_absolute() or ".." in Path(item.filename).parts:
                    raise RuntimeError("Invalid archive path")
            archive.extractall(CACHE)
        extracted = CACHE / f"dinov2-{REVISION}"
        if TARGET.exists():
            backup = CACHE / "facebookresearch_dinov2_main_incompatible_2026"
            if backup.exists():
                raise RuntimeError("Existing backup; inspect source cache before replacing")
            TARGET.rename(backup)
        extracted.rename(TARGET)
        record = {"revision": REVISION, "repository": "https://github.com/facebookresearch/dinov2",
                  "source_url": url, "archive_sha256": hashlib.sha256(response.content).hexdigest(),
                  "python_source_tree_sha256": tree_hash(TARGET),
                  "reason": "AdaJEPA declares Python3.9; unpinned 2026 DINOv2 main uses incompatible PEP604 annotations",
                  "scope": "Auxiliary checkpoint-loader import; actual released visual-shift model is SmallResNetGeM",
                  "cache_alias": "facebookresearch_dinov2_main to satisfy unchanged upstream torch.hub call"}
        MARKER.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
