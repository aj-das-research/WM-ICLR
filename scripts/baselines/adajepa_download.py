#!/usr/bin/env python3
"""Download only the official AdaJEPA PushT checkpoint and released eval goals."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import stat
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/baselines/adajepa"
FILES = [
    ("1xExDiHn0rJ5zfQ4A50wkObjPvaE3pjGm", "pusht_visual_shift.zip", 310210277),
    ("1LNoPl-3XTlBFGSPg6DFqNXF3rv01LbUf", "pushobj_eval.zip", 4147919),
]


def download(spec):
    ident, name, expected = spec
    target = OUTPUT / "downloads" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={ident}&export=download&authuser=0&confirm=t"
    if not target.exists():
        temporary = target.with_suffix(".partial")
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if "html" in response.headers.get("content-type", ""):
                raise RuntimeError(f"Drive returned HTML instead of official archive: {name}")
            received = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > expected:
                        raise RuntimeError("Download exceeds verified public metadata size")
                    handle.write(chunk)
        if received != expected:
            raise RuntimeError(f"Size mismatch {name}: {received} != {expected}")
        temporary.replace(target)
    if target.stat().st_size != expected:
        raise RuntimeError(f"Existing archive size mismatch: {name}")
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Archive CRC failure {bad}")
        members = archive.infolist()
        if sum(member.file_size for member in members) > 5_000_000_000:
            raise RuntimeError("Uncompressed archive exceeds approved 5GB bound")
        for member in members:
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError(f"Unsafe archive member {member.filename}")
        release = OUTPUT / "release"
        archive.extractall(release)
    return {"drive_id": ident, "filename": name, "bytes": expected,
            "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_url": f"https://drive.google.com/file/d/{ident}/view",
            "extracted_files": [{"path": str((OUTPUT / "release" / member.filename).relative_to(ROOT)),
                                 "bytes": member.file_size} for member in members if not member.is_dir()]}


if __name__ == "__main__":
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(download, FILES))
    result = {"status": "downloaded_crc_verified", "seconds": time.time() - started,
              "official_code": "https://github.com/agentic-learning-ai-lab/adajepa",
              "official_commit": "51d8665b7978824bd218decab9e05ddb6eb1f47b", "assets": records}
    report = ROOT / "reports/evidence/adajepa_downloads.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
