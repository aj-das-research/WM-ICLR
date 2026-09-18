#!/usr/bin/env python3
"""Fetch the authors' public DROID-100 release with pinned object generations.

This is a 100-episode subset, not the complete DROID benchmark. No credentials,
generated images, or third-party mirrors are used. Inventory MD5 and size are
checked before each atomic rename; SHA256 receipts identify local inputs.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests


def fetch_object(item, root, prefix):
    name = item["name"]
    if not name.startswith(prefix) or ".." in Path(name).parts:
        raise ValueError("Unexpected object path")
    path = root / name[len(prefix):]
    path.parent.mkdir(parents=True, exist_ok=True)

    def verify(candidate):
        md5, sha = hashlib.md5(), hashlib.sha256()
        size = 0
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                size += len(block)
                md5.update(block)
                sha.update(block)
        if size != int(item["size"]) or base64.b64encode(md5.digest()).decode() != item["md5Hash"]:
            raise ValueError(f"Object checksum mismatch: {name}")
        return {"object": name, "generation": item["generation"],
                "bytes": size, "md5_base64": item["md5Hash"],
                "sha256": sha.hexdigest(), "local_path": str(path)}

    if path.exists():
        return verify(path)
    url = "https://storage.googleapis.com/gresearch/" + quote(name, safe="/")
    partial = path.with_name(path.name + ".partial")
    for attempt in range(3):
        try:
            with requests.get(url, params={"generation": item["generation"]},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for block in response.iter_content(4 << 20):
                        output.write(block)
            receipt = verify(partial)
            partial.replace(path)
            print(json.dumps({"downloaded": name, "bytes": receipt["bytes"]}), flush=True)
            return receipt
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inventory", default="reports/evidence/real_video/droid_100_inventory.json")
    parser.add_argument("--output", default="data/real_video/droid_100/raw")
    parser.add_argument("--prefix", default="robotics/droid_100/")
    args = parser.parse_args()
    inventory_path = Path(args.inventory)
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("nextPageToken"):
        raise ValueError("Incomplete inventory")
    items = inventory["items"]
    root = Path(args.output)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: fetch_object(item, root, args.prefix), items))
    receipt = {"dataset": "official DROID release", "status": "verified",
               "scope": inventory.get("selection", "author-provided 100-episode subset; not full benchmark"),
               "source": "https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md",
               "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
               "total_bytes": sum(record["bytes"] for record in records), "files": records}
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "download_receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(root / "download_receipt.json")
    print(json.dumps({"status": "verified", "files": len(records), "bytes": receipt["total_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
