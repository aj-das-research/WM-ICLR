#!/usr/bin/env python3
"""Fetch a deterministic, verified shard subset of an Open-X-Embodiment RLDS dataset.

Supported: BridgeData V2 (``robotics/bridge/0.1.0``), RT-1 / fractal
(``robotics/fractal20220817_data/0.1.0``) and Language-Table
(``robotics/language_table/0.1.0``) from the public ``gs://gresearch`` bucket,
over plain HTTPS (no credentials). Mirrors scripts/real_video/fetch_droid.py:

1. ``--plan``: list the full prefix through the GCS JSON API (paginated; name, size,
   md5Hash, generation), read ``dataset_info.json`` shard lengths, choose shards with a
   fixed seed until the per-split byte budget is reached, and write the inventory to
   ``reports/evidence/v2/<ds>_inventory.json``.
2. ``--download`` (default after plan): fetch every selected object pinned to its
   generation; size + base64 MD5 are verified before an atomic rename; a receipt with
   SHA256 is written to ``<raw>/download_receipt.json``.

Shard choice per split: ``random.Random(f"{seed}:{split}").shuffle(shard_ids)`` then take
shards in that order while ``cum_bytes + size <= budget`` (skip ones that overflow).
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.parse import quote

import requests

BUCKET = "gresearch"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
PUBLIC = f"https://storage.googleapis.com/{BUCKET}/"
GB = 1e9

DATASETS = {
    # official held-out split 'test' (= BridgeData V2 val, 3475 eps / 50 GB) is NOT small,
    # so it is sub-sampled too.
    "bridge": dict(prefix="robotics/bridge/0.1.0/", name="bridge",
                   budgets={"train": 11.5 * GB, "test": 3.5 * GB}),
    # fractal has only 'train'; test/val are carved by episode hash at conversion time.
    "fractal": dict(prefix="robotics/fractal20220817_data/0.1.0/", name="fractal20220817_data",
                    budgets={"train": 15.0 * GB}),
    # Language-Table (xArm planar block pushing): only 'train' (1024 shards, ~430 GB, 442k
    # short episodes); ~6 GB = ~14 shards; test/val carved by episode hash at conversion.
    "language_table": dict(prefix="robotics/language_table/0.1.0/", name="language_table",
                           budgets={"train": 6.0 * GB}),
}


def list_prefix(prefix):
    items, token = [], None
    while True:
        params = {"prefix": prefix, "fields": "items(name,size,md5Hash,crc32c,generation),nextPageToken",
                  "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(API, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def shard_name(name, split, index, n_shards):
    return f"{name}-{split}.tfrecord-{index:05d}-of-{n_shards:05d}"


def select_shards(split_info, sizes, seed, budget):
    """Deterministic shard subset for one split. sizes: {shard_index: bytes}."""
    n = len(split_info["shardLengths"])
    order = list(range(n))
    random.Random(f"{seed}:{split_info['name']}").shuffle(order)
    chosen, total = [], 0
    for i in order:
        if total + sizes[i] <= budget:
            chosen.append(i)
            total += sizes[i]
    return sorted(chosen), total


def plan(ds, seed, inventory_path):
    cfg = DATASETS[ds]
    listing = {it["name"]: it for it in list_prefix(cfg["prefix"])}
    info = requests.get(PUBLIC + cfg["prefix"] + "dataset_info.json", timeout=60).json()
    selected = [listing[cfg["prefix"] + f] for f in ("dataset_info.json", "features.json")]
    selection = {}
    for split in info["splits"]:
        sname = split["name"]
        if sname not in cfg["budgets"]:
            continue
        lengths = [int(x) for x in split["shardLengths"]]
        n = len(lengths)
        sizes = {}
        for i in range(n):
            key = cfg["prefix"] + shard_name(cfg["name"], sname, i, n)
            if key not in listing:
                raise ValueError(f"missing shard in listing: {key}")
            sizes[i] = int(listing[key]["size"])
        if sum(sizes.values()) != int(split["numBytes"]):
            # dataset_info numBytes counts serialized examples, not TFRecord framing; informative only
            print(json.dumps({"note": "listing bytes != dataset_info numBytes", "split": sname,
                              "listing": sum(sizes.values()), "numBytes": int(split["numBytes"])}))
        chosen, total = select_shards(split, sizes, seed, cfg["budgets"][sname])
        selection[sname] = {"n_shards_total": n, "episodes_total": sum(lengths),
                            "bytes_total": int(split["numBytes"]),
                            "budget_bytes": int(cfg["budgets"][sname]),
                            "shards": chosen, "n_shards": len(chosen), "bytes": total,
                            "episodes": sum(lengths[i] for i in chosen),
                            "shard_lengths": {str(i): lengths[i] for i in chosen}}
        selected += [listing[cfg["prefix"] + shard_name(cfg["name"], sname, i, n)] for i in chosen]
    inventory = {"dataset": ds, "tfds_name": cfg["name"], "version": info.get("version"),
                 "prefix": f"gs://{BUCKET}/{cfg['prefix']}", "seed": seed,
                 "selection_rule": "per split: random.Random(f'{seed}:{split}').shuffle(range(n_shards)); "
                                   "take shards in that order while cum_bytes+size <= budget",
                 "selection": selection, "listing_objects": len(listing),
                 "listed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "items": selected}
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = inventory_path.with_name(inventory_path.name + ".tmp")
    tmp.write_text(json.dumps(inventory, indent=1) + "\n")
    tmp.replace(inventory_path)
    print(json.dumps({"planned": ds, **{s: {k: v for k, v in d.items() if k not in ("shards", "shard_lengths")}
                                        for s, d in selection.items()}}), flush=True)
    return inventory


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
        return {"object": name, "generation": item["generation"], "bytes": size,
                "md5_base64": item["md5Hash"], "sha256": sha.hexdigest(), "local_path": str(path)}

    if path.exists():
        return verify(path)
    url = PUBLIC + quote(name, safe="/")
    partial = path.with_name(path.name + ".partial")
    for attempt in range(5):
        try:
            with requests.get(url, params={"generation": item["generation"]},
                              stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for block in response.iter_content(4 << 20):
                        output.write(block)
            receipt = verify(partial)
            partial.replace(path)
            print(json.dumps({"downloaded": name, "bytes": receipt["bytes"]}), flush=True)
            return receipt
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def download(ds, inventory_path, root, workers):
    inventory = json.loads(inventory_path.read_text())
    prefix = DATASETS[ds]["prefix"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda it: fetch_object(it, root, prefix), inventory["items"]))
    receipt = {"dataset": ds, "status": "verified", "source": inventory["prefix"],
               "scope": "deterministic shard subset, see inventory selection",
               "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
               "total_bytes": sum(r["bytes"] for r in records), "files": records}
    tmp = root / "download_receipt.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1) + "\n")
    tmp.replace(root / "download_receipt.json")
    print(json.dumps({"status": "verified", "dataset": ds, "files": len(records),
                      "bytes": receipt["total_bytes"]}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inventory")
    ap.add_argument("--output")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--replan", action="store_true", help="rebuild inventory even if present")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    inv = Path(a.inventory or f"reports/evidence/v2/{a.dataset}_inventory.json")
    out = Path(a.output or f"data/real_video/{a.dataset}/raw")
    if a.replan or not inv.exists():
        plan(a.dataset, a.seed, inv)
    if not a.plan_only:
        out.mkdir(parents=True, exist_ok=True)
        download(a.dataset, inv, out, a.workers)


if __name__ == "__main__":
    main()
