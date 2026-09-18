#!/usr/bin/env python3
"""Register/download a fresh DROID candidate and inspect identities only.

This deliberately cannot decode images, compute features, or evaluate a model.
The emitted metadata manifest is not accepted by the training data loader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "data/real_video/droid_fresh_v1"
ORIGINAL_MANIFEST = ROOT / "data/real_video/droid_selected/processed/manifest.json"
ORIGINAL_INVENTORY = ROOT / "reports/evidence/real_video/droid_selected_inventory.json"
ORIGINAL_INFO = ROOT / "data/real_video/droid_selected/raw/1.0.0/dataset_info.json"
PROTOCOL = ROOT / "reports/real_droid_fresh_holdout_protocol.md"
FETCHER = ROOT / "scripts/real_video/fetch_droid.py"
SALT = "shiftwm-real-fresh-v1-20260919:"
PREFIX = "robotics/droid/"
SHARD_RE = re.compile(r"robotics/droid/1\.0\.0/r2d2_faceblur-train\.tfrecord-(\d{5})-of-02048")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_new_json(path, value):
    """Never overwrite a prior registration or audit silently."""
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Immutable file already exists: {path}")
        return
    with path.open("x") as stream:
        stream.write(content)


def session_from_metadata(file_path, recording_path):
    # Same site/date grouping as the frozen original extraction. The status
    # directory is checked for schema validity, never used to retain examples.
    def extract(value):
        parts = Path(value).parts
        positions = [(i, part) for i, part in enumerate(parts)
                     if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part)]
        if len(positions) != 1:
            raise ValueError("Cannot identify exactly one recording date")
        i, day = positions[0]
        if i < 2 or parts[i - 1] not in {"success", "failure", "failures"}:
            raise ValueError("Unknown site/status/date schema")
        site = parts[i - 2]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", site):
            raise ValueError("Invalid collection-site key")
        return site + "/" + day
    first, second = extract(file_path), extract(recording_path)
    if first != second:
        raise ValueError("Episode and recording session disagree")
    return first


def select_shards(items, excluded):
    shards = {item["name"]: item for item in items if SHARD_RE.fullmatch(item["name"])}
    if len(shards) != 2048 or not set(excluded).issubset(shards):
        raise ValueError("Expected all 2048 official shards and every original shard")
    candidates = [item for name, item in shards.items() if name not in excluded]
    return sorted(candidates, key=lambda item: hashlib.sha256(
        (SALT + item["name"]).encode()).hexdigest())[:12]


def register():
    import requests
    if (DESTINATION / "registration.json").exists():
        registration = verify_registration()
        print(json.dumps({"status": "existing_registration_verified", "registration": registration}))
        return
    if (DESTINATION / "raw").exists():
        raise ValueError("Raw namespace exists before registration")
    original = json.loads(ORIGINAL_MANIFEST.read_text())
    original_inventory = json.loads(ORIGINAL_INVENTORY.read_text())
    sessions = sorted({row["session_id"] for row in original["episodes"]})
    episode_ids = sorted(row["episode_id"] for row in original["episodes"])
    if original["status"] != "complete" or len(sessions) != 433 or len(episode_ids) != 1126:
        raise ValueError("Unexpected original study identity/size")
    excluded = original_inventory["selection"]["selected_shards"]
    if len(set(excluded)) != 24:
        raise ValueError("Original selection is not exactly 24 unique shards")
    items, pages, token = [], 0, None
    while True:
        params = {"prefix": "robotics/droid/1.0.0/", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        response = requests.get("https://storage.googleapis.com/storage/v1/b/gresearch/o",
                                params=params, timeout=(30, 120))
        response.raise_for_status()
        page = response.json()
        pages += 1
        items.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            break
    current = {item["name"]: item for item in items}
    if len(current) != len(items):
        raise ValueError("Duplicate listing entries")
    for item in original_inventory["items"]:
        now = current[item["name"]]
        if any(now[key] != item[key] for key in ("generation", "size", "md5Hash")):
            raise ValueError("Official release generation changed since original study")
    selected = select_shards(items, excluded)
    metadata = [current["robotics/droid/1.0.0/" + name]
                for name in ("CC-BY-4.0", "dataset_info.json", "features.json")]
    info = json.loads(ORIGINAL_INFO.read_text())
    lengths = next(split["shardLengths"] for split in info["splits"] if split["name"] == "train")
    expected = sum(int(lengths[int(SHARD_RE.fullmatch(item["name"]).group(1))])
                   for item in selected)
    inventory = {"selection": {
        "rule": "lowest 12 SHA256(salt + object name), excluding all original 24 shards",
        "salt": SALT, "selected_shards": [item["name"] for item in selected],
        "excluded_original_shards": excluded, "shards": 12,
        "expected_episodes_before_session_exclusion": expected,
        "scope": "unopened candidate fresh holdout, not full DROID benchmark",
        "bytes": sum(int(item["size"]) for item in metadata + selected)},
        "items": metadata + selected}
    exclusions = {
        "original_manifest_sha256": digest(ORIGINAL_MANIFEST),
        "session_ids": sessions, "episode_ids": episode_ids,
        "serialized_example_sha256": sorted(row["serialized_example_sha256"]
                                             for row in original["episodes"])}
    write_new_json(DESTINATION / "source_inventory.json", {"items": items, "pages": pages})
    write_new_json(DESTINATION / "inventory.json", inventory)
    write_new_json(DESTINATION / "original_exclusions.json", exclusions)
    protocol = f"""# Fresh real-DROID candidate holdout: acquisition and metadata protocol

Registered UTC: {datetime.now(timezone.utc).isoformat()}

Status: acquisition and identity audit only. **Not eligible for evaluation.**
The original DROID study's test results have already been revealed. This separate
candidate is reserved for a future confirmatory experiment after development
choices are frozen using original training/validation data only.

## Deterministic selection before download

From the complete official public GCS `robotics/droid/1.0.0/` listing, exclude all
24 previously downloaded full-release shards. Rank the remaining shards by the
hexadecimal SHA256 of `{SALT}` concatenated with the exact GCS object name and
take the first 12. No resampling or topping up based on retained counts is allowed.
The full listing and selected inventory, including immutable GCS generations,
publisher MD5 and byte lengths, are stored locally before any new shard download.
Expected source episodes before exclusions: **{expected}**. Download bytes,
including release metadata and CC-BY-4.0 license: **{inventory['selection']['bytes']}**.

The official dataset description is https://droid-dataset.github.io/ and the
download specification is https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md.
Selection does not imply the whole published DROID benchmark was evaluated.

## Identity-only audit and exclusions

Verify every downloaded object's generation-bound publisher MD5, size and local
SHA256; validate TFRecord CRCs. Read only the two episode metadata strings from
each TF Example. Derive episode IDs identically to the original study and derive
the same collection-site/calendar-day session key. Do not decode JPEGs, inspect
images, parse rewards/actions, generate features, or run any model or metric.
The status path component is checked only for schema validity and never filters
episodes. Raw path strings are not included in the candidate manifest.

Exclude the union of (a) any of all **433** original session IDs, irrespective of
their original split; (b) any of all **1,126** original episode IDs; (c) any exact
serialized-example hash already present in the original manifest. Report each
reason's count, intersections and union exactly. Duplicate candidate IDs or
ambiguous/mismatched session metadata fail the audit; they are not silently
removed. There are no success, motion, length, appearance or outcome exclusions.

The retained set is session-disjoint under this site/date definition. It is **not
established as scene-disjoint, object-disjoint or unseen physical environments**.
There is no clinical or physical robot control evaluation claim.

## Separation and future eligibility

All new bytes and metadata stay under `data/real_video/droid_fresh_v1`, outside
the original data, feature and training paths. The candidate manifest has
`eligible_for_evaluation: false` and is intentionally not a trainable data manifest.
No data is redistributed or added to GitHub. Preserve source license/attribution.

Before any held-out image decoding, a **separate** evaluation freeze must pin the
selected methods and all method/baseline checkpoints, calibration files, code,
primary camera, horizon, metrics, inference budgets, all secondary analyses and
statistical comparisons. Freeze that using original training/validation only,
then hash and archive the final candidate manifest with the evaluation freeze.
The current preparation script provides no operation to bypass this gate.
If too few sessions remain, report that fact without inspecting performance or
silently enlarging this registered candidate. Any later acquisition is a new
separately registered version.

## Registered files

- Selected inventory SHA256: `{digest(DESTINATION / 'inventory.json')}`
- Full official listing SHA256: `{digest(DESTINATION / 'source_inventory.json')}`
- Original manifest SHA256: `{digest(ORIGINAL_MANIFEST)}`
- Original inventory SHA256: `{digest(ORIGINAL_INVENTORY)}`
- Exclusion identities SHA256: `{digest(DESTINATION / 'original_exclusions.json')}`
- Preparation script SHA256: `{digest(__file__)}`
- Reused immutable downloader SHA256: `{digest(FETCHER)}`

Selected shard object names, in registered rank order:

""" + "\n".join("- `" + item["name"] + "`" for item in selected) + "\n"
    if PROTOCOL.exists():
        raise ValueError("Refusing to overwrite a prior protocol")
    PROTOCOL.write_text(protocol)
    registration = {
        "schema_version": 1, "status": "registered_before_download",
        "eligible_for_evaluation": False, "salt": SALT,
        "expected_episodes_before_exclusion": expected,
        "registered_total_bytes": inventory["selection"]["bytes"],
        "identity_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), FETCHER, PROTOCOL, ORIGINAL_MANIFEST, ORIGINAL_INVENTORY,
            ORIGINAL_INFO, DESTINATION / "inventory.json", DESTINATION / "source_inventory.json",
            DESTINATION / "original_exclusions.json")}}
    write_new_json(DESTINATION / "registration.json", registration)
    print(json.dumps(registration, indent=2))


def verify_registration():
    registration = json.loads((DESTINATION / "registration.json").read_text())
    if registration["eligible_for_evaluation"] is not False:
        raise ValueError("Acquisition registration must never grant evaluation eligibility")
    for relative, expected in registration["identity_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Registered scientific input changed: " + relative)
    return registration


def download():
    verify_registration()
    subprocess.run([sys.executable, str(FETCHER), "--inventory",
                    str(DESTINATION / "inventory.json"), "--output",
                    str(DESTINATION / "raw"), "--prefix", PREFIX], check=True, cwd=ROOT)
    verify_registration()


def inspect_shard(spec):
    os.environ.update(CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3",
                      TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="1")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    path = Path(spec["local_path"])
    if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError("Downloaded shard changed")
    rows = []
    for index, serialized in enumerate(tf.compat.v1.io.tf_record_iterator(str(path))):
        # Protobuf parsing reads the two metadata strings; no JPEG decoding or
        # numerical observation/action/reward/terminal-field access occurs.
        features = tf.train.Example.FromString(serialized).features.feature
        values = {}
        for key in ("file_path", "recording_folderpath"):
            name = "episode_metadata/" + key
            if name not in features:
                raise ValueError("Required identity metadata missing")
            encoded = features[name].bytes_list.value
            if len(encoded) != 1:
                raise ValueError("Identity metadata is not scalar")
            values[key] = encoded[0].decode("utf-8")
        rows.append({
            "episode_id": "droid-" + hashlib.sha256(values["file_path"].encode()).hexdigest()[:24],
            "session_id": session_from_metadata(values["file_path"], values["recording_folderpath"]),
            "source_shard": path.name, "source_shard_sha256": spec["sha256"],
            "source_generation": spec["generation"], "source_episode_index": index,
            "serialized_example_sha256": hashlib.sha256(serialized).hexdigest(),
            "file_path_sha256": hashlib.sha256(values["file_path"].encode()).hexdigest(),
            "recording_path_sha256": hashlib.sha256(values["recording_folderpath"].encode()).hexdigest()})
    if len(rows) != spec["expected_episodes"]:
        raise ValueError("Record count differs from official shard length")
    return rows


def mark_exclusions(rows, original):
    sessions = set(original["session_ids"])
    episode_ids = set(original["episode_ids"])
    original_hashes = set(original["serialized_example_sha256"])
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate episode identity")
    if len({row["serialized_example_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate serialized example")
    for row in rows:
        reasons = []
        if row["session_id"] in sessions:
            reasons.append("original_session")
        if row["episode_id"] in episode_ids:
            reasons.append("original_episode")
        if row["serialized_example_sha256"] in original_hashes:
            reasons.append("original_serialized_example")
        row["excluded_reasons"] = reasons
        row["retained"] = not reasons
    return rows


def audit(workers):
    registration = verify_registration()
    receipt_path = DESTINATION / "raw/download_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    inventory = json.loads((DESTINATION / "inventory.json").read_text())
    if receipt["status"] != "verified" or receipt["inventory_sha256"] != digest(DESTINATION / "inventory.json"):
        raise ValueError("Missing or mismatched download receipt")
    by_name = {row["object"]: row for row in receipt["files"]}
    if set(by_name) != {row["name"] for row in inventory["items"]}:
        raise ValueError("Incomplete or extra downloaded objects")
    for row in inventory["items"]:
        record = by_name[row["name"]]
        if (record["generation"] != row["generation"] or record["bytes"] != int(row["size"])
                or record["md5_base64"] != row["md5Hash"]):
            raise ValueError("Download identity changed")
        if not Path(record["local_path"]).resolve().is_relative_to(DESTINATION / "raw"):
            raise ValueError("Download escaped separate holdout namespace")
    for name, row in by_name.items():
        if not SHARD_RE.fullmatch(name) and digest(row["local_path"]) != row["sha256"]:
            raise ValueError("Downloaded schema/license changed")
    info = json.loads((DESTINATION / "raw/1.0.0/dataset_info.json").read_text())
    lengths = next(split["shardLengths"] for split in info["splits"] if split["name"] == "train")
    specs = [dict(row, expected_episodes=int(lengths[int(SHARD_RE.fullmatch(name).group(1))]))
             for name, row in by_name.items() if SHARD_RE.fullmatch(name)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(inspect_shard, spec): spec for spec in specs}
        for future in as_completed(futures):
            found = future.result()
            rows.extend(found)
            print(json.dumps({"metadata_audited_shard": Path(futures[future]["local_path"]).name,
                              "episodes": len(found), "completed_episodes": len(rows)}), flush=True)
    if len(rows) != registration["expected_episodes_before_exclusion"]:
        raise ValueError("Unexpected candidate source episode count")
    original = json.loads((DESTINATION / "original_exclusions.json").read_text())
    rows = mark_exclusions(sorted(rows, key=lambda row: row["episode_id"]), original)
    retained = [row for row in rows if row["retained"]]
    excluded = [row for row in rows if not row["retained"]]
    combinations = Counter("+".join(row["excluded_reasons"]) for row in excluded)
    reason_counts = Counter(reason for row in excluded for reason in row["excluded_reasons"])
    manifest = {
        "schema_version": 1, "status": "metadata_only_candidate_holdout",
        "eligible_for_evaluation": False, "eligible_for_training": False,
        "dataset": "DROID official real robot recordings; fresh candidate subset",
        "source_episodes": len(rows), "retained_episodes": len(retained),
        "excluded_episodes": len(excluded),
        "source_sessions": len({row["session_id"] for row in rows}),
        "retained_sessions": len({row["session_id"] for row in retained}),
        "excluded_sessions": len({row["session_id"] for row in excluded}),
        "exclusion_reason_counts_nonexclusive": {name: reason_counts[name] for name in (
            "original_session", "original_episode", "original_serialized_example")},
        "exclusion_reason_combinations_exclusive": dict(combinations),
        "no_outcome_motion_or_length_filtering": True,
        "images_decoded": 0, "images_visually_inspected": 0,
        "numerical_actions_rewards_terminal_fields_read": False,
        "outcome_path_component_used_for_selection": False, "model_outputs_computed": 0,
        "session_disjoint_from_original": not ({row["session_id"] for row in retained} & set(original["session_ids"])),
        "episode_disjoint_from_original": not ({row["episode_id"] for row in retained} & set(original["episode_ids"])),
        "scene_or_object_disjointness_established": False,
        "registered_download_bytes": receipt["total_bytes"],
        "registration_sha256": digest(DESTINATION / "registration.json"),
        "download_receipt_sha256": digest(receipt_path),
        "protocol_sha256": digest(PROTOCOL), "auditor_sha256": digest(__file__),
        "raw_license_sha256": digest(DESTINATION / "raw/1.0.0/CC-BY-4.0"),
        "future_gate": "Separate methods/baselines/camera/horizon/metrics/code/checkpoint freeze before image decoding or evaluation",
        "retained": retained, "excluded": excluded}
    write_new_json(DESTINATION / "metadata_manifest.json", manifest)
    summary = {key: value for key, value in manifest.items() if key not in {"retained", "excluded"}}
    summary["metadata_manifest_sha256"] = digest(DESTINATION / "metadata_manifest.json")
    write_new_json(DESTINATION / "metadata_audit.json", summary)
    verify_registration()
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=("register", "download", "audit"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    {"register": register, "download": download, "audit": lambda: audit(args.workers)}[args.operation]()


if __name__ == "__main__":
    main()
