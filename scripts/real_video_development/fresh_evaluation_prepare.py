#!/usr/bin/env python3
"""Decode exactly the metadata-registered fresh episodes after evaluation freeze."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path

from fresh_evaluation_common import (ROOT, FRESH, DECODED, FREEZE, atomic_json,
                                     sha256, validate_fresh_manifest, verify_freeze)


def process_shard(spec):
    os.environ.update(CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3", OMP_NUM_THREADS="1",
                      TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    path = Path(spec["local_path"])
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError("Fresh raw shard changed")
    module_spec = importlib.util.spec_from_file_location("frozen_ingestion", ROOT / "scripts/real_video/prepare_droid.py")
    ingestion = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(ingestion)
    # This candidate is held out as a whole. Change only split assignment,
    # preserving every original frame/action/schema audit and extraction step.
    ingestion.split_for = lambda session: "test"
    wanted = {row["source_episode_index"]: row for row in spec["selected"]}
    rows = []
    for index, serialized in enumerate(tf.compat.v1.io.tf_record_iterator(str(path))):
        if index not in wanted:
            continue
        expected = wanted[index]
        if __import__("hashlib").sha256(serialized).hexdigest() != expected["serialized_example_sha256"]:
            raise ValueError("Fresh metadata/serialized record alignment changed")
        row = ingestion.extract_example(serialized, {"filename": path.name, "sha256": spec["sha256"]},
                                        index, DECODED, spec["identity"])
        if any(row[key] != expected[key] for key in ("episode_id", "session_id", "serialized_example_sha256")):
            raise ValueError("Decoded identity differs from frozen candidate")
        rows.append(row)
    if len(rows) != len(wanted):
        raise ValueError("Missing registered fresh episodes")
    return rows


def main():
    freeze = verify_freeze()
    metadata = json.loads((FRESH / "metadata_manifest.json").read_text())
    receipt = json.loads((FRESH / "raw/download_receipt.json").read_text())
    originals = json.loads((FRESH / "original_exclusions.json").read_text())
    retained = metadata["retained"]
    if ({row["session_id"] for row in retained} & set(originals["session_ids"])) or (
            {row["episode_id"] for row in retained} & set(originals["episode_ids"])):
        raise ValueError("Fresh candidate overlaps original study")
    identity = {"evaluation_freeze_sha256": sha256(FREEZE), "metadata_manifest_sha256": sha256(FRESH / "metadata_manifest.json"),
                "original_extractor_sha256": sha256(ROOT / "scripts/real_video/prepare_droid.py"),
                "wrapper_sha256": sha256(__file__), "split_assignment": "all retained fresh episodes test-only"}
    DECODED.mkdir(parents=True, exist_ok=True)
    identity_path = DECODED / "extraction_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Cannot mix fresh extraction identities")
    atomic_json(identity, identity_path)
    specs = []
    for record in receipt["files"]:
        if ".tfrecord-" not in record["object"]:
            continue
        selected = [row for row in retained if row["source_shard"] == Path(record["local_path"]).name]
        # Read and verify every selected shard even if none of its episodes survive.
        specs.append(dict(record, selected=selected, identity=identity))
    rows = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(process_shard, spec): spec for spec in specs}
        for future in as_completed(futures):
            found = future.result()
            rows.extend(found)
            print(json.dumps({"stage": "fresh_decode", "completed_episodes": len(rows), "total": len(retained)}), flush=True)
    manifest = {"status": "complete", "dataset": "DROID fresh confirmatory recordings",
                "episodes": sorted(rows, key=lambda row: row["episode_id"]), "source_identity": identity,
                "action_block": 5, "action_dim": 35, "history_length": 3,
                "synthetic_corruptions": False, "simulated_images": False,
                "native_steps": sum(row["native_steps"] for row in rows),
                "grouped_steps": sum(row["steps"] for row in rows)}
    validate_fresh_manifest(manifest)
    atomic_json(manifest, DECODED / "manifest.json")
    eligibility = {str(h): {"eligible_episodes": sum(row["steps"] + 1 >= 3 + h for row in rows),
                            "short_episode_ids": [row["episode_id"] for row in rows if row["steps"] + 1 < 3 + h],
                            "window_count": sum(max(0, (row["steps"] + 1 - 3 - h) // 5 + 1) for row in rows)}
                   for h in (5, 10)}
    audit = {"status": "passed", "dataset_manifest_sha256": sha256(DECODED / "manifest.json"),
             "metadata_manifest_sha256": sha256(FRESH / "metadata_manifest.json"),
             "evaluation_freeze_sha256": sha256(FREEZE), "episodes": len(rows),
             "sessions": len({row["session_id"] for row in rows}), "eligibility": eligibility,
             "native_steps": manifest["native_steps"], "grouped_steps": manifest["grouped_steps"],
             "action_semantics": "commanded_cartesian_position_6_plus_gripper_position_1",
             "all_original_extraction_checks_reused": True, "all_retained_episodes_decoded": True,
             "outcome_based_exclusions": 0, "images_visually_inspected_before_evaluation": 0,
             "split_only_adapter": "test-only fresh population; original split hash not applied",
             "timestamps_available": False}
    atomic_json(audit, DECODED / "data_audit.json")
    verify_freeze()
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
