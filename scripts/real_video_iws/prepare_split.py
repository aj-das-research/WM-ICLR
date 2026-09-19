#!/usr/bin/env python3
"""Freeze IWS trajectory identities without decoding images or reading HDF5 values.

Only official training trajectories participate in development selection. The
official validation identities and released handles are reserved unchanged.
This is split preparation, not a model-training or evaluation registration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("pusht", "bimanual_box", "bimanual_rope")
SALT = "shiftwm-iws-internal-development-split-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(records):
    """Use identities only; reject duplicate or overlapping upstream identities."""
    grouped = {task: {"train": [], "val": []} for task in TASKS}
    seen = set()
    for row in records:
        task = row["task"]
        if task not in TASKS:
            continue
        eid, split = row["episode_id"], row["split"]
        if not isinstance(eid, str) or len(eid) != 6 or not eid.isdigit():
            raise ValueError("Invalid trajectory identity")
        if split not in ("train", "val") or (task, eid) in seen:
            raise ValueError("Duplicate or invalid upstream split identity")
        seen.add((task, eid))
        grouped[task][split].append(eid)
    output = {}
    for task, splits in grouped.items():
        if len(splits["train"]) < 5 or not splits["val"]:
            raise ValueError("Missing task or too few trajectories")
        ranked = sorted(splits["train"], key=lambda eid: (
            hashlib.sha256(f"{SALT}/{task}/{eid}".encode()).hexdigest(), eid))
        held_count = math.ceil(len(ranked) / 5)
        output[task] = {
            "internal_train": sorted(ranked[held_count:]),
            "internal_development": sorted(ranked[:held_count]),
            "reserved_official_validation": sorted(splits["val"]),
        }
    return output


def build(root=ROOT):
    metadata_path = root / "reports/evidence/iws_temporal_metadata_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "metadata_only_audit_passed":
        raise ValueError("Require completed metadata audit")
    acquisition_path = root / "configs/real_video_development/iws_acquisition_v1.json"
    acquisition = json.loads(acquisition_path.read_text())
    partitions = partition(metadata["episodes"])
    source_hashes = {
        str(metadata_path.relative_to(root)): sha(metadata_path),
        str(acquisition_path.relative_to(root)): sha(acquisition_path),
    }
    data_root = root / "data/real_video/iws_public_v1"
    rows_by_task = {}
    for task in TASKS:
        split_path = data_root / "extracted/iws_converted" / task / "split_ranges.json"
        original = json.loads(split_path.read_text())
        source_hashes[str(split_path.relative_to(root))] = sha(split_path)
        for split, destinations in (("train", ("internal_train", "internal_development")),
                                    ("val", ("reserved_official_validation",))):
            expected = {f"{i:06d}" for i in range(original[split]["start"], original[split]["end"] + 1)}
            actual = {eid for destination in destinations for eid in partitions[task][destination]}
            if actual != expected:
                raise ValueError("Audited metadata disagrees with official split")
        rows_by_task[task] = [row for row in metadata["episodes"] if row["task"] == task]
        # Hash bytes only for training metadata. Official validation values and
        # all RGB streams remain unopened. Recorded hashes bind their identities.
        for row in rows_by_task[task]:
            if row["split"] == "train" and sha(root / row["file"]) != row["sha256"]:
                raise ValueError("Training metadata changed since independent audit")
    handles = {}
    names = {"pusht": "pusht", "box": "bimanual_box", "rope": "bimanual_rope"}
    for short, task in names.items():
        path = data_root / f"download/eval_handles/iws/handles.{short}.json"
        payload = json.loads(path.read_text())
        if set(payload["handles_by_horizon"]) != {"60"}:
            raise ValueError("Official evaluation horizon changed")
        values = payload["handles_by_horizon"]["60"]
        ids, unique = set(), set()
        for handle in values:
            eid = f'{int(handle["traj_id"]):06d}'
            encoded = json.dumps(handle, sort_keys=True)
            if (handle["group_name"] != task or handle["sampled_horizon"] != 60
                    or eid not in partitions[task]["reserved_official_validation"]
                    or encoded in unique):
                raise ValueError("Invalid or duplicate reserved evaluation handle")
            ids.add(eid); unique.add(encoded)
        if len(values) != 200 or ids != set(partitions[task]["reserved_official_validation"]):
            raise ValueError("Incomplete official handles")
        source_hashes[str(path.relative_to(root))] = sha(path)
        handles[task] = {"path": str(path.relative_to(root)), "sha256": sha(path),
                         "windows": len(values), "trajectories": len(ids), "stored_command_rows": 60}
    return {
        "kind": "shiftwm_iws_trajectory_split_v1",
        "status": "identities_frozen_before_model_training_and_official_validation_video_decoding",
        "scope": "Split preparation only; model, training and evaluation protocols are not registered by this file.",
        "dataset_revision": acquisition["dataset_revision"],
        "upstream_code_revision": acquisition["upstream_code_revision"],
        "source_sha256": sha(Path(__file__)),
        "split_rule": "Per task, rank official train IDs by SHA256(salt/task/id); first ceil(N/5) form internal development.",
        "salt": SALT, "task_order": list(TASKS), "partitions": partitions,
        "counts": {task: {name: len(ids) for name, ids in value.items()} for task, value in partitions.items()},
        "reserved_handles": handles, "source_files": source_hashes,
        "metadata": {task: [{key: row[key] for key in ("episode_id", "split", "file", "sha256", "shapes")}
                            for row in rows_by_task[task]] for task in TASKS},
        "images_decoded_by_preparer": 0, "official_validation_values_read_by_preparer": 0,
        "limits": ["Official validation remains named validation, not a new independent test dataset.",
                   "Session, scene and operator independence is unestablished.",
                   "Recorded command units and physical acquisition timing remain unverified."],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "configs/real_video_iws/split_v1.json")
    args = parser.parse_args()
    payload = build()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != encoded:
            raise ValueError("Refusing to overwrite a different frozen split")
    else:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(json.dumps({"path": str(args.output), "sha256": sha(args.output), "counts": payload["counts"]}))


if __name__ == "__main__":
    main()
