"""Completion-last reserved feature cache; no normalization is fitted here."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from shiftwm.real_video_iws.cache import atomic_json, fsync_directory, writer_lock
from shiftwm.real_video_iws.data import canonical_hash, decode_native_rgb, load_commands, read_json, require, sha
from shiftwm.real_video_iws.features import PREPROCESSING
from .data import ReservedInventory, SPLIT, safe_path, validate_arrays


def immutable_json(value, path):
    path = Path(path)
    if path.exists():
        require(not path.is_symlink() and read_json(path) == value, "Refusing to replace a different committed JSON")
    else:
        atomic_json(value, path)


def extraction_contract(inventory):
    registry = inventory.check_access()
    expected = {"encoder_root": "data/pretrained/dinov2-small", "batch_size": 32,
                "device": "cuda", "precision": "bfloat16", "preprocessing": PREPROCESSING, "tf32": False}
    for key, value in expected.items():
        require(registry["extraction"].get(key) == value, "Extraction contract differs: " + key)
    return expected


def statistics_binding(inventory):
    task = inventory.registration["tasks"][inventory.task]
    relative = task["training_statistics_path"]
    require(relative == f"data/features/iws_{inventory.task}_spatial_v1/training_statistics.json", "Statistics must come from the existing training cache")
    path = safe_path(inventory.root, relative, exists=True)
    require(sha(path) == task["training_statistics_sha256"]
            == inventory.registration["dependencies"].get(relative), "Training statistics changed")
    statistics = read_json(path)
    require(statistics["fit_split"] == "internal_train"
            and statistics["episodes"] == inventory.document["partitions"][inventory.task]["internal_train"],
            "Statistics were not fitted on the frozen training partition")
    return {"path": relative, "sha256": sha(path)}, statistics


def collect_inputs(inventory):
    """First raw hashes are collected only after the reviewed registration check."""
    inventory.check_access()
    records = []
    for eid in inventory.episode_ids:
        row, _, metadata, video = inventory.paths(eid)
        require(sha(metadata) == row["sha256"], "Reserved metadata differs from the original metadata audit")
        records.append({"episode_id": eid, "split": SPLIT, "frames": row["shapes"]["target_qpos"][0],
                        "metadata_path": str(metadata.relative_to(inventory.root)), "metadata_sha256": row["sha256"],
                        "video_path": str(video.relative_to(inventory.root)), "video_sha256": sha(video),
                        "video_bytes": video.stat().st_size})
    return records


def expected_identity(inventory, records, runtime):
    contract = extraction_contract(inventory)
    stats, _ = statistics_binding(inventory)
    return {"schema": "shiftwm_iws_reserved_extraction_identity_v1", "task": inventory.task,
            "registration_sha256": inventory.registration_sha256,
            "split_sha256": inventory.split_sha256, "handles_sha256": inventory.handles_sha256,
            "ordered_handles_sha256": canonical_hash(inventory.handles),
            "input_records_sha256": canonical_hash(records), "extraction": contract,
            "training_statistics": stats, "statistics_fitted": False,
            "pooling_and_storage_precision": "float32", "command_width": inventory.command_width,
            "runtime": runtime}


def receipt_fields(inventory, record, identity_sha256):
    return {"schema": "shiftwm_iws_reserved_episode_v1", "task": inventory.task,
            "episode_id": record["episode_id"], "split": SPLIT, "frames": record["frames"],
            "command_rows": record["frames"], "command_width": inventory.command_width,
            "feature_dim": 6144, "identity_sha256": identity_sha256,
            "source_video_sha256": record["video_sha256"],
            "source_metadata_sha256": record["metadata_sha256"]}


def package_paths(inventory, eid):
    require(eid in inventory.rows, "Unknown cache episode")
    relative = f"data/features/iws_reserved_v1/{inventory.task}/episodes/{eid}"
    directory = safe_path(inventory.root, relative)
    require(directory.is_dir(), "Missing reserved episode package")
    require({p.name for p in directory.iterdir()} == {"arrays.npz", "receipt.json"}, "Incomplete episode package")
    return (safe_path(inventory.root, relative + "/receipt.json", exists=True),
            safe_path(inventory.root, relative + "/arrays.npz", exists=True))


def load_package(inventory, record, identity_sha256, *, payload=True):
    receipt_path, array_path = package_paths(inventory, record["episode_id"])
    receipt = read_json(receipt_path)
    for key, value in receipt_fields(inventory, record, identity_sha256).items():
        require(receipt.get(key) == value, "Reserved episode identity changed: " + key)
    require(all(isinstance(receipt.get(k), str) and len(receipt[k]) == 64
                for k in ("payload_sha256", "command_values_sha256", "decoded_rgb_sha256")), "Missing payload provenance")
    if not payload:
        return receipt, None
    inventory.authorize(record["episode_id"])
    require(sha(array_path) == receipt["payload_sha256"], "Reserved feature payload is corrupt")
    with np.load(array_path, allow_pickle=False) as value:
        arrays = validate_arrays({k: value[k] for k in value.files}, record["frames"], inventory.command_width)
    require(sha(array_path) == receipt["payload_sha256"], "Reserved feature payload changed during reading")
    require(hashlib.sha256(arrays["commands"].tobytes()).hexdigest() == receipt["command_values_sha256"], "Cached native commands changed")
    return receipt, arrays


def write_package(inventory, output, record, identity_sha256, arrays, rgb):
    validate_arrays(arrays, record["frames"], inventory.command_width)
    require(rgb.dtype == np.uint8 and rgb.shape == (record["frames"], 480, 640, 3), "Incomplete native RGB stream")
    parent = safe_path(inventory.root, f"data/features/iws_reserved_v1/{inventory.task}/episodes")
    parent.mkdir(exist_ok=True)
    destination = parent / record["episode_id"]
    require(not destination.exists(), "Refusing to replace an episode package")
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        path = temporary / "arrays.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        receipt = {**receipt_fields(inventory, record, identity_sha256), "payload_sha256": sha(path),
                   "command_values_sha256": hashlib.sha256(arrays["commands"].tobytes()).hexdigest(),
                   "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                   "first_rgb_frame_sha256": hashlib.sha256(rgb[0].tobytes()).hexdigest(),
                   "last_rgb_frame_sha256": hashlib.sha256(rgb[-1].tobytes()).hexdigest()}
        atomic_json(receipt, temporary / "receipt.json")
        with np.load(path, allow_pickle=False) as value:
            require(all(np.array_equal(value[k], arrays[k]) for k in arrays), "Cache serialization changed array values")
        fsync_directory(temporary); os.rename(temporary, destination); fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt


def build_cache(inventory, encoder, runtime, *, stop_after=None):
    """CLI supplies unchanged DINO; injection is restricted to synthetic unit tests."""
    require(isinstance(inventory, ReservedInventory), "Explicit reserved inventory required")
    contract = extraction_contract(inventory)
    output = safe_path(inventory.root, f"data/features/iws_reserved_v1/{inventory.task}")
    with writer_lock(output):
        records = collect_inputs(inventory)
        inputs = {"schema": "shiftwm_iws_reserved_raw_inputs_v1", "task": inventory.task,
                  "registration_sha256": inventory.registration_sha256, "episodes": records}
        immutable_json(inputs, output / "inputs.json")
        identity = expected_identity(inventory, records, runtime)
        immutable_json(identity, output / "identity.json")
        identity_sha256 = sha(output / "identity.json")
        if (output / "manifest.json").exists():
            cache = ReservedFeatureCache(inventory.root, inventory.task, inventory.registration_path)
            for eid in cache.episode_ids:
                cache.episode(eid)
            return cache.manifest
        receipts = []
        for i, record in enumerate(records):
            eid = record["episode_id"]
            if (output / "episodes" / eid).exists():
                receipt, _ = load_package(inventory, record, identity_sha256)
            else:
                commands = load_commands(inventory, eid)
                rgb = decode_native_rgb(inventory, eid, record["video_sha256"])
                features = encoder(rgb, contract["batch_size"])
                arrays = {"features": features, "commands": commands,
                          "frame_indices": np.arange(len(rgb), dtype=np.int64),
                          "command_row_indices": np.arange(len(commands), dtype=np.int64)}
                receipt = write_package(inventory, output, record, identity_sha256, arrays, rgb)
            receipts.append(receipt)
            if stop_after is not None and i + 1 >= stop_after and i + 1 < len(records):
                return {"status": "incomplete", "episodes": i + 1}
        require(collect_inputs(inventory) == records, "Raw inputs changed during extraction")
        index = {"schema": "shiftwm_iws_reserved_episode_index_v1", "identity_sha256": identity_sha256,
                 "episodes": receipts}
        immutable_json(index, output / "episode_index.json")
        inventory.check_access()
        complete = {"schema": "shiftwm_iws_reserved_feature_cache_v1", "status": "complete",
                    "task": inventory.task, "split": SPLIT, "registration_sha256": inventory.registration_sha256,
                    "identity_sha256": identity_sha256, "inputs_sha256": sha(output / "inputs.json"),
                    "episode_index_sha256": sha(output / "episode_index.json"),
                    "handles_sha256": inventory.handles_sha256, "ordered_handles_sha256": canonical_hash(inventory.handles),
                    "episodes": 10, "handles": 200, "native_frames": inventory.manifest["native_frames"],
                    "command_width": inventory.command_width, "feature_dim": 6144,
                    "training_statistics": identity["training_statistics"], "statistics_fitted": False,
                    "model_training_or_evaluation": False}
        immutable_json(complete, output / "manifest.json")
        return ReservedFeatureCache.metadata_only(inventory.root, inventory.task, inventory.registration_path).manifest


class ReservedFeatureCache:
    def __init__(self, root, task, registration_path=None, *, _metadata_only=False):
        self.inventory = ReservedInventory(root, task, registration_path)
        self.root = self.inventory.root; self.task = task; self.metadata_only_access = _metadata_only
        self.handles = self.inventory.handles; self.episode_ids = self.inventory.episode_ids
        self.command_width = self.inventory.command_width
        self.relative = f"data/features/iws_reserved_v1/{task}"
        self.output = safe_path(self.root, self.relative)
        def load(name):
            return read_json(safe_path(self.root, self.relative + "/" + name, exists=True))
        self.manifest = load("manifest.json")
        m = self.manifest
        require(m.get("schema") == "shiftwm_iws_reserved_feature_cache_v1" and m.get("status") == "complete", "Reserved cache is not complete")
        for file, field in [("identity.json", "identity_sha256"), ("inputs.json", "inputs_sha256"), ("episode_index.json", "episode_index_sha256")]:
            require(sha(safe_path(self.root, self.relative + "/" + file, exists=True)) == m[field], "Committed cache metadata changed: " + file)
        inputs = load("inputs.json"); identity = load("identity.json"); index = load("episode_index.json")
        records = inputs["episodes"]
        require(inputs.get("schema") == "shiftwm_iws_reserved_raw_inputs_v1" and inputs.get("task") == task
                and inputs.get("registration_sha256") == self.inventory.registration_sha256, "Raw input record identity differs")
        require([r["episode_id"] for r in records] == self.episode_ids, "Cached population/order differs")
        for record in records:
            eid = record["episode_id"]; row = self.inventory.rows[eid]
            require(record["split"] == SPLIT and record["frames"] == row["shapes"]["target_qpos"][0]
                    and record["metadata_sha256"] == row["sha256"] and record["metadata_path"] == row["file"]
                    and record["video_path"] == f"{row['file'].rsplit('/', 1)[0]}/camera_0_rgb.mp4", "Raw input provenance differs")
        require(identity == expected_identity(self.inventory, records, identity["runtime"]), "Reserved extraction identity differs")
        expected = {"task": task, "split": SPLIT, "registration_sha256": self.inventory.registration_sha256,
                    "episodes": 10, "handles": 200, "native_frames": self.inventory.manifest["native_frames"],
                    "command_width": self.command_width, "feature_dim": 6144,
                    "handles_sha256": self.inventory.handles_sha256,
                    "ordered_handles_sha256": canonical_hash(self.handles), "statistics_fitted": False,
                    "training_statistics": identity["training_statistics"], "model_training_or_evaluation": False}
        require(all(m.get(k) == v for k, v in expected.items()), "Completion manifest differs from reviewed protocol")
        require(index.get("schema") == "shiftwm_iws_reserved_episode_index_v1"
                and index["identity_sha256"] == m["identity_sha256"]
                and [r["episode_id"] for r in index["episodes"]] == self.episode_ids, "Episode index incomplete or reordered")
        self.records = {r["episode_id"]: r for r in records}; self.index = {r["episode_id"]: r for r in index["episodes"]}
        for record in records:
            receipt, _ = load_package(self.inventory, record, m["identity_sha256"], payload=False)
            require(receipt == self.index[record["episode_id"]], "Episode receipt differs from completed index")
        _, self.statistics = statistics_binding(self.inventory)
        self.audit = {**self.inventory.audit, "metadata_only": _metadata_only, "episodes_loaded": [],
                      "manifest_sha256": sha(self.output / "manifest.json"), "identity_sha256": m["identity_sha256"],
                      "source_dependencies": {self.relative + "/" + name: sha(self.output / name)
                                              for name in ("manifest.json", "identity.json", "inputs.json", "episode_index.json")}}
        for eid, row in self.index.items():
            self.audit["source_dependencies"][f"{self.relative}/episodes/{eid}/arrays.npz"] = row["payload_sha256"]
            self.audit["source_dependencies"][f"{self.relative}/episodes/{eid}/receipt.json"] = sha(self.output / "episodes" / eid / "receipt.json")

    @classmethod
    def metadata_only(cls, root, task, registration_path=None):
        return cls(root, task, registration_path, _metadata_only=True)

    def episode(self, episode_id):
        require(not self.metadata_only_access, "Metadata-only cache forbids feature payload access")
        self.inventory.authorize(episode_id)
        require(sha(self.output / "manifest.json") == self.audit["manifest_sha256"], "Cache manifest changed during evaluation")
        receipt, arrays = load_package(self.inventory, self.records[episode_id], self.manifest["identity_sha256"])
        require(receipt == self.index[episode_id], "Episode changed after cache completion")
        self.audit["episodes_loaded"].append(episode_id)
        return arrays
