"""Immutable per-episode packages and a completion-last IWS cache transaction."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from .data import (SPLITS, InternalInventory, canonical_hash, decode_native_rgb,
                   load_commands, read_json, require, sha, training_statistics, validate_arrays)


def fsync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path); fsync_directory(path.parent)


@contextmanager
def writer_lock(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another cache writer holds this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def establish_identity(output, identity):
    path = Path(output) / "identity.json"
    if path.exists():
        require(read_json(path) == identity, "Cannot resume a different input/source/encoder/runtime identity")
    else:
        require(not [p for p in Path(output).iterdir() if p.name != ".writer.lock"], "Nonempty cache lacks its identity")
        atomic_json(identity, path)
    return sha(path)


def expected_receipt(record, identity_sha256):
    return {"schema": "shiftwm_iws_native_episode_v1", "task": "pusht", "episode_id": record["episode_id"],
            "split": record["split"], "frames": record["frames"], "command_rows": record["frames"],
            "command_width": 4, "feature_dim": 6144, "identity_sha256": identity_sha256,
            "source_video_sha256": record["video_sha256"], "source_metadata_sha256": record["metadata_sha256"],
            "native_indexing": "frame[i] and command_row[i] retained; temporal influence/physical units not inferred"}


def load_package(output, record, identity_sha256):
    require(record.get("split") in SPLITS, "Reserved cached episode access rejected")
    eid = record["episode_id"]
    require(isinstance(eid, str) and len(eid) == 6 and eid.isdigit(), "Invalid package identity")
    directory = Path(output) / "episodes" / eid
    require(directory.is_dir() and not directory.is_symlink(), "Missing or unsafe episode package")
    require({p.name for p in directory.iterdir()} == {"arrays.npz", "receipt.json"}, "Incomplete/corrupt committed package")
    require(not any(p.is_symlink() for p in directory.iterdir()), "Package symlinks forbidden")
    receipt = read_json(directory / "receipt.json")
    for key, value in expected_receipt(record, identity_sha256).items():
        require(receipt.get(key) == value, "Episode receipt identity differs: " + key)
    path = directory / "arrays.npz"
    require(sha(path) == receipt.get("payload_sha256"), "Committed feature payload is corrupt")
    with np.load(path, allow_pickle=False) as value:
        arrays = validate_arrays({key: value[key] for key in value.files}, record["frames"])
    require(hashlib.sha256(arrays["commands"].tobytes()).hexdigest() == receipt.get("command_values_sha256"), "Recorded command bytes changed")
    return receipt, arrays


def write_package(output, record, identity_sha256, arrays, rgb):
    validate_arrays(arrays, record["frames"])
    require(rgb.dtype == np.uint8 and rgb.shape == (record["frames"], 480, 640, 3), "Decoded images do not match complete native stream")
    directory = Path(output) / "episodes"; directory.mkdir(exist_ok=True)
    destination = directory / record["episode_id"]
    require(not destination.exists(), "Refusing to replace an existing episode")
    temporary = Path(tempfile.mkdtemp(prefix="." + record["episode_id"] + ".pending-", dir=directory))
    try:
        path = temporary / "arrays.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        receipt = {**expected_receipt(record, identity_sha256), "payload_sha256": sha(path),
                   "command_values_sha256": hashlib.sha256(arrays["commands"].tobytes()).hexdigest(),
                   "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                   "first_rgb_frame_sha256": hashlib.sha256(rgb[0].tobytes()).hexdigest(),
                   "last_rgb_frame_sha256": hashlib.sha256(rgb[-1].tobytes()).hexdigest()}
        atomic_json(receipt, temporary / "receipt.json")
        # Round-trip bytes before the only operation that makes this episode visible.
        with np.load(path, allow_pickle=False) as value:
            checked = validate_arrays({key: value[key] for key in value.files}, record["frames"])
            require(all(np.array_equal(checked[k], arrays[k]) for k in arrays), "Feature package round trip differs")
        fsync_directory(temporary); os.rename(temporary, destination); fsync_directory(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt


def verify_inputs(inventory, records):
    require(len(records) == len(inventory.assignment), "Input population incomplete")
    require({r["episode_id"] for r in records} == set(inventory.assignment), "Wrong cache input population")
    require(len({r["episode_id"] for r in records}) == len(records), "Duplicate cache input records")
    for record in records:
        row, split, metadata, video = inventory.paths(record["episode_id"], record["split"])
        require(record["frames"] == row["shapes"]["target_qpos"][0], "Frame count differs from frozen metadata")
        require(record["metadata_sha256"] == row["sha256"] == sha(metadata), "Input metadata identity changed")
        require(str(metadata.relative_to(inventory.root)) == record["metadata_path"], "Metadata path changed")
        require(str(video.relative_to(inventory.root)) == record["video_path"] and sha(video) == record["video_sha256"], "RGB input identity changed")


def finish_cache(output, records, identity_sha256, inventory, before_complete=None):
    output = Path(output); rows = []
    for record in records:
        receipt, _ = load_package(output, record, identity_sha256)
        rows.append(receipt)
    index = {"schema": "shiftwm_iws_native_episode_index_v1", "status": "features_verified", "identity_sha256": identity_sha256, "episodes": rows}
    atomic_json(index, output / "episode_index.json")

    def training_rows():
        for record in records:
            if record["split"] == "internal_train":
                yield load_package(output, record, identity_sha256)

    statistics = training_statistics(training_rows(), inventory.partitions["internal_train"])
    statistics["episode_index_sha256"] = sha(output / "episode_index.json")
    statistics["identity_sha256"] = identity_sha256
    atomic_json(statistics, output / "training_statistics.json")
    # Statistics are verified before the complete marker is visible.
    require(read_json(output / "training_statistics.json") == statistics, "Statistics round trip differs")
    complete = {"schema": "shiftwm_iws_feature_cache_v1", "status": "complete", "task": "pusht",
                "identity_sha256": identity_sha256, "episode_index_sha256": sha(output / "episode_index.json"),
                "training_statistics_sha256": sha(output / "training_statistics.json"),
                "episodes": len(records), "counts": {split: sum(r["split"] == split for r in records) for split in SPLITS},
                "native_frames": sum(r["frames"] for r in records), "feature_dim": 6144, "command_width": 4,
                "official_validation_payloads_read": 0, "model_training_or_evaluation": False}
    if before_complete is not None:
        before_complete()
    atomic_json(complete, output / "manifest.json")
    return complete


def build_cache(inventory, records, output, identity, encoder, batch_size=32, stop_after=None, before_complete=None):
    """An injected encoder supports meaningful lifecycle tests; CLI uses pinned DINO only."""
    output = Path(output)
    with writer_lock(output):
        require(identity.get("split_sha256") == inventory.split_sha256, "Extraction identity uses another frozen split")
        require(identity.get("input_records_sha256") == canonical_hash(records), "Extraction identity uses another input population")
        verify_inputs(inventory, records)
        identity_sha256 = establish_identity(output, identity)
        finished = 0
        for record in records:
            destination = output / "episodes" / record["episode_id"]
            if destination.exists():
                receipt, _ = load_package(output, record, identity_sha256)
            else:
                commands = load_commands(inventory, record["episode_id"], record["split"])
                rgb = decode_native_rgb(inventory, record["episode_id"], record["video_sha256"], record["split"])
                features = encoder(rgb, batch_size)
                arrays = {"features": features, "commands": commands, "frame_indices": np.arange(len(rgb), dtype=np.int64),
                          "command_row_indices": np.arange(len(commands), dtype=np.int64)}
                receipt = write_package(output, record, identity_sha256, arrays, rgb)
            finished += 1
            print(json.dumps({"event": "verified_episode", "episode_id": record["episode_id"], "split": record["split"], "done": finished, "total": len(records)}), flush=True)
            if stop_after is not None and finished >= stop_after and finished < len(records):
                return {"status": "incomplete", "completed_episodes": finished}
        verify_inputs(inventory, records)
        return finish_cache(output, records, identity_sha256, inventory, before_complete)


class IWSFeatureCache:
    """Validated package loader; no prediction windows or hidden action alignment."""
    def __init__(self, output, inventory, records, expected_static_identity):
        self.output = Path(output); self.inventory = inventory
        self.records = {r["episode_id"]: r for r in records}
        require(set(self.records) == set(inventory.assignment) and len(self.records) == len(records), "Unexpected cached population")
        complete = read_json(self.output / "manifest.json")
        require(complete.get("schema") == "shiftwm_iws_feature_cache_v1" and complete.get("status") == "complete", "Cache is not complete")
        require(complete["identity_sha256"] == sha(self.output / "identity.json"), "Cache identity changed")
        self.identity_sha256 = complete["identity_sha256"]
        identity = read_json(self.output / "identity.json")
        required_static = {"schema", "registration_sha256", "split_sha256", "input_manifest_sha256", "input_records_sha256",
                           "config_sha256", "preprocessing", "pooling_and_storage_precision", "tf32", "batch_size",
                           "encoder_provenance_sha256", "official_validation_payloads_allowed"}
        require(set(expected_static_identity) == required_static, "Full current registration identity must be supplied")
        for key, value in expected_static_identity.items():
            require(identity.get(key) == value, "Cache differs from current registered static identity: " + key)
        require(identity.get("device") in ("cpu", "cuda") and identity.get("encoder_precision") == ("bfloat16" if identity["device"] == "cuda" else "float32"), "Cached encoder precision/device contract differs")
        require(identity.get("split_sha256") == inventory.split_sha256 and identity.get("input_records_sha256") == canonical_hash(records), "Cache belongs to another split/input population")
        require(complete["episodes"] == len(records) and complete["native_frames"] == sum(r["frames"] for r in records), "Completed counts differ")
        require(complete["counts"] == {s: len(inventory.partitions[s]) for s in SPLITS}, "Completed split counts differ")
        require(complete["episode_index_sha256"] == sha(self.output / "episode_index.json"), "Episode index changed")
        require(complete["training_statistics_sha256"] == sha(self.output / "training_statistics.json"), "Training statistics changed")
        index = read_json(self.output / "episode_index.json")
        require(index["identity_sha256"] == self.identity_sha256, "Index belongs to another cache")
        require(len(index["episodes"]) == len(records) and {r["episode_id"] for r in index["episodes"]} == set(self.records), "Incomplete episode index")
        self.index = {r["episode_id"]: r for r in index["episodes"]}
        self.statistics = read_json(self.output / "training_statistics.json")
        require(self.statistics["fit_split"] == "internal_train" and self.statistics["episodes"] == inventory.partitions["internal_train"], "Normalization used wrong split")
        require(self.statistics["episode_index_sha256"] == complete["episode_index_sha256"] and self.statistics["identity_sha256"] == self.identity_sha256, "Normalization identity mismatch")

    def episode(self, episode_id, split):
        self.inventory.authorize(episode_id, split)
        receipt, arrays = load_package(self.output, self.records[episode_id], self.identity_sha256)
        require(receipt == self.index[episode_id], "Committed episode changed after finalization")
        return receipt, arrays
