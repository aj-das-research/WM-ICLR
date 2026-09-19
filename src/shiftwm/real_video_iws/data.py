"""Strict internal-split access and native recorded-row loading for IWS PushT.

No model windowing is defined here. A row index is not a physical timestamp.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("internal_train", "internal_development")
TASK = "pusht"
COMMAND_WIDTH = 4
DATA_RELATIVE = Path("data/real_video/iws_public_v1/extracted/iws_converted/pusht")
SPLIT_RELATIVE = Path("configs/real_video_iws/split_v1.json")
METADATA_RELATIVE = Path("reports/evidence/iws_temporal_metadata_manifest.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_split(document):
    require(document.get("kind") == "shiftwm_iws_trajectory_split_v1", "Wrong split schema")
    require(document.get("status") == "identities_frozen_before_model_training_and_official_validation_video_decoding", "Unfrozen split")
    parts = document["partitions"][TASK]
    require(set(parts) == {*SPLITS, "reserved_official_validation"}, "Unexpected partition")
    seen = set()
    for split, ids in parts.items():
        require(ids and ids == sorted(set(ids)), "Empty, duplicated or unsorted partition")
        require(all(isinstance(eid, str) and len(eid) == 6 and eid.isdigit() for eid in ids), "Invalid episode ID")
        require(not seen.intersection(ids), "Episode crosses partitions")
        require(document["counts"][TASK][split] == len(ids), "Split count mismatch")
        seen.update(ids)
    rows = document["metadata"][TASK]
    require(len(rows) == len(seen) and {r["episode_id"] for r in rows} == seen, "Metadata population mismatch")
    reserved = set(parts["reserved_official_validation"])
    for row in rows:
        eid = row["episode_id"]
        require(row["split"] == ("val" if eid in reserved else "train"), "Upstream split identity mismatch")
        require(row["file"] == str(DATA_RELATIVE / f"traj_{eid}" / "metadata.h5"), "Unexpected metadata path")
        shape = row["shapes"].get("target_qpos", [])
        require(len(shape) == 2 and shape[0] >= 2 and shape[1] == COMMAND_WIDTH, "Wrong native command layout")
        require(all(s and s[0] == shape[0] for s in row["shapes"].values()), "Metadata rows do not align")
    return parts


class InternalInventory:
    """Authorize identities before any raw video/HDF5 open.

    The constructor reads frozen identity metadata, including reserved IDs/shapes;
    it never opens the underlying reserved-validation episode payloads.
    """

    def __init__(self, root, split_path=None):
        self.root = Path(root)
        self.split_path = Path(split_path) if split_path else self.root / SPLIT_RELATIVE
        self.document = read_json(self.split_path)
        self.partitions = validate_split(self.document)
        self.assignment = {eid: split for split in SPLITS for eid in self.partitions[split]}
        self.rows = {row["episode_id"]: row for row in self.document["metadata"][TASK] if row["split"] == "train"}
        require(set(self.rows) == set(self.assignment), "Training inventory mismatch")
        self.split_sha256 = sha(self.split_path)

    def authorize(self, episode_id, split=None):
        # Membership is deliberately checked before resolving or opening a path.
        require(episode_id in self.assignment, "Reserved or unknown episode payload access rejected")
        require(split is None or split in SPLITS, "Only internal training/development may be read")
        assigned = self.assignment[episode_id]
        require(split is None or split == assigned, "Episode does not belong to requested internal split")
        row = self.rows[episode_id]
        require(row["split"] == "train", "Official validation payload is forbidden")
        return row, assigned

    def paths(self, episode_id, split=None):
        row, assigned = self.authorize(episode_id, split)
        directory = self.root / DATA_RELATIVE / f"traj_{episode_id}"
        require(not directory.is_symlink(), "Episode directory symlinks are forbidden")
        metadata = directory / "metadata.h5"
        video = directory / "camera_0_rgb.mp4"
        for path in (metadata, video):
            require(not path.is_symlink(), "Episode payload symlinks are forbidden")
            require(path.is_file(), "Missing authorized episode payload: " + str(path))
            require(path.resolve().parent == directory.resolve(), "Payload escaped episode directory")
        return row, assigned, metadata, video

    def selected(self, split=None):
        require(split is None or split in SPLITS, "Reserved split selection rejected")
        return sorted(eid for eid, value in self.assignment.items() if split is None or value == split)


def load_commands(inventory, episode_id, split=None):
    row, assigned, path, _ = inventory.paths(episode_id, split)
    require(sha(path) == row["sha256"], "Audited training metadata changed")
    expected = tuple(row["shapes"]["target_qpos"])
    with h5py.File(path, "r") as stream:
        require("target_qpos" in stream, "Recorded commands missing")
        value = stream["target_qpos"]
        require(value.shape == expected and value.dtype == np.dtype("float32"), "Command shape/dtype changed")
        commands = value[...]
    require(commands.shape == expected and np.isfinite(commands).all(), "Nonfinite or truncated commands")
    require(sha(path) == row["sha256"], "Training metadata changed while commands were read")
    return commands


def decode_native_rgb(inventory, episode_id, expected_video_sha256, split=None):
    """Decode all rows in sequence, never seek, resample, drop or repeat frames."""
    import cv2
    row, _, _, path = inventory.paths(episode_id, split)
    require(sha(path) == expected_video_sha256, "Training RGB bytes changed")
    expected = row["shapes"]["target_qpos"][0]
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    require(capture.isOpened(), "Video decoder could not open authorized stream")
    frames = []
    try:
        require(capture.getBackendName() == "FFMPEG", "Video backend differs from pinned decoder")
        header_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        require(math.isfinite(header_count) and int(header_count) == header_count == expected, "Container frame count differs from audited metadata")
        header_shape = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 3]
        require(header_shape == [480, 640, 3], "Unexpected IWS RGB dimensions")
        for index in range(expected + 1):
            ok, bgr = capture.read()
            if not ok:
                break
            require(index < expected, "Video has extra decoded frames")
            require(bgr.dtype == np.uint8 and list(bgr.shape) == header_shape, "Decoded RGB shape/dtype changed")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    require(len(frames) == expected, "Decoded frame count differs from audited metadata")
    require(sha(path) == expected_video_sha256, "Training RGB changed during decoding")
    return np.stack(frames)


def validate_arrays(arrays, expected_frames):
    require(set(arrays) == {"features", "commands", "frame_indices", "command_row_indices"}, "Unexpected cache arrays")
    features, commands = arrays["features"], arrays["commands"]
    require(features.dtype == np.float32 and features.shape == (expected_frames, 6144), "Feature shape/dtype mismatch")
    require(commands.dtype == np.float32 and commands.shape == (expected_frames, COMMAND_WIDTH), "Command row/width mismatch")
    require(np.isfinite(features).all() and np.isfinite(commands).all(), "Nonfinite cache arrays")
    native = np.arange(expected_frames, dtype=np.int64)
    for key in ("frame_indices", "command_row_indices"):
        require(arrays[key].dtype == np.int64 and np.array_equal(arrays[key], native), "Native row indexing changed")
    return arrays


class RunningMoments:
    """Float64 parallel-Welford moments; stable for large feature means."""
    def __init__(self, width):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        require(values.ndim == 2 and values.shape[1] == len(self.mean) and len(values) > 0, "Bad statistic dimensions")
        require(np.isfinite(values).all(), "Nonfinite statistic input")
        n = len(values); mean = values.mean(0); m2 = np.square(values - mean).sum(0)
        total = self.count + n; delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * n / total
        self.mean += delta * n / total; self.count = total

    def finish(self):
        require(self.count >= 2, "Insufficient training samples")
        return self.mean, np.maximum(np.sqrt(np.maximum(self.m2 / (self.count - 1), 0)), 1e-5)


def training_statistics(episodes, expected_training_ids):
    """Accept training episodes only; development records cause rejection."""
    expected = set(expected_training_ids)
    require(expected, "Empty training population")
    seen = set(); features = RunningMoments(384); commands = RunningMoments(COMMAND_WIDTH)
    hashes = {}
    for row, arrays in episodes:
        eid = row["episode_id"]
        require(row["split"] == "internal_train" and eid in expected and eid not in seen, "Training statistics reject development, unknown or duplicate records")
        validate_arrays(arrays, row["frames"])
        values = arrays["features"].reshape(-1, 384, 16).transpose(0, 2, 1).reshape(-1, 384)
        features.add(values); commands.add(arrays["commands"]); seen.add(eid)
        hashes[eid] = row["payload_sha256"]
    require(seen == expected, "Training statistics population incomplete")
    fm, fs = features.finish(); am, ast = commands.finish()
    return {"schema": "shiftwm_iws_training_statistics_v1", "fit_split": "internal_train", "episodes": sorted(seen),
            "episode_payload_sha256": hashes, "counts": {"feature_patch_rows": features.count, "command_rows": commands.count},
            "normalization": "shared_per_channel_over_all_internal_train_native_frames_and_16_patches", "ddof": 1, "std_floor": 1e-5,
            "feature_channel_mean": fm.tolist(), "feature_channel_std": fs.tolist(),
            "feature_mean": np.repeat(fm, 16).tolist(), "feature_std": np.repeat(fs, 16).tolist(),
            "command_mean": am.tolist(), "command_std": ast.tolist(),
            "command_units": "unverified recorded coordinates; no physical interpretation", "frame_timebase": "native stored row indices only"}
