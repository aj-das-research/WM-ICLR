"""Strict internal-split access and native recorded-row loading for IWS Box and Rope.

No model windowing is defined here. A row index is not a physical timestamp.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SPLITS = ("internal_train", "internal_development")
TASK_WIDTHS = {"bimanual_box": 14, "bimanual_rope": 8}
DATA_BASE = Path("data/real_video/iws_public_v1/extracted/iws_converted")
SPLIT_RELATIVE = Path("configs/real_video_iws/split_v1.json")
METADATA_RELATIVE = Path("reports/evidence/iws_temporal_metadata_manifest.json")


from shiftwm.real_video_iws.data import (sha, read_json, require, canonical_hash,
    load_commands, decode_native_rgb, RunningMoments)


def task_width(task):
    require(task in TASK_WIDTHS, "Only exact registered Box/Rope task names are supported")
    return TASK_WIDTHS[task]


def validate_split(document, task):
    width = task_width(task)
    data_relative = DATA_BASE / task
    require(document.get("kind") == "shiftwm_iws_trajectory_split_v1", "Wrong split schema")
    require(document.get("status") == "identities_frozen_before_model_training_and_official_validation_video_decoding", "Unfrozen split")
    parts = document["partitions"][task]
    require(set(parts) == {*SPLITS, "reserved_official_validation"}, "Unexpected partition")
    seen = set()
    for split, ids in parts.items():
        require(ids and ids == sorted(set(ids)), "Empty, duplicated or unsorted partition")
        require(all(isinstance(eid, str) and len(eid) == 6 and eid.isdigit() for eid in ids), "Invalid episode ID")
        require(not seen.intersection(ids), "Episode crosses partitions")
        require(document["counts"][task][split] == len(ids), "Split count mismatch")
        seen.update(ids)
    rows = document["metadata"][task]
    require(len(rows) == len(seen) and {r["episode_id"] for r in rows} == seen, "Metadata population mismatch")
    reserved = set(parts["reserved_official_validation"])
    for row in rows:
        eid = row["episode_id"]
        require(row["split"] == ("val" if eid in reserved else "train"), "Upstream split identity mismatch")
        require(row["file"] == str(data_relative / f"traj_{eid}" / "metadata.h5"), "Unexpected metadata path")
        shape = row["shapes"].get("target_qpos", [])
        require(len(shape) == 2 and shape[0] >= 2 and shape[1] == width, "Wrong native command layout")
        require(all(s and s[0] == shape[0] for s in row["shapes"].values()), "Metadata rows do not align")
    return parts


class InternalInventory:
    """Authorize identities before any raw video/HDF5 open.

    The constructor reads frozen identity metadata, including reserved IDs/shapes;
    it never opens the underlying reserved-validation episode payloads.
    """

    def __init__(self, root, task, split_path=None):
        self.task = task
        self.command_width = task_width(task)
        self.data_relative = DATA_BASE / task
        self.root = Path(root)
        self.split_path = Path(split_path) if split_path else self.root / SPLIT_RELATIVE
        self.document = read_json(self.split_path)
        self.partitions = validate_split(self.document, task)
        self.assignment = {eid: split for split in SPLITS for eid in self.partitions[split]}
        self.rows = {row["episode_id"]: row for row in self.document["metadata"][task] if row["split"] == "train"}
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
        directory = self.root / self.data_relative / f"traj_{episode_id}"
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


def validate_arrays(arrays, expected_frames, command_width):
    require(command_width in TASK_WIDTHS.values(), "Unsupported task command width")
    require(set(arrays) == {"features", "commands", "frame_indices", "command_row_indices"}, "Unexpected cache arrays")
    features, commands = arrays["features"], arrays["commands"]
    require(features.dtype == np.float32 and features.shape == (expected_frames, 6144), "Feature shape/dtype mismatch")
    require(commands.dtype == np.float32 and commands.shape == (expected_frames, command_width), "Command row/width mismatch")
    require(np.isfinite(features).all() and np.isfinite(commands).all(), "Nonfinite cache arrays")
    native = np.arange(expected_frames, dtype=np.int64)
    for key in ("frame_indices", "command_row_indices"):
        require(arrays[key].dtype == np.int64 and np.array_equal(arrays[key], native), "Native row indexing changed")
    return arrays


def training_statistics(episodes, expected_training_ids, task):
    """Accept training episodes only; development records cause rejection."""
    width = task_width(task)
    expected = set(expected_training_ids)
    require(expected, "Empty training population")
    seen = set(); features = RunningMoments(384); commands = RunningMoments(width)
    hashes = {}
    for row, arrays in episodes:
        eid = row["episode_id"]
        require(row["split"] == "internal_train" and eid in expected and eid not in seen, "Training statistics reject development, unknown or duplicate records")
        require(row["task"] == task and row["command_width"] == width, "Statistic task identity differs")
        validate_arrays(arrays, row["frames"], width)
        values = arrays["features"].reshape(-1, 384, 16).transpose(0, 2, 1).reshape(-1, 384)
        features.add(values); commands.add(arrays["commands"]); seen.add(eid)
        hashes[eid] = row["payload_sha256"]
    require(seen == expected, "Training statistics population incomplete")
    fm, fs = features.finish(); am, ast = commands.finish()
    return {"schema": "shiftwm_iws_task_training_statistics_v1", "task": task, "command_width": width, "fit_split": "internal_train", "episodes": sorted(seen),
            "episode_payload_sha256": hashes, "counts": {"feature_patch_rows": features.count, "command_rows": commands.count},
            "normalization": "shared_per_channel_over_all_internal_train_native_frames_and_16_patches", "ddof": 1, "std_floor": 1e-5,
            "feature_channel_mean": fm.tolist(), "feature_channel_std": fs.tolist(),
            "feature_mean": np.repeat(fm, 16).tolist(), "feature_std": np.repeat(fs, 16).tolist(),
            "command_mean": am.tolist(), "command_std": ast.tolist(),
            "command_units": "unverified recorded coordinates; no physical interpretation", "frame_timebase": "native stored row indices only"}
