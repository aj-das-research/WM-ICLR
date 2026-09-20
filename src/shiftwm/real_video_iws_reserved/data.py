"""Exact official handle inventory, with an explicit pre-access metadata-only mode."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from shiftwm.real_video_iws.data import canonical_hash, read_json, require, sha

TASK_WIDTHS = {"pusht": 4, "bimanual_box": 14, "bimanual_rope": 8}
SPLIT = "reserved_official_validation"
SPLIT_PATH = "configs/real_video_iws/split_v1.json"
REGISTRATION_PATH = "configs/real_video_iws_reserved_v1/registration.json"
RAW_ROOT = "data/real_video/iws_public_v1/extracted/iws_converted"


def safe_path(root, relative, *, exists=False):
    """Reject traversal and symlinks in every component, even inside the root."""
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and bool(relative.parts)
            and all(p not in ("..", ".") for p in relative.parts), "Unsafe relative path")
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), "Symlink traversal is forbidden")
    require(path.resolve().is_relative_to(root), "Path escaped repository")
    if exists:
        require(path.is_file(), "Missing required file: " + str(relative))
    return path


def registration_relative(root, registration_path=None):
    path = Path(registration_path or REGISTRATION_PATH)
    if path.is_absolute():
        try:
            path = path.relative_to(Path(root).resolve())
        except ValueError as error:
            raise ValueError("Registration must be inside the repository") from error
    return path


def read_metadata(root, task):
    require(task in TASK_WIDTHS, "Unknown reserved task")
    root = Path(root).resolve()
    document = read_json(safe_path(root, SPLIT_PATH, exists=True))
    require(document.get("kind") == "shiftwm_iws_trajectory_split_v1", "Wrong split schema")
    require(document.get("status") == "identities_frozen_before_model_training_and_official_validation_video_decoding", "Unfrozen split")
    require(set(document["partitions"]) == set(TASK_WIDTHS), "Incomplete three-task metadata")
    parts = document["partitions"][task]
    require(set(parts) == {"internal_train", "internal_development", SPLIT}, "Unexpected partitions")
    seen = set()
    for name, ids in parts.items():
        require(ids == sorted(set(ids)) and bool(ids), "Duplicate/unsorted/empty partition")
        require(all(isinstance(e, str) and len(e) == 6 and e.isdigit() for e in ids), "Invalid episode identity")
        require(not seen.intersection(ids), "Episode crosses partitions")
        require(document["counts"][task][name] == len(ids), "Partition count differs")
        seen.update(ids)
    require(len(parts[SPLIT]) == 10, "Reserved task must contain exactly ten trajectories")
    rows = document["metadata"][task]
    require(len(rows) == len(seen) and {r["episode_id"] for r in rows} == seen, "Metadata population differs")
    reserved = set(parts[SPLIT]); selected = {}
    for row in rows:
        eid = row["episode_id"]
        require(row["split"] == ("val" if eid in reserved else "train"), "Upstream split changed")
        require(row["file"] == f"{RAW_ROOT}/{task}/traj_{eid}/metadata.h5", "Metadata path differs")
        shape = row["shapes"]["target_qpos"]
        require(len(shape) == 2 and type(shape[0]) is int and shape[0] > 60 and shape[1] == TASK_WIDTHS[task], "Invalid native command shape")
        require(all(s and s[0] == shape[0] for s in row["shapes"].values()), "Native row counts differ")
        if eid in reserved:
            selected[eid] = row
    binding = document["reserved_handles"][task]
    require(binding["stored_command_rows"] == 60 and binding["windows"] == 200
            and binding["trajectories"] == 10, "Wrong reserved handle contract")
    handle_path = safe_path(root, binding["path"], exists=True)
    require(sha(handle_path) == binding["sha256"], "Original handle JSON changed")
    original = read_json(handle_path)
    require(original["meta"]["horizons"] == [60] and original["meta"]["total_handles"] == 200
            and original["meta"]["handles_per_horizon"] == 200, "Wrong handle metadata")
    require(set(original["handles_by_horizon"]) == {"60"}, "Extra official horizon")
    handles = []
    for h in original["handles_by_horizon"]["60"]:
        eid, start = h["traj_id"], h["frame_id"]
        require(eid in selected and type(start) is int and start >= 0, "Unknown episode or invalid start")
        frames = selected[eid]["shapes"]["target_qpos"][0]
        require(start + 60 < frames, "Handle exceeds original feasible-horizon contract")
        require(h["version"] == 1 and h["dataset_idx"] == 0 and h["group_name"] == task
                and h["sampled_horizon"] == 60 and h["feasible_horizon"] == frames - start - 1
                and h["camera_keys"] == ["camera_0"] and h["rgb_variant"] == "base"
                and h["interaction_frame_indices"] is None
                and h["dataset_root"] == f"data/iws_converted/{task}", "Handle identity/temporal contract differs")
        handles.append({"episode_id": eid, "start": start})
    require(len(handles) == 200 and len({(h["episode_id"], h["start"]) for h in handles}) == 200,
            "Missing or duplicate reserved handles")
    require({h["episode_id"] for h in handles} == reserved, "Handle population differs from reserved trajectories")
    return document, selected, handles


class ReservedInventory:
    """Metadata can be inspected before freeze; payload paths cannot be obtained."""
    def __init__(self, root, task, registration_path=None, *, _metadata_only=False):
        self.root = Path(root).resolve(); self.task = task
        self.registration_path = registration_relative(self.root, registration_path)
        self.metadata_only_access = _metadata_only
        self.document, self.rows, self.handles = read_metadata(self.root, task)
        self.episode_ids = sorted(self.rows)
        self.command_width = TASK_WIDTHS[task]
        self.split_sha256 = sha(self.root / SPLIT_PATH)
        self.handles_sha256 = self.document["reserved_handles"][task]["sha256"]
        self.manifest = {"task": task, "split": SPLIT, "episode_ids": self.episode_ids,
                         "handles": self.handles, "command_width": self.command_width,
                         "native_frames": sum(r["shapes"]["target_qpos"][0] for r in self.rows.values())}
        self.audit = {"task": task, "trajectories": 10, "handles": 200,
                      "handles_sha256": self.handles_sha256, "split_sha256": self.split_sha256,
                      "ordered_handles_sha256": canonical_hash(self.handles),
                      "native_frames": self.manifest["native_frames"],
                      "handles_per_trajectory": dict(sorted(Counter(h["episode_id"] for h in self.handles).items())),
                      "payload_paths_authorized": 0, "metadata_only": _metadata_only}
        self.registration = None
        if not _metadata_only:
            self.check_access()

    @classmethod
    def metadata_only(cls, root, task, registration_path=None):
        return cls(root, task, registration_path, _metadata_only=True)

    def check_access(self):
        require(not self.metadata_only_access, "Metadata-only inventory forbids reserved payload access")
        from .protocol import checked_registration
        registry = checked_registration(self.root, registration_path=self.registration_path, require_review=True)
        binding = registry["tasks"][self.task]
        expected = {"command_width": self.command_width, "episode_ids": self.episode_ids,
                    "expected_handles": 200, "expected_trajectories": 10,
                    "native_frames": self.manifest["native_frames"],
                    "handles_path": self.document["reserved_handles"][self.task]["path"],
                    "handles_sha256": self.handles_sha256,
                    "cache_root": f"data/features/iws_reserved_v1/{self.task}"}
        for key, value in expected.items():
            require(binding.get(key) == value, "Registered reserved population differs: " + key)
        require(registry["dependencies"].get(SPLIT_PATH) == self.split_sha256, "Registration uses another split")
        path = safe_path(self.root, self.registration_path, exists=True)
        digest = sha(path)
        if self.registration is not None:
            require(digest == self.registration_sha256, "Registration changed during reserved access")
        self.registration, self.registration_sha256 = registry, digest
        return registry

    def authorize(self, episode_id, split=None):
        require(episode_id in self.rows, "Non-reserved or unknown episode access rejected")
        require(split is None or split == SPLIT, "Only the registered reserved split may be read")
        self.check_access()
        return self.rows[episode_id], SPLIT

    def paths(self, episode_id, split=None):
        row, assigned = self.authorize(episode_id, split)
        directory = f"{RAW_ROOT}/{self.task}/traj_{episode_id}"
        metadata = safe_path(self.root, directory + "/metadata.h5", exists=True)
        video = safe_path(self.root, directory + "/camera_0_rgb.mp4", exists=True)
        self.audit["payload_paths_authorized"] += 1
        return row, assigned, metadata, video


def validate_arrays(arrays, frames, command_width):
    require(set(arrays) == {"features", "commands", "frame_indices", "command_row_indices"}, "Unexpected reserved cache arrays")
    require(arrays["features"].dtype == np.float32 and arrays["features"].shape == (frames, 6144), "Feature shape/dtype differs")
    require(arrays["commands"].dtype == np.float32 and arrays["commands"].shape == (frames, command_width), "Command shape/dtype differs")
    require(np.isfinite(arrays["features"]).all() and np.isfinite(arrays["commands"]).all(), "Nonfinite cache values")
    native = np.arange(frames, dtype=np.int64)
    for key in ("frame_indices", "command_row_indices"):
        require(arrays[key].dtype == np.int64 and np.array_equal(arrays[key], native), "Native row alignment changed")
    return arrays
