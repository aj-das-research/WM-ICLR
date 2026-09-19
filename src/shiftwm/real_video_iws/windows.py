"""Strict native IWS windows over an already validated immutable feature cache.

The caller must construct the task's frozen IWSFeatureCache through its complete
registration/static-identity gate. This module opens no raw RGB/HDF5 and never
opens official-validation cache payloads. Native rows are not physical time.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import SPLITS, require

TASK_WIDTHS = {"pusht": 4, "bimanual_box": 14, "bimanual_rope": 8}


def eligible_starts(frame_count, horizon=60, stride=5):
    """Preserve upstream unused-tail rule s+H<N, not s+H<=N."""
    require(isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count >= 0,
            "Invalid native frame count")
    require(isinstance(horizon, int) and not isinstance(horizon, bool) and horizon >= 2,
            "Horizon must supply at least two native command rows")
    require(isinstance(stride, int) and not isinstance(stride, bool) and stride >= 1,
            "Window-start stride must be a positive integer")
    return list(range(0, frame_count-horizon, stride))


class IWSWindowDataset(Dataset):
    """One observation, H native commands, H-1 loss-only targets per window.

    ``episodes`` and ``episode_ids`` retain all authorized trajectories, including
    those with zero eligible windows. ``audit`` explicitly records their status.
    Payloads are loaded once, after all selected identities have been authorized.
    """
    def __init__(self, cache, split, horizon=60, stride=5):
        require(split in SPLITS, "IWS windows reject reserved/test splits before payload access")
        eligible_starts(0, horizon, stride)
        self.cache, self.split = cache, split
        self.horizon, self.stride = horizon, stride
        self.statistics = cache.statistics
        require(self.statistics.get("fit_split") == "internal_train", "Normalization must be internal-training-only")
        require(self.statistics.get("episodes") == cache.inventory.partitions["internal_train"],
                "Normalization trajectory population differs from frozen training split")
        assigned = cache.inventory.assignment
        require(set(cache.index) == set(assigned), "Cache index contains missing, unknown or reserved identities")
        for eid, row in cache.index.items():
            require(row.get("episode_id") == eid and row.get("split") == assigned[eid] and row.get("split") in SPLITS,
                    "Cache index split identity mismatch")
        self.episode_ids = cache.inventory.selected(split)
        require(self.episode_ids == sorted(set(self.episode_ids)) and self.episode_ids,
                "Selected trajectory population is empty, duplicated or unsorted")
        authorized = []
        for eid in self.episode_ids:
            metadata, actual_split = cache.inventory.authorize(eid, split)
            receipt = cache.index[eid]
            task = receipt.get("task")
            require(task in TASK_WIDTHS and receipt.get("command_width") == TASK_WIDTHS[task],
                    "Unknown task or altered native command width")
            require(actual_split == split and receipt.get("frames") == metadata["shapes"]["target_qpos"][0]
                    and metadata["shapes"]["target_qpos"][1] == TASK_WIDTHS[task],
                    "Cache frame/command layout differs from authorized metadata")
            require(receipt.get("feature_dim") == 6144 and receipt.get("command_rows") == receipt["frames"],
                    "IWS requires N native feature rows and N command rows")
            authorized.append((eid, receipt))
        tasks = {receipt["task"] for _, receipt in authorized}
        require(len(tasks) == 1, "A dataset cannot mix task-specific command semantics")
        self.task = next(iter(tasks)); self.command_width = TASK_WIDTHS[self.task]
        self.episodes, self.windows, records = [], [], []
        for eid, expected in authorized:
            receipt, arrays = cache.episode(eid, split)
            require(receipt == expected, "Episode receipt changed after window authorization")
            frames = receipt["frames"]
            require(set(arrays) == {"features", "commands", "frame_indices", "command_row_indices"},
                    "Unexpected IWS payload arrays")
            features, commands = arrays["features"], arrays["commands"]
            require(features.dtype == np.float32 and features.shape == (frames, 6144), "Invalid native feature array")
            require(commands.dtype == np.float32 and commands.shape == (frames, self.command_width), "Invalid native command array")
            require(np.isfinite(features).all() and np.isfinite(commands).all(), "Nonfinite native cache arrays")
            native = np.arange(frames, dtype=np.int64)
            require(all(arrays[k].dtype == np.int64 and np.array_equal(arrays[k], native)
                        for k in ("frame_indices", "command_row_indices")), "Native row indexing changed")
            index = len(self.episodes)
            self.episodes.append({**receipt, "features": features, "commands": commands})
            starts = eligible_starts(frames, horizon, stride)
            self.windows.extend((index, start) for start in starts)
            records.append({"episode_id": eid, "episode_index": index, "split": split,
                            "frames": frames, "windows": len(starts), "first_start": starts[0] if starts else None,
                            "last_start": starts[-1] if starts else None,
                            "status": "eligible" if starts else "no_window_under_strict_unused_tail_rule"})
        self.audit = {"task": self.task, "split": split, "horizon_command_rows": horizon,
                      "predicted_offsets": list(range(1, horizon)), "window_start_stride": stride,
                      "eligibility": "range(0,N-H,stride), requiring s+H<N",
                      "episodes": len(self.episodes), "eligible_episodes": sum(r["windows"] > 0 for r in records),
                      "ineligible_episodes": sum(r["windows"] == 0 for r in records),
                      "windows": len(self.windows), "records": records}
        # An empty dataset still exposes the complete exclusion audit. Training
        # must reject len(dataset)==0 rather than fabricate a padded query.

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        episode = self.episodes[episode_index]
        end = start + self.horizon
        return {"initial_features": torch.from_numpy(episode["features"][start]),
                "commands": torch.from_numpy(episode["commands"][start:end]),
                "targets": torch.from_numpy(episode["features"][start+1:end]),
                "episode_index": episode_index, "episode_id": episode["episode_id"], "window_start": start}
