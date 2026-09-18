"""Episode-disjoint windows from immutable, real-video feature caches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def validate_manifest(manifest):
    if manifest.get("status") != "complete" or not manifest.get("episodes"):
        raise ValueError("Incomplete or empty real-video manifest")
    sessions, identifiers = {}, set()
    for row in manifest["episodes"]:
        if row["episode_id"] in identifiers:
            raise ValueError("Duplicate episode identifier")
        identifiers.add(row["episode_id"])
        session, split = row["session_id"], row["split"]
        if not session or split not in ("train", "val", "test"):
            raise ValueError("Unknown split or empty recording session")
        if session in sessions and sessions[session] != split:
            raise ValueError("A recording session crosses splits")
        sessions[session] = split
    if set(sessions.values()) != {"train", "val", "test"}:
        raise ValueError("All three session-disjoint splits are required")


class RealVideoDataset(Dataset):
    def __init__(self, root, split, horizon=5, stride=2,
                 camera="exterior_image_1_left", verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        validate_manifest(self.manifest)
        if horizon < 1 or stride < 1:
            raise ValueError("Invalid horizon/stride")
        self.history_length = 3
        self.horizon = horizon
        self.camera = camera
        self.episodes, self.windows = [], []
        sessions = {}
        for episode in self.manifest["episodes"]:
            session, assigned = episode["session_id"], episode["split"]
            if session in sessions and sessions[session] != assigned:
                raise ValueError("A recording session crosses splits")
            sessions[session] = assigned
            if assigned != split or camera not in episode["cameras"]:
                continue
            entry = episode["cameras"][camera]
            path = self.root / entry["file"]
            if verify and sha256(path) != entry["sha256"]:
                raise ValueError(f"Feature payload changed: {path}")
            with np.load(path, allow_pickle=False) as data:
                features = np.array(data["features"], dtype=np.float32)
                actions = np.array(data["actions"], dtype=np.float32)
                frame_indices = np.array(data["frame_indices"])
            if features.ndim != 2 or features.shape[1] != self.manifest["feature_dim"]:
                raise ValueError("Wrong feature shape")
            if actions.shape != (len(features)-1, self.manifest["action_dim"]):
                raise ValueError("Actions must join every adjacent stored image")
            if not np.isfinite(features).all() or not np.isfinite(actions).all():
                raise ValueError("Nonfinite input")
            if (frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer)
                    or len(frame_indices) != len(features) or len(frame_indices) < 1
                    or frame_indices[0] != 0 or not np.all(np.diff(frame_indices) == 5)):
                raise ValueError("Grouped-frame spacing changed")
            index = len(self.episodes)
            self.episodes.append({**episode, "features": features, "actions": actions,
                                  "frame_indices": frame_indices})
            length = self.history_length + horizon
            self.windows.extend((index, start) for start in range(0, len(features)-length+1, stride))
        if not self.windows:
            raise ValueError(f"No complete {split}/{camera} prediction windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        episode = self.episodes[episode_index]
        end = start + self.history_length + self.horizon
        return {"features": torch.from_numpy(episode["features"][start:end]),
                "actions": torch.from_numpy(episode["actions"][start:end-1]),
                "episode_index": episode_index, "window_start": start}


def training_statistics(cache_root):
    root = Path(cache_root)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    sums, squares, counts = {}, {}, {}
    for episode in manifest["episodes"]:
        if episode["split"] != "train":
            continue
        record = episode["cameras"]["exterior_image_1_left"]
        path = root / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Training payload changed")
        with np.load(path, allow_pickle=False) as values:
            features, actions = values["features"], values["actions"]
            if (features.ndim != 2 or features.shape[1] != manifest["feature_dim"]
                    or actions.shape != (len(features)-1,manifest["action_dim"])
                    or not np.isfinite(features).all() or not np.isfinite(actions).all()):
                raise ValueError("Invalid finite training feature/action arrays")
            for key, field in (("feature", "features"), ("action", "actions")):
                array = values[field].astype(np.float64)
                sums[key] = sums.get(key, 0) + array.sum(0)
                squares[key] = squares.get(key, 0) + np.square(array).sum(0)
                counts[key] = counts.get(key, 0) + len(array)
    result = {"fit_split": "train", "camera": "exterior_image_1_left",
              "cache_manifest_sha256": sha256(root / "manifest.json"),
              "std_floor": 1e-5, "ddof": 1, "counts": counts}
    for key in sums:
        if counts[key] < 2:
            raise ValueError("Insufficient training data")
        mean = sums[key] / counts[key]
        variance = np.maximum((squares[key] - counts[key] * mean**2)/(counts[key]-1), 0)
        result[key + "_mean"] = mean.tolist()
        result[key + "_std"] = np.maximum(np.sqrt(variance), 1e-5).tolist()
        result[key + "_constant_dimensions"] = np.flatnonzero(np.sqrt(variance) < 1e-5).tolist()
    if set(sums) != {"feature", "action"}:
        raise ValueError("Missing complete training statistics")
    atomic_json(result, root / "training_statistics.json")
    return result
