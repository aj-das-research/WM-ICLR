"""Train/validation-only patch caches; windowing reused from real_video.data."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from shiftwm.real_video.data import atomic_json, sha256


def validate_spatial_manifest(manifest):
    if manifest.get("status") != "complete" or not manifest.get("episodes"):
        raise ValueError("Incomplete spatial cache")
    sessions, ids = {}, set()
    for row in manifest["episodes"]:
        if row["split"] not in ("train", "val") or not row["session_id"] or row["episode_id"] in ids:
            raise ValueError("Invalid or held-out spatial episode")
        if row["session_id"] in sessions and sessions[row["session_id"]] != row["split"]:
            raise ValueError("Recording session crosses splits")
        if set(row["cameras"]) != {"exterior_image_1_left"}:
            raise ValueError("Spatial development uses camera one only")
        sessions[row["session_id"]] = row["split"]; ids.add(row["episode_id"])
    if set(sessions.values()) != {"train", "val"}:
        raise ValueError("Both train and validation are required")
    if manifest.get("feature_dim") != 6144 or manifest.get("action_dim") != 35:
        raise ValueError("Wrong spatial dimensions")

class SpatialDataset(Dataset):
    def __init__(self, root, split, horizon=5, stride=2,
                 camera="exterior_image_1_left", verify=True):
        if split not in ("train", "val"):
            raise ValueError("Spatial development rejects test payload access")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        validate_spatial_manifest(self.manifest)
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


def statistics_from_episodes(episodes):
    sums, squares, counts = {}, {}, {}
    for episode in episodes:
        features, actions = episode["features"], episode["actions"]
        if features.ndim != 2 or features.shape[1] != 6144 or actions.shape != (len(features)-1,35):
            raise ValueError("Bad spatial feature/action shape")
        for key, values in (("feature",features.reshape(-1,384,16).transpose(0,2,1).reshape(-1,384)),("action",actions)):
            values=values.astype(np.float64)
            if not np.isfinite(values).all(): raise ValueError("Nonfinite training arrays")
            sums[key]=sums.get(key,0)+values.sum(0)
            squares[key]=squares.get(key,0)+np.square(values).sum(0)
            counts[key]=counts.get(key,0)+len(values)
    result={"counts":counts}
    for key in sums:
        n=counts[key]
        if n<2: raise ValueError("Insufficient training samples")
        mean=sums[key]/n; std=np.maximum(np.sqrt(np.maximum((squares[key]-n*mean**2)/(n-1),0)),1e-5)
        result[key+"_mean"]=(np.repeat(mean,16) if key=="feature" else mean).tolist()
        result[key+"_std"]=(np.repeat(std,16) if key=="feature" else std).tolist()
    return result


def training_statistics(cache_root):
    root=Path(cache_root); manifest=json.loads((root/"manifest.json").read_text()); validate_spatial_manifest(manifest)
    def arrays():
        for row in manifest["episodes"]:
            if row["split"] != "train": continue
            record=row["cameras"]["exterior_image_1_left"]; path=root/record["file"]
            if sha256(path)!=record["sha256"]: raise ValueError("Training feature hash changed")
            with np.load(path,allow_pickle=False) as data: yield {"features":data["features"],"actions":data["actions"]}
    result={"fit_split":"train","camera":"exterior_image_1_left","cache_manifest_sha256":sha256(root/"manifest.json"),
        "std_floor":1e-5,"ddof":1,"normalization":"shared_per_channel_over_train_frames_and_patches",**statistics_from_episodes(arrays())}
    atomic_json(result,root/"training_statistics.json"); return result
