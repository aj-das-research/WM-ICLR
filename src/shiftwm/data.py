"""Trajectory-safe crossed-shift data with no held-out combination leakage.

The stored pixels are canonical simulator renders. Appearance changes are
deterministic, pointwise RGB transforms, so paired views refer to exactly the
same physical state. These are photometric shifts, not new camera viewpoints.
Factor identifiers are training metadata and are never model inputs.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


APPEARANCES = {
    0: {"name": "canonical", "gain": [1.0, 1.0, 1.0], "bias": [0.0, 0.0, 0.0]},
    1: {"name": "warm", "gain": [0.92, 0.78, 0.68], "bias": [0.05, 0.03, 0.02]},
    2: {"name": "cool", "gain": [0.65, 0.80, 0.95], "bias": [0.02, 0.03, 0.04]},
    3: {"name": "dim_extrapolation", "gain": [0.40, 0.48, 0.55], "bias": [0.0, 0.0, 0.0]},
}
TRAIN_COMBINATIONS = ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2))
DEVELOPMENT_COMBINATIONS = ((1, 1),)
HELDOUT_COMBINATIONS = ((2, 2),)


def appearance_transform(images: torch.Tensor, appearance_id: int) -> torch.Tensor:
    """Map CHW or TCHW float [0,1] images; transforms preserve pixel positions."""
    spec = APPEARANCES[int(appearance_id)]
    shape = (1,) * (images.ndim - 3) + (3, 1, 1)
    gain = images.new_tensor(spec["gain"]).reshape(shape)
    bias = images.new_tensor(spec["bias"]).reshape(shape)
    return (images * gain + bias).clamp(0, 1)


def split_combinations(split: str) -> tuple[tuple[int, int], ...]:
    if split in {"train", "val"}:
        return TRAIN_COMBINATIONS
    if split == "development":
        return DEVELOPMENT_COMBINATIONS
    if split == "test":
        return tuple((o, d) for d in range(3) for o in range(3))
    if split == "extrapolation":
        return ((3, 0), (0, 3), (3, 3))
    raise ValueError(f"Unknown split {split!r}")


def split_family(split: str) -> str:
    return {"development": "development", "extrapolation": "test"}.get(split, split)


def validate_manifest(manifest: dict) -> None:
    """Reject trajectory leakage and non-prespecified train combination changes."""
    seen: dict[str, str] = {}
    seen_seeds: dict[int, str] = {}
    for episode in manifest["episodes"]:
        uid, family = episode["trajectory_id"], episode["split"]
        if uid in seen:
            raise ValueError(f"Duplicate trajectory {uid}")
        seen[uid] = family
        seed = int(episode["seed"])
        if seed in seen_seeds and seen_seeds[seed] != family:
            raise ValueError(f"Seed {seed} crosses trajectory split boundary")
        seen_seeds[seed] = family
        if episode["steps"] < 1:
            raise ValueError(f"Empty trajectory {uid}")
    if tuple(map(tuple, manifest["train_combinations"])) != TRAIN_COMBINATIONS:
        raise ValueError("Manifest changed prespecified train combinations")


class TrajectoryDataset(Dataset):
    """Sliding chronological windows; every physical trajectory stays in one split.

    ``sequence_length`` counts images, so the action sequence is one shorter.
    Actions concatenate five native two-dimensional controls per transition.
    On train/validation, the paired view is restricted to TRAIN_COMBINATIONS.
    Canonical targets are privileged training/evaluation targets, not inputs to
    the inference context or policy.
    """

    def __init__(self, root: str | Path, split: str = "train", sequence_length: int = 8,
                 stride: int = 1, cache_size: int = 4, max_episodes: int | None = None,
                 feature_cache: str | Path | None = None, preload_features: bool = False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        validate_manifest(self.manifest)
        if sequence_length < 2 or stride < 1:
            raise ValueError("Need sequence_length >=2 and stride >=1")
        self.split = split
        self.feature_cache = Path(feature_cache) if feature_cache else None
        if self.feature_cache:
            cache_manifest = json.loads((self.feature_cache / "manifest.json").read_text())
            import hashlib
            manifest_sha = hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest()
            if cache_manifest["dataset_manifest_sha256"] != manifest_sha:
                raise ValueError("Feature cache belongs to a different dataset manifest")
        self.sequence_length = sequence_length
        combos = split_combinations(split)
        self.episodes = [ep for ep in self.manifest["episodes"] if ep["split"] == split_family(split)]
        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]
        self.windows = []
        for i, ep in enumerate(self.episodes):
            observations = [o for o, d in combos if d == ep["dynamics_id"]]
            for appearance in observations:
                # Identical pairing restrictions apply to baseline and method data.
                for start in range(0, ep["steps"] + 2 - sequence_length, stride):
                    self.windows.append((i, start, appearance, tuple(observations)))
        if not self.windows:
            raise ValueError(f"No windows for {split=} {sequence_length=} at {root}")
        self.preloaded = {}
        if preload_features:
            if self.feature_cache is None:
                raise ValueError("preload_features requires a feature cache; raw RGB preloading is disabled")
            self.preloaded = {i: self._read_episode_uncached(i) for i in range(len(self.episodes))}
            self._read_episode = self.preloaded.__getitem__
        else:
            self._read_episode = lru_cache(maxsize=cache_size)(self._read_episode_uncached)

    def _read_episode_uncached(self, index: int) -> dict[str, np.ndarray]:
        if self.feature_cache:
            with np.load(self.feature_cache / self.episodes[index]["file"], allow_pickle=False) as episode:
                return {key: episode[key] for key in ("features", "actions")}
        with np.load(self.root / self.episodes[index]["file"], allow_pickle=False) as episode:
            return {key: episode[key] for key in ("images", "actions")}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        ep_idx, start, appearance, allowed = self.windows[index]
        ep = self._read_episode(ep_idx)
        stop = start + self.sequence_length
        # Each paired image corresponds to the same timepoint as the primary.
        alternative = allowed[(allowed.index(appearance) + 1) % len(allowed)]
        sample = {
            "actions": torch.from_numpy(ep["actions"][start:stop - 1].copy()).float(),
            "observation_id": torch.tensor(appearance, dtype=torch.long),
            "paired_observation_id": torch.tensor(alternative, dtype=torch.long),
            "dynamics_id": torch.tensor(self.episodes[ep_idx]["dynamics_id"], dtype=torch.long),
            "trajectory_id": self.episodes[ep_idx]["trajectory_id"],
            "seed": torch.tensor(self.episodes[ep_idx]["seed"], dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }
        if self.feature_cache:
            for key, obs in (("features", appearance), ("paired_features", alternative), ("reference_features", 0)):
                sample[key] = torch.from_numpy(ep["features"][obs, start:stop].copy()).float()
        else:
            reference = torch.from_numpy(ep["images"][start:stop].copy()).permute(0, 3, 1, 2).float().div_(255)
            sample.update(images=appearance_transform(reference, appearance),
                          paired_images=appearance_transform(reference, alternative), reference_images=reference)
        return sample


def pixels_to_tensor(images: np.ndarray, appearance_id: int = 0) -> torch.Tensor:
    """Convert THWC or HWC uint8 render(s) into floating point channel-first."""
    data = torch.from_numpy(np.asarray(images).copy())
    data = data.permute(0, 3, 1, 2) if data.ndim == 4 else data.permute(2, 0, 1)
    return appearance_transform(data.float() / 255.0, appearance_id)
