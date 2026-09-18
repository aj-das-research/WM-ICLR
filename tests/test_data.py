"""Tests for scientific data validity, not benchmark performance claims."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shiftwm.data import (APPEARANCES, TRAIN_COMBINATIONS, TrajectoryDataset,
                          appearance_transform, validate_manifest)
from shiftwm.generate import (collect_episode, make_env, restore_simulator_state,
                              simulator_state, collection_action, render_env)


def test_photometric_pairs_do_not_move_pixels():
    frame = torch.rand(4, 3, 19, 17)
    for appearance, spec in APPEARANCES.items():
        changed = appearance_transform(frame, appearance)
        gain = frame.new_tensor(spec["gain"]).reshape(1, 3, 1, 1)
        bias = frame.new_tensor(spec["bias"]).reshape(1, 3, 1, 1)
        torch.testing.assert_close(changed, (frame * gain + bias).clamp(0, 1))
    torch.testing.assert_close(appearance_transform(frame, 0), frame)


def test_manifest_rejects_seed_leakage():
    manifest = {"train_combinations": TRAIN_COMBINATIONS, "episodes": [
        {"trajectory_id": "train-1", "split": "train", "seed": 1, "steps": 9},
        {"trajectory_id": "test-1", "split": "test", "seed": 1, "steps": 9},
    ]}
    with pytest.raises(ValueError, match="crosses trajectory"):
        validate_manifest(manifest)


@pytest.mark.parametrize("name", ["pusht", "reacher"])
def test_real_simulator_state_restore_and_physics_change(name):
    initial_states, trajectories = [], []
    for dynamics in (0, 2):
        env = make_env(name, dynamics, 31005, image_size=32, action_interface="absolute" if name == "pusht" else "torque")
        try:
            original = simulator_state(env, name)
            rng = np.random.default_rng(995)
            env.step(collection_action(env, name, rng))
            restore_simulator_state(env, name, original)
            np.testing.assert_allclose(simulator_state(env, name), original, atol=1e-9)
            assert render_env(env, name, 32).shape == (32, 32, 3)
            initial_states.append(original)
            track = []
            rng = np.random.default_rng(887)
            for step in range(60):
                # Same executed targets/torques in both conditions; PushT first
                # targets the block to guarantee interaction, then displaces it.
                if name == "pusht":
                    target = original[4:6] + [40.0 * np.sin(step / 8), 30.0 * np.cos(step / 8)]
                    action = np.clip(target, 0, 512).astype(np.float32)
                else:
                    action = rng.uniform(-1, 1, 2).astype(np.float32)
                env.step(action)
                track.append(simulator_state(env, name))
            trajectories.append(np.asarray(track))
        finally:
            env.close()
    np.testing.assert_allclose(initial_states[0], initial_states[1], atol=1e-9)
    assert np.max(np.abs(trajectories[0] - trajectories[1])) > 1e-3


def test_dataset_prohibits_heldout_primary_and_paired_views(tmp_path):
    episodes = []
    for dynamics in range(3):
        episode = collect_episode({"env": "pusht", "output": str(tmp_path), "split": "train",
                                   "seed": 31005, "dynamics_id": dynamics, "image_size": 32,
                                   "steps": 7, "action_block": 5})
        episodes.append(episode)
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": episodes,
                                                       "train_combinations": TRAIN_COMBINATIONS}))
    dataset = TrajectoryDataset(tmp_path, sequence_length=5, cache_size=1)
    assert len(dataset) == len(TRAIN_COMBINATIONS) * 4
    found = set()
    for i in range(len(dataset)):
        item = dataset[i]
        dynamics = int(item["dynamics_id"])
        for key in ("observation_id", "paired_observation_id"):
            combination = (int(item[key]), dynamics)
            assert combination in TRAIN_COMBINATIONS
        found.add((int(item["observation_id"]), dynamics))
        assert item["images"].shape == (5, 3, 32, 32)
        assert item["actions"].shape == (4, 10)
        np.testing.assert_array_less(-1.0001, item["actions"].numpy())
        np.testing.assert_array_less(item["actions"].numpy(), 1.0001)
    assert found == set(TRAIN_COMBINATIONS)
