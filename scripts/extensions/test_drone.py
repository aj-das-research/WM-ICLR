"""Physical, rendering, and data-isolation invariants for the drone extension."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from shiftwm.extensions.drone import DroneConfig, VisualDroneEnv

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("collect_drone", ROOT / "scripts/extensions/collect_drone.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def run_commands(gain, commands):
    env = VisualDroneEnv(gain, DroneConfig(max_steps=len(commands)))
    initial, _ = env.reset(seed=728)
    initial_state = env.simulator_state()
    images, states = [initial], [initial_state]
    try:
        for command in commands:
            image, _, terminated, _, info = env.step(command)
            assert not terminated, info
            images.append(image)
            states.append(env.simulator_state())
        return np.asarray(images), np.asarray(states)
    finally:
        env.close()


def test_repeat_seed_actions_exact_rgb_and_state():
    actions = np.tile(np.array([0.3, -0.2], np.float32), (20, 1))
    first = run_commands(0, actions)
    second = run_commands(0, actions)
    assert np.array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.count_nonzero(first[0][-1] != first[0][0]) > 50


def test_hidden_gain_changes_real_displacement_under_same_commands():
    actions = np.tile(np.array([0.4, 0.0], np.float32), (25, 1))
    trajectories = [run_commands(gain, actions)[1] for gain in (1, 0, 2)]
    displacements = [states[-1, 0] - states[0, 0] for states in trajectories]
    assert 0 < displacements[0] < displacements[1] < displacements[2]
    assert displacements[2] > 1.3 * displacements[0]
    for states in trajectories:
        assert np.max(np.abs(states[:, 2] - 0.65)) < 0.005


def test_grouped_model_data_excludes_privileged_state_and_reachable_goal(tmp_path):
    task = {"output": str(tmp_path), "split": "train", "seed": 990,
            "dynamics_id": 2, "native_steps": 40, "action_block": 5, "image_size": 64}
    metadata = collector.collect(task)
    with np.load(tmp_path / metadata["file"], allow_pickle=False) as model, \
         np.load(tmp_path / metadata["audit_file"], allow_pickle=False) as audit:
        assert set(model.files) == {"images", "actions"}
        np.testing.assert_array_equal(model["images"], audit["images"][::5])
        np.testing.assert_array_equal(model["actions"], audit["actions"].reshape(-1, 10))
        np.testing.assert_array_equal(audit["goal_image"], audit["images"][-1])
        np.testing.assert_array_equal(audit["goal_state"], audit["simulator_states"][-1])
        np.testing.assert_allclose(audit["executed_actions"], 1.25 * audit["actions"])
        assert audit["low_level_rpm"].shape == (40, 12, 4)
    assert metadata["terminal_goal_metrics"]["success"]
    # Idempotent resumability checks both model and privileged audit hashes.
    assert collector.collect(task) == metadata
    with (tmp_path / metadata["audit_file"]).open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        collector.collect(task)


@pytest.mark.parametrize("bad", [np.zeros(3), np.array([np.nan, 0]), np.array([2, 0])])
def test_invalid_actions_fail_without_clipping_silently(bad):
    env = VisualDroneEnv()
    env.reset(seed=0)
    try:
        with pytest.raises(ValueError):
            env.step(bad)
    finally:
        env.close()
