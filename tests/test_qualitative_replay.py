"""Exact physical replay, native stopping semantics, and rejection of bad evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qualitative_replay_test", ROOT / "scripts/diagnose_qualitative_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def archived_fixture(seed=2031024, mode="factorized"):
    data = ROOT / "data/world/pusht_relative"
    manifest = json.loads((data / "manifest.json").read_text())
    episode = next(ep for ep in manifest["episodes"] if ep["seed"] == seed
                   and ep["split"] == "development" and ep["dynamics_id"] == 1)
    directory = ROOT / "results/development_official_budget" / f"pusht_{mode}_s0"
    result = json.loads((directory / "planning_development.json").read_text())
    row = next(row for row in result["planning"]["records"] if row["seed"] == seed)
    with np.load(data / episode["file"], allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    with np.load(directory / "videos" / f"{row['trajectory_id']}-o1.npz", allow_pickle=False) as archive:
        method = {"record": row, "frames": archive["frames"], "goal": archive["goal_image"]}
    return manifest, episode, stored, method


def test_pusht_criterion_is_combined_position_not_block_only():
    goal = np.zeros(7)
    current = np.array([19., 0., 10., 0., 0., 0., 0.])
    errors, metadata = replay.criterion("pusht", current, goal)
    assert errors[0] == pytest.approx(np.sqrt(19**2 + 10**2))
    assert errors[0] > metadata["thresholds"][0]
    current[4] = 2 * np.pi - .1
    assert replay.criterion("pusht", current, goal)[0][1] == pytest.approx(.1)


def test_reacher_criterion_is_per_joint_and_unwrapped():
    errors, metadata = replay.criterion("reacher", np.array([.044, .044]), np.zeros(2))
    assert np.linalg.norm(errors) > .05
    assert np.all(errors < metadata["thresholds"])
    errors, _ = replay.criterion("reacher", np.array([2 * np.pi - .01, 0]), np.zeros(2))
    assert errors[0] > 6
    boundary, _ = replay.criterion("reacher", np.array([.05, 0]), np.zeros(2))
    assert not np.all(boundary < metadata["thresholds"])


@pytest.mark.parametrize("seed,mode,terminal", [(2031024, "factorized", 21), (2031000, "framewise", 48)])
def test_actual_archived_pusht_reproduces_every_frame_and_terminal(seed, mode, terminal):
    manifest, episode, stored, method = archived_fixture(seed, mode)
    arrays, details = replay.replay_method(manifest, episode, stored, method, "pusht", 1)
    assert details["validation"]["status"] == "exact"
    assert details["validation"]["first_success_native_step"] == terminal
    assert arrays["trace_native_times"].tolist() == list(range(terminal + 1))
    assert arrays["native_times"][:3].tolist() == [0, 5, 10]
    assert arrays["native_times"][-1] == terminal
    assert arrays["executed_native_actions"].shape == (terminal, 2)
    assert arrays["success_flags"][1:-1].sum() == 0
    assert arrays["success_flags"][-1]
    assert arrays["criterion_margin"][-1] < 1
    np.testing.assert_array_equal(arrays["shifted_frames"][2:], method["frames"])
    np.testing.assert_array_equal(arrays["canonical_frames"][:3], stored["images"][:3])


@pytest.mark.parametrize("which", ["goal", "frames"])
def test_changed_archived_pixel_prevents_claim_eligibility(which):
    manifest, episode, stored, method = archived_fixture()
    method = deepcopy(method)
    array = method[which]
    array.reshape(-1)[0] ^= 1
    with pytest.raises(ValueError, match="Exact pixel replay failed"):
        replay.replay_method(manifest, episode, stored, method, "pusht", 1)


def test_changed_recorded_endpoint_is_rejected():
    manifest, episode, stored, method = archived_fixture()
    method = deepcopy(method)
    method["record"]["final_distance"] += .1
    with pytest.raises(ValueError, match="Endpoint/support mismatch final_distance"):
        replay.replay_method(manifest, episode, stored, method, "pusht", 1)


def test_extra_action_after_success_is_rejected_before_synthetic_endpoint():
    manifest, episode, stored, method = archived_fixture()
    method = deepcopy(method)
    method["record"]["executed_action_blocks"][-1] += [0., 0.]
    method["record"]["native_steps"] += 1
    with pytest.raises(ValueError, match="continue after first upstream success"):
        replay.replay_method(manifest, episode, stored, method, "pusht", 1)


def test_source_mutation_invalidates_replay(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    hashes = {"source.py": replay.sha(source)}
    replay.verify_sources(hashes, tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Source changed"):
        replay.verify_sources(hashes, tmp_path)
