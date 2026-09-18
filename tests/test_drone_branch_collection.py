"""Branch supervision contracts; dummy traces are not experiment results."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drone_branch_collector", ROOT / "scripts/extensions/collect_drone_branches.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def source_manifest():
    rows = []
    for seed in range(64):
        for gain in range(3):
            rows.append({"trajectory_id": f"train-{seed}-{gain}", "split": "train", "seed": seed,
                         "dynamics_id": gain, "action_block": 5, "native_steps": 200,
                         "protocol": {"response_gain": collector.GAINS[gain]}})
    rows.append({"trajectory_id": "test-100-0", "split": "test", "seed": 100, "dynamics_id": 0})
    return {"environment": "drone", "episodes": rows}


def test_selection_is_first32_train_seeds_crossed_three_gains_without_outcome_filtering():
    manifest = source_manifest()
    manifest["episodes"] = list(reversed(manifest["episodes"]))
    selected = collector.select_contexts(manifest)
    assert [(row["seed"], row["dynamics_id"]) for row in selected] == [(s, g) for s in range(32) for g in range(3)]
    assert len(selected) == 96 and {row["split"] for row in selected} == {"train"}


@pytest.mark.parametrize("mutation", ["cross_split_seed", "missing_gain", "duplicate", "gain_mismatch"])
def test_selection_rejects_ambiguous_or_leaking_source(mutation):
    manifest = source_manifest()
    if mutation == "cross_split_seed":
        manifest["episodes"][-1]["seed"] = 0
    elif mutation == "missing_gain":
        manifest["episodes"].pop(0)
    elif mutation == "duplicate":
        manifest["episodes"].append(deepcopy(manifest["episodes"][0]))
    else:
        manifest["episodes"][0]["protocol"]["response_gain"] = .125
    with pytest.raises(ValueError):
        collector.select_contexts(manifest)


def test_candidates_match_preoutcome_diagnostic_registration_exactly():
    labels, native = collector.candidate_library()
    previous = json.loads((ROOT / "reports/evidence/drone_action_ranking/registration.json").read_text())
    assert labels == previous["candidate_labels"]
    np.testing.assert_array_equal(native, np.array(previous["candidate_commands"], np.float32))
    assert native.shape == (32, 25, 2) and native.dtype == np.float32
    assert np.isfinite(native).all() and np.abs(native).max() <= 1
    assert np.all(native[0] == 0)
    assert np.all(native[9:13, 5:] == 0)
    assert len(set(collector.array_sha256(value) for value in native)) == 32


class DummyEnv:
    def __init__(self, *, failure_at=None, failure_kind="crash", success_at=None,
                 changed_rgb_at=None, changed_state_at=None, hidden=9999):
        self.failure_at = failure_at
        self.failure_kind = failure_kind
        self.success_at = success_at
        self.changed_rgb_at = changed_rgb_at
        self.changed_state_at = changed_state_at
        self.hidden = hidden
        self.reset_count = 0
        self.commands_by_branch = []

    def reset(self, seed):
        self.n = 0
        self.reset_count += 1
        self.commands_by_branch.append([])
        return self.image(), {"seed": seed}

    def image(self):
        image = np.full((128, 128, 3), self.n, np.uint8)
        if self.n == self.changed_rgb_at:
            image[0, 0, 0] += 1
        return image

    def simulator_state(self):
        state = np.zeros(20, np.float64)
        state[0] = self.n
        state[1] = self.hidden
        if self.n == self.changed_state_at:
            state[2] = .00000000001
        return state

    def set_goal(self, state):
        self.goal = state.copy()

    def step(self, command):
        self.n += 1
        self.commands_by_branch[-1].append(np.array(command, np.float32))
        self.last_executed_action = np.asarray(command, np.float32) * 1.25
        self.last_rpm = np.full((12, 4), self.hidden, np.float32)
        failure = self.n == self.failure_at
        info = {"crash": failure and self.failure_kind == "crash",
                "workspace_escape": failure and self.failure_kind == "workspace_escape",
                "success": self.success_at is not None and self.n >= self.success_at,
                "goal_distance_m": float(self.hidden), "speed_m_s": 0., "altitude_error_m": 0.}
        return self.image(), 0., failure, self.n == 35, info


def support(hidden=9999):
    env = DummyEnv(hidden=hidden)
    initial, _ = env.reset(17)
    images, states = [initial], [env.simulator_state()]
    commands = np.linspace(-.5, .5, 20, dtype=np.float32).reshape(10, 2)
    for command in commands:
        image, *_ = env.step(command)
        images.append(image)
        states.append(env.simulator_state())
    return {"seed": 17, "images": np.stack(images), "states": np.stack(states),
            "actions": commands, "goal_state": np.full(20, hidden, np.float64)}


def test_every_branch_fresh_reset_replays_all_original_support_commands():
    recorded = support()
    native = collector.candidate_library()[1][[0, 1, 9]]
    env = DummyEnv(success_at=3)
    main, audit, outcomes = collector.simulate_context(env, recorded, native)
    assert env.reset_count == 3
    assert set(main) == collector.MODEL_FIELDS
    for actual, future in zip(env.commands_by_branch, native):
        np.testing.assert_array_equal(actual, np.concatenate([recorded["actions"], future]))
    np.testing.assert_array_equal(main["support_images"], recorded["images"][[0, 5, 10]])
    np.testing.assert_array_equal(main["candidate_actions"], native.reshape(3, 5, 10))
    assert np.all(main["executed_lengths"] == 25) and main["executed_mask"].all() and main["valid_mask"].all()
    assert np.all(main["terminal_images"] == 35)
    assert main["future_image_mask"].all()
    for boundary, native_index in enumerate((15, 20, 25, 30, 35)):
        assert np.all(main["future_images"][:, boundary] == native_index)
    assert all(row["reason"] == "full_horizon" for row in outcomes)
    # Native success during support or future never shortens this supervised branch.
    assert all(len(row["metrics_per_native_step"]) == 25 for row in outcomes)
    assert audit["native_states"].shape == (3, 26, 20)


@pytest.mark.parametrize("at", [0, 1, 4, 5, 6, 9, 10])
@pytest.mark.parametrize("field", ["rgb", "state"])
def test_every_support_image_and_state_must_match_not_just_seed_or_block_endpoints(at, field):
    env = DummyEnv(**{f"changed_{field}_at": at})
    with pytest.raises(ValueError, match="Exact support"):
        collector.simulate_context(env, support(), collector.candidate_library()[1][:1])


@pytest.mark.parametrize("kind", ["crash", "workspace_escape"])
@pytest.mark.parametrize("failure_at", [11, 13, 15, 35])
def test_unsafe_branch_retained_with_actual_terminal_and_exact_prefix(kind, failure_at):
    env = DummyEnv(failure_at=failure_at, failure_kind=kind)
    main, audit, outcomes = collector.simulate_context(env, support(), collector.candidate_library()[1][:1])
    length = failure_at - 10
    assert main["executed_lengths"][0] == length
    assert main["executed_mask"].sum() == length
    assert not main["valid_mask"][0]
    assert np.all(main["terminal_images"][0] == failure_at)
    np.testing.assert_array_equal(main["future_image_mask"][0], np.arange(5, 26, 5) <= length)
    assert np.all(main["future_images"][~main["future_image_mask"]] == 0)
    assert outcomes[0]["reason"] == kind
    assert len(outcomes[0]["metrics_per_native_step"]) == length
    assert np.isfinite(audit["native_states"][0, :length + 1]).all()
    assert np.isnan(audit["native_states"][0, length + 1:]).all()
    assert np.isnan(audit["gain_scaled_commands"][0, length:]).all()


def test_support_failure_aborts_without_silently_creating_branch():
    with pytest.raises(ValueError, match="observed support became unsafe"):
        collector.simulate_context(DummyEnv(failure_at=3), support(), collector.candidate_library()[1][:1])


def test_privileged_values_cannot_enter_model_shard():
    candidates = collector.candidate_library()[1][:1]
    main_a, audit_a, _ = collector.simulate_context(DummyEnv(hidden=912), support(hidden=912), candidates)
    main_b, audit_b, _ = collector.simulate_context(DummyEnv(hidden=648), support(hidden=648), candidates)
    for key in collector.MODEL_FIELDS:
        np.testing.assert_array_equal(main_a[key], main_b[key])
    assert not np.array_equal(audit_a["native_states"], audit_b["native_states"])
    bad = dict(main_a, hidden_state=np.zeros(20))
    with pytest.raises(ValueError, match="privileged fields"):
        collector.validate_model_arrays(bad, 1)


def test_partial_mask_and_validity_are_checked():
    main, _, _ = collector.simulate_context(DummyEnv(failure_at=13), support(), collector.candidate_library()[1][:1])
    bad = deepcopy(main)
    bad["valid_mask"][0] = True
    with pytest.raises(ValueError, match="Partial branch"):
        collector.validate_model_arrays(bad, 1)
    bad = deepcopy(main)
    bad["executed_mask"][0, 7] = True
    with pytest.raises(ValueError, match="exactly the actual prefix"):
        collector.validate_model_arrays(bad, 1)


def test_source_payload_hash_corruption_is_rejected(tmp_path):
    payload = tmp_path / "source.npz"
    payload.write_bytes(b"original")
    row = {"split": "train", "file": "source.npz", "sha256": collector.sha256(payload)}
    payload.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        collector.load_support(tmp_path, row)


def test_nontraining_source_cannot_be_loaded(tmp_path):
    with pytest.raises(ValueError, match="another split"):
        collector.load_support(tmp_path, {"split": "development"})


@pytest.mark.skipif(os.getenv("SHIFTWM_BRANCH_REAL") != "1", reason="Explicit bounded real simulator contract")
def test_real_original_training_support_exact_all_frames_and_states():
    from shiftwm.extensions.drone import DroneConfig, VisualDroneEnv
    manifest = json.loads((ROOT / "data/extensions/drone_v1/manifest.json").read_text())
    row = collector.select_contexts(manifest)[0]
    recorded = collector.load_support(ROOT / "data/extensions/drone_v1", row)
    recorded["seed"] = row["seed"]
    env = VisualDroneEnv(row["dynamics_id"], DroneConfig(max_steps=35))
    try:
        main, audit, outcomes = collector.simulate_context(env, recorded, collector.candidate_library()[1][:1])
    finally:
        env.close()
    assert main["valid_mask"].all() and main["executed_lengths"][0] == 25
    assert np.isfinite(audit["native_states"]).all()


def collected_fixture(tmp_path, monkeypatch):
    import shiftwm.extensions.drone as drone
    output = tmp_path / "branches"
    output.mkdir()
    registration = {"data_root": str(tmp_path / "source"), "simulator_config": {"max_steps": 35}}
    collector.atomic_json(output / "registration.json", registration)
    monkeypatch.setattr(collector, "verify_sources", lambda registration: None)
    monkeypatch.setattr(collector, "load_support", lambda *args: support())
    monkeypatch.setattr(drone, "VisualDroneEnv", lambda *args: DummyEnv())
    monkeypatch.setattr(DummyEnv, "close", lambda self: None, raising=False)
    row = {"trajectory_id": "dummy-train-17-0", "seed": 17, "dynamics_id": 0, "split": "train"}
    task = {"output": str(output), "row": row,
            "registration_sha256": collector.sha256(output / "registration.json")}
    metadata = collector.collect_context(task)
    return output, task, metadata


def test_resume_reuses_identical_registered_artifacts_without_new_simulation(tmp_path, monkeypatch):
    output, task, first = collected_fixture(tmp_path, monkeypatch)
    import shiftwm.extensions.drone as drone
    def forbidden(*args):
        raise AssertionError("Resume created a simulator")
    monkeypatch.setattr(drone, "VisualDroneEnv", forbidden)
    second = collector.collect_context(task)
    assert first == second
    with np.load(output / second["file"], allow_pickle=False) as archive:
        assert set(archive.files) == collector.MODEL_FIELDS


@pytest.mark.parametrize("file_key", ["file", "audit_file", "audit_json"])
def test_resume_rejects_corrupted_existing_artifact(tmp_path, monkeypatch, file_key):
    output, task, metadata = collected_fixture(tmp_path, monkeypatch)
    (output / metadata[file_key]).write_bytes(b"corrupted after collection")
    with pytest.raises(ValueError, match="Existing context payload changed"):
        collector.collect_context(task)


def test_resume_rechecks_original_support_payload_and_registration(tmp_path, monkeypatch):
    output, task, _ = collected_fixture(tmp_path, monkeypatch)
    broken = dict(task, registration_sha256="not the registered hash")
    with pytest.raises(ValueError, match="Registration changed"):
        collector.collect_context(broken)
    def bad_source(*args):
        raise ValueError("Source payload hash mismatch")
    monkeypatch.setattr(collector, "load_support", bad_source)
    with pytest.raises(ValueError, match="Source payload hash mismatch"):
        collector.collect_context(task)


def test_missing_future_targets_cannot_be_marked_available_or_filled():
    main, _, _ = collector.simulate_context(DummyEnv(failure_at=13), support(), collector.candidate_library()[1][:1])
    bad = deepcopy(main)
    bad["future_image_mask"][0, 0] = True
    with pytest.raises(ValueError, match="executed block boundaries"):
        collector.validate_model_arrays(bad, 1)
    bad = deepcopy(main)
    bad["future_images"][0, 0] = 13
    with pytest.raises(ValueError, match="zero with a false mask"):
        collector.validate_model_arrays(bad, 1)
