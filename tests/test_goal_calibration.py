"""Check isolated diagnostics against the actual planning setup, not outcomes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch import nn

from shiftwm.data import TRAIN_COMBINATIONS
from shiftwm.generate import collect_episode
import shiftwm.evaluate as evaluation


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_goal_calibration.py"
spec = importlib.util.spec_from_file_location("goal_calibration_test", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class TraceModel(nn.Module):
    """Deterministic test-only encoder/dynamics with inspectable model inputs."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(history_length=3)
        self.action_dim = 10
        self.context_inputs = []
        self.rollout_inputs = []

    def encode_images(self, images, reference=False):
        return images.mean((-1, -2))

    def infer_context(self, features, actions):
        self.context_inputs.append((features.clone(), actions.clone()))
        return features.mean(1) * .1, features.mean(1) * 0

    def correct_observations(self, features, context):
        return features + context[:, None]

    def goal_embedding(self, goal, context):
        return self.correct_observations(self.encode_images(goal)[:, None], context)[:, 0]

    def rollout_features(self, features, past, future, contexts=None):
        self.rollout_inputs.append((features.clone(), past.clone(), future.clone()))
        return features[:, -1:] + future.sum(-1, keepdim=True).cumsum(1)


def make_fixture(tmp_path):
    episode = collect_episode({"env": "pusht", "output": str(tmp_path), "split": "development",
                               "seed": 31005, "dynamics_id": 1, "image_size": 32,
                               "steps": 8, "action_block": 5, "action_interface": "relative"})
    manifest = {"environment": "pusht", "image_size": 32, "action_block": 5,
                "action_interface": "relative", "episodes": [episode],
                "train_combinations": TRAIN_COMBINATIONS}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with np.load(tmp_path / episode["file"], allow_pickle=False) as archive:
        stored = {key: archive[key] for key in ("simulator_states", "task_states", "actions")}
    return episode, manifest, stored


@pytest.mark.parametrize("early_success", [False, True])
def test_same_actual_goal_pixels_history_and_support_status_as_evaluator(tmp_path, monkeypatch, early_success):
    """One real simulator fixture compares both setup implementations directly."""
    episode, manifest, stored = make_fixture(tmp_path)
    model = TraceModel()
    if early_success:
        # Exercise the otherwise rare stop halfway through a support action block.
        monkeypatch.setattr(diagnostic, "task_distance_success", lambda *args: (0.0, True))
        monkeypatch.setattr(evaluation, "task_distance_success", lambda *args: (0.0, True))
    acquired = diagnostic.acquire_planning_support(model, manifest, episode, stored, 1)
    renders = []
    original_render = evaluation.render_env

    def capture(*args):
        pixels = original_render(*args)
        renders.append(pixels.copy())
        return pixels

    monkeypatch.setattr(evaluation, "render_env", capture)
    result = evaluation.evaluate_planning(model, tmp_path, split="development", episodes_per_dynamics=1,
                                         policy="replay_oracle", samples=2, iterations=1, elites=2)
    assert len(result["records"]) == 1
    row = result["records"][0]
    assert acquired["success_during_context"] == row["success_during_context"]
    assert acquired["policy_eligible"] == row["policy_eligible"]
    assert acquired["goal_index"] == row["goal_index"] == 7
    assert acquired["initial_distance_after_context"] == pytest.approx(row["initial_distance_after_context"], abs=1e-9)
    np.testing.assert_array_equal(acquired["goal_pixels"], renders[0])
    np.testing.assert_array_equal(acquired["support_pixels"], np.stack(renders[1:1 + len(acquired["support_pixels"])]))
    assert acquired["support_native_steps"] == (1 if early_success else 10)
    if early_success:
        assert diagnostic.measure_episode(model, acquired, stored, 1) == {"diagnostic_status": "excluded_support_success"}
        assert not model.context_inputs


def synthetic_support():
    return {"policy_eligible": 1, "history_features": torch.arange(9).reshape(1, 3, 3).float() / 10,
            "past_actions": torch.arange(20).reshape(1, 2, 10).float() / 100,
            "goal_image": torch.full((1, 3, 8, 8), .3), "goal_pixels": np.full((8, 8, 3), 90, np.uint8),
            "support_pixels": np.full((3, 8, 8, 3), 60, np.uint8), "goal_index": 7}


def test_privileged_targets_and_future_actions_never_enter_context():
    model, acquired = TraceModel(), synthetic_support()
    stored = {"actions": np.arange(90, dtype=np.float32).reshape(9, 10) / 100}
    first = diagnostic.measure_episode(model, acquired, stored, 1)
    expected = torch.tensor(stored["actions"][2:7])[None]
    torch.testing.assert_close(model.rollout_inputs[0][2], expected)
    torch.testing.assert_close(model.rollout_inputs[1][2], torch.zeros_like(expected))
    changed, changed_stored = deepcopy(acquired), deepcopy(stored)
    changed["goal_pixels"][:] = 220
    changed["support_pixels"][:] = 240
    changed_stored["actions"][2:] *= 7
    second = diagnostic.measure_episode(model, changed, changed_stored, 1)
    for before, after in zip(model.context_inputs[0], model.context_inputs[1]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert first["context_observation_sha256"] == second["context_observation_sha256"]
    assert first["goal_calibration_mse"] != second["goal_calibration_mse"]
    assert first["terminal_action_sensitivity_mse"] != second["terminal_action_sensitivity_mse"]


def test_metrics_use_same_goal_with_distinct_canonical_and_adapted_coordinates():
    model, acquired = TraceModel(), synthetic_support()
    stored = {"actions": np.zeros((9, 10), np.float32)}
    metrics = diagnostic.measure_episode(model, acquired, stored, 1)
    expected_goal = model.encode_images(acquired["goal_image"]) + acquired["history_features"].mean(1) * .1
    canonical_goal = torch.full((1, 3), 90 / 255)
    terminal = acquired["history_features"][:, -1]
    assert metrics["goal_calibration_mse"] == pytest.approx(diagnostic.mse(expected_goal, canonical_goal))
    assert metrics["recorded_prediction_to_adapted_goal_mse"] == pytest.approx(diagnostic.mse(terminal, expected_goal))
    assert metrics["recorded_prediction_to_canonical_goal_mse"] == pytest.approx(diagnostic.mse(terminal, canonical_goal))
    assert metrics["terminal_action_sensitivity_mse"] == 0


def test_selection_is_development_only_in_evaluator_order():
    episodes = [{"trajectory_id": f"{split}-{seed}-{d}", "split": split, "seed": seed,
                 "dynamics_id": d, "steps": 8} for split, seed in [("development", 9), ("test", 8), ("train", 1), ("development", 3)]
                for d in [2, 1, 0]]
    selected = diagnostic.selected_episodes({"episodes": episodes, "train_combinations": TRAIN_COMBINATIONS}, count=2)
    assert [(e["seed"], e["dynamics_id"]) for e in selected] == [(3, 1), (9, 1)]
    with pytest.raises(ValueError, match="exactly 32"):
        diagnostic.selected_episodes({"episodes": episodes, "train_combinations": TRAIN_COMBINATIONS})


def test_completed_result_comparison_rejects_changed_support_or_sources():
    identity = {"checkpoint_sha256": "weights", "data_manifest_sha256": "data", "evaluator_sha256": "code"}
    row = {"trajectory_id": "dev1", "observation_id": 1, "goal_index": 7, "success_during_context": 0,
           "policy_eligible": 1, "initial_distance_after_context": 2.0, "initial_errors": {"joint_error": .5}}
    planning_row = {**row, "initial_joint_error": .5}
    planning = {"status": "complete", **identity,
                "planning": {"split": "development", "protocol": {"goal_offset": 5}, "records": [planning_row]}}
    assert diagnostic.planning_agreement({"records": [row]}, planning, identity)["status"] == "passed"
    planning["planning"]["records"][0]["success_during_context"] = 1
    with pytest.raises(ValueError, match="success_during_context"):
        diagnostic.planning_agreement({"records": [row]}, planning, identity)
    planning["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        diagnostic.planning_agreement({"records": [row]}, planning, identity)


def test_source_rejects_incomplete_training(tmp_path):
    package = tmp_path / "best"
    package.mkdir()
    (package / "model.pt").write_bytes(b"not loaded in this eligibility test")
    (package / "config.json").write_text(json.dumps({"model_config": {"mode": "framewise"}}))
    (tmp_path / "run_config.json").write_text(json.dumps({"epochs": 30}))
    (tmp_path / "training_summary.json").write_text(json.dumps({"status": "completed", "completed_epochs": 2}))
    with pytest.raises(ValueError, match="fully completed"):
        diagnostic.verify_completed_source(package)


def test_archived_pixel_comparison_has_exact_frame_indices_and_units():
    images = np.zeros((8, 2, 2, 3), dtype=np.uint8)
    support = images[:3].copy()
    support[1, 0, 0, 0] = 6
    goal = images[7].copy()
    goal[0, 0] = [3, 6, 9]
    acquired = {"policy_eligible": 1, "support_native_steps": 10, "goal_index": 7,
                "support_pixels": support, "goal_pixels": goal}
    result = diagnostic.archived_render_comparison(acquired, {"images": images})
    assert result["support_grouped_frame_indices"] == [0, 1, 2]
    assert result["goal_grouped_frame_index"] == 7
    assert result["support"]["mean_absolute_channel_difference_0_255"] == pytest.approx(6 / 36)
    assert result["support"]["fraction_changed_channel_values"] == pytest.approx(1 / 36)
    assert result["support"]["fraction_changed_spatial_pixels"] == pytest.approx(1 / 12)
    assert result["support_by_frame"][1]["maximum_absolute_channel_difference_0_255"] == 6
    assert result["goal"]["mean_absolute_channel_difference_0_255"] == 18 / 12
    assert result["goal"]["fraction_changed_spatial_pixels"] == .25
    acquired["policy_eligible"] = 0
    assert diagnostic.archived_render_comparison(acquired, {}) == {"status": "excluded_support_success"}
    with pytest.raises(ValueError, match="uint8"):
        diagnostic.pixel_difference(goal.astype(float), goal)


def test_renderer_metadata_never_creates_a_new_context():
    class NoContextPhysics:
        _contexts = None

        @property
        def contexts(self):
            raise AssertionError("Accessing this property would create a GL context")

    env = SimpleNamespace(env=SimpleNamespace(physics=NoContextPhysics()))
    result = diagnostic.existing_renderer_metadata(env, "reacher")
    assert result["status"] == "unknown"
    assert result["reason"] == "no_existing_GL_context"


def test_renderer_strings_are_queried_only_with_existing_context_current(monkeypatch):
    current = []

    @contextmanager
    def make_current():
        current.append(True)
        try:
            yield
        finally:
            current.clear()

    def get_string(key):
        assert current == [True]
        return {1: b"test renderer", 2: b"test vendor"}[key]

    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=SimpleNamespace(
        GL_RENDERER=1, GL_VENDOR=2, glGetString=get_string)))
    physics = SimpleNamespace(_contexts=SimpleNamespace(gl=SimpleNamespace(make_current=make_current)))
    env = SimpleNamespace(env=SimpleNamespace(physics=physics))
    result = diagnostic.existing_renderer_metadata(env, "reacher")
    assert result == {"status": "available", "renderer": "test renderer", "vendor": "test vendor"}
    assert not current
