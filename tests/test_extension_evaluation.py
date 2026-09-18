"""Independent evaluation-contract checks; synthetic traces are not results."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from shiftwm.extensions.evaluate import accept_execution, evaluate_task, tasks_for_split
from shiftwm.extensions.rpc import decode_image, SimulatorClient


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("action_mean", torch.zeros(10))
        self.register_buffer("action_std", torch.ones(10))
        self.context_inputs = []
        self.encoded_images = []

    def encode_images(self, image):
        assert image.shape == (1, 3, 128, 128)
        assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
        self.encoded_images.append(image.clone())
        return image.mean((-1, -2))

    def infer_context(self, history, actions):
        assert history.shape == (1, 3, 3)
        assert actions.shape == (1, 2, 10)
        self.context_inputs.append((history.clone(), actions.clone()))
        return torch.zeros(1, 2), torch.zeros(1, 2)

    def goal_embedding(self, goal, observation_context):
        assert goal.shape == (1, 3, 128, 128)
        assert observation_context.shape == (1, 2)
        return goal.mean((-1, -2))

    def rollout_features(self, history, actions, future, contexts):
        return history[:, -1:, :].expand(-1, future.shape[1], -1)


class DummySolver:
    calls = []

    def __init__(self, **kwargs):
        self.device = kwargs["device"]

    def configure(self, **kwargs):
        assert kwargs["config"].action_block == 5

    def solve(self, info, init_action=None):
        assert set(info) == {"history_features", "past_actions", "observation_context", "dynamics_context", "goal_features"}
        assert all(torch.is_tensor(value) for value in info.values())
        self.calls.append({k: value.clone() for k, value in info.items()})
        return {"actions": torch.full((1, 5, 10), .2, device=self.device)}


class DummyRPC:
    def __init__(self, *, success_at=None, failure_at=None, secret=800000, initial_mismatch=False):
        self.success_at = success_at
        self.failure_at = failure_at
        self.secret = secret
        self.initial_mismatch = initial_mismatch
        self.step_requests = []
        self.n = 0

    def metrics(self):
        success = self.success_at is not None and self.n >= self.success_at
        invalid = self.failure_at is not None and self.n >= self.failure_at
        return {"success": success, "distance_m": .0 if success else .01,
                "goal_distance_m": .0 if success else .1,
                "valid_action": not invalid, "stable_deformation": True,
                "hidden_state": [self.secret] * 20, "hidden_gain": self.secret,
                "native_step": self.n}

    def image(self):
        return np.full((128, 128, 3), self.n % 256, np.uint8)

    def request(self, req):
        if req["op"] == "reset":
            self.n, self.limit = 0, req["max_steps"]
            initial = self.image()
            if self.initial_mismatch:
                initial[0, 0, 0] = 1
            return {"ok": True, "image": initial, "metrics": self.metrics()}
        assert req["op"] == "step"
        self.step_requests.append(deepcopy(req))
        executed, metrics = [], []
        for action in req["actions"]:
            if self.n >= self.limit:
                break
            self.n += 1
            executed.append(action)
            metrics.append(self.metrics())
            if metrics[-1]["success"] or not metrics[-1]["valid_action"]:
                break
        current = self.metrics()
        failure = not current["valid_action"]
        terminal = current["success"] or failure
        return {"ok": True, "image": self.image(), "metrics": current,
                "metrics_per_step": metrics, "executed_commands": executed,
                "terminated": terminal, "truncated": self.n >= self.limit and not terminal,
                "stop_reason": "invalid_action" if failure else "success" if terminal else "budget" if self.n >= self.limit else None}


def task():
    commands = np.random.default_rng(912).uniform(-.8, .8, (200, 2)).astype(np.float32)
    return {"trajectory_id": "synthetic-contract-fixture", "seed": 10,
            "dynamics_id": 1, "observation_id": 0, "goal_state": [987654.] * 3,
            "initial_image": np.zeros((128, 128, 3), np.uint8),
            "goal_image": np.full((128, 128, 3), 240, np.uint8),
            "commands": commands}


@pytest.fixture
def dummy_solver(monkeypatch):
    import stable_worldmodel.planning.solver.cem as cem
    DummySolver.calls = []
    monkeypatch.setattr(cem, "CEMSolver", DummySolver)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("policy", ["world_model", "random", "replay_oracle"])
def test_native_budget_includes_support_and_partial_final_block(dummy_solver, policy):
    fixture, model, client = task(), RecordingModel(), DummyRPC()
    result, arrays = evaluate_task(model, client, fixture, "surgery", policy, native_budget=17)
    assert result["native_calls"] == 17 and not result["success"]
    assert len(result["metrics_per_native_step"]) == 17
    assert result["stop_reason"] == "budget"
    np.testing.assert_array_equal(arrays["image_native_steps"], [0, 5, 10, 15, 17])
    np.testing.assert_array_equal(arrays["commands"][:10], fixture["commands"][:10])
    assert [len(x["actions"]) for x in client.step_requests] == [5, 5, 5, 2]
    if policy == "world_model":
        assert len(model.context_inputs) == 2
        np.testing.assert_array_equal(model.context_inputs[0][1].numpy(), fixture["commands"][:10].reshape(1, 2, 10))


@pytest.mark.parametrize("success_at", [1, 3, 5, 7, 10])
def test_early_support_success_never_invokes_learned_planning(dummy_solver, success_at):
    model = RecordingModel()
    result, arrays = evaluate_task(model, DummyRPC(success_at=success_at), task(), "surgery", native_budget=200)
    assert result["native_calls"] == success_at
    assert result["success"] and result["success_during_support"]
    assert result["first_success_native_step"] == success_at
    assert result["decisions"] == [] and model.context_inputs == [] and DummySolver.calls == []
    assert arrays["commands"].shape == (success_at, 2)


def test_success_in_partial_planned_block_logs_only_executed_actions(dummy_solver):
    result, arrays = evaluate_task(RecordingModel(), DummyRPC(success_at=12), task(), "surgery", native_budget=200)
    assert result["success"] and not result["success_during_support"]
    assert result["native_calls"] == 12 and result["first_success_native_step"] == 12
    assert len(result["decisions"]) == 1
    assert len(result["decisions"][0]["executed_commands"]) == 2
    np.testing.assert_array_equal(arrays["image_native_steps"], [0, 5, 10, 12])


@pytest.mark.parametrize("at", [3, 12])
def test_goal_proximity_on_invalid_action_is_failure(dummy_solver, at):
    result, _ = evaluate_task(RecordingModel(), DummyRPC(success_at=at, failure_at=at), task(), "surgery", native_budget=200)
    assert not result["success"] and not result["success_during_support"]
    assert result["first_success_native_step"] is None
    assert result["stop_reason"] == "invalid_action" and result["native_calls"] == at


def test_metrics_and_goal_coordinates_cannot_enter_model_context(dummy_solver):
    models = [RecordingModel(), RecordingModel()]
    cases = [task(), task()]
    cases[1]["goal_state"] = [-1234567.] * 3
    for model, case, hidden in zip(models, cases, [800000, -900000]):
        evaluate_task(model, DummyRPC(secret=hidden), case, "surgery", native_budget=17)
    assert len(models[0].context_inputs) == len(models[1].context_inputs) == 2
    for left, right in zip(models[0].context_inputs, models[1].context_inputs):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(models[0].encoded_images) == len(models[1].encoded_images)
    for a, b in zip(models[0].encoded_images, models[1].encoded_images):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_changed_live_initial_rgb_is_rejected_before_interaction(dummy_solver):
    client = DummyRPC(initial_mismatch=True)
    with pytest.raises(ValueError, match="Initial simulator RGB"):
        evaluate_task(RecordingModel(), client, task(), "surgery")
    assert client.step_requests == []


@pytest.mark.parametrize("fault", ["dropped", "changed", "metrics_count", "extra"])
def test_execution_accounting_rejects_silent_protocol_faults(fault):
    requested = np.zeros((5, 2), np.float32)
    response = {"executed_commands": requested.tolist(), "metrics_per_step": [{}] * 5,
                "metrics": {"success": False}, "terminated": False, "truncated": False}
    if fault == "dropped":
        response["executed_commands"] = requested[:3].tolist(); response["metrics_per_step"] = [{}] * 3
    elif fault == "changed":
        response["executed_commands"][0][0] = .1
    elif fault == "metrics_count":
        response["metrics_per_step"] = [{}] * 4
    else:
        response["executed_commands"] += [[0, 0]]; response["metrics_per_step"] += [{}]
    with pytest.raises(ValueError):
        accept_execution(response, requested)


def test_rpc_image_decoder_rejects_shape_and_payload_corruption():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    payload = {"shape": [2, 3, 3], "dtype": "uint8", "data": base64.b64encode(image.tobytes()).decode()}
    np.testing.assert_array_equal(decode_image(payload), image)
    for altered in [{**payload, "shape": [2, 3, 4]}, {**payload, "dtype": "float32"},
                    {**payload, "data": base64.b64encode(image.tobytes()[:-1]).decode()},
                    {**payload, "data": "not:base64!"}]:
        with pytest.raises((ValueError, TypeError)):
            decode_image(altered)


@pytest.mark.skipif(os.environ.get("SHIFTWM_REAL_EXTENSION_CONTRACT") != "1", reason="explicit real simulator integration run")
@pytest.mark.parametrize("domain", ["drone", "surgery"])
@pytest.mark.parametrize("policy", ["replay_oracle", "random"])
def test_real_simulator_evaluation_contract(domain, policy, tmp_path):
    root = Path("data/extensions") / f"{domain}_v1"
    tasks, _ = tasks_for_split(domain, root, "development", 1)
    fixture = tasks[0]
    with SimulatorClient(domain, tmp_path / f"{domain}-{policy}.log") as client:
        result, arrays = evaluate_task(RecordingModel(), client, fixture, domain,
                                      policy=policy, native_budget=200)
    assert 0 < result["native_calls"] <= 200
    assert arrays["image_native_steps"][-1] == result["native_calls"]
    assert len(result["metrics_per_native_step"]) == result["native_calls"]
    assert arrays["commands"].shape == (result["native_calls"], 2)
    np.testing.assert_array_equal(arrays["images"][0], fixture["initial_image"])
    if policy == "replay_oracle":
        assert result["success"], "Recorded-action reference should be reachable within its verified budget"
        np.testing.assert_array_equal(arrays["commands"], fixture["commands"][:result["native_calls"]])
    report_dir = Path("reports/extension_evaluation_contract")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "engineering_contract_with_no_trained_model_not_benchmark_results",
              "domain": domain, "policy": policy, "trajectory_id": fixture["trajectory_id"],
              "native_calls": result["native_calls"], "success": result["success"],
              "support_success": result["success_during_support"],
              "final_distance_m": result["final_distance_m"],
              "exact_initial_rgb": True, "accounting_verified": True}
    (report_dir / f"{domain}-{policy}.json").write_text(json.dumps(report, indent=2) + "\n")
