"""Controlled-rollout correctness fixtures, not research benchmark results."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from shiftwm.checkpoint import resume_training, save_package
from shiftwm.rollout_revision import (PACKAGE_KIND, RolloutRevision, RolloutRevisionConfig,
                                      load_rollout_package)

# Reuse the established dropout/BN donor rather than constructing an easier
# surrogate that would miss accidental donor training-mode changes.
from test_dynamics_revision import TinyBase, batch, donor


@pytest.fixture(autouse=True)
def small_cpu_fixture():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def revision(**overrides):
    return RolloutRevision(donor(), {"context_dim": 4, "context_hidden": 16, **overrides})


def teacher_donor_predictions(base, b):
    corrected = base.correct_observations(b["features"], torch.zeros(4, 4))
    states = torch.stack([corrected[:, i:i + 3] for i in range(5)], 1)
    actions = torch.stack([b["actions"][:, i:i + 3] for i in range(5)], 1)
    return base.predict_features(states.reshape(20, 3, 8), actions.reshape(20, 3, 4),
                                 torch.zeros(20, 4))[:, -1].reshape(4, 5, 8)


@pytest.mark.parametrize("objective", ["teacher_forced", "recursive"])
@pytest.mark.parametrize("context_mode", ["inferred", "constant"])
def test_zero_initialization_is_exact_donor_in_both_objectives(objective, context_mode):
    model = revision(objective=objective, context_mode=context_mode)
    b = batch()
    for train in (False, True):
        model.train(train)
        expected = (teacher_donor_predictions(model.donor, b) if objective == "teacher_forced"
                    else model.donor.rollout_features(b["features"][:, :3], b["actions"][:, :2],
                                                      b["actions"][:, 2:]))
        torch.testing.assert_close(model(b)["predictions"], expected, rtol=0, atol=0)
        torch.testing.assert_close(model.recursive_validation(b)["predictions"],
                                   model.donor.rollout_features(b["features"][:, :3], b["actions"][:, :2],
                                                                 b["actions"][:, 2:]), rtol=0, atol=0)
    goals = torch.rand(4, 3, 8, 8)
    torch.testing.assert_close(model.goal_embedding(goals, torch.randn(4, 4)),
                               model.donor.goal_embedding(goals, torch.randn(4, 4)), rtol=0, atol=0)


def test_teacher_forced_boundary_windows_actions_and_targets(monkeypatch):
    model = revision(objective="teacher_forced")
    b = batch()
    seen = []

    def predict(features, actions, context):
        seen.append((features, actions, context))
        return features + actions[..., :1] + 1

    monkeypatch.setattr(model, "predict_features", predict)
    out = model(b)
    assert out["predictions"].shape == (4, 5, 8)
    torch.testing.assert_close(out["targets"], b["reference_features"][:, 3:8], rtol=0, atol=0)
    assert len(seen) == 1
    states, actions, contexts = seen[0]
    states, actions = states.reshape(4, 5, 3, 8), actions.reshape(4, 5, 3, 4)
    corrected = model.correct_observations(b["features"], out["observation_context"])
    for step in range(5):
        torch.testing.assert_close(states[:, step], corrected[:, step:step + 3], rtol=0, atol=0)
        torch.testing.assert_close(actions[:, step], b["actions"][:, step:step + 3], rtol=0, atol=0)
        expected = corrected[:, step + 2] + b["actions"][:, step + 2, :1] + 1
        torch.testing.assert_close(out["predictions"][:, step], expected, rtol=0, atol=0)
    torch.testing.assert_close(contexts.reshape(4, 5, 4),
                               out["dynamics_context"][:, None].expand(-1, 5, -1), rtol=0, atol=0)


def test_recursive_boundary_horizon_actions_and_no_recalibration(monkeypatch):
    model = revision(objective="recursive")
    b = batch()
    seen = []

    def predict(features, actions, context):
        seen.append((features.clone(), actions.clone()))
        return features + actions[..., :1] + 1

    monkeypatch.setattr(model, "predict_features", predict)
    out = model(b)
    assert len(seen) == 5
    state = model.correct_observations(b["features"][:, :3], out["observation_context"])
    for step, (features, actions) in enumerate(seen):
        torch.testing.assert_close(features, state[:, -3:], rtol=0, atol=0)
        torch.testing.assert_close(actions, b["actions"][:, step:step + 3], rtol=0, atol=0)
        expected = state[:, -1:] + b["actions"][:, step + 2:step + 3, :1] + 1
        torch.testing.assert_close(out["predictions"][:, step:step + 1], expected, rtol=0, atol=0)
        state = torch.cat((state, expected), 1)
    torch.testing.assert_close(out["targets"], b["reference_features"][:, 3:8], rtol=0, atol=0)


@pytest.mark.parametrize("objective", ["teacher_forced", "recursive"])
def test_context_reads_only_corrected_support_and_executed_past(objective):
    model = revision(objective=objective)
    b = batch()
    captured = []
    hook = model.dynamics_context.register_forward_pre_hook(lambda module, args: captured.append(args))
    first = model(b)
    hook.remove()
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][0], model.correct_observations(b["features"][:, :3],
                                                                       first["observation_context"]), rtol=0, atol=0)
    torch.testing.assert_close(captured[0][1], model.normalize_actions(b["actions"][:, :2]), rtol=0, atol=0)
    other = deepcopy(b)
    other["features"][:, 3:] += 100
    other["actions"][:, 2:] += 100
    other["reference_features"] += 1000
    other["paired_features"] += 1000
    other["observation_id"][:] = 9
    torch.testing.assert_close(first["dynamics_context"], model(other)["dynamics_context"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="executed action"):
        model.infer_context(b["features"][:, :3], b["actions"][:, :3])


def test_recursive_predictions_have_no_query_or_target_leakage():
    model = revision(objective="recursive")
    with torch.no_grad():
        model.dynamics_adapter.affine.weight.normal_(std=.05)
    b = batch()
    b["features"].requires_grad_()
    b["reference_features"].requires_grad_()
    out = model(b)
    other = {key: value.detach().clone() for key, value in b.items()}
    other["features"][:, 3:] += 100
    other["reference_features"] += 1000
    other["paired_features"] -= 1000
    other["observation_id"][:] = 9
    changed = model(other)
    torch.testing.assert_close(out["predictions"], changed["predictions"], rtol=0, atol=0)
    assert not torch.equal(out["loss"], changed["loss"])
    out["loss"].backward()
    assert b["features"].grad[:, :3].count_nonzero() > 0
    assert b["features"].grad[:, 3:].count_nonzero() == 0
    assert b["reference_features"].grad is None


def test_last_recursive_loss_backpropagates_through_first_prediction(monkeypatch):
    model = revision().train()
    captured = []
    original = model.predict_features

    def predict(*args):
        out = original(*args)
        out.retain_grad()
        captured.append(out)
        return out

    monkeypatch.setattr(model, "predict_features", predict)
    output = model(batch())
    output["predictions"][:, -1].square().mean().backward()
    assert len(captured) == 5
    assert captured[0].grad[:, -1].abs().sum() > 0
    assert torch.isfinite(captured[0].grad).all()
    assert model.dynamics_adapter.affine.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.donor.parameters())


def test_constant_control_has_same_architecture_but_zero_network_inputs():
    inferred = revision()
    constant = revision(context_mode="constant")
    constant.load_state_dict(inferred.state_dict(), strict=True)
    assert [(n, p.shape, p.requires_grad) for n, p in constant.named_parameters()] == [
        (n, p.shape, p.requires_grad) for n, p in inferred.named_parameters()]
    captured = []
    hook = constant.dynamics_context.register_forward_pre_hook(lambda module, args: captured.append(args))
    b = batch()
    first = constant.infer_context(b["features"][:, :3], b["actions"][:, :2])[1]
    other = constant.infer_context(b["features"][:, :3] * 100 + 9, b["actions"][:, :2] - 200)[1]
    hook.remove()
    assert all(value.count_nonzero() == 0 for args in captured for value in args)
    torch.testing.assert_close(first, other, rtol=0, atol=0)
    torch.testing.assert_close(first, first[:1].expand_as(first), rtol=0, atol=0)
    # It is a learned constant, not a zero context or a frozen network.
    assert first.count_nonzero() > 0
    first.square().mean().backward()
    assert constant.dynamics_context.readout[-1].bias.grad.abs().sum() > 0
    assert constant.dynamics_context.gru.weight_ih_l0.grad.count_nonzero() == 0


@pytest.mark.parametrize("objective", ["teacher_forced", "recursive"])
@pytest.mark.parametrize("context_mode", ["inferred", "constant"])
def test_donor_state_dropout_and_goal_calibration_remain_fixed(objective, context_mode):
    model = revision(objective=objective, context_mode=context_mode).train()
    expected = {name for name, _ in model.named_parameters()
                if name.startswith(("dynamics_context.", "dynamics_adapter."))}
    assert {name for name, p in model.named_parameters() if p.requires_grad} == expected
    assert not any(module.training for module in model.donor.modules())
    before = deepcopy(model.donor.state_dict())
    context_before = deepcopy(model.dynamics_context.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
    b, goals = batch(), torch.rand(4, 3, 8, 8)
    goal_before = model.goal_embedding(goals, torch.randn(4, 4)).detach()
    torch.testing.assert_close(model(b)["predictions"], model(b)["predictions"], rtol=0, atol=0)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(b)
        assert out["loss"] is out["prediction_loss"]
        out["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.donor.parameters())
        optimizer.step()
    assert any(not torch.equal(v, model.dynamics_context.state_dict()[k]) for k, v in context_before.items())
    assert model.dynamics_adapter.affine.weight.count_nonzero() > 0
    for key, value in before.items():
        torch.testing.assert_close(model.donor.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(model.goal_embedding(goals, torch.randn(4, 4) * 100), goal_before, rtol=0, atol=0)


def test_common_recursive_validation_is_independent_of_training_objective():
    teacher = revision(objective="teacher_forced")
    recursive = revision(objective="recursive")
    recursive.load_state_dict(teacher.state_dict(), strict=True)
    b = batch()
    teacher_output, recursive_output = teacher(b), recursive(b)
    assert not torch.equal(teacher_output["predictions"], recursive_output["predictions"])
    for key, value in teacher.recursive_validation(b).items():
        torch.testing.assert_close(value, recursive.recursive_validation(b)[key], rtol=0, atol=0)
        torch.testing.assert_close(value, recursive_output[key], rtol=0, atol=0)
    expected = (recursive_output["predictions"] - b["reference_features"][:, 3:8]).square().mean()
    torch.testing.assert_close(recursive_output["prediction_loss"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("overrides", [
    {"history_length": 2}, {"context_dim": 0}, {"context_hidden": 1.5},
    {"context_mode": "target"}, {"objective": "old"}, {"mode": "framewise_dynamics_revision"},
    {"freeze_visual": False}, {"alignment_weight": 1}, {"dynamics_consistency_weight": .1},
    {"observation_consistency_weight": .1},
])
def test_invalid_protocol_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        RolloutRevisionConfig(**overrides)


@pytest.mark.parametrize("key,selection", [("features", slice(0, 7)),
                                            ("reference_features", slice(0, 7)),
                                            ("actions", slice(0, 6))])
def test_malformed_training_horizon_is_rejected(key, selection):
    b = batch()
    b[key] = b[key][:, selection]
    with pytest.raises(ValueError):
        revision()(b)


def test_portable_full_package_and_exact_optimizer_resume(tmp_path, monkeypatch):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    model = revision(objective="teacher_forced", context_mode="constant").train()
    model.provenance["revision_donor"] = {"path": "/nonexistent/original/donor"}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4)
    generator = torch.Generator().manual_seed(9)
    b = batch()
    model(b)["loss"].backward()
    optimizer.step(); optimizer.zero_grad(); scheduler.step()
    save_package(model, tmp_path, optimizer=optimizer, scheduler=scheduler, loader_generator=generator,
                 epoch=1, step=1)
    expected_random, expected_numpy = torch.rand(3), np.random.rand(3)
    model(b)["loss"].backward(); optimizer.step(); scheduler.step()
    expected = deepcopy(model.state_dict())
    loaded, saved = load_rollout_package(tmp_path)
    assert saved["config"]["package_kind"] == PACKAGE_KIND
    assert loaded.config.objective == "teacher_forced" and loaded.config.context_mode == "constant"
    assert not any(module.training for module in loaded.modules())
    optim2 = torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad], lr=.001)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, 4)
    resume_training(tmp_path, loaded, optim2, sched2, generator)
    torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
    loaded.train(); loaded(b)["loss"].backward(); optim2.step(); sched2.step()
    for key, value in expected.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert scheduler.state_dict() == sched2.state_dict()
    # Direct file loading is equally portable and strict.
    reloaded, _ = load_rollout_package(tmp_path / "model.pt")
    assert reloaded.config == loaded.config


@pytest.mark.parametrize("damage", ["kind", "format", "config_alias", "objective", "context_mode",
                                     "missing_key", "extra_key", "wrong_shape", "donor_mode"])
def test_loader_rejects_incompatible_package(tmp_path, monkeypatch, damage):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    save_package(revision(), tmp_path)
    state = torch.load(tmp_path / "model.pt", weights_only=True)
    if damage == "kind":
        state["config"]["package_kind"] = "frozen_framewise_with_dynamics_residual_v1"
    elif damage == "format":
        state["config"]["format_version"] = 2
    elif damage == "config_alias":
        state["config"]["model_config"]["objective"] = "teacher_forced"
    elif damage in {"objective", "context_mode"}:
        for alias in ("model_config", "revision_config"):
            state["config"][alias][damage] = "unsupported"
    elif damage == "missing_key":
        state["state_dict"].pop("dynamics_adapter.affine.bias")
    elif damage == "extra_key":
        state["state_dict"]["unrecognized"] = torch.zeros(1)
    elif damage == "wrong_shape":
        state["state_dict"]["dynamics_adapter.affine.bias"] = torch.zeros(3)
    elif damage == "donor_mode":
        state["config"]["donor_config"]["model_config"]["mode"] = "single"
    torch.save(state, tmp_path / "model.pt")
    with pytest.raises((ValueError, RuntimeError)):
        load_rollout_package(tmp_path)
