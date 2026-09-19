"""Bound-removal isolation tests; no dataset, benchmark, or model training runs."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from shiftwm.real_video_iws import model as bounded
from shiftwm.real_video_iws_unbounded import model as unbounded


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def statistics(width):
    return (torch.linspace(-.2, .2, 384).repeat_interleave(16),
            torch.linspace(.5, 1.5, 384).repeat_interleave(16),
            torch.linspace(-.1, .1, width), torch.linspace(.7, 1.1, width))


def make(module=unbounded, width=4, depth=1, learned=True):
    torch.manual_seed(173)
    mode = "unbounded_spatial_mix" if module is unbounded else "bounded_spatial_mix"
    model = module.SingleObservationWorldModel(
        {"mode": mode, "action_dim": width, "depth": depth}, *statistics(width)).eval()
    if learned:
        with torch.no_grad():
            model.output_projection[-1].weight.normal_(0, .025)
            model.output_projection[-1].bias.normal_(0, .015)
            for block in model.predictor.transformer.layers:
                block.adaLN_modulation[-1].weight.normal_(0, .04)
                block.adaLN_modulation[-1].bias.normal_(0, .02)
            model.gate.weight.normal_(0, .04)
    return model


def inputs(width=4, horizon=8):
    generator = torch.Generator().manual_seed(71)
    return (torch.randn(1, 6144, generator=generator),
            torch.randn(1, horizon, width, generator=generator))


def test_frozen_base_is_unchanged_and_shared_helpers_are_exact_copies():
    assert hashlib.sha256(Path(bounded.__file__).read_bytes()).hexdigest() == unbounded.SOURCE_BASE_SHA256
    for name in ("tokens", "flatten", "normalize_features", "normalize_commands", "_encode", "predict", "forward"):
        assert inspect.getsource(getattr(bounded.SingleObservationWorldModel, name)) == inspect.getsource(
            getattr(unbounded.SingleObservationWorldModel, name))


@pytest.mark.parametrize("width", [4, 8, 14])
def test_exact_seeded_production_state_and_active_parameter_parity(width):
    reference = make(bounded, width, depth=4, learned=False)
    variant = make(unbounded, width, depth=4, learned=False)
    assert reference.state_dict().keys() == variant.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, variant.state_dict()[name], rtol=0, atol=0)
    assert reference.parameter_counts == variant.parameter_counts
    assert {name: p.requires_grad for name, p in reference.named_parameters()} == {
        name: p.requires_grad for name, p in variant.named_parameters()}
    assert variant.transport_query.weight.requires_grad and variant.gate.weight.requires_grad
    initial, commands = inputs(width, horizon=3)
    # Zero residual gives identical initial predictions despite distinct packages.
    torch.testing.assert_close(reference.predict(initial, commands), variant.predict(initial, commands), rtol=0, atol=0)


@pytest.mark.parametrize("width", [4, 8, 14])
def test_only_innovation_changes_with_identical_nonzero_weights(width):
    reference = make(bounded, width)
    variant = make(unbounded, width)
    variant.load_state_dict(reference.state_dict(), strict=True)
    initial, commands = inputs(width, horizon=6)
    z = reference.normalize_features(initial)
    _, old = reference._predict_normalized(z, commands, True)
    prediction, new = variant._predict_normalized(z, commands, True)
    anchor = variant.tokens(z)
    for a, b in zip(old, new):
        torch.testing.assert_close(a["transport"], b["transport"], rtol=0, atol=0)
        torch.testing.assert_close(a["gate"], b["gate"], rtol=0, atol=0)
        torch.testing.assert_close(a["innovation"], b["innovation"].tanh(), rtol=0, atol=0)
        expected_mix = (1-b["gate"])*anchor + b["gate"]*(b["transport"] @ anchor)
        torch.testing.assert_close(b["prediction"], expected_mix+b["innovation"], rtol=0, atol=0)
        torch.testing.assert_close(b["prediction"]-a["prediction"],
                                   b["innovation"]-b["innovation"].tanh(), atol=6e-7, rtol=1e-5)
    restored_raw = prediction*variant.feature_std+variant.feature_mean
    torch.testing.assert_close(variant.predict(initial, commands), restored_raw, atol=0, rtol=0)


def test_large_projection_can_leave_the_old_unit_envelope():
    model = make(learned=False)
    with torch.no_grad():
        model.output_projection[-1].bias.fill_(2.5)
    initial = model.feature_mean[None].clone()  # Every standardized source is zero.
    commands = torch.zeros(1, 3, 4)
    result, details = model._predict_normalized(model.normalize_features(initial), commands, True)
    torch.testing.assert_close(result, torch.full_like(result, 2.5), rtol=0, atol=0)
    assert all(torch.equal(d["innovation"], torch.full_like(d["innovation"], 2.5)) for d in details)
    assert result.min() > 1


def test_future_commands_and_targets_cannot_enter_earlier_predictions():
    model = make()
    initial, commands = inputs(horizon=8)
    commands.requires_grad_()
    predicted = model.predict(initial, commands)
    changed = commands.detach().clone(); changed[:, 3:] += 100
    torch.testing.assert_close(predicted[:, :2], model.predict(initial, changed)[:, :2], rtol=0, atol=0)
    gradient = torch.autograd.grad(predicted[:, 1].square().mean(), commands, retain_graph=True)[0]
    assert gradient[:, 3:].count_nonzero() == 0
    assert (gradient[:, :3].abs().sum((0, 2)) > 0).all()
    target = torch.randn_like(predicted, requires_grad=True)
    batch = {"initial_features": initial, "commands": commands, "targets": target}
    before, after = model(batch), model({**batch, "targets": target+300})
    torch.testing.assert_close(before["predictions"], after["predictions"], rtol=0, atol=0)
    assert before["loss"] != after["loss"]
    assert torch.autograd.grad(before["predictions"].sum(), target, allow_unused=True)[0] is None


def test_full_h60_cpu_prefixes_and_fixed_observation_register():
    model = make(depth=4)
    initial, commands = inputs(horizon=60)
    seen = []
    original = model._encode
    def capture(value):
        seen.append(value.detach().clone())
        return original(value)
    model._encode = capture
    with torch.inference_mode():
        full = model.predict(initial, commands)
        assert full.shape == (1, 59, 6144)
        assert len(seen) == 1
        assert torch.equal(seen[0][:, 0], seen[0][:, 1]) and torch.equal(seen[0][:, 1], seen[0][:, 2])
        for h in (15, 30, 45):
            prefix = model.predict(initial, commands[:, :h])
            torch.testing.assert_close(prefix, full[:, :h-1], rtol=1e-5, atol=2e-5)
    assert not any("context" in name or "film" in name.lower() for name, _ in model.named_modules())


@pytest.mark.parametrize("width", [4, 8, 14])
def test_final_command_all_dimensions_and_all_active_parameters_get_gradients(width):
    model = make(width=width)
    initial, commands = inputs(width, horizon=4)
    commands.requires_grad_()
    output = model.predict(initial, commands)
    (output[:, -1]-torch.randn_like(output[:, -1])).square().mean().backward()
    assert (commands.grad[:, -1].abs().sum(0) > 0).all()
    assert commands.grad[:, 0].abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("mode", ["autoregressive", "anchored_additive", "bounded_spatial_mix"])
def test_old_modes_are_rejected_not_redirected(mode):
    with pytest.raises(ValueError, match="Invalid unbounded"):
        unbounded.SingleObservationWorldModel({"mode": mode}, *statistics(4))


def test_mutated_mode_or_bound_cannot_enable_an_old_branch():
    for field, value in (("mode", "autoregressive"), ("innovation_bound", 2.)):
        model = make()
        setattr(model.config, field, value)
        with pytest.raises(ValueError, match="Only the fixed-source"):
            model.predict(*inputs())


@pytest.mark.parametrize("field", ["feature_mean", "feature_std"])
def test_position_specific_normalization_rejected(field):
    config = make().package_config
    config[field][1] += .1
    with pytest.raises(ValueError, match="shared per-channel"):
        unbounded.from_config(config)


def test_strict_weights_only_reload_and_bidirectional_package_rejection(tmp_path):
    model = make()
    initial, commands = inputs()
    prediction = model.predict(initial, commands).detach()
    config = model.package_config
    assert config["package_kind"] == unbounded.PACKAGE_KIND
    assert config["model_config"]["mode"] == "unbounded_spatial_mix"
    (tmp_path/"config.json").write_text(json.dumps(config))
    torch.save(model.state_dict(), tmp_path/"model.pt")
    restored = unbounded.from_config(json.loads((tmp_path/"config.json").read_text())).eval()
    state = torch.load(tmp_path/"model.pt", weights_only=True, map_location="cpu")
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored.predict(initial, commands), prediction, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unsupported IWS"):
        bounded.from_config(config)
    with pytest.raises(ValueError, match="Unsupported unbounded"):
        unbounded.from_config(make(bounded).package_config)
    wrong = deepcopy(config); wrong["input_contract"] = "three_observations"
    with pytest.raises(ValueError, match="Unsupported unbounded"):
        unbounded.from_config(wrong)
    missing = dict(state); missing.pop("gate.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(missing, strict=True)
