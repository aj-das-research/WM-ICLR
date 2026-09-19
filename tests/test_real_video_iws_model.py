"""Engineering contracts for the single-observation model; no benchmark runs."""
import inspect
import json
import shutil

import pytest
import torch

from shiftwm.real_video_iws.model import (IWSConfig, SingleObservationWorldModel,
                                         PACKAGE_KIND, from_config)


@pytest.fixture(autouse=True)
def small_cpu_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make(mode="bounded_spatial_mix", width=4, learned=True, depth=1):
    torch.manual_seed(173)
    fm = torch.linspace(-.2, .2, 384).repeat_interleave(16)
    fs = torch.linspace(.5, 1.5, 384).repeat_interleave(16)
    model = SingleObservationWorldModel({"mode": mode, "action_dim": width, "depth": depth},
        fm, fs, torch.linspace(-.1, .1, width), torch.linspace(.7, 1.1, width)).eval()
    if learned:
        # Both residual and conditional-transformer zero heads must be nonzero:
        # otherwise command-causality tests would be vacuous at initialization.
        with torch.no_grad():
            model.output_projection[-1].weight.normal_(0, .025)
            model.output_projection[-1].bias.normal_(0, .015)
            for block in model.predictor.transformer.layers:
                block.adaLN_modulation[-1].weight.normal_(0, .04)
                block.adaLN_modulation[-1].bias.normal_(0, .02)
            model.gate.weight.normal_(0, .04)
    return model


def inputs(width=4, horizon=8, batch=1):
    generator = torch.Generator().manual_seed(71)
    return (torch.randn(batch, 6144, generator=generator),
            torch.randn(batch, horizon, width, generator=generator))


@pytest.mark.parametrize("mode", SingleObservationWorldModel.MODES)
def test_nonzero_weights_causal_prefix_gradients_and_target_invariance(mode):
    model = make(mode)
    initial, commands = inputs()
    commands.requires_grad_()
    prediction = model.predict(initial, commands)
    changed = commands.detach().clone(); changed[:, 3:] += 100
    torch.testing.assert_close(prediction[:, :2], model.predict(initial, changed)[:, :2], rtol=0, atol=0)
    # FP32 kernels may round differently when GRU sequence extent changes.
    torch.testing.assert_close(prediction[:, :2], model.predict(initial, commands[:, :3]), rtol=1e-6, atol=5e-7)
    allowed_gradient = torch.autograd.grad(prediction[:, 1].square().mean(), commands, retain_graph=True)[0]
    assert torch.equal(allowed_gradient[:, 3:], torch.zeros_like(allowed_gradient[:, 3:]))
    assert all(allowed_gradient[:, k].abs().sum() > 0 for k in range(3))
    assert allowed_gradient[:, 2].abs().sum() > 0  # Current row really conditions prediction offset2.
    targets = torch.randn_like(prediction, requires_grad=True)
    batch = {"initial_features": initial, "commands": commands, "targets": targets}
    before = model(batch)
    after = model({**batch, "targets": targets+300})
    torch.testing.assert_close(before["predictions"], after["predictions"], rtol=0, atol=0)
    assert not torch.equal(before["loss"], after["loss"])
    assert torch.autograd.grad(before["predictions"].sum(), targets, allow_unused=True)[0] is None


@pytest.mark.parametrize("horizon", [2, 15, 30, 45, 60])
def test_exact_native_horizon_shapes_and_last_command_inclusion(horizon):
    model = make()
    initial, commands = inputs(horizon=horizon)
    commands.requires_grad_()
    out = model.predict(initial, commands)
    assert out.shape == (1, horizon-1, 6144)
    gradient = torch.autograd.grad(out[:, -1].square().mean(), commands)[0]
    assert gradient[:, -1].abs().sum() > 0
    assert gradient[:, 0].abs().sum() > 0


@pytest.mark.parametrize("width", [4, 8, 14])
def test_all_native_command_widths_are_real_inputs(width):
    model = make(width=width)
    initial, commands = inputs(width=width, horizon=3)
    commands.requires_grad_()
    out = model.predict(initial, commands)
    assert out.shape == (1, 2, 6144)
    gradient = torch.autograd.grad(out[:, -1].square().mean(), commands)[0]
    assert (gradient.abs().sum((0, 1)) > 0).all()


def test_single_observation_api_and_no_manufactured_context():
    assert list(inspect.signature(SingleObservationWorldModel.predict).parameters) == ["self", "initial_features", "native_commands"]
    model = make()
    assert not any("context" in name or "film" in name.lower() for name, _ in model.named_modules())
    initial, commands = inputs()
    with pytest.raises(ValueError, match="one initial"):
        model.predict(initial[:, None].expand(-1, 3, -1), commands)
    with pytest.raises(ValueError, match="one initial"):
        model.predict(initial, commands[:, :1])
    with pytest.raises(ValueError, match="Nonfinite"):
        model.predict(initial*float("nan"), commands)
    with pytest.raises(ValueError, match="Targets"):
        model({"initial_features": initial, "commands": commands, "targets": initial[:, None]})


def test_matched_initialization_active_heads_and_zero_residual_behavior():
    models = [make(mode, learned=False) for mode in SingleObservationWorldModel.MODES]
    ref = models[0].state_dict()
    for model in models[1:]:
        assert ref.keys() == model.state_dict().keys()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(ref[key], value, rtol=0, atol=0)
    assert len({m.parameter_counts["total"] for m in models}) == 1
    assert models[0].parameter_counts["trainable"] == models[1].parameter_counts["trainable"]
    delta = models[2].parameter_counts["trainable"]-models[0].parameter_counts["trainable"]
    assert delta == 2*96*96 + 96+1
    initial, commands = inputs(horizon=4)
    normalized = models[0].normalize_features(initial)
    for model in models[:2]:
        torch.testing.assert_close(model._predict_normalized(normalized, commands), normalized[:, None].expand(-1, 3, -1), rtol=0, atol=0)
        assert not model.gate.weight.requires_grad and not model.transport_key.weight.requires_grad
    output, details = models[2]._predict_normalized(normalized, commands, True)
    assert not torch.equal(output[:, 0], normalized)  # 4I bias is not an exact identity map.
    for detail in details:
        assert torch.count_nonzero(detail["innovation"]) == 0


@pytest.mark.parametrize("mode", SingleObservationWorldModel.MODES)
def test_repeated_initial_register_and_all_active_parameter_gradients(mode):
    model = make(mode)
    initial, commands = inputs(horizon=4)
    seen = []
    original = model._encode
    def capture(x):
        seen.append(x.detach().clone())
        return original(x)
    model._encode = capture
    result = model.predict(initial, commands)
    assert torch.equal(seen[0][:, 0], seen[0][:, 1]) and torch.equal(seen[0][:, 1], seen[0][:, 2])
    if mode == "autoregressive":
        assert len(seen) == 3
        normalized_first = model.tokens(model.normalize_features(result[:, 0]))
        torch.testing.assert_close(seen[1][:, -1], normalized_first, rtol=1e-6, atol=5e-7)
    else:
        assert len(seen) == 1
    (result-torch.randn_like(result)).square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name


def test_full_h60_ar_keeps_early_prediction_in_last_offset_graph():
    # Actual production depth4, full59 outputs; no detach/truncated backprop.
    model = make("autoregressive", depth=4)
    initial, commands = inputs(horizon=60)
    commands.requires_grad_()
    out, details = model._predict_normalized(model.normalize_features(initial), commands, True)
    details[0]["prediction"].retain_grad()
    out[:, -1].square().mean().backward()
    assert len(details) == 59
    assert details[0]["prediction"].grad.abs().sum() > 0
    assert commands.grad[:, -1].abs().sum() > 0
    assert model.output_projection[-1].weight.grad.abs().sum() > 0


def test_mixture_simplex_and_channelwise_innovation_bound_under_stress():
    model = make()
    initial, commands = inputs(horizon=4)
    with torch.no_grad():
        model.output_projection[-1].weight.mul_(500)
        model.output_projection[-1].bias.fill_(200)
    normalized = model.normalize_features(initial)
    result, details = model._predict_normalized(normalized, commands, True)
    anchor = model.tokens(normalized)
    envelope = anchor.abs().amax(1)[:, None, :] + model.config.innovation_bound
    assert torch.all(model.tokens(result).abs() <= envelope[:, None]+1e-6)
    for d in details:
        torch.testing.assert_close(d["transport"].sum(-1), torch.ones(1,16), atol=1e-6, rtol=1e-6)
        assert (d["transport"] >= 0).all()
        assert ((d["gate"] >= 0) & (d["gate"] <= 1)).all()
        assert d["innovation"].abs().max() <= 1
        mixed = (1-d["gate"])*anchor+d["gate"]*(d["transport"]@anchor)
        torch.testing.assert_close(d["prediction"]-mixed,d["innovation"],rtol=1e-6,atol=5e-7)


@pytest.mark.parametrize("field", ["feature_mean", "feature_std"])
def test_position_specific_stats_rejected_and_buffers_fixed(field):
    model = make()
    config = model.package_config
    config[field][1] += .1
    with pytest.raises(ValueError, match="shared per-channel"):
        from_config(config)
    stats = torch.zeros(6144, requires_grad=True)
    check = SingleObservationWorldModel(IWSConfig(depth=1),stats,torch.ones(6144),torch.zeros(4),torch.ones(4))
    assert not check.feature_mean.requires_grad


@pytest.mark.parametrize("mode", SingleObservationWorldModel.MODES)
def test_relocated_weights_only_reload_and_package_type_rejection(tmp_path, mode):
    model = make(mode)
    initial, commands = inputs(horizon=4)
    expected = model.predict(initial, commands).detach()
    source = tmp_path/'old';source.mkdir()
    (source/'config.json').write_text(json.dumps(model.package_config))
    torch.save(model.state_dict(), source/'model.pt')
    relocated = tmp_path/'new'/'model';relocated.parent.mkdir();shutil.copytree(source,relocated);shutil.rmtree(source)
    config = json.loads((relocated/'config.json').read_text())
    restored = from_config(config).eval()
    restored.load_state_dict(torch.load(relocated/'model.pt',weights_only=True,map_location='cpu'),strict=True)
    torch.testing.assert_close(restored.predict(initial,commands),expected,rtol=0,atol=0)
    assert config['package_kind'] == PACKAGE_KIND
    for field, bad in [('package_kind','shiftwm_real_video_spatial_v1'),('input_contract','three_observations')]:
        with pytest.raises(ValueError,match='Unsupported IWS'):
            from_config({**config,field:bad})
