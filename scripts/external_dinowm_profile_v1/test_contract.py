"""Synthetic contracts for a real external predictor, no dataset access."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapter


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    return adapter.synthetic_model().eval()


def test_full_external_architecture_and_mask(model):
    assert model.predictor.pos_embedding.shape == (1, 48, 394)
    assert len(model.predictor.transformer.layers) == 6
    expected = torch.arange(48)[:, None] // 16 >= torch.arange(48)[None, :] // 16
    for attention, feedforward in model.predictor.transformer.layers:
        assert attention.heads == 16
        assert attention.to_qkv.weight.shape == (3072, 394)
        assert feedforward.net[1].weight.shape == (2048, 394)
        assert torch.equal(attention.bias[0, 0].bool(), expected)
        assert attention.bias.device.type == "cpu"
    assert not any(key.endswith(".bias") and "layers" in key and "to_" not in key and "net" not in key and "norm" not in key for key in model.predictor.state_dict())
    assert all(name not in model.predictor.state_dict() for name, _ in model.predictor.named_buffers())
    assert model.action_encoder.patch_embed.weight.shape == (10, 35, 1)
    assert not hasattr(model, "proprio_encoder")


def test_channel_major_layout_roundtrip():
    x = torch.arange(2 * 6144).reshape(2, 6144)
    value = adapter.tokens(x)
    assert value[0, 7, 11] == x[0, 11 * 16 + 7]
    assert torch.equal(adapter.flatten(value), x)


def test_same_external_predict_as_official_world_model(model):
    path = adapter.UPSTREAM / "models/visual_world_model.py"
    assert adapter.hashlib.sha256(path.read_bytes()).hexdigest() == "931d44c0d541d95be8236f65247de9b40ac3843fe047c1a44d4a2482d1bf636f"
    spec = importlib.util.spec_from_file_location("official_dinowm_contract_vwm", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    batch = adapter.synthetic_batch(1)
    features = batch["features"][:, :3]
    actions = model.action_encoder(batch["actions"][:, :3])
    embedded = torch.cat((adapter.tokens(features), actions[:, :, None].expand(-1, -1, 16, -1)), -1)
    with torch.no_grad():
        expected = official.VWorldModel.predict(types.SimpleNamespace(predictor=model.predictor), embedded)
        actual = model.predict_three_slots(features, batch["actions"][:, :3])
    assert torch.equal(actual, adapter.flatten(expected[..., :384]))


def test_mask_future_slots_cannot_affect_earlier_prediction(model):
    x = torch.randn(1, 3, 6144, requires_grad=True)
    actions = torch.randn(1, 3, 35, requires_grad=True)
    prediction = model.predict_three_slots(x, actions)
    gx, ga = torch.autograd.grad(prediction[:, 0].square().mean(), (x, actions))
    assert torch.count_nonzero(gx[:, 1:]) == 0
    assert torch.count_nonzero(ga[:, 1:]) == 0
    assert gx[:, 0].abs().sum() > 0 and ga[:, 0].abs().sum() > 0


def test_recursive_prefix_and_allowed_action_effect(model):
    batch = adapter.synthetic_batch(1)
    past = batch["actions"][:, :2]
    future = batch["actions"][:, 2:].clone().requires_grad_()
    x = batch["features"][:, :3]
    predicted = model.predict_normalized(x, past, future)
    short = model.predict_normalized(x, past, future[:, :3])
    assert torch.equal(predicted[:, :3], short)
    gradient, = torch.autograd.grad(predicted[:, 0].square().mean(), (future,))
    assert gradient[:, 0].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 1:]) == 0


def test_recursive_predictions_do_not_receive_future_targets(model):
    batch = adapter.synthetic_batch(1)
    batch["features"].requires_grad_()
    result = model(batch, "matched_recursive_h10")
    gradient, = torch.autograd.grad(result["standardized_predictions"].square().mean(), (batch["features"],))
    assert gradient[:, :3].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 3:]) == 0
    assert torch.equal(result["standardized_targets"], batch["features"][:, 3:13])
    assert not result["standardized_targets"].requires_grad


def test_native_loss_uses_all_three_shifted_slots_only(model):
    batch = adapter.synthetic_batch(1)
    with torch.no_grad():
        result = model(batch, "official_one_step_shifted")
        expected_prediction = model.predict_three_slots(batch["features"][:, :3], batch["actions"][:, :3])
        assert torch.equal(result["standardized_predictions"], expected_prediction)
        assert torch.equal(result["standardized_targets"], batch["features"][:, 1:4])
        expected_loss = (expected_prediction - batch["features"][:, 1:4]).square().mean()
        torch.testing.assert_close(result["loss"], expected_loss, rtol=0, atol=0)
        changed = {k: v.clone() for k, v in batch.items()}
        changed["features"][:, 4:] += 100
        changed["actions"][:, 3:] -= 100
        assert torch.equal(model(changed, "official_one_step_shifted")["loss"], result["loss"])


def test_recursive_graph_retains_earlier_predicted_state(model):
    batch = adapter.synthetic_batch(1)
    captured = []
    def retain(_module, _arguments, output):
        output.retain_grad()
        captured.append(output)
    handle = model.predictor.register_forward_hook(retain)
    try:
        model.zero_grad(set_to_none=True)
        result = model.predict_normalized(batch["features"][:, :3], batch["actions"][:, :2], batch["actions"][:, 2:5])
        result[:, -1].square().mean().backward()
        assert len(captured) == 3
        assert captured[0].grad[:, -16:, :384].abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def test_bad_shapes_objective_and_normalization_rejected(model):
    batch = adapter.synthetic_batch(1)
    with pytest.raises(ValueError): model(batch, "new_tuned_loss")
    with pytest.raises(ValueError): model({"features": batch["features"][:, :-1], "actions": batch["actions"]})
    wrong = torch.ones(6144); wrong[1] = 2
    with pytest.raises(ValueError, match="shared per channel"):
        adapter.ExternalDinoWM(torch.zeros(6144), wrong, torch.zeros(35), torch.ones(35))


def test_profile_review_gate_precedes_upstream_import_and_cuda(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("external_dinowm_profile_contract", HERE / "profile_cuda.py")
    profile = importlib.util.module_from_spec(spec); spec.loader.exec_module(profile)
    registration = tmp_path / "registration.json"; review = tmp_path / "review.json"
    registration.write_text(json.dumps({"source_sha256": {"unavailable-source": "dead"}, "spec": {}}))
    review.write_text(json.dumps({"status": "passed", "registration_sha256": "stale", "source_sha256": {}}))
    monkeypatch.setattr(profile, "REGISTRATION", registration); monkeypatch.setattr(profile, "REVIEW", review)
    monkeypatch.setattr(profile, "upstream_identity", lambda: pytest.fail("Upstream touched before review rejected"))
    with pytest.raises(ValueError, match="stale"):
        profile.verify_reviewed_registration()


def test_runtime_projection_keeps_objective_specific_step_cost():
    spec = importlib.util.spec_from_file_location("external_dinowm_profile_estimate", HERE / "profile_cuda.py")
    profile = importlib.util.module_from_spec(spec); spec.loader.exec_module(profile)
    config = json.loads((HERE / "profile_spec.json").read_text())
    value = profile.estimate(2., 1., config)
    assert value["training_batches_per_epoch"] == 146
    assert value["validation_batches_per_epoch"] == 13
    assert value["estimated_30epoch_hours_one_seed_without_io"] == 305 * 30 / 3600
