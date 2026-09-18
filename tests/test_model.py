"""Scientific correctness tests; fixtures are not reported experiments."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from shiftwm.model import ModelConfig, ShiftWorldModel
from shiftwm.checkpoint import load_package, resume_training, save_package


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, pixels, **kwargs):
        return SimpleNamespace(last_hidden_state=self.linear(pixels.mean((-1, -2)))[:, None])


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.projector = nn.Identity()
        self.action_encoder = nn.Linear(4, 8)
        self.predictor = nn.Linear(8, 8)
        self.pred_proj = nn.Identity()

    def predict(self, emb, actions):
        return self.predictor(torch.nn.functional.dropout(emb + actions, p=.15, training=self.training))


BASE_CONFIG = {"predictor": {"input_dim": 8, "num_frames": 3},
               "action_encoder": {"input_dim": 4}, "encoder": {"image_size": 8}}


def make_model(mode="factorized", **kwargs):
    return ShiftWorldModel(TinyBase(), BASE_CONFIG,
                           ModelConfig(mode=mode, context_hidden=16, context_dim=4, **kwargs),
                           [0.] * 4, [1.] * 4)


def feature_batch():
    torch.manual_seed(11)
    return {"features": torch.randn(4, 7, 8), "reference_features": torch.randn(4, 7, 8),
            "paired_features": torch.randn(4, 7, 8), "actions": torch.randn(4, 6, 4),
            "observation_id": torch.tensor([0, 0, 1, 1])}


def test_context_cannot_read_query_frames_or_intervention_ids():
    model = make_model().eval()
    batch = feature_batch()
    out = model(batch)
    other = deepcopy(batch)
    other["features"][:, 3:] += 100
    other["actions"][:, 2:] += 100
    other["reference_features"] += 1000
    other["observation_id"][:] = 9
    changed = model(other)
    for name in ("observation_context", "dynamics_context"):
        torch.testing.assert_close(out[name], changed[name], rtol=0, atol=0)


def test_frozen_targets_do_not_move_when_online_encoder_changes():
    model = make_model(freeze_visual=False)
    images = torch.rand(2, 3, 3, 8, 8)
    before = model.encode_images(images, reference=True).detach().clone()
    with torch.no_grad():
        model.base.encoder.linear.weight.add_(5)
    after = model.encode_images(images, reference=True)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.allclose(before, model.encode_images(images))


def test_frozen_visual_stays_eval_during_training():
    model = make_model().train()
    assert not model.reference_encoder.training
    assert not model.base.encoder.training
    assert model.base.predictor.training
    assert all(not p.requires_grad for p in model.reference_encoder.parameters())


def test_visual_finetune_batchnorm_cannot_mix_future_into_support():
    base = TinyBase()
    base.projector = nn.BatchNorm1d(8)
    model = ShiftWorldModel(base, BASE_CONFIG, ModelConfig(freeze_visual=False), [0.] * 4, [1.] * 4).train()
    assert not model.base.projector.training
    images = torch.rand(4, 7, 3, 8, 8)
    first = model.encode_images(images)[:, :3]
    images[:, 3:] = torch.rand_like(images[:, 3:]) * 100
    second = model.encode_images(images)[:, :3]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_rollout_uses_executed_history_and_strictly_future_actions():
    model = make_model("frozen").eval()
    feature = torch.randn(2, 3, 8)
    past, future = torch.randn(2, 2, 4), torch.randn(2, 4, 4)
    output = model.rollout_features(feature, past, future)
    expected_first = model.base.predict(feature, model.base.action_encoder(
        torch.cat((past, future[:, :1]), 1)))[:, -1]
    torch.testing.assert_close(output[:, 0], expected_first)
    changed = future.clone()
    changed[:, -1] += 10
    after = model.rollout_features(feature, past, changed)
    torch.testing.assert_close(output[:, :3], after[:, :3], rtol=0, atol=0)
    assert not torch.allclose(output[:, -1], after[:, -1])


def test_goal_uses_same_nonprivileged_observation_correction():
    model = make_model().eval()
    with torch.no_grad():
        model.observation_adapter.affine.bias[8:] = 2
    images = torch.rand(2, 3, 8, 8)
    context = torch.zeros(2, 4)
    torch.testing.assert_close(model.goal_embedding(images, context), model.encode_images(images) + 2)


@pytest.mark.parametrize("mode", ["plain", "framewise", "single", "factorized", "factorized_unpaired", "observation", "dynamics"])
def test_training_has_finite_gradients_and_updates(mode):
    model = make_model(mode).train()
    batch = feature_batch()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    before = model.base.predictor.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        outputs = model(batch)
        outputs["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(before, model.base.predictor.weight)
    assert all(p.grad is None for p in model.reference_encoder.parameters())


def test_paired_constraint_only_used_in_factorized_mode():
    for mode in ("single", "plain", "factorized_unpaired"):
        model = make_model(mode).eval()
        batch = feature_batch()
        before = model(batch)["loss"]
        batch["paired_features"] += 100
        batch["observation_id"][:] = 7
        torch.testing.assert_close(before, model(batch)["loss"], rtol=0, atol=0)


def test_cached_features_disallowed_when_encoder_trainable():
    with pytest.raises(ValueError, match="Cached features"):
        make_model(freeze_visual=False)(feature_batch())


def test_raw_and_cached_feature_predictions_match():
    model = make_model().eval()
    images = torch.rand(4, 7, 3, 8, 8)
    batch = {"images": images, "reference_images": images, "paired_images": images,
             "actions": torch.randn(4, 6, 4)}
    features = model.encode_images(images).detach()
    cached = {"features": features, "reference_features": features, "paired_features": features,
              "actions": batch["actions"]}
    torch.testing.assert_close(model(batch)["predictions"], model(cached)["predictions"])


@pytest.mark.parametrize("mode", ["factorized", "framewise"])
def test_weights_only_checkpoint_and_exact_resume(tmp_path, monkeypatch, mode):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    model = make_model(mode).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    generator = torch.Generator().manual_seed(17)
    batch = feature_batch()
    model(batch)["loss"].backward(); optimizer.step(); optimizer.zero_grad()
    save_package(model, tmp_path, optimizer=optimizer, epoch=1, step=1, loader_generator=generator)
    expected_rand = torch.rand(4)
    expected_numpy = np.random.rand(4)
    model(batch)["loss"].backward(); optimizer.step(); optimizer.zero_grad()
    expected_params = deepcopy(model.state_dict())
    loaded, state = load_package(tmp_path)
    loaded.train()
    optim2 = torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad], lr=.001)
    resume_training(tmp_path, loaded, optim2, loader_generator=generator)
    torch.testing.assert_close(torch.rand(4), expected_rand, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(4), expected_numpy)
    loaded(batch)["loss"].backward(); optim2.step()
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected_params[key], rtol=0, atol=0)


def test_interrupted_training_resumes_exact_batch_and_epoch(tmp_path, monkeypatch):
    """Compare complete two-epoch optimizer path with interruption after batch1."""
    import json
    import shiftwm.train as trainer
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            generator = torch.Generator().manual_seed(41)
            self.items = [{"features": torch.randn(7, 8, generator=generator),
                           "reference_features": torch.randn(7, 8, generator=generator),
                           "paired_features": torch.randn(7, 8, generator=generator),
                           "actions": torch.randn(6, 4, generator=generator),
                           "observation_id": torch.tensor(i % 2)} for i in range(10)]
        def __len__(self):
            return len(self.items)
        def __getitem__(self, index):
            return self.items[index]
    monkeypatch.setattr(trainer, "make_dataset", lambda cfg, split: Dataset())
    monkeypatch.setattr(trainer, "load_base", lambda path: (TinyBase(), BASE_CONFIG, {"weights_sha256": "test"}))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"mean": [0] * 4, "std": [1] * 4}))
    config = {"pretrained_dir": "unused-test-fixture", "data_root": str(tmp_path), "action_stats": str(stats),
              "seed": 13, "epochs": 2, "batch_size": 3, "num_workers": 0, "cpu_threads": 1,
              "device": "cpu", "model": {"context_dim": 4, "context_hidden": 16},
              "output_dir": str(tmp_path / "complete")}
    trainer.train(config)
    complete = torch.load(tmp_path / "complete/last/model.pt", weights_only=True)
    config["output_dir"] = str(tmp_path / "interrupted")
    config["max_runtime_seconds"] = 1e-12
    result = trainer.train(config)
    assert result["batches_completed_in_epoch"] == 1
    config.pop("max_runtime_seconds")
    config["resume"] = str(tmp_path / "interrupted/last")
    trainer.train(config)
    resumed = torch.load(tmp_path / "interrupted/last/model.pt", weights_only=True)
    assert resumed["step"] == complete["step"] == 8
    for key in complete["state_dict"]:
        torch.testing.assert_close(resumed["state_dict"][key], complete["state_dict"][key], rtol=0, atol=0)


def test_bundled_upstream_hashes_and_import_without_external_checkout(tmp_path, monkeypatch):
    import hashlib
    import json
    import shiftwm.upstream as upstream
    manifest = json.loads((upstream.VENDOR / "NOTICE.json").read_text())
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((upstream.VENDOR / name).read_bytes()).hexdigest() == expected_hash
    monkeypatch.setattr(upstream, "ROOT", tmp_path)
    module = upstream._source_module("shiftwm_test_portable_module", "module.py")
    jepa = upstream._source_module("shiftwm_test_portable_jepa", "jepa.py")
    assert module.ARPredictor is not None and jepa.JEPA is not None


def test_framewise_starts_identity_and_has_no_history_or_context_dependence():
    model = make_model("framewise").eval()
    features = torch.randn(3, 7, 8)
    context_a, context_b = torch.randn(3, 4), torch.randn(3, 4)
    torch.testing.assert_close(model.correct_observations(features, context_a), features, rtol=0, atol=0)
    with torch.no_grad():
        model.framewise_adapter[-1].weight.normal_(0, .1)
    first = model.correct_observations(features, context_a)
    torch.testing.assert_close(first, model.correct_observations(features, context_b), rtol=0, atol=0)
    altered = features.clone()
    altered[:, :-1] = 100 * torch.randn_like(altered[:, :-1])
    torch.testing.assert_close(first[:, -1], model.correct_observations(altered, context_b)[:, -1], rtol=0, atol=0)
    obs, dyn = model.infer_context(features[:, :3], torch.randn(3, 2, 4))
    assert torch.count_nonzero(obs) == 0 and torch.count_nonzero(dyn) == 0


def test_framewise_goal_and_history_share_exact_same_calibration():
    model = make_model("framewise").eval()
    with torch.no_grad():
        model.framewise_adapter[-1].weight.normal_(0, .1)
        model.framewise_adapter[-1].bias.fill_(.3)
    goal = torch.rand(3, 3, 8, 8)
    context = torch.randn(3, 4)
    raw = model.encode_images(goal)
    corrected = model.correct_observations(raw[:, None], context)[:, 0]
    torch.testing.assert_close(model.goal_embedding(goal, context), corrected, rtol=0, atol=0)
    assert not torch.allclose(raw, corrected)


def test_framewise_alignment_updates_only_calibrator_on_fixed_features():
    model = make_model("framewise").train()
    batch = feature_batch()
    model(batch)["alignment_loss"].backward()
    assert model.framewise_adapter[-1].weight.grad.abs().sum() > 0
    assert model.base.predictor.weight.grad is None
    assert all(p.grad is None for p in model.observation_context.parameters())
    assert all(p.grad is None for p in model.dynamics_context.parameters())


def test_existing_modes_do_not_gain_framewise_checkpoint_keys():
    for mode in ("frozen", "plain", "single", "factorized", "factorized_unpaired", "observation", "dynamics"):
        model = make_model(mode)
        assert not hasattr(model, "framewise_adapter")
        assert not any(key.startswith("framewise_adapter.") for key in model.state_dict())
