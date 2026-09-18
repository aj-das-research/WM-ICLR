"""Geometry ablation contracts; tiny CPU fixtures are not research results."""
from copy import deepcopy
import json
import math

import pytest
import torch

from shiftwm.model import ResidualFiLM
from shiftwm.extensions import checkpoint as storage
from shiftwm.extensions import geometry_revision as geometry
from shiftwm.extensions import model as original
from shiftwm.extensions import train as training
from test_extension_model import model, batch
from test_extension_training import config, TinyDataset
from test_extension_checkpoint import refresh_manifest
from test_model import TinyBase


@pytest.fixture(autouse=True)
def tiny_cpu(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(original, "create_base", lambda config: TinyBase())
    yield
    torch.set_num_threads(old)


def settings(output, mode="factorized"):
    value = config(output, mode)
    value["geometry_revision"] = geometry.GEOMETRY_KIND
    return value


def test_identity_and_scale_translation_jacobians_match_original():
    old = ResidualFiLM(2, 3).double()
    revised = geometry.WideObservationFiLM(2, 3).double()
    value = torch.tensor([[[-2., 3.], [4., -1.]]], dtype=torch.float64)
    context = torch.randn(1, 3, dtype=torch.float64)
    for adapter in (old, revised):
        output = adapter(value, context)
        torch.testing.assert_close(output, value, rtol=0, atol=0)
        output.sum().backward()
    for old_parameter, new_parameter in zip(old.parameters(), revised.parameters()):
        torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=1e-13, atol=1e-13)
    assert revised.affine.bias.grad[:2].tolist() == pytest.approx([.2, .2])
    assert revised.affine.bias.grad[2:].tolist() == [2., 2.]


def test_wider_goal_separation_and_gain_bounds_with_unchanged_translation():
    old = ResidualFiLM(2, 3)
    revised = geometry.WideObservationFiLM(2, 3)
    with torch.no_grad():
        for module in (old, revised):
            module.affine.bias[:2].copy_(torch.tensor([1e6, -1e6]))
            module.affine.bias[2:].fill_(7.)
    value = torch.tensor([[[0., 0.], [1., 1.]]])
    context = torch.zeros(1, 3)
    before, after = old(value, context), revised(value, context)
    torch.testing.assert_close(after[:, 0], torch.full((1, 2), 7.))
    torch.testing.assert_close(after[:, 1] - after[:, 0], torch.tensor([[4., .25]]))
    torch.testing.assert_close(before[:, 1] - before[:, 0], torch.tensor([[1.1, .9]]))
    assert (after[:, 1] - after[:, 0]).square().sum() > (before[:, 1] - before[:, 0]).square().sum()


@pytest.mark.parametrize("mode", geometry.MODES)
def test_initial_full_model_equivalence_same_shapes_rng_and_action_adapter(mode):
    old = model(mode)
    rng = torch.get_rng_state().clone()
    revised = geometry.from_extension(old)
    assert torch.equal(rng, torch.get_rng_state())
    assert set(old.state_dict()) == set(revised.state_dict())
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in revised.parameters())
    assert {n for n, p in old.named_parameters() if p.requires_grad} == {
        n for n, p in revised.named_parameters() if p.requires_grad}
    for name, value in old.state_dict().items():
        torch.testing.assert_close(revised.state_dict()[name], value, rtol=0, atol=0)
    assert type(revised.dynamics_adapter) is ResidualFiLM
    inputs = batch()
    a, b = old(inputs), revised(inputs)
    for name in ("predictions", "targets", "prediction_loss", "loss"):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    revised.train()
    revised(inputs)["loss"].backward()
    assert revised.observation_adapter.affine.weight.grad is not None
    assert all(p.grad is None for p in revised.reference_encoder.parameters())


@pytest.mark.parametrize("mode", geometry.MODES)
def test_portable_v2_loader_preserves_nonzero_gain_and_v1_loader_refuses(tmp_path, mode):
    revised = geometry.from_extension(model(mode))
    with torch.no_grad():
        revised.observation_adapter.affine.bias[:8].fill_(8.)
    path = tmp_path / "best"
    geometry.save_package(revised, path)
    loaded, state = geometry.load_geometry_package(path)
    assert isinstance(loaded.observation_adapter, geometry.WideObservationFiLM)
    assert state["config"]["extension_format_version"] == 2
    assert state["config"]["geometry_revision"] == geometry.GEOMETRY_CONFIG
    inputs = batch()
    torch.testing.assert_close(loaded(inputs)["predictions"], revised(inputs)["predictions"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unsupported extension checkpoint format"):
        storage.load_package(path)


@pytest.mark.parametrize("damage", ["tag", "version", "bounds", "mode_alias"])
def test_v2_loader_rejects_changed_contract_even_with_updated_file_hashes(tmp_path, damage):
    path = tmp_path / "best"
    geometry.save_package(geometry.from_extension(model("factorized")), path)
    state = torch.load(path / "model.pt", weights_only=True)
    if damage == "tag":
        state["config"].pop("geometry_revision")
    elif damage == "version":
        state["config"]["extension_format_version"] = 1
    elif damage == "bounds":
        state["config"]["geometry_revision"]["gain_bounds"] = [.5, 2.]
    else:
        state["config"]["model_config"]["mode"] = "single"
    torch.save(state, path / "model.pt")
    (path / "config.json").write_text(json.dumps(state["config"]))
    refresh_manifest(path)
    with pytest.raises(ValueError):
        geometry.load_geometry_package(path)


def test_full30_epoch_reuse_of_frozen_loop_and_exact_interrupted_resume(tmp_path):
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    torch.manual_seed(41)
    completed = training.fit(geometry.from_extension(model("constant_dynamics")),
                             settings(full, "constant_dynamics"), TinyDataset(), TinyDataset())
    assert geometry.validate_completed(full, completed["training_identity"]) == completed
    with pytest.raises(ValueError):
        training.validate_completed(full, completed["training_identity"])
    partial_config = settings(interrupted, "constant_dynamics")
    partial_config["max_runtime_seconds"] = 1e-12
    torch.manual_seed(41)
    partial = training.fit(geometry.from_extension(model("constant_dynamics")), partial_config,
                           TinyDataset(), TinyDataset())
    assert partial["completed_epochs"] == 0 and partial["batches_completed_in_epoch"] == 1
    partial_config.pop("max_runtime_seconds")
    partial_config["resume"] = str(interrupted / "last")
    resumed = training.fit(geometry.from_extension(model("constant_dynamics")), partial_config,
                           TinyDataset(), TinyDataset())
    assert geometry.validate_completed(interrupted, resumed["training_identity"]) == resumed
    full_state = storage.read_package(full / "last")[1]
    resumed_state = storage.read_package(interrupted / "last")[1]
    for name, value in full_state["state_dict"].items():
        torch.testing.assert_close(resumed_state["state_dict"][name], value, rtol=0, atol=0)
    assert [(r["train"], r["val"]) for r in training.metric_rows(full)] == [
        (r["train"], r["val"]) for r in training.metric_rows(interrupted)]
    summary = json.loads((full / "training_summary.json").read_text())
    summary["validation_metric"] = "teacher_forced"
    (full / "training_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="identity/criterion"):
        geometry.validate_completed(full, completed["training_identity"])


def test_registration_guards_before_initialization(tmp_path):
    for change in ({"geometry_revision": "wrong"}, {"architecture": "gru"},
                   {"mode": "framewise"}, {"seed": 3},
                   {"output_dir": str(geometry.Path(__file__).resolve().parents[1] / "runs/extensions/danger")}):
        with pytest.raises(ValueError):
            geometry.initialize({**settings(tmp_path), **change})
