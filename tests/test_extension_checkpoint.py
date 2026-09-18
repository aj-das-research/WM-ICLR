"""Portable/transactional correctness fixtures, not benchmark measurements."""
from copy import deepcopy
import json
import random
import shutil

import numpy as np
import pytest
import torch

from shiftwm.extensions import checkpoint as cp
from shiftwm.extensions import model as models
from test_extension_model import model, batch
from test_model import TinyBase, BASE_CONFIG


@pytest.fixture(autouse=True)
def offline_tiny_base(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(models, "create_base", lambda config: TinyBase())
    # Offline loading must never use the pretrained-file loader.
    monkeypatch.setattr(models, "load_base", lambda *args: pytest.fail("Loading attempted external weights"))
    yield
    torch.set_num_threads(old)


def refresh_manifest(directory):
    directory = directory.resolve()
    path = directory / "package_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {name: cp.file_sha256(directory / name) for name in manifest["files"]}
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("mode", models.MODES)
def test_weights_only_roundtrip_offline_and_real_directory_copy(tmp_path, mode):
    m = model(mode)
    inputs = batch()
    expected = m(inputs)["predictions"].detach()
    path = tmp_path / "best"
    cp.save_package(m, path, epoch=4, step=12, best_metric=.15)
    assert path.is_symlink() and not path.readlink().is_absolute()
    loaded, state = cp.load_package(path)
    assert state["epoch"] == 4 and state["step"] == 12
    torch.testing.assert_close(loaded(inputs)["predictions"], expected, rtol=0, atol=0)
    assert not (path / "training_state.pt").exists()
    copied = tmp_path / "portable"
    shutil.copytree(path, copied)
    loaded2, _ = cp.load_package(copied)
    torch.testing.assert_close(loaded2(inputs)["predictions"], expected, rtol=0, atol=0)


def test_resume_restores_optimizer_scheduler_sampler_and_all_rng(tmp_path):
    m = model("factorized").train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    generator = torch.Generator().manual_seed(18)
    inputs = batch()
    m(inputs)["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    path = tmp_path / "last"
    cp.save_package(m, path, optimizer=optimizer, scheduler=scheduler,
                    loader_generator=generator, epoch=1, step=1,
                    metadata={"training_identity": "fixture"})
    torch_values, numpy_values, python_value = torch.rand(3), np.random.rand(3), random.random()
    permutation = torch.randperm(10, generator=generator)
    m(inputs)["loss"].backward()
    optimizer.step()
    scheduler.step()
    expected = deepcopy(m.state_dict())
    cp.resume_training(path, m, optimizer, scheduler, generator)
    torch.testing.assert_close(torch.rand(3), torch_values, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(3), numpy_values)
    assert random.random() == python_value
    assert torch.equal(torch.randperm(10, generator=generator), permutation)
    optimizer.zero_grad()
    m(inputs)["loss"].backward()
    optimizer.step()
    scheduler.step()
    for name, value in expected.items():
        torch.testing.assert_close(m.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("mode", models.MODES)
def test_gru_package_reconstructs_the_correct_family_and_recursive_rollout(tmp_path, monkeypatch, mode):
    class DualInputBase(TinyBase):
        def predict(self, features, actions):
            return self.predictor(features, actions)

    monkeypatch.setattr(models, "create_base", lambda config: DualInputBase())
    base = models.configure_predictor(DualInputBase(), BASE_CONFIG, "gru")
    m = models.ExtensionWorldModel(base, BASE_CONFIG, mode, [0.] * 4, [1.] * 4,
                                   architecture="gru", model_config={"context_dim": 4, "context_hidden": 16}).eval()
    inputs = batch()
    expected = m(inputs)["predictions"].detach()
    assert expected.shape == (3, 5, 8)
    path = tmp_path / "best"
    cp.save_package(m, path)
    loaded, _ = cp.load_package(path)
    assert isinstance(loaded.base.predictor, models.GRUPredictor)
    torch.testing.assert_close(loaded(inputs)["predictions"], expected, rtol=0, atol=0)


def test_interrupted_generation_never_replaces_previous_complete_package(tmp_path, monkeypatch):
    m, path = model("framewise"), tmp_path / "last"
    cp.save_package(m, path, epoch=1, step=3)
    original = path.resolve()
    old_replace = cp.os.replace

    def fail_pointer(source, target):
        if target == path.absolute():
            raise OSError("simulated interruption before commit")
        return old_replace(source, target)

    monkeypatch.setattr(cp.os, "replace", fail_pointer)
    with pytest.raises(OSError, match="simulated interruption"):
        cp.save_package(m, path, epoch=2, step=6)
    assert path.resolve() == original
    assert cp.read_package(path)[1]["epoch"] == 1
    assert len(list((tmp_path / ".last.generations").iterdir())) == 1


def test_generations_keep_current_and_previous_only(tmp_path):
    m, path = model("framewise"), tmp_path / "last"
    for epoch in range(4):
        cp.save_package(m, path, epoch=epoch)
    assert cp.read_package(path)[1]["epoch"] == 3
    assert len(list((tmp_path / ".last.generations").iterdir())) == 2


@pytest.mark.parametrize("damage", ["hash", "sidecar", "package_kind", "mode_alias", "state_shape", "resume_step"])
def test_rejects_corrupt_or_inconsistent_packages(tmp_path, damage):
    m, path = model("framewise"), tmp_path / "best"
    optimizer = torch.optim.AdamW(m.parameters())
    cp.save_package(m, path, optimizer=optimizer, metadata={"training_identity": "fixture"})
    if damage in {"hash", "sidecar"}:
        config = json.loads((path / "config.json").read_text())
        config["mode"] = "factorized"
        (path / "config.json").write_text(json.dumps(config))
    elif damage == "resume_step":
        training = torch.load(path / "training_state.pt", weights_only=True)
        training["step"] = 10
        torch.save(training, path / "training_state.pt")
    else:
        state = torch.load(path / "model.pt", weights_only=True)
        if damage == "package_kind":
            state["config"]["package_kind"] = "old-model"
        elif damage == "mode_alias":
            state["config"]["model_config"]["mode"] = "factorized"
        else:
            state["state_dict"]["action_mean"] = torch.ones(13)
        torch.save(state, path / "model.pt")
        (path / "config.json").write_text(json.dumps(state["config"]))
    if damage != "hash":
        refresh_manifest(path)
    with pytest.raises((ValueError, RuntimeError)):
        if damage == "resume_step":
            cp.read_package(path, require_training=True)
        else:
            cp.load_package(path)
