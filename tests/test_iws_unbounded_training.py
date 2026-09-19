"""Private reuse, strict namespace, registration and exact continuation tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from shiftwm.real_video_iws import training as original
from shiftwm.real_video_iws_unbounded import training as adapter

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def toy(monkeypatch):
    fixture = load("tests/test_iws_training.py", "_unbounded_test_only_frozen_fixtures")
    fixture.trainer = adapter.engine
    def construct(config):
        model = fixture.ToyModel(dropout=True)
        model.config.mode = "unbounded_spatial_mix"
        return model
    monkeypatch.setattr(adapter.engine.base, "from_config", construct)
    return fixture


def inputs(toy, path):
    model, config, train, val, identity = toy.fit_inputs(path)
    model.config.mode = "unbounded_spatial_mix"
    config["scientific_config"]["mode"] = "unbounded_spatial_mix"
    return model, config, train, val, identity


def test_private_engine_does_not_mutate_original():
    assert original.PACKAGE_KIND == "shiftwm_iws_single_observation_v1"
    assert original.base.PACKAGE_KIND == "shiftwm_iws_single_observation_v1"
    assert adapter.PACKAGE_KIND == "shiftwm_iws_single_observation_unbounded_v1"
    assert adapter.engine is not original and adapter.engine.base is not original.base
    for name in ("fit", "epoch_pass", "validate_completed", "read_package"):
        assert getattr(adapter.engine, name).__code__.co_code == getattr(original, name).__code__.co_code


def test_exact_resume_preserves_optimizer_rng_and_selector(tmp_path, toy):
    full, full_config, train, val, identity = inputs(toy, tmp_path / "full")
    adapter.fit(full, full_config, train, val, identity)
    part, part_config, train2, val2, identity2 = inputs(toy, tmp_path / "resumed")
    part_config["max_runtime_seconds"] = 0
    result = adapter.fit(part, part_config, train2, val2, identity2)
    assert result["completed_epochs"] == 1
    _, state = adapter.read_package(tmp_path / "resumed/last", require_training=True)
    assert state["config"]["package_kind"] == adapter.PACKAGE_KIND
    with pytest.raises(ValueError, match="Unsupported"):
        original.read_package(tmp_path / "resumed/last")
    resumed = toy.ToyModel(dropout=True); resumed.config.mode = "unbounded_spatial_mix"
    part_config["resume_if_present"] = True; part_config.pop("max_runtime_seconds")
    result = adapter.fit(resumed, part_config, train2, val2, identity2)
    assert result["status"] == "completed" and result["completed_epochs"] == 30
    for pointer in ("best", "last"):
        _, a = adapter.read_package(tmp_path / "full" / pointer, require_training=True)
        _, b = adapter.read_package(tmp_path / "resumed" / pointer, require_training=True)
        toy.assert_tree_equal(a, b)
    reread, _ = adapter.load_package(tmp_path / "resumed/best")
    assert reread.config.mode == "unbounded_spatial_mix"


def test_modified_resume_identity_is_rejected(tmp_path, toy):
    model, config, train, val, identity = inputs(toy, tmp_path / "run")
    config["max_runtime_seconds"] = 0; adapter.fit(model, config, train, val, identity)
    config["resume_if_present"] = True
    changed = deepcopy(identity); changed["extra_identity"] = "different"
    with pytest.raises(ValueError, match="identity"):
        adapter.fit(model, config, train, val, changed)


def test_full_grid_recipe_and_unfrozen_training_gate(tmp_path, monkeypatch):
    campaign = load("scripts/real_video_iws_unbounded/campaign.py", "_unbounded_test_campaign")
    config = campaign.validate_config(campaign.read(campaign.CONFIG))
    assert len(campaign.grid(config)) == 9
    assert {r["mode"] for r in campaign.grid(config)} == {"unbounded_spatial_mix"}
    assert config["training"] == campaign.read(campaign.base.CONFIG)["training"]
    altered = deepcopy(config); altered["training"]["epochs"] = 2
    with pytest.raises(ValueError): campaign.validate_config(altered)
    monkeypatch.setattr(campaign, "REGISTRATION", tmp_path / "not_registered.json")
    with pytest.raises(ValueError, match="candidate only"):
        campaign.check_registration()
    assert not campaign.REGISTRATION.exists()


def test_training_and_profile_reject_old_output_namespace(monkeypatch):
    runner = load("scripts/real_video_iws_unbounded/train.py", "_unbounded_test_runner")
    campaign = load("scripts/real_video_iws_unbounded/campaign.py", "_unbounded_test_campaign_for_paths")
    config = campaign.read(campaign.CONFIG)
    monkeypatch.setattr(campaign, "check_registration", lambda p: {"runs": campaign.grid(config)})
    monkeypatch.setattr(runner, "_module", lambda *a: campaign)
    with pytest.raises(ValueError, match="new-namespace"):
        runner.train(campaign.CONFIG, "pusht", "unbounded_spatial_mix", 0,
                     ROOT / "runs/real_video_iws/v1/pusht_bounded_spatial_mix_s0")
    with pytest.raises(ValueError, match="new ablation"):
        runner.profile(campaign.CONFIG, "pusht", "all", 64,
                       ROOT / "reports/real_video_iws/full_shape_profile_final.json")


def test_candidate_cannot_register_without_all_profiles(tmp_path, monkeypatch):
    campaign = load("scripts/real_video_iws_unbounded/campaign.py", "_unbounded_test_freeze")
    monkeypatch.setattr(campaign, "REGISTRATION", tmp_path / "registration.json")
    monkeypatch.setattr(campaign, "source_dependencies", lambda: {})
    with pytest.raises(ValueError, match="All three"):
        campaign.register(campaign.CONFIG, [], tmp_path / "review.json")
    assert not campaign.REGISTRATION.exists()
