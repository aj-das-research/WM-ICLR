"""Scientific launch guards: omitted arms, changed horizons and stale recipes."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iws_campaign_test_module", ROOT / "scripts/real_video_iws/campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def config():
    return json.loads(campaign.CONFIG.read_text())


def test_complete_grid_and_information_budget(config):
    campaign.validate_config(config)
    rows = campaign.grid(config)
    assert len(rows) == len({row["name"] for row in rows}) == 27
    assert len({row["output"] for row in rows}) == 27
    assert {(row["task"], row["mode"], row["seed"]) for row in rows} == {
        (task, mode, seed) for task in campaign.TASKS for mode in campaign.MODES for seed in range(3)}


@pytest.mark.parametrize("change", [
    lambda c: c["modes"].pop(), lambda c: c["seeds"].pop(),
    lambda c: c["tasks"].pop("bimanual_rope"),
    lambda c: c["tasks"]["bimanual_box"].update(action_dim=4),
    lambda c: c["training"].update(horizon=15),
    lambda c: c["training"].update(epochs=1),
    lambda c: c["training"].update(accumulation_steps=c["training"]["accumulation_steps"]+1),
    lambda c: c["training"].update(selector="best_any_horizon"),
    lambda c: c["evaluation"].update(official_validation_allowed_during_training=True),
    lambda c: c.update(package_kind="shiftwm_real_video_spatial_v1"),
    lambda c: c.update(split_sha256="unverified"),
])
def test_incomplete_or_changed_recipe_rejected(config, change):
    candidate = copy.deepcopy(config)
    change(candidate)
    with pytest.raises(ValueError):
        campaign.validate_config(candidate)


def test_resource_change_preserves_effective_batch(config):
    config["training"].update(microbatch_size=4, accumulation_steps=16)
    campaign.validate_config(config)


def test_registration_rejects_stale_dependency_and_config(config, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    dependency = tmp_path / "model.py"
    dependency.write_text("original")
    registration_path = tmp_path / "registration.json"
    reg = {"status": "registered_before_predictor_training", "schema": "shiftwm_iws_training_registration_v1",
           "config_sha256": campaign.sha(config_path), "runs": campaign.grid(config),
           "dependencies": {str(dependency): campaign.sha(dependency)}, "expected_runs": 27, "epochs_per_run": 30}
    registration_path.write_text(json.dumps(reg))
    monkeypatch.setattr(campaign, "REGISTRATION", registration_path)
    assert campaign.check_registration(config_path) == reg
    dependency.write_text("changed_after_registration")
    with pytest.raises(ValueError, match="dependency changed"):
        campaign.check_registration(config_path)
    dependency.write_text("original")
    config["training"].update(microbatch_size=4, accumulation_steps=16)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="registered recipe"):
        campaign.check_registration(config_path)
