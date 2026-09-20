"""New-namespace contracts on synthetic tensors; no dataset payload reads."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("external_train_test_" + name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


@pytest.fixture(scope="module")
def trainer():
    torch.set_num_threads(2)
    return module("train")


@pytest.fixture(scope="module")
def model(trainer):
    return trainer.models.ExternalDinoWM(
        {**trainer.models.ARCHITECTURE, "mode": "official_one_step_shifted"},
        feature_mean=torch.zeros(6144), feature_std=torch.ones(6144),
        action_mean=torch.zeros(35), action_std=torch.ones(35)).eval()


def test_exact_package_reload_preserves_forecasts(trainer, model, tmp_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    counts = {"total": 19412420, "trainable": 19412420}
    metadata = {"training_identity": "synthetic-only", "parameter_counts": counts}
    trainer.base.save_package(model, tmp_path / "best", optimizer=optimizer,
        scheduler=scheduler, epoch=1, step=1, best_metric=1., metadata=metadata,
        generator=torch.Generator().manual_seed(0), history=[])
    restored, state = trainer.load_package(tmp_path / "best", "cpu")
    assert state["config"]["package_kind"] == "adapted_official_dinowm_droid_v1"
    assert restored.package_config == model.package_config
    batch = trainer.models.adapter.synthetic_batch(1)
    with torch.inference_mode():
        for objective in trainer.models.adapter.OBJECTIVES:
            assert torch.equal(model(batch, objective)["standardized_predictions"],
                               restored(batch, objective)["standardized_predictions"])


class RoutingModel(torch.nn.Module):
    def __init__(self, mode):
        super().__init__(); self.config = SimpleNamespace(mode=mode)
        self.scale = torch.nn.Parameter(torch.tensor(1.)); self.calls = []

    def forward(self, batch, objective):
        self.calls.append((objective, torch.is_grad_enabled(), self.training))
        target = batch["features"][:, 1:4] if objective == "official_one_step_shifted" else batch["features"][:, 3:13]
        return {"standardized_predictions": torch.ones_like(target) * self.scale,
                "standardized_targets": target.detach()}


@pytest.mark.parametrize("mode,steps", [("official_one_step_shifted", 3), ("matched_recursive_h10", 10)])
def test_training_routes_declared_objective_but_validation_always_h10(trainer, mode, steps):
    model = RoutingModel(mode)
    batches = [{"features": torch.zeros(n, 13, 6144), "episode_index": torch.tensor(ids)}
               for n, ids in ((2, [0, 0]), (1, [1]))]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    training = trainer.epoch_pass(model, batches, torch.device("cpu"), optimizer)
    validation = trainer.epoch_pass(model, batches, torch.device("cpu"))
    assert training["elements"] == 3 * steps * 6144
    assert training["target_grid_indices"] == ([1, 2, 3] if steps == 3 else list(range(3, 13)))
    assert validation["elements"] == 3 * 10 * 6144 and validation["episodes"] == 2
    assert validation["standardized_mse"] == 1.
    assert model.calls == [(mode, True, True)] * 2 + [("matched_recursive_h10", False, False)] * 2


def test_validation_is_window_weighted_not_equal_episode(trainer):
    model = RoutingModel("official_one_step_shifted")
    batches = [{"features": torch.zeros(2, 13, 6144), "episode_index": torch.tensor([0, 0])},
               {"features": torch.full((1, 13, 6144), 4.), "episode_index": torch.tensor([1])}]
    value = trainer.epoch_pass(model, batches, torch.device("cpu"))
    assert value["standardized_mse"] == pytest.approx(11 / 3)
    assert value["equal_episode_diagnostic_mse"] == 5.


@pytest.mark.parametrize("field,value", [("epochs", 29), ("batch_size", 64), ("train_horizon", 5),
                                         ("validation_stride", 2), ("cache_root", "reserved"), ("seed", True)])
def test_recipe_and_population_changes_rejected(field, value):
    registry = module("registry")
    config = registry.expected_config("official_one_step_shifted", 0); config[field] = value
    with pytest.raises(ValueError): registry.validate_config(config)


def test_review_rejection_precedes_source_and_data_access(monkeypatch, tmp_path):
    registry = module("registry")
    registration = tmp_path / "registration.json"; review = tmp_path / "review.json"
    registration.write_text(json.dumps({"dependencies": {"forbidden-payload": "bad"}}))
    review.write_text(json.dumps({"status": "passed", "registration_sha256": "stale"}))
    monkeypatch.setattr(registry, "REG", registration); monkeypatch.setattr(registry, "REVIEW", review)
    monkeypatch.setattr(registry, "sources", lambda: pytest.fail("Sources touched before review rejection"))
    monkeypatch.setattr(registry, "profile_gate", lambda: pytest.fail("Profile touched before review rejection"))
    with pytest.raises(ValueError, match="review missing or stale"): registry.verify()


def test_completion_rejects_partial_population_and_wrong_objective(trainer, monkeypatch, tmp_path):
    registry = module("registry")
    (tmp_path / "training_config.json").write_text(json.dumps(registry.expected_config("official_one_step_shifted", 0)))
    def part(steps, n, objective):
        return {"supervised_grids": steps, "windows": n, "elements": steps * n * 6144,
                "objective": objective, "batches": (n + 127) // 128,
                "target_grid_indices": [1, 2, 3] if steps == 3 else list(range(3, 13)), "episodes": 141}
    row = {"train": part(3, 18660, "official_one_step_shifted"), "val": part(10, 1631, "matched_recursive_h10")}
    monkeypatch.setattr(trainer.base, "validate_completed", lambda *a: {"parameter_counts": {"total": 19412420, "trainable": 19412420}})
    rows = [row]
    monkeypatch.setattr(trainer.base, "metric_rows", lambda _: rows)
    trainer.validate_completed(tmp_path)
    rows = [copy.deepcopy(row)]; rows[0]["val"]["windows"] -= 1
    with pytest.raises(ValueError, match="population"): trainer.validate_completed(tmp_path)
    rows = [copy.deepcopy(row)]; rows[0]["val"]["objective"] = "official_one_step_shifted"
    with pytest.raises(ValueError, match="population"): trainer.validate_completed(tmp_path)


def test_duplicate_submission_cannot_enter_training(monkeypatch, tmp_path):
    import fcntl
    campaign = module("campaign")
    directory = tmp_path / "run"; directory.mkdir()
    fake = SimpleNamespace(verify=lambda: {"runs": [{"name": "run", "config": "config.json"}]},
                           read=lambda _: {"output_dir": str(directory)})
    monkeypatch.setattr(campaign, "module", lambda name: fake if name == "registry" else pytest.fail("Training entered while locked"))
    with (directory / ".training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): campaign.run(index=0)


def test_execution_receipt_binds_actual_visible_hardware_and_scheduler(monkeypatch):
    campaign = module("campaign")
    props = SimpleNamespace(name="synthetic device", total_memory=123456, major=8, minor=9)
    fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test-cuda"),
                           cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                get_device_properties=lambda _: props))
    monkeypatch.setenv("SLURM_JOB_ID", "123"); monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    value = campaign.execution_identity(fake)
    assert value["slurm"]["SLURM_JOB_ID"] == "123" and value["slurm"]["SLURM_ARRAY_TASK_ID"] == "4"
    assert value["visible_gpus"] == [{"visible_index": 0, "name": "synthetic device",
                                      "total_memory_bytes": 123456, "compute_capability": [8, 9]}]
    assert value["hostname"] and value["torch_version"] == "test" and value["torch_cuda_build"] == "test-cuda"
