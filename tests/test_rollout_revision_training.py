"""CPU correctness fixtures for the isolated trainer; not experiment results."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch

from test_dynamics_revision import TinyBase, TinyDataset, script
from test_rollout_revision import revision


trainer = script("train_rollout_revision")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration(output):
    return {"seed": 0, "epochs": 30, "batch_size": 3, "num_workers": 0, "device": "cpu",
            "lr": .001, "min_lr": .00001, "weight_decay": .001, "bf16": True,
            "output_dir": str(output)}


def metric_rows(output):
    return [json.loads(line) for line in (Path(output) / "metrics.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    output = tmp_path_factory.mktemp("completed_rollout_training")
    config = configuration(output)
    torch.manual_seed(42)
    model = revision(objective="teacher_forced", context_mode="constant")
    trainer.fit(model, config, TinyDataset(), TinyDataset())
    torch.set_num_threads(previous)
    return output


@pytest.mark.parametrize("objective", ["teacher_forced", "recursive"])
def test_complete_loop_uses_common_fp32_recursive_selection(tmp_path, monkeypatch, objective):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    model = revision(objective=objective)
    config = configuration(tmp_path)
    original_forward, original_validation = model.forward, model.recursive_validation
    validations = []

    def forward(batch):
        assert model.training, "Validation must not use the selected training objective"
        return original_forward(batch)

    def validation(batch):
        assert not model.training and not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")
        output = original_validation(batch)
        assert output["predictions"].dtype == torch.float32
        assert output["targets"].shape[1] == 5
        validations.append((len(batch["actions"]), float(output["prediction_loss"])))
        return output

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(model, "recursive_validation", validation)
    summary = trainer.fit(model, config, TinyDataset(), TinyDataset())
    rows = metric_rows(tmp_path)
    assert len(rows) == 30 and len(validations) == 90
    assert [row["epoch"] for row in rows] == list(range(1, 31))
    assert summary["completed_epochs"] == 30 and summary["step"] == 90
    for epoch, row in enumerate(rows):
        observed = validations[epoch * 3:epoch * 3 + 3]
        expected = sum(n * value for n, value in observed) / sum(n for n, _ in observed)
        assert row["val"]["prediction_loss"] == expected
    best = min(rows, key=lambda row: row["val"]["prediction_loss"])
    state = torch.load(tmp_path / "best/model.pt", weights_only=True)
    assert state["epoch"] == best["epoch"]
    assert summary["best_validation_prediction_loss"] == state["best_metric"] == best["val"]["prediction_loss"]
    metadata = state["config"]["metadata"]
    assert metadata["validation_precision"] == "float32"
    assert metadata["training_objective"] == objective
    assert trainer.validate_completed(tmp_path, summary["training_identity"]) == summary


def test_interruption_then_resume_matches_full_30_epoch_path(completed_run, tmp_path, monkeypatch):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    config = configuration(tmp_path)
    config["max_runtime_seconds"] = 1e-12
    torch.manual_seed(42)
    stopped = trainer.fit(revision(objective="teacher_forced", context_mode="constant"), config,
                          TinyDataset(), TinyDataset())
    assert stopped["status"] == "interrupted_checkpoint_saved"
    assert stopped["batches_completed_in_epoch"] == 1 and stopped["completed_epochs"] == 0
    config.pop("max_runtime_seconds")
    config["resume"] = str(tmp_path / "last")
    result = trainer.fit(revision(objective="teacher_forced", context_mode="constant"), config,
                         TinyDataset(), TinyDataset())
    baseline = torch.load(completed_run / "last/model.pt", weights_only=True)
    resumed = torch.load(tmp_path / "last/model.pt", weights_only=True)
    assert baseline["step"] == resumed["step"] == 90
    assert baseline["epoch"] == resumed["epoch"] == 30
    for key, value in baseline["state_dict"].items():
        torch.testing.assert_close(resumed["state_dict"][key], value, rtol=0, atol=0)
    assert [(r["train"], r["val"]) for r in metric_rows(completed_run)] == [
        (r["train"], r["val"]) for r in metric_rows(tmp_path)]
    assert trainer.validate_completed(tmp_path, result["training_identity"]) == result


@pytest.mark.parametrize("damage", [
    "summary_epochs", "summary_identity", "summary_criterion", "summary_best", "missing_epoch",
    "duplicate_epoch", "nonfinite_metric", "sidecar_config", "package_identity", "selection",
    "validation_protocol", "validation_precision", "training_objective", "context_mode",
    "wrong_best_epoch", "wrong_last_epoch", "wrong_last_step", "wrong_package_best",
])
def test_completion_rejects_tampered_protocol_or_history(completed_run, tmp_path, monkeypatch, damage):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    shutil.copytree(completed_run, tmp_path, dirs_exist_ok=True)
    summary_path = tmp_path / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    identity = summary["training_identity"]
    if damage.startswith("summary_"):
        key, value = {"summary_epochs": ("completed_epochs", 29),
                      "summary_identity": ("training_identity", "tampered"),
                      "summary_criterion": ("validation_metric", "teacher_forced_mse"),
                      "summary_best": ("best_validation_prediction_loss", -1)}[damage]
        summary[key] = value
        summary_path.write_text(json.dumps(summary))
    elif damage in {"missing_epoch", "duplicate_epoch", "nonfinite_metric"}:
        rows = metric_rows(tmp_path)
        if damage == "missing_epoch":
            rows.pop()
        elif damage == "duplicate_epoch":
            rows[-1]["epoch"] = rows[-2]["epoch"]
        else:
            rows[-1]["val"]["prediction_loss"] = float("nan")
        (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        directory = tmp_path / ("last" if damage in {"wrong_last_epoch", "wrong_last_step"} else "best")
        state = torch.load(directory / "model.pt", weights_only=True)
        if damage == "sidecar_config":
            sidecar = deepcopy(state["config"])
            sidecar["metadata"]["training_identity"] = "tampered"
            (directory / "config.json").write_text(json.dumps(sidecar))
        else:
            if damage == "package_identity":
                state["config"]["metadata"]["training_identity"] = "tampered"
            elif damage in {"selection", "validation_protocol", "validation_precision",
                             "training_objective", "context_mode"}:
                state["config"]["metadata"][damage] = "tampered"
            elif damage == "wrong_best_epoch":
                state["epoch"] = 31
            elif damage == "wrong_last_epoch":
                state["epoch"] = 29
            elif damage == "wrong_last_step":
                state["step"] -= 1
            elif damage == "wrong_package_best":
                state["best_metric"] = -1
            torch.save(state, directory / "model.pt")
            (directory / "config.json").write_text(json.dumps(state["config"]))
    with pytest.raises(ValueError):
        trainer.validate_completed(tmp_path, identity)


def test_resume_rejects_changed_scientific_settings(completed_run, monkeypatch):
    config = configuration(completed_run)
    config["resume"] = str(completed_run / "last")
    config["lr"] = .5
    with pytest.raises(ValueError, match="Resume sources"):
        trainer.fit(revision(objective="teacher_forced", context_mode="constant"), config,
                    TinyDataset(), TinyDataset())


def test_validation_nonfinite_is_not_saved_as_complete(tmp_path, monkeypatch):
    model = revision()
    config = configuration(tmp_path)
    original = model.recursive_validation

    def invalid(batch):
        result = original(batch)
        result["prediction_loss"] = torch.tensor(float("nan"))
        return result

    monkeypatch.setattr(model, "recursive_validation", invalid)
    with pytest.raises(FloatingPointError, match="Nonfinite revision epoch"):
        trainer.fit(model, config, TinyDataset(), TinyDataset())
    assert not (tmp_path / "training_summary.json").exists()
    assert not (tmp_path / "best/model.pt").exists()


@pytest.mark.parametrize("config", [{"seed": 1, "epochs": 30}, {"seed": 0, "epochs": 29}])
def test_public_initialization_requires_seed0_and_full30epochs(config):
    with pytest.raises(ValueError, match="seed0 and all 30"):
        trainer.initialize(config)
