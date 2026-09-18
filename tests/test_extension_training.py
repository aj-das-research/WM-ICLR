"""Full-epoch CPU fixtures check selection and exact interrupted continuation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from shiftwm.extensions import train as trainer
from shiftwm.extensions import model as models
from shiftwm.extensions import checkpoint as cp
from shiftwm.data import TRAIN_COMBINATIONS
from test_extension_model import model
from test_extension_checkpoint import refresh_manifest
from test_model import TinyBase, BASE_CONFIG


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(17)
        self.data = {"features": torch.randn(7, 8, 8, generator=g),
                     "reference_features": torch.randn(7, 8, 8, generator=g),
                     "paired_features": torch.randn(7, 8, 8, generator=g),
                     "actions": torch.randn(7, 7, 4, generator=g),
                     "observation_id": torch.tensor([0, 0, 1, 0, 1, 1, 0])}

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {key: value[index].clone() for key, value in self.data.items()}


def config(output, mode="factorized"):
    return {"seed": 0, "epochs": 30, "batch_size": 3, "num_workers": 0, "device": "cpu",
            "lr": .001, "min_lr": .00001, "weight_decay": .001, "grad_clip": 1., "bf16": True,
            "output_dir": str(output), "architecture": "transformer", "mode": mode,
            "sequence_length": 8, "stride": 1, "dataset_kwargs": {"feature_cache": "fixture-cache"}}


@pytest.fixture(autouse=True)
def tiny_threads_and_loader(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(models, "create_base", lambda config: TinyBase())
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("extension-completed")
    torch.manual_seed(42)
    trainer.fit(model("factorized"), config(path), TinyDataset(), TinyDataset())
    torch.set_num_threads(old)
    return path


@pytest.mark.parametrize("mode", models.MODES)
def test_full30_epochs_selects_common_fp32_recursive_mse(tmp_path, monkeypatch, mode):
    m = model(mode)
    forward, validation = m.forward, m.recursive_validation
    observed = []

    def val(batch):
        assert not m.training and torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda")
        out = validation(batch)
        assert out["predictions"].dtype == torch.float32 and out["targets"].shape[1] == 5
        observed.append((len(batch["actions"]), float(out["prediction_loss"])))
        # Force total loss to disagree with the selection metric. Selection
        # must use prediction_loss, never aligned/regularized aggregate loss.
        out["loss"] = 1000 - out["prediction_loss"]
        return out

    monkeypatch.setattr(m, "recursive_validation", val)
    result = trainer.fit(m, config(tmp_path, mode), TinyDataset(), TinyDataset())
    rows = trainer.metric_rows(tmp_path)
    assert result["completed_epochs"] == len(rows) == 30
    assert result["step"] == 90 and len(observed) == 90
    for i, row in enumerate(rows):
        expected = sum(n * x for n, x in observed[3 * i:3 * i + 3]) / 7
        assert row["val"]["prediction_loss"] == expected
    _, selected = cp.read_package(tmp_path / "best")
    best = min(rows, key=lambda row: row["val"]["prediction_loss"])
    assert selected["epoch"] == best["epoch"]
    assert selected["best_metric"] == best["val"]["prediction_loss"]
    assert trainer.validate_completed(tmp_path, result["training_identity"]) == result


def test_mid_epoch_resume_matches_every_metric_and_model_tensor(completed, tmp_path):
    settings = config(tmp_path)
    settings["max_runtime_seconds"] = 1e-12
    torch.manual_seed(42)
    interrupted = trainer.fit(model("factorized"), settings, TinyDataset(), TinyDataset())
    assert interrupted["completed_epochs"] == 0 and interrupted["batches_completed_in_epoch"] == 1
    settings.pop("max_runtime_seconds")
    settings["resume"] = str(tmp_path / "last")
    result = trainer.fit(model("factorized"), settings, TinyDataset(), TinyDataset())
    expected = cp.read_package(completed / "last")[1]
    actual = cp.read_package(tmp_path / "last")[1]
    assert expected["epoch"] == actual["epoch"] == 30 and expected["step"] == actual["step"] == 90
    for name, tensor in expected["state_dict"].items():
        torch.testing.assert_close(actual["state_dict"][name], tensor, rtol=0, atol=0)
    assert [(r["train"], r["val"]) for r in trainer.metric_rows(completed)] == [
        (r["train"], r["val"]) for r in trainer.metric_rows(tmp_path)]
    assert trainer.validate_completed(tmp_path, result["training_identity"]) == result


@pytest.mark.parametrize("damage", ["missing_epoch", "duplicate_epoch", "nan", "summary_epochs",
                                   "summary_identity", "summary_criterion", "last_epoch", "best_epoch",
                                   "method", "selection", "precision", "source_identity"])
def test_complete_validation_rejects_tampered_evidence(completed, tmp_path, damage):
    # Dereference generations: a portable copied directory is supported by
    # readers and must still receive all integrity/semantic validations.
    shutil.copytree(completed, tmp_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*.generations"))
    summary_path = tmp_path / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    identity = summary["training_identity"]
    if damage in {"missing_epoch", "duplicate_epoch", "nan"}:
        rows = trainer.metric_rows(tmp_path)
        if damage == "missing_epoch":
            rows.pop()
        elif damage == "duplicate_epoch":
            rows[-1]["epoch"] = 29
        else:
            rows[-1]["val"]["prediction_loss"] = float("nan")
        (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif damage.startswith("summary_"):
        key, value = {"summary_epochs": ("completed_epochs", 29),
                      "summary_identity": ("training_identity", "bad"),
                      "summary_criterion": ("validation_metric", "one_step")}[damage]
        summary[key] = value
        summary_path.write_text(json.dumps(summary))
    else:
        path = tmp_path / ("last" if damage == "last_epoch" else "best")
        state = torch.load(path / "model.pt", weights_only=True)
        if damage == "last_epoch":
            state["epoch"] = 29
        elif damage == "best_epoch":
            state["epoch"] = 31
        elif damage == "method":
            state["config"]["mode"] = "constant_dynamics"
        elif damage == "selection":
            state["config"]["metadata"]["selection"] = "training_loss"
        elif damage == "precision":
            state["config"]["metadata"]["validation_precision"] = "bfloat16"
        else:
            state["config"]["provenance"]["tampered"] = True
        torch.save(state, path / "model.pt")
        (path / "config.json").write_text(json.dumps(state["config"]))
        refresh_manifest(path)
    with pytest.raises(ValueError):
        trainer.validate_completed(tmp_path, identity)


def test_resume_rejects_changed_learning_rule_before_mutating_evidence(completed):
    settings = config(completed)
    settings.update(resume=str(completed / "last"), lr=.002)
    before = (completed / "metrics.jsonl").read_bytes()
    with pytest.raises(ValueError, match="Resume sources"):
        trainer.fit(model("factorized"), settings, TinyDataset(), TinyDataset())
    assert (completed / "metrics.jsonl").read_bytes() == before


def setup_sources(tmp_path, monkeypatch):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    pretrained, data, cache = (tmp_path / name for name in ("pretrained", "data", "cache"))
    for p in (pretrained, data, cache):
        p.mkdir()
    (pretrained / "config.json").write_text(json.dumps(BASE_CONFIG))
    torch.save(TinyBase().state_dict(), pretrained / "weights.pt")
    manifest = {"environment": "fixture", "train_combinations": TRAIN_COMBINATIONS,
                "episodes": [{"trajectory_id": split, "seed": seed, "split": split,
                              "steps": 8, "dynamics_id": 0, "file": split + ".npz"}
                             for seed, split in enumerate(("train", "val"))]}
    (data / "manifest.json").write_text(json.dumps(manifest))
    (cache / "manifest.json").write_text(json.dumps({
        "dataset_manifest_sha256": cp.file_sha256(data / "manifest.json"),
        "encoder_weights_sha256": cp.file_sha256(pretrained / "weights.pt")}))
    for split in ("train", "val"):
        np.savez(cache / (split + ".npz"), features=np.zeros((4, 9, 8), np.float32),
                 actions=np.zeros((8, 4), np.float32))
    stats = tmp_path / "action_stats.json"
    stats.write_text(json.dumps({"mean": [0] * 4, "std": [1] * 4}))
    settings = config(tmp_path / "output")
    settings.update(pretrained_dir=str(pretrained), data_root=str(data), action_stats=str(stats),
                    dataset_kwargs={"feature_cache": str(cache)})
    return settings, cache


def test_identity_pins_consumed_feature_payloads_and_source_code(tmp_path, monkeypatch):
    settings, cache = setup_sources(tmp_path, monkeypatch)
    before = trainer.initialize(settings)
    assert len(before.provenance["extension_training_feature_hashes"]) == 2
    assert "extensions/train.py" in before.provenance["extension_source_hashes"]
    np.savez(cache / "train.npz", features=np.ones((4, 9, 8), np.float32),
             actions=np.zeros((8, 4), np.float32))
    after = trainer.initialize(settings)
    assert trainer.training_identity(settings, before) != trainer.training_identity(settings, after)


@pytest.mark.parametrize("damage", ["cache_encoder", "manifest", "zero_std", "subsample", "epochs"])
def test_initialize_rejects_invalid_data_and_training_contract(tmp_path, monkeypatch, damage):
    settings, cache = setup_sources(tmp_path, monkeypatch)
    if damage in {"cache_encoder", "manifest"}:
        p = cache / "manifest.json"
        x = json.loads(p.read_text())
        x["encoder_weights_sha256" if damage == "cache_encoder" else "dataset_manifest_sha256"] = "bad"
        p.write_text(json.dumps(x))
    elif damage == "zero_std":
        Path(settings["action_stats"]).write_text(json.dumps({"mean": [0] * 4, "std": [0] * 4}))
    elif damage == "subsample":
        settings["dataset_kwargs"]["max_episodes"] = 1
    else:
        settings["epochs"] = 1
    with pytest.raises(ValueError):
        trainer.initialize(settings)
