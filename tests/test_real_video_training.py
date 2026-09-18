"""Real-video lifecycle checks use full 30-epoch CPU fixtures, not GPU smoke jobs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("real_video_training", ROOT / "scripts/real_video/train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(67)
        self.features = torch.randn(5, 8, 8, generator=generator)
        self.actions = torch.randn(5, 7, 10, generator=generator)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {"features": self.features[index], "actions": self.actions[index]}


def model(mode="factorized"):
    trainer.seed_everything(0)
    return trainer.RealVideoWorldModel({"feature_dim": 8, "action_dim": 10, "hidden_dim": 12,
        "context_dim": 4, "context_hidden": 8, "depth": 1, "mode": mode},
        [0.] * 8, [2.] * 8, [0.] * 10, [1.] * 10)


def settings(directory, mode="factorized"):
    return {"epochs": 30, "seed": 0, "mode": mode, "output_dir": str(directory),
            "device": "cpu", "batch_size": 3, "lr": .001, "min_lr": .00001,
            "weight_decay": .01, "grad_clip": 1., "bf16": True}


def identity(config):
    return {"scientific_config": trainer.scientific_config(config), "dependencies": {}}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("realvideo-completed")
    cfg = settings(directory)
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    torch.set_num_threads(previous)
    return directory, result


def test_all30_epochs_and_offline_best_selection(completed, monkeypatch):
    directory, result = completed
    assert result["completed_epochs"] == 30 and result["step"] == 60
    rows = trainer.metric_rows(directory)
    best = min(rows, key=lambda row: row["val"]["standardized_mse"])
    loaded, state = trainer.load_package(directory / "best")
    assert state["epoch"] == best["epoch"]
    assert state["best_metric"] == best["val"]["standardized_mse"]
    assert result["parameter_counts"]["total"] > result["parameter_counts"]["trainable"] > 0
    assert trainer.validate_completed(directory) == result
    assert not loaded.training
    assert torch.load(directory / "best/training_state.pt", weights_only=True)["epoch"] == state["epoch"]


def test_validation_is_fp32_fullquery_metric_not_total_loss():
    m = model()
    original = m.forward
    expected = []
    def forward(batch):
        assert not m.training and torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda")
        out = original(batch)
        assert out["standardized_targets"].shape[1:] == (5, 8)
        assert out["standardized_predictions"].dtype == torch.float32
        values = (out["standardized_predictions"] - out["standardized_targets"]).square()
        expected.append((values.mean().item(), values.numel()))
        out["loss"] = -torch.ones(())  # never a validation selection criterion
        return out
    m.forward = forward
    actual = trainer.epoch_pass(m, torch.utils.data.DataLoader(TinyDataset(), batch_size=3), torch.device("cpu"))
    assert actual["standardized_mse"] == sum(v*n for v,n in expected)/sum(n for _,n in expected)
    assert actual["elements"] == 5*5*8


def test_interrupted_completed_epoch_resume_is_exact(completed, tmp_path):
    cfg = settings(tmp_path)
    cfg["max_runtime_seconds"] = 1e-12
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    assert result["status"] == "interrupted" and result["completed_epochs"] == 1
    # Simulate stale external journal: committed model history wins on resume.
    (tmp_path / "metrics.jsonl").write_text("")
    cfg.pop("max_runtime_seconds")
    cfg["resume_if_present"] = True
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    assert result["status"] == "completed"
    expected = trainer.read_package(completed[0] / "last")[1]
    actual = trainer.read_package(tmp_path / "last")[1]
    assert expected["history"] == actual["history"]
    for key, value in expected["state_dict"].items():
        torch.testing.assert_close(actual["state_dict"][key], value, rtol=0, atol=0)


def copy_completed(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".*.generations", ".training.lock"))


@pytest.mark.parametrize("damage", ["history", "summary", "selection", "identity", "file", "epochs", "mode"])
def test_completion_rejects_changed_evidence(completed, tmp_path, damage):
    copy_completed(completed[0], tmp_path)
    if damage == "history":
        rows = trainer.metric_rows(tmp_path)
        trainer.write_history(rows[:-1], tmp_path)
    elif damage in ("summary", "selection", "identity"):
        path = tmp_path / "training_summary.json"
        value = json.loads(path.read_text())
        key, new = {"summary": ("completed_epochs", 29), "selection": ("validation_metric", "train_loss"),
                    "identity": ("training_identity", "bad")}[damage]
        value[key] = new
        trainer.atomic_json(value, path)
    elif damage == "file":
        with (tmp_path / "best/model.pt").open("ab") as handle:
            handle.write(b"corrupt")
    else:
        path = tmp_path / "training_config.json"
        value = json.loads(path.read_text())
        value["epochs" if damage == "epochs" else "mode"] = 29 if damage == "epochs" else "framewise"
        trainer.atomic_json(value, path)
    with pytest.raises(ValueError):
        trainer.validate_completed(tmp_path)


def test_resume_rejects_changed_hyperparameters(completed, tmp_path):
    copy_completed(completed[0], tmp_path)
    # Copied packages are intentionally immutable; mismatch is rejected before save.
    cfg = settings(tmp_path)
    cfg.update(resume_if_present=True, lr=.002)
    with pytest.raises(ValueError, match="identity"):
        trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))


def test_final_checkpoint_repairs_interrupted_completion_marker(completed, tmp_path):
    copy_completed(completed[0], tmp_path)
    (tmp_path / "training_summary.json").unlink()
    (tmp_path / "metrics.jsonl").write_text("")
    cfg = settings(tmp_path)
    cfg["resume_if_present"] = True
    result = trainer.fit(model(), cfg, TinyDataset(), TinyDataset(), identity(cfg))
    assert result["completed_epochs"] == 30 and result["status"] == "completed"
    assert trainer.metric_rows(tmp_path) == trainer.metric_rows(completed[0])


def audited_fixture(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    processed.mkdir(); cache.mkdir()
    trainer.atomic_json({"source": "real fixture manifest"}, processed / "manifest.json")
    source_sha = trainer.sha256(processed / "manifest.json")
    trainer.atomic_json({"status": "passed", "dataset_manifest_sha256": source_sha}, processed / "data_audit.json")
    manifest = {"status": "complete", "feature_dim": 8, "action_dim": 10,
                "identity": {"dataset_manifest_sha256": source_sha},
                "episodes": [{"episode_id": str(i), "session_id": str(i), "split": split, "cameras": {}}
                             for i, split in enumerate(("train", "val", "test"))]}
    trainer.atomic_json(manifest, cache / "manifest.json")
    stats = {"fit_split": "train", "camera": trainer.PRIMARY_CAMERA,
             "cache_manifest_sha256": trainer.sha256(cache / "manifest.json"), "ddof": 1, "std_floor": 1e-5,
             "feature_mean": [0.]*8, "feature_std": [1.]*8, "action_mean": [0.]*10, "action_std": [1.]*10}
    trainer.atomic_json(stats, cache / "training_statistics.json")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Registered fixture protocol")
    monkeypatch.setattr(trainer, "source_files", lambda: {})
    return {"cache_root": str(cache), "metadata_audit": str(processed / "data_audit.json"),
            "protocol_path": str(protocol)}


@pytest.mark.parametrize("damage", ["audit", "dataset_hash", "crossed_session", "statistics", "statistics_hash"])
def test_audit_rejects_unverified_or_leaking_data(tmp_path, monkeypatch, damage):
    cfg = audited_fixture(tmp_path, monkeypatch)
    trainer.audit_inputs(cfg)
    if damage in ("audit", "dataset_hash"):
        path = Path(cfg["metadata_audit"])
        value = json.loads(path.read_text())
        value["status" if damage == "audit" else "dataset_manifest_sha256"] = "bad"
    elif damage == "crossed_session":
        path = Path(cfg["cache_root"]) / "manifest.json"
        value = json.loads(path.read_text())
        value["episodes"][1]["session_id"] = value["episodes"][0]["session_id"]
    else:
        path = Path(cfg["cache_root"]) / "training_statistics.json"
        value = json.loads(path.read_text())
        value["fit_split" if damage == "statistics" else "cache_manifest_sha256"] = "test"
    trainer.atomic_json(value, path)
    with pytest.raises(ValueError):
        trainer.audit_inputs(cfg)


def test_statistics_are_recomputed_only_from_training_episode_arrays():
    arrays = np.arange(24, dtype=np.float32).reshape(3,8)
    dataset = type("Data", (), {"episodes": [{"features": arrays, "actions": arrays[:2]}]})()
    stats = {"counts": {"feature": 3, "action": 2}, "feature_mean": arrays.mean(0),
             "feature_std": arrays.std(0,ddof=1), "action_mean": arrays[:2].mean(0),
             "action_std": arrays[:2].astype(np.float64).std(0,ddof=1)}
    trainer.verify_training_statistics(dataset, stats)
    stats["feature_mean"] = arrays.mean(0) + .1
    with pytest.raises(ValueError):
        trainer.verify_training_statistics(dataset, stats)
