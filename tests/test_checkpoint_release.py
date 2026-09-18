"""A release must not present partial training or a last model as best."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_checkpoint_release.py"
spec = importlib.util.spec_from_file_location("shiftwm_release_builder_test", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "pusht_factorized_s0"
    data, cache = tmp_path / "data", tmp_path / "features"
    write(data / "manifest.json", {"environment": "pusht"})
    write(run / "best/config.json", {"model_config": {"mode": "factorized"}, "provenance": {"weights_sha256": "fixture"}})
    (run / "best/model.pt").write_bytes(b"unused eligibility-test placeholder")
    write(run / "run_config.json", {"epochs": 3, "data_root": str(data),
                                   "dataset_kwargs": {"feature_cache": str(cache)}})
    write(run / "training_summary.json", {"status": "completed", "completed_epochs": 3,
                                           "best_validation_prediction_loss": .2})
    rows = [{"epoch": epoch, "step": epoch * 10, "val": {"prediction_loss": value}}
            for epoch, value in [(1, .5), (2, .2), (3, .3)]]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    return run


def test_completed_run_records_actual_best_epoch(source):
    record = release.inspect_source(source)
    assert record["completed_epochs"] == 3
    assert record["best_epochs"] == [2]
    assert record["best_prediction"] == .2


def test_interrupted_run_cannot_be_published(source):
    write(source / "training_summary.json", {"status": "interrupted_checkpoint_saved", "completed_epochs": 2})
    with pytest.raises(ValueError, match="fully completed"):
        release.inspect_source(source)


def test_missing_epoch_is_not_completed_evidence(source):
    rows = [{"epoch": 1, "val": {"prediction_loss": .5}}, {"epoch": 3, "val": {"prediction_loss": .2}}]
    (source / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="missing a completed epoch"):
        release.inspect_source(source)


def test_summary_cannot_replace_measured_best_metric(source):
    write(source / "training_summary.json", {"status": "completed", "completed_epochs": 3,
                                             "best_validation_prediction_loss": .1})
    with pytest.raises(ValueError, match="disagrees"):
        release.inspect_source(source)


def test_last_checkpoint_directory_is_rejected(source):
    last = source / "last"
    last.mkdir()
    (last / "model.pt").write_bytes(b"not best")
    with pytest.raises(ValueError, match="validation-best"):
        release.inspect_source(last)


def test_frozen_requires_explicit_separate_kind(source):
    write(source / "best/config.json", {"model_config": {"mode": "frozen"}})
    with pytest.raises(ValueError, match="explicit --frozen"):
        release.inspect_source(source)


def test_portable_check_rejects_last_epoch_with_carried_best_metric(source, monkeypatch):
    """A last checkpoint's best_metric field alone does not make its weights best."""
    record = release.inspect_source(source)
    class Model:
        def eval(self):
            return self
        def requires_grad_(self, enabled):
            return self
    state = {"config": record["config"], "epoch": 3, "best_metric": .2}
    monkeypatch.setattr(release, "load_package", lambda *args, **kwargs: (Model(), state))
    with pytest.raises(ValueError, match="did not attain"):
        release.verify_portable_forward(source / "best", record)
