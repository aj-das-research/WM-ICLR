"""Release scheduling tests with fake packages; no model load or training."""
import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / "scripts" / "watch_artifacts.py"
    spec = importlib.util.spec_from_file_location("watch_artifacts", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, status="completed"):
    run = module()
    run_dir = tmp_path / "runs" / "example"
    (run_dir / "best").mkdir(parents=True)
    for name in ("run_config.json", "metrics.jsonl", "best/model.pt", "best/config.json"):
        (run_dir / name).write_text("fixture")
    run.atomic_json(run_dir / "training_summary.json", {"status": status, "completed_epochs": 30})
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.json").write_text("{}")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "manifest.json").write_text("{}")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "build_checkpoint_release.py").write_text("# fixture only")
    config = {"output_dir": "runs/example", "epochs": 30, "data_root": "data",
              "dataset_kwargs": {"feature_cache": "cache"}}
    run.atomic_json(tmp_path / "config.json", config)
    run.atomic_json(tmp_path / "campaign.json", {"tasks": [{"id": "example", "config": "config.json"}]})
    return run, run_dir


def fake_release(command, log, root, **kwargs):
    assert Path(command[1]).name == "build_checkpoint_release.py"
    assert "aggregate_results.py" not in " ".join(command)
    destination = Path(command[command.index("--output") + 1])
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("model.pt", "config.json", "MODEL_CARD.md", "verification.json"):
        (destination / name).write_text("fixture")
        files[name] = "fixture-hash"
    (destination / "release_manifest.json").write_text(json.dumps({"status": "ready", "package_files": files}))
    log.write_text("fixture builder completed\n")
    return 0


def test_release_waits_for_fully_completed_training(tmp_path):
    run, run_dir = fixture(tmp_path, status="interrupted_checkpoint_saved")
    calls = []
    watcher = run.ReleaseWatcher(tmp_path, "campaign.json", "releases", "state/status.json",
                                 invoke=lambda *args, **kwargs: calls.append(args))
    assert not watcher.scan_once()
    assert calls == []
    assert watcher.state["runs"]["example"]["status"] == "pending"
    run.atomic_json(run_dir / "training_summary.json", {"status": "completed", "completed_epochs": 29})
    assert not watcher.scan_once()
    assert calls == []
    assert watcher.state["runs"]["example"]["reason"] == "completed_epoch_count_mismatch"


def test_successful_release_is_idempotent_across_scans_and_watcher_restart(tmp_path):
    run, _ = fixture(tmp_path)
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return fake_release(command, **kwargs)
    kwargs = dict(invoke=invoke)
    watcher = run.ReleaseWatcher(tmp_path, "campaign.json", "releases", "state/status.json", **kwargs)
    assert watcher.scan_once()
    assert watcher.scan_once()
    restarted = run.ReleaseWatcher(tmp_path, "campaign.json", "releases", "state/status.json", **kwargs)
    assert restarted.scan_once()
    assert len(calls) == 1
    persisted = json.loads((tmp_path / "state" / "status.json").read_text())
    record = persisted["runs"]["example"]
    assert record["status"] == "ready" and record["attempts"] == 1
    assert Path(record["log"]).read_text() == "fixture builder completed\n"
    assert not list((tmp_path / "state").glob("*.tmp"))


def test_failed_builder_does_not_publish_success_or_retry_every_scan(tmp_path):
    run, _ = fixture(tmp_path)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return 7
    watcher = run.ReleaseWatcher(tmp_path, "campaign.json", "releases", "state/status.json", invoke=fail)
    assert not watcher.scan_once()
    assert not watcher.scan_once()
    assert len(calls) == 1
    assert watcher.state["runs"]["example"]["status"] == "failed"
    assert watcher.state["runs"]["example"]["returncode"] == 7


def test_missing_packaged_files_invalidate_previous_ready_status(tmp_path):
    run, _ = fixture(tmp_path)
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return fake_release(command, **kwargs) if len(calls) == 1 else 8
    watcher = run.ReleaseWatcher(tmp_path, "campaign.json", "releases", "state/status.json", invoke=invoke)
    assert watcher.scan_once()
    (tmp_path / "releases" / "example" / "model.pt").unlink()
    assert not watcher.scan_once()
    assert len(calls) == 2
    assert watcher.state["runs"]["example"]["status"] == "failed"
