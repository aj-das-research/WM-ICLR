"""Lease recovery with fake run files; no model training is performed."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts" / "run_training_campaign.py"
    spec = importlib.util.spec_from_file_location("training_campaign", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_files(tmp_path, state):
    run = module()
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    states = tmp_path / "states"
    states.mkdir()
    run.write_json(states / "example.json", state)
    task = {"id": "example", "config": "unused.json"}
    cfg = {"action_stats": str(stats), "dataset_kwargs": {"feature_cache": str(cache)}}
    return run, task, cfg, states


def test_legacy_active_slurm_worker_without_task_lock_is_not_reclaimed(tmp_path):
    state = {"status": "running", "job_id": "199379", "node": "legacy-node"}
    run, task, cfg, states = fixture_files(tmp_path, state)
    handle, reason = run.claim_task(task, cfg, states, lambda job: job == "199379")
    assert handle is None and reason == "active_owner"
    assert json.loads((states / "example.json").read_text()) == state


def test_confirmed_dead_slurm_owner_is_reclaimed_and_locked(tmp_path):
    state = {"status": "running", "job_id": "finished-job", "node": "old-node"}
    run, task, cfg, states = fixture_files(tmp_path, state)
    handle, reason = run.claim_task(task, cfg, states, lambda job: False)
    assert handle is not None and reason == "claimed"
    recovered = json.loads((states / "example.json").read_text())
    assert recovered["lease_version"] == 2 and recovered["recovery_count"] == 1
    assert recovered["recovered_owner"]["job_id"] == "finished-job"
    second, reason = run.claim_task(task, cfg, states, lambda job: False)
    assert second is None and reason == "busy"
    handle.close()


def test_scheduler_failure_conservatively_protects_legacy_owner(tmp_path):
    state = {"status": "running", "job_id": "199380", "node": "old-node"}
    run, task, cfg, states = fixture_files(tmp_path, state)
    handle, reason = run.claim_task(task, cfg, states, lambda job: None)
    assert handle is None and reason == "active_owner"


def test_local_pid_is_checked_and_dead_local_owner_can_be_recovered():
    run = module()
    state = {"status": "running", "node": "local", "pid": 12345}
    assert run.owner_is_active(state, lambda job: False, node="local", process_activity=lambda pid: True)
    assert not run.owner_is_active(state, lambda job: False, node="local", process_activity=lambda pid: False)


def test_complete_and_failed_states_preserve_completion_rules(tmp_path):
    run, task, cfg, states = fixture_files(tmp_path, {"status": "complete"})
    handle, reason = run.claim_task(task, cfg, states, lambda job: False)
    assert handle is None and reason == "complete"
    run.write_json(states / "example.json", {"status": "failed"})
    handle, reason = run.claim_task(task, cfg, states, lambda job: False)
    assert handle is None and reason == "failed"


def test_child_inherits_lease_after_parent_handle_closes(tmp_path):
    run, task, cfg, states = fixture_files(tmp_path, {"status": "pending"})
    handle, reason = run.claim_task(task, cfg, states, lambda job: False)
    assert reason == "claimed"
    child = subprocess.Popen([sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
                             pass_fds=(handle.fileno(),), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        handle.close()
        with (states / "example.lock").open("a+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child.stdin.write("finish\n")
            child.stdin.flush()
            assert child.wait(timeout=5) == 0
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_cached_scheduler_absence_requires_fresh_confirmation(monkeypatch):
    run = module()
    activity = run.SlurmActivity()
    activity.active = {"old-job"}
    activity.updated = run.time.monotonic()
    queries = []
    def query(*args, **kwargs):
        queries.append(args)
        return SimpleNamespace(returncode=0, stdout="old-job\nnew-job\n")
    monkeypatch.setattr(run.subprocess, "run", query)
    assert activity("old-job") is True
    assert queries == []
    assert activity("new-job") is True
    assert len(queries) == 1


def campaign_fixture(tmp_path, monkeypatch, *, state=None, exit_when_not_ready=False):
    run, task, cfg, states = fixture_files(tmp_path, state or {"status": "pending"})
    cfg["output_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    task["config"] = str(config_path)
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"tasks": [task]}))
    args = ["training_campaign", "--campaign", str(campaign), "--state", str(states), "--max-seconds", "1000"]
    if exit_when_not_ready:
        args.append("--exit-when-not-ready")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(run, "SlurmActivity", lambda: lambda job: True)
    return run, task, cfg, states, campaign


@pytest.mark.parametrize("reason", ["busy", "active_owner", "not_ready"])
def test_exit_when_not_ready_releases_idle_worker_without_waiting(tmp_path, monkeypatch, reason):
    state = {"status": "running", "job_id": "other-worker", "node": "elsewhere"} if reason == "active_owner" else None
    run, task, cfg, states, _ = campaign_fixture(tmp_path, monkeypatch, state=state, exit_when_not_ready=True)
    held = None
    if reason == "busy":
        held = (states / "example.lock").open("a+")
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif reason == "not_ready":
        Path(cfg["action_stats"]).unlink()
    original_state = (states / "example.json").read_bytes()
    monkeypatch.setattr(run.time, "sleep", lambda seconds: pytest.fail("Idle worker should exit without sleeping"))
    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: pytest.fail("No task is ready to train"))
    try:
        assert run.main() == 0
        assert (states / "example.json").read_bytes() == original_state
    finally:
        if held:
            held.close()


def test_exit_flag_still_processes_ready_work_after_a_busy_task(tmp_path, monkeypatch):
    run, task, cfg, states, campaign = campaign_fixture(tmp_path, monkeypatch, exit_when_not_ready=True)
    campaign.write_text(json.dumps({"tasks": [{"id": "busy", "config": task["config"]}, task]}))
    output = Path(cfg["output_dir"])
    (output / "last").mkdir(parents=True)
    (output / "last/training_state.pt").write_bytes(b"fixture resume marker")
    calls = []
    def train(command, **kwargs):
        runtime = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert runtime["resume"] == str(output / "last")
        assert kwargs["pass_fds"]  # The actual task lease still reaches its child.
        calls.append(command)
        (output / "training_summary.json").write_text(json.dumps({"status": "completed"}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run.subprocess, "run", train)
    monkeypatch.setattr(run.time, "sleep", lambda seconds: pytest.fail("Ready work must be handled, then the idle worker exits"))
    with (states / "busy.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run.main() == 0
    assert len(calls) == 1
    assert json.loads((states / "example.json").read_text())["status"] == "complete"


def test_default_waits_and_processes_a_task_after_its_other_owner_releases_it(tmp_path, monkeypatch):
    run, task, cfg, states, _ = campaign_fixture(tmp_path, monkeypatch)
    held = (states / "example.lock").open("a+")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sleeps, calls = [], []
    def release_on_wait(seconds):
        sleeps.append(seconds)
        assert len(sleeps) == 1
        held.close()
    def train(command, **kwargs):
        calls.append(command)
        (Path(cfg["output_dir"]) / "training_summary.json").write_text(json.dumps({"status": "completed"}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run.time, "sleep", release_on_wait)
    monkeypatch.setattr(run.subprocess, "run", train)
    try:
        assert run.main() == 0
    finally:
        held.close()
    assert sleeps == [15] and len(calls) == 1
    assert json.loads((states / "example.json").read_text())["status"] == "complete"
