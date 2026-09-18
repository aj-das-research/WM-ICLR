"""Campaign reuse must avoid expensive rereads without masking identity changes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load_campaign_module():
    path = Path(__file__).parents[1] / "scripts" / "run_evaluation_campaign.py"
    spec = importlib.util.spec_from_file_location("evaluation_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_digest_cache_avoids_rereads_and_invalidates_equal_size_changes(tmp_path, monkeypatch):
    cache = load_campaign_module().FileDigestCache()
    path = tmp_path / "weights.pt"
    path.write_bytes(b"initial weights")
    first = cache.sha256(path)
    original_open = Path.open
    reads = []

    def record_open(opened_path, *args, **kwargs):
        if args and args[0] == "rb":
            reads.append(opened_path)
        return original_open(opened_path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    assert cache.sha256(path) == first
    assert cache.sha256(tmp_path / "." / "weights.pt") == first
    assert reads == []
    old_stat = path.stat()
    path.write_bytes(b"updated weights")  # Equal length must still invalidate.
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))
    second = cache.sha256(path)
    assert second == hashlib.sha256(b"updated weights").hexdigest()
    assert first != second
    assert len(reads) == 1
    assert cache.sha256(path) == second
    assert len(reads) == 1
    path.write_bytes(b"a different-sized checkpoint")
    assert cache.sha256(path) == hashlib.sha256(b"a different-sized checkpoint").hexdigest()
    assert len(reads) == 2


def campaign_fixture(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"pretrained_dir": "pretrained/pusht", "output_dir": "runs/pusht_factorized_s0",
                                  "data_root": "data/pusht", "dataset_kwargs": {"feature_cache": "features/pusht"}}))
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"tasks": [{"id": "pusht_factorized_s0", "config": str(config)}]}))
    return campaign


def planning_fixture(module, policy="world_model", split="test"):
    budget = module.OFFICIAL_PLANNER_BUDGET if policy == "world_model" else module.CONTROL_METADATA_BUDGET
    task = {"kind": "planning", "policy": policy, "split": split, "planner_budget": budget.copy()}
    result = {"checkpoint_sha256": "weights", "data_manifest_sha256": "data", "evaluator_sha256": "evaluator"}
    protocol = {key: budget[key] for key in ("samples", "iterations", "elites", "horizon", "native_budget")}
    protocol.update(episodes_per_dynamics=64, policy=policy, split=split, seed=1701, goal_offset=5,
                    evaluator_sha256="evaluator", run_identity={key: result[key] for key in ("checkpoint_sha256", "data_manifest_sha256")},
                    search_coordinates=module.PLANNER_IDENTITY["search_coordinates"])
    result["planning"] = {"policy": policy, "split": split, "protocol": protocol,
                          "planner": {**budget, **module.PLANNER_IDENTITY},
                          "signature": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}
    return task, result


def test_main_commands_lock_official_sampling_controls_keep_unused_defaults(tmp_path):
    module = load_campaign_module()
    tasks = module.evaluation_tasks(campaign_fixture(tmp_path))
    assert len(tasks) == 12  # two checkpoints x two forecasts/planning + four controls
    for task in tasks:
        command = module.evaluation_command(task, 64, 3000)
        if task["kind"] == "planning" and task["policy"] == "world_model":
            for flag, value in (("--samples", "300"), ("--iterations", "30"), ("--elites", "30"),
                                ("--horizon", "5"), ("--native-budget", "50")):
                assert command[command.index(flag) + 1] == value
            assert task["uses_model_planner"] is True
        else:
            assert "--samples" not in command and "--iterations" not in command and "--elites" not in command
            if task["kind"] == "planning":
                assert task["planner_budget"]["samples"] == 128
                assert task["uses_model_planner"] is False


@pytest.mark.parametrize("key,value", [("samples", 128), ("iterations", 5), ("elites", 16), ("horizon", 4), ("native_budget", 60)])
def test_completed_main_budget_cannot_be_reused_with_changed_search(key, value):
    module = load_campaign_module()
    task, result = planning_fixture(module)
    module.validate_completed_planning(result, task, 64)
    result["planning"]["planner"][key] = value
    with pytest.raises(RuntimeError, match="planning budget differs"):
        module.validate_completed_planning(result, task, 64)


def test_summary_budget_relabeling_and_inconsistent_signatures_are_rejected():
    module = load_campaign_module()
    task, result = planning_fixture(module)
    result["planning"]["protocol"]["samples"] = 128
    with pytest.raises(RuntimeError, match="hashed protocol budget"):
        module.validate_completed_planning(result, task, 64)
    result["planning"]["protocol"]["samples"] = 300
    result["planning"]["signature"] = "wrong"
    with pytest.raises(RuntimeError, match="signature is inconsistent"):
        module.validate_completed_planning(result, task, 64)


@pytest.mark.parametrize("policy", ["random", "replay_oracle"])
def test_completed_controls_ignore_unused_cem_sampling_fields(policy):
    module = load_campaign_module()
    task, result = planning_fixture(module, policy=policy)
    task["planner_budget"] = module.OFFICIAL_PLANNER_BUDGET.copy()
    module.validate_completed_planning(result, task, 64)
    result["planning"]["planner"]["native_budget"] = 60
    with pytest.raises(RuntimeError, match="native_budget"):
        module.validate_completed_planning(result, task, 64)


def test_slurm_membership_does_not_imply_dedicated_timing(monkeypatch):
    module = load_campaign_module()
    task, _ = planning_fixture(module)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    context = module.execution_context(task, "unspecified", {}, "runner-sha")
    assert context["job_id"] == "123"
    assert context["main_efficiency_claim_eligible"] is False
    assert context["shared_gpu_with_training"] is None


def test_only_explicitly_dedicated_entire_episode_history_is_eligible_for_timing():
    module = load_campaign_module()
    task, _ = planning_fixture(module)
    first = module.execution_context(task, "dedicated_gpu_campaign", {}, "runner-sha")
    assert first["main_efficiency_claim_eligible"] is True
    assert first["shared_gpu_with_training"] is False
    resumed = module.execution_context(task, "dedicated_gpu_campaign", {"records": [{}, {}], "execution_context": first}, "runner-sha")
    assert resumed["main_efficiency_claim_eligible"] is True
    assert resumed["resumed_episode_count"] == 2
    unknown_history = module.execution_context(task, "dedicated_gpu_campaign", {"records": [{}]}, "runner-sha")
    assert unknown_history["main_efficiency_claim_eligible"] is False
    shared_history = module.execution_context(task, "dedicated_gpu_campaign", {"records": [{}], "execution_context": {
        "main_efficiency_claim_eligible": False, "shared_gpu_with_training": True}}, "runner-sha")
    assert shared_history["shared_gpu_with_training"] is True
    assert shared_history["main_efficiency_claim_eligible"] is False
    task["split"] = "development"
    assert module.execution_context(task, "dedicated_gpu_campaign", {}, "runner-sha")["main_efficiency_claim_eligible"] is False


def test_retry_failed_attempts_each_task_once_per_worker_invocation(tmp_path, monkeypatch):
    module = load_campaign_module()
    task = module.evaluation_tasks(campaign_fixture(tmp_path))[0]
    task["output"] = str(tmp_path / "results" / "forecast_test.json")
    state = tmp_path / "state"
    module.write_json(state / (task["task_id"] + ".json"), {"status": "failed"})
    monkeypatch.setattr(module, "evaluation_tasks", lambda *args, **kwargs: [task])
    monkeypatch.setattr(module, "ready", lambda task: True)
    monkeypatch.setattr(module.FileDigestCache, "sha256", lambda *args: "fixture-sha")
    calls = []
    def fails(command, **kwargs):
        calls.append(command)
        assert len(calls) == 1, "Persistent task failure was retried within the same worker"
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(module.subprocess, "run", fails)
    monkeypatch.setattr(sys, "argv", ["campaign", "--state", str(state), "--retry-failed", "--max-seconds", "1000"])
    assert module.main() == 1
    assert len(calls) == 1
    assert json.loads((state / (task["task_id"] + ".json")).read_text())["status"] == "failed"


@pytest.mark.parametrize("flag,selection,count", [("--forecasts-only", "forecasts_only", 64),
    ("--frozen-controls-only", "frozen_controls_only", 12), (None, "all", 136)])
def test_worker_filters_preserve_the_full_grid_without_evaluating(tmp_path, monkeypatch, flag, selection, count):
    module = load_campaign_module()
    training = []
    for environment in ("pusht", "reacher"):
        for mode in ("plain", "single", "factorized", "framewise", "factorized_unpaired"):
            for seed in range(3):
                run = f"{environment}_{mode}_s{seed}"
                config = tmp_path / f"{run}.json"
                config.write_text(json.dumps({"pretrained_dir": f"pretrained/{environment}",
                    "output_dir": f"runs/{run}", "data_root": f"data/{environment}",
                    "dataset_kwargs": {"feature_cache": f"features/{environment}"}}))
                training.append({"id": run, "config": str(config)})
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"tasks": training}))
    state = tmp_path / "state"
    args = ["campaign", "--campaign", str(campaign), "--state", str(state), "--max-seconds", "0"]
    if flag:
        args.append(flag)
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(module.FileDigestCache, "sha256", lambda *args: "fixture-sha")
    def cannot_evaluate(*args, **kwargs):
        pytest.fail("Grid-selection test must never start an evaluation")
    monkeypatch.setattr(module.subprocess, "run", cannot_evaluate)
    assert module.main() == 0
    grid = json.loads((state / "task_grid.json").read_text())
    assert len(grid["tasks"]) == 136
    assert grid["worker_selection"] == selection
    assert len(grid["selected_task_ids"]) == count
    selected = [task for task in grid["tasks"] if task["task_id"] in grid["selected_task_ids"]]
    if flag == "--forecasts-only":
        assert {task["kind"] for task in selected} == {"forecast"}
        assert len({task["id"] for task in selected}) == 32
        assert {task["split"] for task in selected} == {"test", "extrapolation"}
    assert list(state.glob("*.json")) == [state / "task_grid.json"]


def test_worker_filters_are_mutually_exclusive(monkeypatch):
    module = load_campaign_module()
    monkeypatch.setattr(sys, "argv", ["campaign", "--forecasts-only", "--frozen-controls-only"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def completed_output_fixture(module, tmp_path, kind="planning"):
    task, result = planning_fixture(module)
    result.update(status="complete", environment="pusht")
    task["environment"] = "pusht"
    if kind == "forecast":
        task["kind"] = "forecast"
        task.pop("planner_budget")
        result.pop("planning")
        result["forecast"] = {"kind": "fixed_reference_forecasting", "split": "test",
                              "records": [{"large_record": "no-cache" * 1000}]}
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(result))
    return task, result, path


@pytest.mark.parametrize("kind", ["forecast", "planning"])
def test_complete_validation_cache_parses_once_and_keeps_no_raw_records(tmp_path, monkeypatch, kind):
    module = load_campaign_module()
    task, _, path = completed_output_fixture(module, tmp_path, kind)
    cache = module.CompletedEvaluationCache()
    original_loads = module.json.loads
    parsed = []

    def count_parse(raw, *args, **kwargs):
        parsed.append(len(raw))
        return original_loads(raw, *args, **kwargs)

    monkeypatch.setattr(module.json, "loads", count_parse)
    for _ in range(3):
        assert cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    assert cache.is_complete(path.parent / "." / path.name, task, 64, "weights", "data", "evaluator")
    assert len(parsed) == 1
    assert "no-cache" not in repr(cache._entries)
    assert len(repr(cache._entries)) < 2000


def test_complete_validation_cache_revalidates_equal_size_mutation(tmp_path, monkeypatch):
    module = load_campaign_module()
    task, result, path = completed_output_fixture(module, tmp_path)
    cache = module.CompletedEvaluationCache()
    assert cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    before = path.stat()
    result["checkpoint_sha256"] = "invalid"  # Same byte length as "weights".
    path.write_text(json.dumps(result))
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    with pytest.raises(RuntimeError, match="Immutable completed evaluation changed"):
        cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    assert not cache._entries


def test_complete_validation_cache_rejects_atomic_replacement_with_old_mtime(tmp_path):
    module = load_campaign_module()
    task, result, path = completed_output_fixture(module, tmp_path)
    cache = module.CompletedEvaluationCache()
    assert cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    before = path.stat()
    result["checkpoint_sha256"] = "invalid"
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(result))
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_ino != before.st_ino
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    with pytest.raises(RuntimeError, match="Immutable completed evaluation changed"):
        cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    assert not cache._entries


@pytest.mark.parametrize("changed", ["checkpoint", "data", "evaluator", "split", "policy", "kind",
                                    "episodes", "samples", "planner_identity", "environment"])
def test_cached_completion_does_not_validate_a_different_scientific_request(tmp_path, changed):
    module = load_campaign_module()
    task, _, path = completed_output_fixture(module, tmp_path)
    cache = module.CompletedEvaluationCache()
    assert cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    checkpoint, data, evaluator, episodes = "weights", "data", "evaluator", 64
    if changed == "checkpoint":
        checkpoint = "different-weights"
    elif changed == "data":
        data = "different-data"
    elif changed == "evaluator":
        evaluator = "different-evaluator"
    elif changed == "episodes":
        episodes = 32
    elif changed == "samples":
        task["planner_budget"]["samples"] = 128
    elif changed == "planner_identity":
        module.PLANNER_IDENTITY["search_coordinates"] = "different-coordinates"
    else:
        task[changed] = {"split": "extrapolation", "policy": "random", "kind": "forecast",
                         "environment": "reacher"}[changed]
    with pytest.raises(RuntimeError):
        cache.is_complete(path, task, episodes, checkpoint, data, evaluator)
    assert not cache._entries


@pytest.mark.parametrize("changed", ["split", "policy"])
def test_forecast_cache_checks_requested_split_and_policy(tmp_path, changed):
    module = load_campaign_module()
    task, _, path = completed_output_fixture(module, tmp_path, "forecast")
    cache = module.CompletedEvaluationCache()
    assert cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    task[changed] = "extrapolation" if changed == "split" else "random"
    with pytest.raises(RuntimeError, match="forecast policy/split differs"):
        cache.is_complete(path, task, 64, "weights", "data", "evaluator")


def test_file_changed_during_read_is_never_cached(tmp_path, monkeypatch):
    module = load_campaign_module()
    task, _, path = completed_output_fixture(module, tmp_path)
    cache = module.CompletedEvaluationCache()
    original_read = Path.read_text

    def mutate_after_read(opened, *args, **kwargs):
        raw = original_read(opened, *args, **kwargs)
        if opened == path:
            opened.write_text(raw + " ")
        return raw

    monkeypatch.setattr(Path, "read_text", mutate_after_read)
    with pytest.raises(RuntimeError, match="File changed while validating"):
        cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    assert not cache._entries
    monkeypatch.setattr(Path, "read_text", original_read)
    assert cache.is_complete(path, task, 64, "weights", "data", "evaluator")


def test_partial_and_malformed_results_are_not_cached(tmp_path):
    module = load_campaign_module()
    task, result, path = completed_output_fixture(module, tmp_path)
    cache = module.CompletedEvaluationCache()
    result["status"] = "interrupted"
    path.write_text(json.dumps(result))
    assert cache.is_complete(path, task, 64, "weights", "data", "evaluator") is False
    assert not cache._entries
    path.write_text('{"status": "complete",')
    with pytest.raises(json.JSONDecodeError):
        cache.is_complete(path, task, 64, "weights", "data", "evaluator")
    assert not cache._entries
    result["status"] = "complete"
    path.write_text(json.dumps(result))
    assert cache.is_complete(path, task, 64, "weights", "data", "evaluator")


def test_completion_reuse_preserves_provenance_and_does_not_rewrite_unchanged_state(tmp_path, monkeypatch):
    module = load_campaign_module()
    state_path = tmp_path / "task.json"
    output = tmp_path / "complete.json"
    original = {"status": "complete", "output": str(output), "checkpoint_sha256": "weights",
                "job_id": "199428", "node": "gpu-01", "start_unix": 100.0, "end_unix": 200.0,
                "returncode": 0, "log": "original.log", "attempt_wall_seconds": 100.0,
                "execution_context": {"shared_gpu_with_training": False}}
    module.write_json(state_path, original)
    module.mark_completed_reused(state_path, original, output, "weights")
    reused = json.loads(state_path.read_text())
    assert reused == {**original, "reused": True}
    previous_stat = state_path.stat()

    def no_write(*args, **kwargs):
        pytest.fail("Unchanged complete state was rewritten")

    monkeypatch.setattr(module, "write_json", no_write)
    module.mark_completed_reused(state_path, reused, output, "weights")
    assert state_path.stat().st_mtime_ns == previous_stat.st_mtime_ns
    assert original.get("reused") is None  # Caller state was not mutated.


def test_worker_repeated_claim_scans_reuse_cache_and_preserve_completion(tmp_path, monkeypatch):
    module = load_campaign_module()
    task, _, output = completed_output_fixture(module, tmp_path, "forecast")
    task.update(task_id="done_forecast", output=str(output), checkpoint="unused/checkpoint",
                config={"data_root": "unused/data"}, training_summary=None)
    unavailable = {**task, "task_id": "not_ready"}
    state_dir = tmp_path / "state"
    state_path = state_dir / "done_forecast.json"
    original_state = {"status": "complete", "job_id": "original-job", "start_unix": 10, "end_unix": 20}
    module.write_json(state_path, original_state)
    monkeypatch.setattr(module, "evaluation_tasks", lambda *args, **kwargs: [task, unavailable])
    monkeypatch.setattr(module, "ready", lambda selected: selected is task)

    def digest(self, path):
        return "weights" if str(path).endswith("model.pt") else "data" if str(path).endswith("manifest.json") else "evaluator"

    monkeypatch.setattr(module.FileDigestCache, "sha256", digest)
    original_read = Path.read_text
    output_reads = []

    def count_read(opened, *args, **kwargs):
        if opened == output:
            output_reads.append(opened)
        return original_read(opened, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_read)
    clock = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 100))
    monkeypatch.setattr(sys, "argv", ["campaign", "--state", str(state_dir), "--max-seconds", "500"])
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("No evaluation should launch"))
    assert module.main() == 0
    assert len(output_reads) == 1  # Four claim scans while another task is not ready.
    reused = json.loads(state_path.read_text())
    for key, value in original_state.items():
        assert reused[key] == value
    assert reused["reused"] is True
