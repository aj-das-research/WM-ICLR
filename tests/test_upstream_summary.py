"""Protect the upstream result collector from false completion and unsafe text."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_upstream_planning as report


CONFIG = {"seed": 42, "solver": {"seed": 42, "num_samples": 300, "n_steps": 30, "topk": 30},
          "eval": {"num_eval": 50, "goal_offset_steps": 25, "eval_budget": 50, "dataset_name": "pusht_expert_train"},
          "world": {"num_envs": 50}, "plan_config": {"horizon": 5, "receding_horizon": 5, "action_block": 5},
          "output": {"filename": "pusht_results.txt"}}


def output(success_count=7, count=50, config=None):
    flags = [i < success_count for i in range(count)]
    array = "array([" + ",\n".join(map(str, flags)) + "])"
    metrics = "{'success_rate': " + str(100*success_count/50) + ", 'episode_successes': " + array + ", 'seeds': None}"
    return "\n==== CONFIG ====\n" + yaml.safe_dump(config or CONFIG) + "\n==== RESULTS ====\nmetrics: " + metrics + "\nevaluation_time: 123.5 seconds\n"


def test_multiline_numpy_repr_preserves_counts_and_percent():
    parsed = report.parse_output(output())
    assert parsed["success_count"] == 7
    assert parsed["success_rate_percent"] == 14
    assert parsed["evaluation_seconds_including_video"] == 123.5
    assert len(parsed["successes"]) == 50


def test_completed_zero_success_is_a_valid_measured_zero():
    assert report.parse_output(output(0))["success_count"] == 0


@pytest.mark.parametrize("count", [49, 51])
def test_wrong_episode_denominators_rejected(count):
    with pytest.raises(report.InvalidOutput, match="50 Boolean"):
        report.parse_output(output(count=count))


def test_integers_are_not_boolean_outcomes():
    text = output().replace("True", "1").replace("False", "0")
    with pytest.raises(report.InvalidOutput, match="50 Boolean"):
        report.parse_output(text)


def test_inconsistent_rate_rejected():
    with pytest.raises(report.InvalidOutput, match="percentage disagrees"):
        report.parse_output(output().replace("'success_rate': 14.0", "'success_rate': 12.0"))


def test_truncated_footer_is_incomplete():
    with pytest.raises(report.IncompleteOutput):
        report.parse_output(output().split("evaluation_time:")[0])


def test_multiple_or_appended_partial_blocks_are_ambiguous():
    for text in (output()+output(), output()+"\n==== CONFIG ====\nseed: 42"):
        with pytest.raises(report.InvalidOutput, match="Multiple"):
            report.parse_output(text)


def test_arbitrary_python_calls_never_execute(tmp_path):
    sentinel = tmp_path / "must_not_exist"
    payload = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    with pytest.raises(report.InvalidOutput):
        report.parse_output(output().replace("'seeds': None", "'seeds': " + payload))
    assert not sentinel.exists()


def test_unsafe_yaml_and_duplicate_keys_rejected():
    for text in (output().replace("seed: 42", "seed: 42\nseed: 17"),
                 output().replace("seed: 42", "seed: !!python/object:builtins.object {}")):
        with pytest.raises(report.InvalidOutput): report.parse_output(text)


def test_only_official_simple_interpolations_resolve():
    config = copy.deepcopy(CONFIG)
    config["world"]["num_envs"] = "${eval.num_eval}"
    config["solver"]["seed"] = "${seed}"
    parsed = report.parse_output(output(config=config))
    report.validate_config(parsed["config"], CONFIG, "pusht")
    config["seed"] = "${oc.env:HOME}"
    with pytest.raises(report.InvalidOutput): report.parse_output(output(config=config))


def test_full_config_mismatch_rejected():
    actual = copy.deepcopy(CONFIG)
    actual["plan_config"]["receding_horizon"] = 1
    with pytest.raises(report.InvalidOutput, match="fixed official protocol"):
        report.validate_config(actual, CONFIG, "pusht")


def test_changed_preflight_cannot_redefine_fixed_protocol():
    actual = copy.deepcopy(CONFIG)
    actual["eval"]["num_eval"] = 32
    with pytest.raises(report.InvalidOutput, match="fixed official protocol"):
        report.validate_config(actual, actual, "pusht")


def prepared_run(tmp_path, status="complete"):
    job = {"job_id": 123, "environment": "pusht", "log": "logs/upstream-123.out"}
    tasks = [{"row": i*11, "episode": i, "start_step": 2, "goal_step": 27} for i in range(50)]
    preflight = {"environments": {"pusht": {"config": CONFIG, "official_data": {"selected_tasks": tasks}}}}
    now = dt.datetime.now(dt.timezone.utc)
    record = {"job_id": "123", "command": ["bash", "scripts/run_upstream_planning_reproduction.sh", "pusht"],
              "status": status, "exit_code": 0 if status == "complete" else 1,
              "start_utc": (now-dt.timedelta(seconds=10)).isoformat(), "end_utc": (now+dt.timedelta(seconds=10)).isoformat()}
    receipt = tmp_path / "runs/jobs/123/record.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(record))
    result = tmp_path / "artifacts/upstream_planning_reproduction/cache/pusht/pusht_results.txt"
    result.parent.mkdir(parents=True)
    result.write_text(output())
    log = tmp_path / job["log"]
    log.parent.mkdir(parents=True)
    log.write_text("logging before\n[" + " ".join(str(t["row"]) for t in tasks) + "]\nlogging after\n")
    return job, preflight, result, receipt


def collect_fixture(tmp_path, job, preflight, state="COMPLETED"):
    return report.summarize_job(tmp_path, job, preflight, report.FileDigestCache(), lambda _: {"JobState": state})


def test_complete_receipt_and_exact_logged_tasks_bind_outcomes(tmp_path):
    job, preflight, _, _ = prepared_run(tmp_path)
    result = collect_fixture(tmp_path, job, preflight)
    assert result["status"] == "complete" and result["success_count"] == 7
    assert len(result["task_outcomes"]) == 50 and result["task_outcomes"][0]["success"] is True


@pytest.mark.parametrize("state", ["PENDING", "FAILED", "UNKNOWN"])
def test_queued_failed_or_unknown_without_receipt_never_becomes_zero(tmp_path, state):
    job = {"job_id": 123, "environment": "pusht", "log": "unused"}
    result = collect_fixture(tmp_path, job, {}, state)
    assert result["status"] != "complete"
    assert result["success_count"] is None and result["success_rate_percent"] is None


def test_failed_job_cannot_publish_even_complete_result_file(tmp_path):
    job, preflight, _, _ = prepared_run(tmp_path, "failed")
    result = collect_fixture(tmp_path, job, preflight, "FAILED")
    assert result["status"] == "failed" and result["success_count"] is None


def test_unrelated_or_stale_results_rejected(tmp_path):
    job, preflight, result_path, _ = prepared_run(tmp_path)
    os.utime(result_path, (1,1))
    result = collect_fixture(tmp_path, job, preflight)
    assert result["status"] == "invalid_output" and result["success_count"] is None


def test_wrong_row_selection_rejected(tmp_path):
    job, preflight, _, _ = prepared_run(tmp_path)
    (tmp_path / job["log"]).write_text("[1 2 3]\n")
    result = collect_fixture(tmp_path, job, preflight)
    assert result["status"] == "invalid_output" and result["success_count"] is None


def test_config_only_job_cannot_qualify_as_full_run(tmp_path):
    job, preflight, _, receipt = prepared_run(tmp_path)
    record = json.loads(receipt.read_text())
    record["command"].append("--config-only")
    receipt.write_text(json.dumps(record))
    result = collect_fixture(tmp_path, job, preflight)
    assert result["status"] == "invalid_output" and result["success_count"] is None
