"""Development reporting must preserve pairing, support exclusions and provenance."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_goal_intervention.py"
spec = importlib.util.spec_from_file_location("goal_intervention_reporting_test", SCRIPT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def expectations(root, environment):
    keys = [(f"development-s{i}-d1", 1) for i in range(32)]
    sources = {mode: {"hashes": {"model_sha256": mode, "config_sha256": mode + "config"}, "best_epochs": [4]}
               for mode in ("factorized", "single", "framewise")}
    return {"keys": keys, "tasks": {key: {"seed": i, "dynamics_id": 1} for i, key in enumerate(keys)},
            "data_manifest_sha256": "data", "evaluator_sha256": "evaluator", "sources": sources,
            "hybrid_identity": {"fixed_checkpoint_sha256": "factorized", "donor_checkpoint_sha256": "framewise",
                                "intervention_source_sha256": "source", "evaluator_sha256": "evaluator"}}


def fixture_record(environment, mode):
    expected = expectations(None, environment)
    identity = expected["hybrid_identity"] if mode == "hybrid" else {
        "checkpoint_sha256": mode, "data_manifest_sha256": "data"}
    protocol = {**reporter.contract.PROTOCOL, "run_identity": identity, "evaluator_sha256": "evaluator",
                "search_coordinates": reporter.contract.SEARCH_COORDINATES}
    successes = {"factorized": {0, 1, 2}, "single": {0, 2, 3},
                 "framewise": {0, 2, 3}, "hybrid": {0, 2, 4, 5}}[mode]
    rows = []
    for i, key in enumerate(expected["keys"]):
        rows.append({"trajectory_id": key[0], "observation_id": key[1], "seed": i, "dynamics_id": 1,
                     "success": int(i in successes), "final_success": int(i in successes),
                     "success_during_context": int(i == 0), "policy_eligible": int(i != 0),
                     "native_steps": 10 if i == 0 else 50, "num_replans": 0 if i == 0 else 8,
                     "goal_index": 7, "policy": "world_model", "initial_distance_after_context": 0.5 + i,
                     "initial_block_translation_error_px": 2.0 + i,
                     "initial_block_angle_error_rad": .3 + i,
                     "initial_agent_position_error_px": 10.0 + i,
                     "initial_wrapped_joint_error_rad": .1 + i})
    record = {"status": "complete", "environment": environment, "model_mode": mode,
              "training_seed": 0, "checkpoint_epoch": 4, "checkpoint_sha256": mode,
              "data_manifest_sha256": "data", "evaluator_sha256": "evaluator",
              "planning": {"status": "complete", "split": "development", "policy": "world_model",
                           "protocol": protocol, "planner": deepcopy(reporter.contract.PLANNER_METADATA),
                           "signature": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                           # Deliberately wrong aggregate: the report must derive counts from the rows.
                           "summary": {"success_rate": 1.0, "successes": 32}, "records": rows}}
    if mode == "hybrid":
        record.update(model_mode="factorized_with_framewise_goal", run_identity=identity,
                      fixed_epoch=4, donor_epoch=4)
    return record


def result_path(root, environment, mode):
    if mode == "hybrid":
        return root / "results/development_goal_intervention" / f"{environment}_factorized_s0_framewise_goal_s0" / "planning_development.json"
    return root / "results/development_official_budget" / f"{environment}_{mode}_s0" / "planning_development.json"


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def populate(root, modes=reporter.MODES):
    for environment in ("pusht", "reacher"):
        for mode in modes:
            save(result_path(root, environment, mode), fixture_record(environment, mode))


def test_counts_are_record_derived_support_excluded_and_bootstrap_is_deterministic(tmp_path):
    populate(tmp_path)
    first = reporter.build_report(tmp_path, expectations)
    second = reporter.build_report(tmp_path, expectations)
    assert first == second
    assert first["status"] == "complete"
    hybrid = next(row for row in first["methods"] if row["environment"] == "pusht" and row["mode"] == "hybrid")
    assert hybrid["counts"] == {"raw_successes": 4, "raw_total": 32, "support_successes": 1,
                                "eligible_successes": 3, "eligible_total": 31}
    for row in first["comparisons"]:
        assert row["eligible_total"] == row["clusters"] == 31
        assert row["hybrid_only_successes"] == 2
        assert row["baseline_only_successes"] == 1
        assert row["both_successes"] == 1 and row["neither_successes"] == 27
        assert row["success_difference"] == pytest.approx(1 / 31)
        assert row["ci95"][0] <= 1 / 31 <= row["ci95"][1]
        assert all(pair["seed"] != 0 for pair in row["pairs"])
    assert "exploratory" in first["scope"]
    text = reporter.render_tex(first)
    assert "4/32" in text and "3/31" in text and "+3.2" in text
    assert "post-hoc" in text and "training seeds" in text


def test_missing_and_incomplete_are_pending_not_zero_and_main_test_is_never_read(tmp_path):
    populate(tmp_path, ("factorized", "single", "framewise"))
    incomplete = fixture_record("reacher", "hybrid")
    incomplete["status"] = "interrupted"
    save(result_path(tmp_path, "reacher", "hybrid"), incomplete)
    # A malformed main-test output would fail if the reporter scanned generic results.
    ignored = tmp_path / "results/planning_main/reacher_factorized_s0/planning_test.json"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("This must never be parsed.")
    report = reporter.build_report(tmp_path, expectations)
    assert report["status"] == "pending"
    hybrids = [row for row in report["methods"] if row["mode"] == "hybrid"]
    assert [row["reason"] for row in hybrids] == ["result_missing", "run_incomplete"]
    assert all(row["counts"] is None for row in hybrids)
    assert all(row["success_difference"] is None and row["ci95"] is None for row in report["comparisons"])
    assert r"\textit{pending}" in reporter.render_tex(report)


@pytest.mark.parametrize("mode", ["factorized", "hybrid"])
@pytest.mark.parametrize("change", ["source", "key", "duplicate", "protocol", "support_status", "support_distance", "seed", "epoch", "test_split"])
def test_reject_changed_sources_protocol_keys_and_initial_support(tmp_path, mode, change):
    populate(tmp_path)
    path = result_path(tmp_path, "pusht", mode)
    record = json.loads(path.read_text())
    if change == "source":
        if mode == "hybrid":
            record["run_identity"]["intervention_source_sha256"] = "changed"
        else:
            record["evaluator_sha256"] = "changed"
    elif change == "key":
        record["planning"]["records"][1]["trajectory_id"] = "another_development_trajectory"
    elif change == "duplicate":
        record["planning"]["records"][1] = deepcopy(record["planning"]["records"][0])
    elif change == "protocol":
        # A self-consistent signature must not permit another planning budget.
        record["planning"]["protocol"]["samples"] = 30
        record["planning"]["signature"] = hashlib.sha256(json.dumps(record["planning"]["protocol"], sort_keys=True).encode()).hexdigest()
    elif change == "support_status":
        row = record["planning"]["records"][0]
        row.update(success_during_context=0, policy_eligible=1)
    elif change == "support_distance":
        record["planning"]["records"][0]["initial_distance_after_context"] += 1
    elif change == "seed":
        record["planning"]["records"][0]["seed"] = 9999
    elif change == "epoch":
        record["fixed_epoch" if mode == "hybrid" else "checkpoint_epoch"] = 1
    else:
        record["planning"]["split"] = "test"
    save(path, record)
    with pytest.raises(ValueError):
        reporter.build_report(tmp_path, expectations)


def test_row_order_does_not_change_the_paired_estimator(tmp_path):
    populate(tmp_path)
    report = reporter.build_report(tmp_path, expectations)
    for environment in ("pusht", "reacher"):
        path = result_path(tmp_path, environment, "hybrid")
        record = json.loads(path.read_text())
        record["planning"]["records"].reverse()
        save(path, record)
    assert reporter.build_report(tmp_path, expectations)["comparisons"] == report["comparisons"]


@pytest.mark.parametrize("mode", ["factorized", "hybrid"])
def test_incomplete_file_with_changed_sources_is_rejected_not_hidden_as_pending(tmp_path, mode):
    populate(tmp_path)
    record = fixture_record("pusht", mode)
    record["status"] = "interrupted"
    if mode == "hybrid":
        record["run_identity"]["donor_checkpoint_sha256"] = "other_donor"
    else:
        record["checkpoint_sha256"] = "other_model"
    save(result_path(tmp_path, "pusht", mode), record)
    with pytest.raises(ValueError, match="[Cc]hanged"):
        reporter.build_report(tmp_path, expectations)


def interrupted_record(environment, mode):
    record = fixture_record(environment, mode)
    record["status"] = "interrupted"
    # Match evaluate_planning's actual early return, which omits planner/summary.
    planning = record["planning"]
    record["planning"] = {key: planning[key] for key in ("split", "policy", "protocol", "signature")}
    record["planning"].update(status="interrupted", kind="closed_loop_goal_image_planning", records=planning["records"][:7])
    return record


@pytest.mark.parametrize("mode", ["factorized", "hybrid"])
def test_real_interrupted_shape_without_planner_summary_stays_pending(tmp_path, mode):
    populate(tmp_path)
    save(result_path(tmp_path, "pusht", mode), interrupted_record("pusht", mode))
    report = reporter.build_report(tmp_path, expectations)
    row = next(row for row in report["methods"] if row["environment"] == "pusht" and row["mode"] == mode)
    assert row["status"] == "pending" and row["reason"] == "run_incomplete" and row["counts"] is None
    assert report["status"] == "pending"
    paired = next(row for row in report["comparisons"] if row["environment"] == "pusht" and row["comparison"] == "hybrid_minus_factorized")
    assert paired["status"] == "pending" and paired["success_difference"] is None and paired["ci95"] is None


@pytest.mark.parametrize("mode", ["factorized", "hybrid"])
@pytest.mark.parametrize("metadata", [None, {}, {"samples": 2}])
def test_partial_output_with_malformed_present_planner_summary_is_rejected(tmp_path, mode, metadata):
    populate(tmp_path)
    record = interrupted_record("pusht", mode)
    record["planning"]["planner"] = metadata
    save(result_path(tmp_path, "pusht", mode), record)
    with pytest.raises(ValueError, match="planner metadata"):
        reporter.build_report(tmp_path, expectations)


@pytest.mark.parametrize("mode", ["factorized", "hybrid"])
@pytest.mark.parametrize("change", ["source", "budget", "signature"])
def test_real_partial_outputs_still_enforce_sources_and_hashed_protocol(tmp_path, mode, change):
    populate(tmp_path)
    record = interrupted_record("pusht", mode)
    if change == "source":
        if mode == "hybrid":
            record["run_identity"]["donor_checkpoint_sha256"] = "wrong_donor"
        else:
            record["checkpoint_sha256"] = "wrong_model"
    elif change == "budget":
        record["planning"]["protocol"]["samples"] = 2
        record["planning"]["signature"] = hashlib.sha256(json.dumps(record["planning"]["protocol"], sort_keys=True).encode()).hexdigest()
    else:
        record["planning"]["signature"] = "invalid"
    save(result_path(tmp_path, "pusht", mode), record)
    with pytest.raises(ValueError, match="[Cc]hanged"):
        reporter.build_report(tmp_path, expectations)


@pytest.mark.parametrize("mode", ["factorized", "hybrid"])
def test_complete_output_still_requires_full_planner_metadata(tmp_path, mode):
    populate(tmp_path)
    record = fixture_record("pusht", mode)
    record["planning"].pop("planner")
    save(result_path(tmp_path, "pusht", mode), record)
    with pytest.raises(ValueError, match="planner metadata"):
        reporter.build_report(tmp_path, expectations)
