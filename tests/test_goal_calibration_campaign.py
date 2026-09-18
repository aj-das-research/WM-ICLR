"""Completed diagnostic reuse must preserve exact development provenance."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def worker_module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("calibration_campaign_test", scripts / "run_goal_calibration_campaign.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def fixture(worker, tmp_path):
    identity = {key: key + "-fixture" for key in ("checkpoint_sha256", "config_sha256", "data_manifest_sha256",
        "evaluator_sha256", "model_source_sha256", "generation_source_sha256", "diagnostic_source_sha256")}
    diagnostic_rows, planning_rows = [], []
    for index in range(32):
        eligible = int(index > 0)
        row = {"trajectory_id": f"dev-{index}", "seed": 2031000 + index, "observation_id": 1,
               "dynamics_id": 1, "goal_index": 7, "success_during_context": 1 - eligible,
               "policy_eligible": eligible, "initial_distance_after_context": float(index),
               "initial_errors": {"joint_error": float(index)},
               "diagnostic_status": "measured" if eligible else "excluded_support_success"}
        diagnostic_rows.append(row)
        planning_rows.append({**row, "initial_joint_error": float(index)})
    coordinates = "checkpoint_training_action_z_scores; native actions clipped to environment bounds"
    planner = {**worker.OFFICIAL_PLANNER_BUDGET, "search_coordinates": coordinates,
               "implementation": "stable_worldmodel.planning.solver.cem.CEMSolver", "goal_offset_from_end_of_history": 5}
    protocol = {key: planner[key] for key in ("samples", "iterations", "elites", "horizon", "native_budget")}
    protocol.update(policy="world_model", split="development", episodes_per_dynamics=32, goal_offset=5, seed=1701,
                    evaluator_sha256=identity["evaluator_sha256"], search_coordinates=coordinates,
                    run_identity={key: identity[key] for key in ("checkpoint_sha256", "data_manifest_sha256")})
    planning = {"status": "complete", **identity, "planning": {"policy": "world_model", "split": "development",
        "planner": planner, "protocol": protocol, "signature": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "records": planning_rows}}
    path = tmp_path / "planning_development.json"
    path.write_text(json.dumps(planning))
    result = {"status": "complete", "kind": "development_goal_calibration_diagnostic", "split": "development", **identity,
        "counts": {"selected": 32, "measured": 31, "support_success": 1}, "records": diagnostic_rows,
        "selected_keys": [[row["trajectory_id"], row["observation_id"]] for row in diagnostic_rows],
        "planning_setup_agreement": {"status": "passed", "compared_records": 32,
            "source": str(path.resolve()), "source_sha256": worker.sha256(path)}}
    return result, planning, identity, path


def test_unchanged_completed_fixture_is_accepted_without_mutation(tmp_path, monkeypatch):
    worker = worker_module(monkeypatch)
    result, planning, identity, path = fixture(worker, tmp_path)
    before_result, before_planning, before_bytes = deepcopy(result), deepcopy(planning), path.read_bytes()
    worker.validate_completed_diagnostic(result, planning, identity, path)
    assert result == before_result and planning == before_planning and path.read_bytes() == before_bytes


@pytest.mark.parametrize("change", ["policy", "budget", "signature"])
def test_changed_planning_policy_budget_or_signature_is_rejected(tmp_path, monkeypatch, change):
    worker = worker_module(monkeypatch)
    result, planning, identity, path = fixture(worker, tmp_path)
    if change == "policy":
        planning["planning"]["policy"] = "random"
    elif change == "budget":
        planning["planning"]["protocol"]["samples"] = 2
    else:
        planning["planning"]["signature"] = "invalid"
    path.write_text(json.dumps(planning))
    result["planning_setup_agreement"]["source_sha256"] = worker.sha256(path)
    with pytest.raises(RuntimeError):
        worker.validate_completed_diagnostic(result, planning, identity, path)


def test_current_source_sha_must_match_recorded_planning_linkage(tmp_path, monkeypatch):
    worker = worker_module(monkeypatch)
    result, planning, identity, path = fixture(worker, tmp_path)
    path.write_text(path.read_text() + "\n")  # Same parseable protocol; different cited artifact.
    with pytest.raises(ValueError, match="Planning source linkage"):
        worker.validate_completed_diagnostic(result, planning, identity, path)


@pytest.mark.parametrize("change", ["appended_duplicate", "replaced_duplicate", "count", "classification"])
def test_unique_rows_and_derived_counts_are_required(tmp_path, monkeypatch, change):
    worker = worker_module(monkeypatch)
    result, planning, identity, path = fixture(worker, tmp_path)
    if change == "appended_duplicate":
        result["records"].append(deepcopy(result["records"][0]))
    elif change == "replaced_duplicate":
        result["records"][-1] = deepcopy(result["records"][0])
    elif change == "count":
        result["counts"].update(measured=32, support_success=0)
    else:
        result["records"][0]["diagnostic_status"] = "measured"
    with pytest.raises(ValueError, match="Diagnostic"):
        worker.validate_completed_diagnostic(result, planning, identity, path)
