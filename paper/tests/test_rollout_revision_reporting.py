"""Evidence guards for the eight-arm study; fixture bytes are not model weights."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


report = load("rollout_report_test", Path(__file__).parents[1] / "scripts/render_rollout_revision.py")
legacy = load("rollout_reporting_fixture", Path(__file__).with_name("test_dynamics_revision_reporting.py"))
write = legacy.write


def planned_configs(root):
    for arm in report.ARMS:
        config = json.loads((report.PROJECT / "configs/rollout_revision" / (arm["run_id"] + ".json")).read_text())
        write(root / "configs/rollout_revision" / (arm["run_id"] + ".json"), config)


@pytest.fixture
def completed(tmp_path):
    root, old_result, _, old_directory, control_path = legacy.completed.__wrapped__(tmp_path)
    planned_configs(root)
    arm = report.ARMS[0]
    directory = root / "runs/rollout_revision" / arm["run_id"]
    run = report.read(old_directory / "run_config.json")
    run.update(output_dir=str(directory), revision={"objective": arm["objective"], "context_mode": arm["context_mode"]})
    planned_path = root / "configs/rollout_revision" / (arm["run_id"] + ".json")
    write(planned_path, run)
    old = report.read(old_directory / "best/config.json")
    provenance = deepcopy(old["provenance"])
    provenance["revision_implementation_hashes"] = {
        name: report.fingerprint(report.PROJECT / ("scripts" if name == "train_rollout_revision.py" else "src/shiftwm") / name)
        for name in report.contract.IMPLEMENTATION_FILES}
    revision = asdict(report.RolloutRevisionConfig(**run["revision"]))
    exported = {"format_version": 1, "package_kind": report.PACKAGE_KIND,
                "revision_config": revision, "model_config": revision,
                "provenance": provenance, "donor_config": old["donor_config"]}
    identity = report.training.training_identity(run, SimpleNamespace(provenance=provenance, export_config=lambda: exported))
    exported["metadata"] = {"config": run, "training_identity": identity,
        "training_objective": arm["objective"], "context_mode": arm["context_mode"],
        "selection": report.SELECTION, "validation_protocol": report.VALIDATION_PROTOCOL,
        "validation_precision": "float32"}
    for name in ("best", "last"):
        write(directory / name / "config.json", exported)
        (directory / name / "model.pt").write_bytes(("rollout fixture " + name).encode())
    write(directory / "run_config.json", run)
    legacy.epochs(directory / "metrics.jsonl")
    write(directory / "training_summary.json", {"status": "completed", "completed_epochs": 30, "step": 300,
        "validation_metric": "recursive_mse_all_query_steps", "best_validation_prediction_loss": .01,
        "training_identity": identity})
    record = {"status": "complete", "kind": "posthoc_rollout_objective_context_revision",
              **arm, "model_mode": "framewise_rollout_revision", "checkpoint": str(directory / "best"),
              "checkpoint_epoch": 30}
    expected = report.build_expectations(root, arm, run, record)
    record.update(run_identity=expected["identity"], selected_keys=[list(k) for k in expected["keys"]])
    record["planning"] = legacy.make_planning(expected["identity"], success_seeds=[0, 1, 3, 4, 5, 6])
    record["comparison"] = report.contract.paired_counts(record["planning"]["records"], expected["control"]["planning"]["records"])
    result_path = root / "results/development_rollout_revision" / arm["run_id"] / "planning_development.json"
    write(result_path, record)
    return SimpleNamespace(root=root, arm=arm, run=run, directory=directory, config_path=planned_path,
                           result=record, result_path=result_path, control_path=control_path)


def test_validated_complete_and_trained_states_do_not_deserialize_tensors(completed, monkeypatch):
    f = completed
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No tensor loading in reporter"))
    ledger = report.collect(f.root)
    assert (ledger["trained_records"], ledger["completed_records"], ledger["expected_records"]) == (1, 1, 8)
    row = ledger["rows"][0]
    assert row["status"] == "complete" and row["training"]["recursive_validation_mse"] == .01
    assert row["counts"]["eligible_successes"] == 4 and row["control_counts"]["eligible_successes"] == 3
    assert "\\positivegain{+3.33}" in report.render_tex(ledger)
    f.result_path.unlink()
    trained = report.collect(f.root)["rows"][0]
    assert trained["status"] == "trained" and trained["training"]["selected_epoch"] is None
    assert trained["training"]["best_epoch_candidates_from_history"] == [30]
    assert trained["counts"] is None and trained["paired"] is None


def test_planned_rows_and_partial_training_have_no_measurements(tmp_path):
    planned_configs(tmp_path)
    arm = report.ARMS[0]
    write(tmp_path / "runs/rollout_revision" / arm["run_id"] / "training_summary.json",
          {"status": "interrupted_checkpoint_saved", "completed_epochs": 29})
    ledger = report.collect(tmp_path)
    assert len(ledger["rows"]) == 8 and ledger["trained_records"] == ledger["completed_records"] == 0
    assert all(r["status"] == "planned" and r["training"] is None for r in ledger["rows"])
    tex = report.render_tex(ledger)
    assert "0/32" not in tex and tex.count(r"\missing") == 24
    assert "not statistical significance" in tex and "equal effective capacity" in tex


@pytest.mark.parametrize("field,value", [("objective", "recursive"), ("context_mode", "constant"),
                                        ("training_seed", 1), ("run_id", "wrong"),
                                        ("checkpoint_epoch", 29), ("checkpoint_epoch", None),
                                        ("model_mode", "factorized")])
def test_mislabelled_completed_arm_is_rejected(completed, field, value):
    f = completed
    f.result[field] = value
    write(f.result_path, f.result)
    with pytest.raises(ValueError): report.collect(f.root)


@pytest.mark.parametrize("change", ["weights", "donor_weights", "control", "run_config", "planned_config",
                                    "source_hash", "validation_metric", "validation_precision", "selection",
                                    "missing_epoch", "nonfinite", "wrong_best", "missing_last"])
def test_false_training_or_changed_sources_are_rejected(completed, change):
    f = completed
    if change in {"weights", "donor_weights", "control", "run_config"}:
        path = {"weights": f.directory / "best/model.pt", "donor_weights": f.root / "runs/world/pusht_framewise_s0/best/model.pt",
                "control": f.control_path, "run_config": f.directory / "run_config.json"}[change]
        with path.open("ab") as handle: handle.write(b" ")
    elif change == "planned_config":
        value = report.read(f.config_path); value["lr"] *= 2; write(f.config_path, value)
    elif change in {"missing_epoch", "nonfinite"}:
        p = f.directory / "metrics.jsonl"; values = [json.loads(line) for line in p.read_text().splitlines()]
        if change == "missing_epoch": values.pop()
        else: values[4]["val"]["prediction_loss"] = float("nan")
        p.write_text("\n".join(json.dumps(v) for v in values))
    elif change == "missing_last": (f.directory / "last/model.pt").unlink()
    elif change in {"validation_metric", "wrong_best"}:
        p = f.directory / "training_summary.json"; value = report.read(p)
        if change == "validation_metric": value["validation_metric"] = "teacher_forced_loss"
        else: value["best_validation_prediction_loss"] = .001
        write(p, value)
    else:
        p = f.directory / "best/config.json"; value = report.read(p)
        if change == "source_hash": value["provenance"]["revision_implementation_hashes"]["rollout_revision.py"] = "changed"
        else: value["metadata"][change] = "incorrect"
        write(p, value)
    with pytest.raises((ValueError, FileNotFoundError)): report.collect(f.root)


@pytest.mark.parametrize("change", ["missing", "duplicate", "support", "initial", "summary", "comparison", "binary_float",
                                    "budget", "signature", "nondevelopment", "selected_order"])
def test_task_protocol_pairing_and_summary_corruption_rejected(completed, change):
    f = completed; p = f.result["planning"]; rows = p["records"]
    if change == "missing": rows.pop()
    elif change == "duplicate": rows[-1] = deepcopy(rows[0])
    elif change == "support": rows[4]["success_during_context"] = 1; rows[4]["policy_eligible"] = 0
    elif change == "initial": rows[4]["initial_block_angle_error_rad"] += .1
    elif change == "summary": p["eligible_summary"]["all"]["success"]["mean"] = .9
    elif change == "comparison": f.result["comparison"]["revision_only_successes"] += 1
    elif change == "binary_float": rows[4]["success"] = 1.0
    elif change == "budget": p["protocol"]["samples"] = 128; legacy.sign(p)
    elif change == "signature": p["signature"] = "bad"
    elif change == "nondevelopment": p["split"] = "test"
    elif change == "selected_order": f.result["selected_keys"].reverse()
    write(f.result_path, f.result)
    with pytest.raises(ValueError): report.collect(f.root)


def test_interrupted_planning_has_no_partial_success_value(completed):
    f = completed
    f.result["status"] = f.result["planning"]["status"] = "interrupted"
    f.result["planning"]["records"] = f.result["planning"]["records"][:9]
    f.result["comparison"] = {"status": "pending"}
    write(f.result_path, f.result)
    row = report.collect(f.root)["rows"][0]
    assert row["status"] == "trained" and row["reason"] == "planning_interrupted"
    assert row["counts"] is row["paired"] is None


@pytest.mark.parametrize("successes,positive", [([0, 1], False), ([0, 1, 2, 3, 4], False),
                                               ([0, 1, 2, 3, 4, 5], True)])
def test_green_only_marks_strictly_positive_validated_difference(completed, successes, positive):
    f = completed
    for i, row in enumerate(f.result["planning"]["records"]):
        row["success"] = row["final_success"] = int(i in successes)
    legacy.refresh_summaries(f.result["planning"])
    control = report.read(f.control_path)
    f.result["comparison"] = report.contract.paired_counts(f.result["planning"]["records"], control["planning"]["records"])
    write(f.result_path, f.result)
    assert (r"\positivegain{" in report.render_tex(report.collect(f.root))) is positive


def test_zero_eligible_population_is_undefined(completed):
    f = completed
    control = report.read(f.control_path)
    for record in (f.result, control):
        for row in record["planning"]["records"]:
            row.update(success=1, final_success=1, success_during_context=1, policy_eligible=0,
                       native_steps=5, num_replans=0)
        legacy.refresh_summaries(record["planning"])
    write(f.control_path, control)
    expected = report.build_expectations(f.root, f.arm, f.run, f.result)
    f.result["run_identity"] = expected["identity"]
    f.result["planning"]["protocol"]["run_identity"] = expected["identity"]
    legacy.sign(f.result["planning"])
    f.result["comparison"] = report.contract.paired_counts(f.result["planning"]["records"], control["planning"]["records"])
    write(f.result_path, f.result)
    ledger = report.collect(f.root)
    assert ledger["rows"][0]["paired"]["eligible_difference_percentage_points"] is None
    tex = report.render_tex(ledger)
    assert "n/a" in tex and "0/0" not in tex and r"\positivegain{" not in tex
