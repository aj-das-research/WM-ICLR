"""Development revision reporting guards; fixture bytes are not trained weights."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "dynamics_revision_report", Path(__file__).parents[1] / "scripts/render_dynamics_revision.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def epochs(path):
    rows = [{"epoch": i, "step": 10 * i, "val": {"prediction_loss": (31 - i) / 100}}
            for i in range(1, 31)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def refresh_summaries(planning):
    rows = planning["records"]
    eligible = [r for r in rows if r["policy_eligible"]]
    def summary(n, total):
        return {"mean": n / total, "observations": total, "clusters": total}
    planning["summary"] = {"all": {
        "success": summary(sum(r["success"] for r in rows), len(rows)),
        "success_during_context": summary(sum(r["success_during_context"] for r in rows), len(rows))}}
    planning["eligible_summary"] = ({"all": {"success": summary(sum(r["success"] for r in eligible), len(eligible))}}
                                    if eligible else {})


def sign(planning):
    planning["signature"] = hashlib.sha256(json.dumps(planning["protocol"], sort_keys=True).encode()).hexdigest()


def make_planning(identity, success_seeds=range(5)):
    protocol = {**report.intervention.PROTOCOL,
                "search_coordinates": report.intervention.SEARCH_COORDINATES,
                "evaluator_sha256": report.fingerprint(report.PROJECT / "src/shiftwm/evaluate.py"),
                "run_identity": identity}
    rows = [{"trajectory_id": f"dev_{i}", "seed": i, "observation_id": 1, "dynamics_id": 1,
             "goal_index": 7, "policy": "world_model", "success": int(i in success_seeds),
             "final_success": int(i in success_seeds), "success_during_context": int(i < 2),
             "policy_eligible": int(i >= 2), "native_steps": 5 if i < 2 else 35,
             "num_replans": 0 if i < 2 else 5, "initial_distance_after_context": 0.1 + i,
             "initial_block_translation_error_px": 1. + i,
             "initial_block_angle_error_rad": 0.01 * i, "initial_agent_position_error_px": 0.1 * i}
            for i in range(32)]
    planning = {"status": "complete", "split": "development", "policy": "world_model",
                "planner": deepcopy(report.intervention.PLANNER_METADATA), "protocol": protocol, "records": rows}
    refresh_summaries(planning); sign(planning)
    return planning


@pytest.fixture
def completed(tmp_path):
    root = tmp_path
    data = root / "data/world/pusht_relative"
    train_combinations = [[0, 0], [1, 0], [2, 0], [0, 1], [2, 1], [0, 2], [1, 2]]
    write(data / "manifest.json", {"environment": "pusht", "action_block": 5, "action_interface": "relative",
        "train_combinations": train_combinations,
        "episodes": [{"trajectory_id": f"dev_{i}", "seed": i, "dynamics_id": 1,
                      "split": "development", "steps": 64} for i in range(32)]})
    cache = root / "data/features/pusht"
    write(cache / "manifest.json", {"encoder_weights_sha256": "base-weights"})
    stats = root / "data/action_stats.json"; write(stats, {"mean": [0, 0], "std": [1, 1]})
    provenance = {"weights_sha256": "base-weights", "data_manifest_sha256": report.fingerprint(data / "manifest.json"),
        "cache_manifest_sha256": report.fingerprint(cache / "manifest.json"),
        "action_stats_sha256": report.fingerprint(stats), "download": {"repo": "quentinll/lewm-pusht"}}
    settings = {"seed": 0, "epochs": 30, "sequence_length": 8, "stride": 1, "batch_size": 128,
                "lr": 5e-5, "min_lr": 1e-6, "weight_decay": .001, "grad_clip": 1., "bf16": True}
    donor_dir = root / "runs/world/pusht_framewise_s0"
    donor_config = {"format_version": 1, "model_config": {"mode": "framewise", "freeze_visual": True},
                    "provenance": deepcopy(provenance), "metadata": {}}
    write(donor_dir / "best/config.json", donor_config)
    (donor_dir / "best/model.pt").write_bytes(b"fixture donor: never deserialize")
    write(donor_dir / "run_config.json", settings)
    epochs(donor_dir / "metrics.jsonl")
    write(donor_dir / "training_summary.json", {"status": "completed", "completed_epochs": 30,
                                               "best_validation_prediction_loss": .01})
    donor = report.completed_framewise_source(donor_dir / "best")
    control_path = root / "results/development_official_budget/pusht_framewise_s0/planning_development.json"
    control = {"status": "complete", "environment": "pusht", "model_mode": "framewise", "training_seed": 0,
               "checkpoint_epoch": 30, "checkpoint_sha256": donor["hashes"]["model_sha256"],
               "data_manifest_sha256": provenance["data_manifest_sha256"],
               "evaluator_sha256": report.fingerprint(report.PROJECT / "src/shiftwm/evaluate.py")}
    control["planning"] = make_planning({k: control[k] for k in ("checkpoint_sha256", "data_manifest_sha256")})
    write(control_path, control)
    directory = root / "runs/dynamics_revision/pusht_s0"
    run = {**settings, "output_dir": str(directory), "data_root": str(data), "donor_checkpoint": str(donor_dir / "best"),
           "dataset_kwargs": {"feature_cache": str(cache), "preload_features": True}, "action_stats": str(stats)}
    provenance.update(revision_environment="pusht", revision_donor={"hashes": donor["hashes"], "epoch": 30, "best_metric": .01})
    provenance["revision_implementation_hashes"] = {name: report.fingerprint(report.PROJECT / "src/shiftwm" / name)
        for name in ("dynamics_revision.py", "model.py", "checkpoint.py", "upstream.py", "data.py", "train.py")}
    provenance["revision_implementation_hashes"]["train_dynamics_revision.py"] = report.fingerprint(report.PROJECT / "scripts/train_dynamics_revision.py")
    revision = asdict(report.RevisionConfig())
    package_config = {"format_version": 1, "package_kind": report.PACKAGE_KIND, "provenance": provenance,
        "revision_config": revision, "model_config": revision,
        "donor_config": {k: v for k, v in donor_config.items() if k not in {"format_version", "metadata"}}}
    identity = report.training.training_identity(run, SimpleNamespace(provenance=provenance, export_config=lambda: package_config))
    package_config["metadata"] = {"config": run, "training_identity": identity}
    for name in ("best", "last"):
        write(directory / name / "config.json", package_config)
        (directory / name / "model.pt").write_bytes(("fixture revision " + name).encode())
    write(directory / "run_config.json", run)
    epochs(directory / "metrics.jsonl")
    write(directory / "training_summary.json", {"status": "completed", "completed_epochs": 30, "step": 300,
        "best_validation_prediction_loss": .01, "training_identity": identity})
    result = {"status": "complete", "kind": "posthoc_trainable_dynamics_revision", "environment": "pusht",
        "model_mode": "framewise_dynamics_revision", "training_seed": 0,
        "checkpoint": str(directory / "best"), "checkpoint_epoch": 30}
    expected = report.build_expectations(root, "pusht", result)
    result.update(run_identity=expected["identity"], selected_keys=[list(k) for k in expected["keys"]])
    result["planning"] = make_planning(expected["identity"], success_seeds=[0, 1, 3, 4, 5, 6])
    result["comparison"] = report.contract.paired_counts(result["planning"]["records"], control["planning"]["records"])
    result_path = root / "results/development_dynamics_revision/pusht_s0/planning_development.json"
    write(result_path, result)
    return root, result, result_path, directory, control_path


def test_completed_reporting_uses_metadata_only_and_recomputes_paired_counts(completed, monkeypatch):
    root, _, _, _, _ = completed
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Reporter must not deserialize tensors"))
    ledger = report.collect(root)
    row = ledger["rows"][0]
    assert row["status"] == "complete" and ledger["completed_records"] == 1
    assert row["counts"] == {"raw_successes": 6, "raw_total": 32, "support_successes": 2,
                             "eligible_successes": 4, "eligible_total": 30}
    assert row["control_counts"]["eligible_successes"] == 3
    assert row["paired"]["revision_only_successes"] == 2
    assert row["paired"]["control_only_successes"] == 1
    assert ledger["rows"][1]["status"] == "pending"
    tex = report.render_tex(ledger)
    for qualifier in ("post", "30 additional", "not matched training", "32 development", "300 CEM", "30 iterations", "30 elites"):
        assert qualifier.lower() in tex.lower()


def test_missing_and_interrupted_are_pending_without_training_or_inference(tmp_path):
    def forbidden(*args):
        pytest.fail("Incomplete outputs must not request model/training sources")
    assert report.collect(tmp_path, forbidden)["completed_records"] == 0
    path = tmp_path / "results/development_dynamics_revision/pusht_s0/planning_development.json"
    write(path, {"status": "interrupted", "planning": {"status": "interrupted", "split": "development"}})
    ledger = report.collect(tmp_path, forbidden)
    assert ledger["rows"][0]["counts"] is None
    assert ledger["rows"][0]["reason"] == "run_incomplete"
    assert "0/32" not in report.render_tex(ledger)


@pytest.mark.parametrize("change", ["weights", "checkpoint_config", "training_summary", "training_metrics", "donor_weights", "control", "run_config"])
def test_changed_sources_cannot_reuse_a_completed_result(completed, change):
    root, _, _, directory, control = completed
    paths = {"weights": directory / "best/model.pt", "checkpoint_config": directory / "best/config.json",
             "training_summary": directory / "training_summary.json", "training_metrics": directory / "metrics.jsonl",
             "donor_weights": root / "runs/world/pusht_framewise_s0/best/model.pt", "control": control,
             "run_config": directory / "run_config.json"}
    path = paths[change]
    if change == "run_config":
        value = json.loads(path.read_text()); value["lr"] = .0002; write(path, value)
    else:
        with path.open("ab") as handle: handle.write(b" ")
    with pytest.raises(ValueError):
        report.collect(root)


@pytest.mark.parametrize("change", ["summary_pending", "missing_epoch", "duplicate_epoch", "nonfinite", "wrong_best_epoch", "wrong_identity", "missing_last", "last_config", "last_step"])
def test_false_training_completion_is_rejected(completed, change):
    root, record, path, directory, _ = completed
    if change in {"missing_epoch", "duplicate_epoch", "nonfinite"}:
        metrics = directory / "metrics.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if change == "missing_epoch": rows.pop()
        elif change == "duplicate_epoch": rows[-1]["epoch"] = 29
        else: rows[3]["val"]["prediction_loss"] = float("nan")
        metrics.write_text("\n".join(json.dumps(r) for r in rows))
    elif change == "wrong_best_epoch":
        record["checkpoint_epoch"] = 29; write(path, record)
    elif change == "missing_last":
        (directory / "last/model.pt").unlink()
    elif change == "last_config":
        p = directory / "last/config.json"; value = json.loads(p.read_text())
        value["metadata"]["training_identity"] = "wrong"; write(p, value)
    else:
        p = directory / "training_summary.json"; value = json.loads(p.read_text())
        value[{"summary_pending": "status", "wrong_identity": "training_identity", "last_step": "step"}[change]] = {
            "summary_pending": "interrupted_checkpoint_saved", "wrong_identity": "wrong", "last_step": 299}[change]
        write(p, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        report.collect(root)


@pytest.mark.parametrize("change", ["missing", "duplicate", "seed", "goal", "factor", "support", "initial", "binary_float", "binary_string", "binary_two", "summary", "comparison", "budget", "signature", "outer_complete", "identity"])
def test_corrupt_completed_population_protocol_or_counts_are_rejected(completed, change):
    root, result, path, _, _ = completed
    planning = result["planning"]; rows = planning["records"]
    if change == "missing": rows.pop()
    elif change == "duplicate": rows[-1] = deepcopy(rows[0])
    elif change == "seed": rows[4]["seed"] = 999
    elif change == "goal": rows[4]["goal_index"] = 8
    elif change == "factor": rows[4]["observation_id"] = 2
    elif change == "support": rows[4]["success_during_context"] = 1; rows[4]["policy_eligible"] = 0
    elif change == "initial": rows[4]["initial_block_angle_error_rad"] += .1
    elif change.startswith("binary_"): rows[4]["success"] = {"binary_float": 1.0, "binary_string": "1", "binary_two": 2}[change]
    elif change == "summary": planning["summary"]["all"]["success"]["mean"] = 0.
    elif change == "comparison": result["comparison"]["revision_only_successes"] += 1
    elif change == "budget": planning["protocol"]["samples"] = 128; sign(planning)
    elif change == "signature": planning["signature"] = "bad"
    elif change == "outer_complete": planning["status"] = "interrupted"
    elif change == "identity": result["run_identity"]["new_training"] = False
    write(path, result)
    with pytest.raises(ValueError):
        report.collect(root)


def test_no_eligible_tasks_are_explicitly_undefined(completed):
    root, result, path, _, control_path = completed
    control = json.loads(control_path.read_text())
    for record in (result, control):
        for row in record["planning"]["records"]:
            row.update(success=1, final_success=1, success_during_context=1, policy_eligible=0,
                       native_steps=5, num_replans=0)
        refresh_summaries(record["planning"])
    write(control_path, control)
    expected = report.build_expectations(root, "pusht", result)
    result["run_identity"] = expected["identity"]
    result["planning"]["protocol"]["run_identity"] = expected["identity"]; sign(result["planning"])
    result["comparison"] = report.contract.paired_counts(result["planning"]["records"], control["planning"]["records"])
    write(path, result)
    ledger = report.collect(root)
    assert ledger["rows"][0]["paired"]["eligible_difference_percentage_points"] is None
    assert "n/a" in report.render_tex(ledger) and "0/0" not in report.render_tex(ledger)
