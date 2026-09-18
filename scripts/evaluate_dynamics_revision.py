#!/usr/bin/env python3
"""Development-only evaluation of the trained frozen-framewise dynamics revision.

Calls the unchanged campaign evaluator on the original 32 development tasks.
The completed framewise donor is the fixed control. No main-test CLI exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import torch

from shiftwm.checkpoint import load_package
from shiftwm.dynamics_revision import (completed_framewise_source, file_sha256,
                                      load_revision_package, validate_donor_state)
import shiftwm.dynamics_revision as revision_source
import shiftwm.evaluate as evaluation_source
import shiftwm.generate as generation_source
from shiftwm.evaluate import atomic_json, evaluate_planning


def local_script(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("revision_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the already tested development selector/protocol and locking checks.
intervention = local_script("evaluate_goal_calibration_intervention")
training = local_script("train_dynamics_revision")
PROTOCOL = intervention.PROTOCOL


def read(path):
    return json.loads(Path(path).read_text())


def validate_control(record, donor_source, expected_keys, environment, data_sha):
    if (record.get("status") != "complete" or record.get("environment") != environment
            or record.get("model_mode") != "framewise" or record.get("training_seed") != 0
            or record.get("checkpoint_sha256") != donor_source["hashes"]["model_sha256"]
            or record.get("data_manifest_sha256") != data_sha
            or record.get("evaluator_sha256") != file_sha256(evaluation_source.__file__)):
        raise ValueError("Framewise development control has incompatible source identity")
    if record.get("checkpoint_epoch") not in donor_source["best_epochs"]:
        raise ValueError("Control is not the donor validation-best epoch")
    planning = record["planning"]
    protocol = planning["protocol"]
    for key, value in PROTOCOL.items():
        if protocol.get(key) != value:
            raise ValueError(f"Control changed the development protocol: {key}")
    if (planning.get("status") != "complete" or planning.get("split") != "development"
            or planning.get("policy") != "world_model"
            or protocol.get("search_coordinates") != intervention.SEARCH_COORDINATES):
        raise ValueError("Invalid control planning metadata")
    if any(planning.get("planner", {}).get(k) != v for k, v in intervention.PLANNER_METADATA.items()):
        raise ValueError("Control planning budget differs")
    if protocol.get("evaluator_sha256") != record["evaluator_sha256"] or protocol.get("run_identity") != {
            "checkpoint_sha256": donor_source["hashes"]["model_sha256"], "data_manifest_sha256": data_sha}:
        raise ValueError("Control planning sources differ")
    expected = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if planning.get("signature") != expected:
        raise ValueError("Control protocol signature is invalid")
    keys = [(r["trajectory_id"], r["observation_id"]) for r in planning["records"]]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys):
        raise ValueError("Control development task selection differs")


def validate_support(records, control_records, *, complete=False):
    """Compare only initial support/task identity, never require equal outcomes."""
    control = {(row["trajectory_id"], row["observation_id"]): row for row in control_records}
    keys = [(row["trajectory_id"], row["observation_id"]) for row in records]
    if len(keys) != len(set(keys)) or not set(keys) <= set(control):
        raise ValueError("Revision records contain duplicate or unexpected development tasks")
    if complete and set(keys) != set(control):
        raise ValueError("Completed revision must contain every control task")
    for row, key in zip(records, keys):
        other = control[key]
        for field in ("seed", "dynamics_id", "goal_index", "success_during_context", "policy_eligible", "policy"):
            if row.get(field) != other.get(field):
                raise ValueError(f"Revision and control initial support differ: {field}")
        initial = {k for k in other if k.startswith("initial_")}
        if initial != {k for k in row if k.startswith("initial_")}:
            raise ValueError("Revision support diagnostics differ")
        for field in initial:
            if not math.isclose(float(row[field]), float(other[field]), rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Revision and control task initialization differ: {field}")


def paired_counts(records, control_records):
    control = {(r["trajectory_id"], r["observation_id"]): r for r in control_records}
    eligible = [r for r in records if r["policy_eligible"]]
    wins = sum(r["success"] > control[r["trajectory_id"], r["observation_id"]]["success"] for r in eligible)
    losses = sum(r["success"] < control[r["trajectory_id"], r["observation_id"]]["success"] for r in eligible)
    return {"raw_successes": sum(r["success"] for r in records), "tasks": len(records),
            "support_successes": sum(r["success_during_context"] for r in records),
            "eligible_successes": sum(r["success"] for r in eligible), "eligible_tasks": len(eligible),
            "revision_only_successes": wins, "control_only_successes": losses,
            "eligible_difference_percentage_points": 100 * (wins - losses) / len(eligible) if eligible else None,
            "scope": "exploratory development, one training seed, additional revision optimization"}


def validate_progress(partial, identity, control_records):
    protocol = partial["protocol"]
    if (protocol.get("run_identity") != identity
            or protocol.get("evaluator_sha256") != identity["evaluator_sha256"]
            or protocol.get("search_coordinates") != intervention.SEARCH_COORDINATES
            or any(protocol.get(k) != v for k, v in PROTOCOL.items())):
        raise ValueError("Existing revision progress belongs to another run or protocol")
    expected = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if partial.get("signature") != expected:
        raise ValueError("Revision progress protocol signature is invalid")
    validate_support(partial["records"], control_records)


def run(args):
    package = Path(args.checkpoint)
    if package.name != "best":
        raise ValueError("Evaluate only a fully trained validation-best revision")
    model, state = load_revision_package(package, device="cpu")
    run_config = read(package.parent / "run_config.json")
    identity_training = training.training_identity(run_config, model)
    training.validate_completed(package.parent, identity_training)
    provenance = model.provenance
    if provenance.get("revision_environment") != args.environment or int(run_config["seed"]) != 0:
        raise ValueError("Wrong environment or training seed for this development revision")
    for name, expected in provenance["revision_implementation_hashes"].items():
        path = Path(__file__).with_name(name) if name == "train_dynamics_revision.py" else Path(revision_source.__file__).with_name(name)
        if file_sha256(path) != expected:
            raise ValueError(f"Revision training implementation changed: {name}")
    data = Path(run_config["data_root"])
    manifest = read(data / "manifest.json")
    data_sha = file_sha256(data / "manifest.json")
    if (manifest["environment"] != args.environment or manifest["action_block"] != 5
            or provenance.get("data_manifest_sha256") != data_sha
            or (args.environment == "pusht" and manifest.get("action_interface") != "relative")):
        raise ValueError("Revision dataset identity or action interface differs")
    expected_keys = intervention.development_keys(manifest)
    donor_source = completed_framewise_source(run_config["donor_checkpoint"])
    if donor_source["hashes"] != provenance["revision_donor"]["hashes"]:
        raise ValueError("The completed framewise donor changed")
    donor, donor_state = load_package(run_config["donor_checkpoint"], device="cpu")
    validate_donor_state(donor_state, donor_source)
    intervention.equal_modules(model.donor, donor, "complete frozen framewise donor")
    if file_sha256(run_config["action_stats"]) != provenance["action_stats_sha256"]:
        raise ValueError("Revision action statistics changed")
    control_path = Path(args.control_root) / f"{args.environment}_framewise_s0/planning_development.json"
    control = read(control_path)
    validate_control(control, donor_source, expected_keys, args.environment, data_sha)
    identity = {"experiment": "posthoc_development_frozen_framewise_dynamics_revision",
                "checkpoint_sha256": file_sha256(package / "model.pt"),
                "checkpoint_config_sha256": file_sha256(package / "config.json"),
                "training_summary_sha256": file_sha256(package.parent / "training_summary.json"),
                "training_metrics_sha256": file_sha256(package.parent / "metrics.jsonl"),
                "training_identity": identity_training, "donor_source_hashes": donor_source["hashes"],
                "control_source_sha256": file_sha256(control_path), "data_manifest_sha256": data_sha,
                "evaluator_sha256": file_sha256(evaluation_source.__file__),
                "generation_source_sha256": file_sha256(generation_source.__file__),
                "revision_source_sha256": file_sha256(revision_source.__file__),
                "evaluation_script_sha256": file_sha256(__file__),
                "protocol_helper_sha256": file_sha256(intervention.__file__),
                "goal_input": "unchanged_frozen_framewise_available_shifted_goal", "new_training": True}
    directory = Path(args.output_root) / f"{args.environment}_s0"
    with intervention.exclusive_output_lock(directory):
        output = directory / "planning_development.json"
        progress = directory / "planning_development.progress.json"
        if output.exists():
            previous = read(output)
            if previous.get("run_identity") != identity:
                raise ValueError("Existing revision result belongs to another run")
            if previous.get("environment") != args.environment or previous.get("model_mode") != model.config.mode:
                raise ValueError("Existing revision summary metadata differs")
            validate_progress(previous["planning"], identity, control["planning"]["records"])
            validate_support(previous["planning"]["records"], control["planning"]["records"],
                             complete=previous["status"] == "complete")
            if previous["status"] == "complete":
                intervention.validate_completed_result(previous, identity, expected_keys)
                return previous
        if progress.exists():
            partial = read(progress)
            validate_progress(partial, identity, control["planning"]["records"])
        model.to(args.device).eval()
        planning = evaluate_planning(model, data, **PROTOCOL, progress_path=progress,
                                     save_video=args.save_video, max_runtime_seconds=args.max_runtime_seconds,
                                     run_identity=identity)
        validate_support(planning["records"], control["planning"]["records"],
                         complete=planning["status"] == "complete")
        record = {"status": planning["status"], "kind": "posthoc_trainable_dynamics_revision",
                  "model_mode": model.config.mode, "environment": args.environment,
                  "run_identity": identity, "checkpoint": str(package.resolve()),
                  "checkpoint_epoch": state["epoch"], "training_seed": 0,
                  "selected_keys": [list(k) for k in expected_keys], "planning": planning,
                  "comparison": paired_counts(planning["records"], control["planning"]["records"])
                  if planning["status"] == "complete" else {"status": "pending"},
                  "execution_context": {"device": args.device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                        "main_efficiency_claim_eligible": False}}
        if record["status"] == "complete":
            intervention.validate_completed_result(record, identity, expected_keys)
        atomic_json(record, output)
        return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["pusht", "reacher"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/development_dynamics_revision"))
    parser.add_argument("--control-root", type=Path, default=Path("results/development_official_budget"))
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args(argv)
    if args.max_runtime_seconds is not None and (not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds <= 0):
        parser.error("max-runtime-seconds must be finite and positive")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    result = run(args)
    print(json.dumps({"event": "revision_development_" + result["status"],
                      "environment": args.environment, "comparison": result.get("comparison")}), flush=True)
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
