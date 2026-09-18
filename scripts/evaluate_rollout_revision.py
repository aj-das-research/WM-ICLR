#!/usr/bin/env python3
"""Evaluate the isolated objective/context study on the fixed development tasks.

No test split is exposed. Completed outputs are reusable only with identical
variant, training, source, donor, data, task, support, and planner identities.
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
                                      validate_donor_state)
from shiftwm.rollout_revision import load_rollout_package
import shiftwm.rollout_revision as revision_source
import shiftwm.evaluate as evaluation_source
import shiftwm.generate as generation_source
from shiftwm.evaluate import atomic_json, evaluate_planning


def local_script(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("rollout_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Preserve the tested source/control, initial-support, and resumability rules.
previous = local_script("evaluate_dynamics_revision")
training = local_script("train_rollout_revision")
intervention = previous.intervention
PROTOCOL = previous.PROTOCOL
validate_control = previous.validate_control
validate_support = previous.validate_support
paired_counts = previous.paired_counts
validate_progress = previous.validate_progress

IMPLEMENTATION_FILES = frozenset({
    "rollout_revision.py", "dynamics_revision.py", "model.py", "checkpoint.py",
    "upstream.py", "data.py", "train.py", "train_rollout_revision.py",
})


def read(path):
    return json.loads(Path(path).read_text())


def variant_identity(model, run_config, package, environment):
    """Prevent a valid checkpoint from being mislabeled as another study arm."""
    config = model.config
    if (config.mode != "framewise_rollout_revision"
            or config.objective not in {"teacher_forced", "recursive"}
            or config.context_mode not in {"inferred", "constant"}):
        raise ValueError("Unsupported rollout revision variant")
    if (environment not in {"pusht", "reacher"} or int(run_config.get("seed", -1)) != 0
            or model.provenance.get("revision_environment") != environment):
        raise ValueError("Wrong environment or training seed for rollout revision")
    saved_config = run_config.get("revision", {})
    if (saved_config.get("objective") != config.objective
            or saved_config.get("context_mode") != config.context_mode):
        raise ValueError("Training configuration and loaded rollout variant differ")
    run_id = f"{environment}_{config.objective}_{config.context_mode}_s0"
    package = Path(package)
    if (package.name != "best" or package.parent.name != run_id
            or Path(run_config.get("output_dir", "")).resolve() != package.parent.resolve()):
        raise ValueError("Rollout revision checkpoint/output directory differs from its variant")
    return {"run_id": run_id, "model_mode": config.mode,
            "objective": config.objective, "context_mode": config.context_mode,
            "training_seed": 0}


def validate_implementation(provenance):
    hashes = provenance.get("revision_implementation_hashes", {})
    if set(hashes) != IMPLEMENTATION_FILES:
        raise ValueError("Rollout revision implementation identity is incomplete or unexpected")
    for name, expected in hashes.items():
        path = (Path(__file__).with_name(name) if name == "train_rollout_revision.py"
                else Path(revision_source.__file__).with_name(name))
        if file_sha256(path) != expected:
            raise ValueError(f"Rollout revision training implementation changed: {name}")


def validate_selection(state, summary):
    metadata = state["config"].get("metadata", {})
    expected = {
        "selection": "minimum_validation_recursive_mse_targets_H_to_T-1_over_completed_training_epochs",
        "validation_protocol": "recursive_from_observed_support_all_query_steps",
        "validation_precision": "float32",
    }
    if (any(metadata.get(k) != v for k, v in expected.items())
            or summary.get("validation_metric") != "recursive_mse_all_query_steps"):
        raise ValueError("Rollout revision must use the common FP32 recursive validation selection")


def validate_result(record, identity, variant, expected_keys, control_records, environment):
    if (record.get("run_identity") != identity or record.get("environment") != environment
            or record.get("kind") != "posthoc_rollout_objective_context_revision"
            or any(record.get(k) != v for k, v in variant.items())
            or record.get("checkpoint_epoch") != identity["checkpoint_epoch"]
            or record.get("selected_keys") != [list(k) for k in expected_keys]
            or record.get("status") not in {"complete", "interrupted"}):
        raise ValueError("Existing rollout result belongs to another variant or protocol")
    if record["planning"].get("status") != record["status"]:
        raise ValueError("Rollout result status contradicts planning status")
    validate_progress(record["planning"], identity, control_records)
    complete = record["status"] == "complete"
    validate_support(record["planning"]["records"], control_records, complete=complete)
    if complete:
        intervention.validate_completed_result(record, identity, expected_keys)
        if record.get("comparison") != paired_counts(record["planning"]["records"], control_records):
            raise ValueError("Rollout comparison does not match its paired task records")


def run(args):
    package = Path(args.checkpoint)
    if package.name != "best":
        raise ValueError("Evaluate only a completed validation-best rollout revision")
    model, state = load_rollout_package(package, device="cpu")
    run_config = read(package.parent / "run_config.json")
    variant = variant_identity(model, run_config, package, args.environment)
    training_identity = training.training_identity(run_config, model)
    summary = training.validate_completed(package.parent, training_identity)
    validate_selection(state, summary)
    provenance = model.provenance
    validate_implementation(provenance)
    data = Path(run_config["data_root"])
    manifest = read(data / "manifest.json")
    data_sha = file_sha256(data / "manifest.json")
    if (manifest["environment"] != args.environment or manifest["action_block"] != 5
            or provenance.get("data_manifest_sha256") != data_sha
            or (args.environment == "pusht" and manifest.get("action_interface") != "relative")):
        raise ValueError("Rollout revision dataset identity or action interface differs")
    expected_keys = intervention.development_keys(manifest)
    donor_source = completed_framewise_source(run_config["donor_checkpoint"])
    if donor_source["hashes"] != provenance["revision_donor"]["hashes"]:
        raise ValueError("The completed framewise donor changed")
    donor, donor_state = load_package(run_config["donor_checkpoint"], device="cpu")
    validate_donor_state(donor_state, donor_source)
    intervention.equal_modules(model.donor, donor, "complete frozen framewise donor")
    if any(p.requires_grad for p in model.donor.parameters()) or model.donor.training:
        raise ValueError("Rollout revision donor must remain frozen and in evaluation mode")
    if file_sha256(run_config["action_stats"]) != provenance["action_stats_sha256"]:
        raise ValueError("Rollout revision action statistics changed")
    control_path = Path(args.control_root) / f"{args.environment}_framewise_s0/planning_development.json"
    control = read(control_path)
    validate_control(control, donor_source, expected_keys, args.environment, data_sha)
    control_records = control["planning"]["records"]
    identity = {
        "experiment": "posthoc_development_rollout_objective_context_study", **variant,
        "checkpoint_sha256": file_sha256(package / "model.pt"),
        "checkpoint_epoch": state["epoch"],
        "checkpoint_config_sha256": file_sha256(package / "config.json"),
        "training_run_config_sha256": file_sha256(package.parent / "run_config.json"),
        "training_summary_sha256": file_sha256(package.parent / "training_summary.json"),
        "training_metrics_sha256": file_sha256(package.parent / "metrics.jsonl"),
        "training_identity": training_identity,
        "training_implementation_hashes": provenance["revision_implementation_hashes"],
        "donor_source_hashes": donor_source["hashes"],
        "control_source_sha256": file_sha256(control_path), "data_manifest_sha256": data_sha,
        "evaluator_sha256": file_sha256(evaluation_source.__file__),
        "generation_source_sha256": file_sha256(generation_source.__file__),
        "revision_source_sha256": file_sha256(revision_source.__file__),
        "evaluation_script_sha256": file_sha256(__file__),
        "development_validation_helper_sha256": file_sha256(previous.__file__),
        "protocol_helper_sha256": file_sha256(intervention.__file__),
        "goal_input": "unchanged_frozen_framewise_available_shifted_goal",
        "new_training": True,
    }
    directory = Path(args.output_root) / variant["run_id"]
    with intervention.exclusive_output_lock(directory):
        output = directory / "planning_development.json"
        progress = directory / "planning_development.progress.json"
        if output.exists():
            existing = read(output)
            validate_result(existing, identity, variant, expected_keys, control_records, args.environment)
            if existing["status"] == "complete":
                return existing
        if progress.exists():
            validate_progress(read(progress), identity, control_records)
        model.to(args.device).eval()
        planning = evaluate_planning(model, data, **PROTOCOL, progress_path=progress,
                                     save_video=args.save_video, max_runtime_seconds=args.max_runtime_seconds,
                                     run_identity=identity)
        record = {
            "status": planning["status"], "kind": "posthoc_rollout_objective_context_revision",
            **variant, "environment": args.environment, "run_identity": identity,
            "checkpoint": str(package.resolve()), "checkpoint_epoch": state["epoch"],
            "selected_keys": [list(k) for k in expected_keys], "planning": planning,
            "comparison": paired_counts(planning["records"], control_records)
            if planning["status"] == "complete" else {"status": "pending"},
            "execution_context": {"device": args.device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                  "main_efficiency_claim_eligible": False},
        }
        validate_result(record, identity, variant, expected_keys, control_records, args.environment)
        atomic_json(record, output)
        return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["pusht", "reacher"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/development_rollout_revision"))
    parser.add_argument("--control-root", type=Path, default=Path("results/development_official_budget"))
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args(argv)
    try:
        device = torch.device(args.device)
    except (ValueError, RuntimeError):
        parser.error("A valid CUDA device is required for development planning")
    if device.type != "cuda":
        parser.error("Development planning requires a CUDA GPU; CPU rendering is not the fixed protocol")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if args.max_runtime_seconds is not None and (not math.isfinite(args.max_runtime_seconds)
                                                or args.max_runtime_seconds <= 0):
        parser.error("max-runtime-seconds must be finite and positive")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    result = run(args)
    print(json.dumps({"event": "rollout_development_" + result["status"],
                      "run_id": result["run_id"], "comparison": result.get("comparison")}), flush=True)
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
