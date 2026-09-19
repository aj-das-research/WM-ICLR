#!/usr/bin/env python3
"""Candidate/freeze/run gate for a separate exploratory nine-run IWS ablation.

No submission command exists here. Parent review and a train-only full-shape
profile must precede immutable registration and any externally submitted job.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/real_video_iws_unbounded/training_v1.json"
REGISTRATION = ROOT / "configs/real_video_iws_unbounded/registration_v1.json"
CANDIDATE = ROOT / "configs/real_video_iws_unbounded/registration_candidate_v1.json"
REPORT = ROOT / "reports/real_video_iws_unbounded"
MODE = "unbounded_spatial_mix"
KIND = "shiftwm_iws_single_observation_unbounded_v1"
TASKS = {"pusht":4, "bimanual_box":14, "bimanual_rope":8}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


base = module(ROOT / "scripts/real_video_iws/campaign.py", "_unbounded_frozen_baseline_campaign")
sha, read, now, atomic_json = base.sha, base.read, base.now, base.atomic_json


def expected_config():
    config = deepcopy(read(base.CONFIG))
    config.update(schema="shiftwm_iws_unbounded_innovation_training_v1", package_kind=KIND,
        modes=[MODE], scope="Exploratory after development: IWS mixing without the innovation tanh bound; internal train/development only",
        intervention="Only innovation changes from tanh(residual) to residual; all source mixing, gates, inputs, initialization and optimization remain matched",
        baseline_registration_sha256=sha(base.REGISTRATION))
    config["evaluation"].update(primary_method=MODE, primary_comparator="bounded_spatial_mix",
        secondary_comparators=["anchored_additive", "autoregressive", "persistence"],
        inference_device="cpu", scope="internal_development_exploratory_after_v1",
        baseline_finalization="reports/real_video_iws/development_finalization.json",
        official_access_gate="This ablation registration cannot unlock reserved validation",
        qualitative_selection="No new outcome-selected qualitative cases are authorized by this ablation",
        interval_scope="H60 all four metrics and all four comparators; shared paired seed/trajectory draws; prefix summaries descriptive only")
    config["resource_policy"].update(slurm_time="08:00:00", epoch_boundary_requeue_after_seconds=26100)
    return config


def validate_config(config):
    base.check_registration()
    if config != expected_config():
        raise ValueError("The ablation must differ only by the declared intervention/namespace/reporting scope")
    return config


def grid(config):
    return [{"name": f"{task}_{MODE}_s{seed}", "task":task, "mode":MODE, "seed":seed,
             "output":f"runs/real_video_iws_unbounded/v1/{task}_{MODE}_s{seed}"}
            for seed in config["seeds"] for task in TASKS]


def trainer():
    return module(Path(__file__).with_name("train.py"), "_unbounded_campaign_runner")


def source_dependencies():
    original = base.check_registration()
    result = dict(original["dependencies"])
    paths = ["scripts/real_video_iws_unbounded/campaign.py", "scripts/real_video_iws_unbounded/train.py",
             "scripts/real_video_iws_unbounded/train.slurm", "scripts/real_video_iws_unbounded/evaluate.py",
             "scripts/real_video_iws_unbounded/profile.slurm",
             "scripts/real_video_iws_unbounded/finalize.py", "tests/test_iws_unbounded_finalization.py",
             "src/shiftwm/real_video_iws_unbounded/__init__.py", "src/shiftwm/real_video_iws_unbounded/model.py",
             "src/shiftwm/real_video_iws_unbounded/training.py", "tests/test_real_video_iws_unbounded_model.py",
             "tests/test_iws_unbounded_training.py", "reports/real_video_iws_unbounded/design.md"]
    for path in paths:
        result[path] = sha(ROOT / path)
    result[str(base.REGISTRATION.relative_to(ROOT))] = sha(base.REGISTRATION)
    return result


def candidate(config_path=CONFIG):
    config_path = Path(config_path)
    config = validate_config(read(config_path))
    if REGISTRATION.exists():
        raise ValueError("Candidate cannot replace an already frozen campaign")
    value = {"schema":"iws_unbounded_registration_candidate_v1", "status":"candidate_not_registered_not_authorized_to_train",
             "created_utc":now(), "config_sha256":sha(config_path), "runs":grid(config),
             "expected_runs":9, "epochs_per_run":30, "source_dependencies":source_dependencies(),
             "scope":config["scope"], "official_validation_payloads_allowed":False,
             "pending":["allocated train-only H60/batch64 resource profile for all three tasks", "independent source/tests/profile review", "parent approval and explicit immutable registration"]}
    atomic_json(value, CANDIDATE)
    return value


def check_registration(config_path=CONFIG):
    config_path = Path(config_path).resolve()
    config = validate_config(read(config_path))
    if not REGISTRATION.is_file():
        raise ValueError("Unbounded ablation is a candidate only; training has no registration")
    reg = read(REGISTRATION)
    if (reg.get("schema") != "iws_unbounded_training_registration_v1"
            or reg.get("status") != "registered_before_ablation_training"
            or reg.get("config_sha256") != sha(config_path) or reg.get("runs") != grid(config)
            or reg.get("expected_runs") != 9 or reg.get("epochs_per_run") != 30
            or reg.get("official_validation_payloads_allowed") is not False):
        raise ValueError("Wrong ablation registration")
    for path, value in reg["dependencies"].items():
        if sha(ROOT / path) != value:
            raise ValueError("Ablation source changed: " + path)
    return reg


def register(config_path, profiles, review_path):
    if REGISTRATION.exists():
        raise ValueError("Registration is immutable and already exists")
    config_path = Path(config_path).resolve()
    config = validate_config(read(config_path)); deps = source_dependencies()
    if len(profiles) != 3:
        raise ValueError("All three native task widths must be profiled")
    seen = set(); profile_bindings = {}
    for filename in profiles:
        path = Path(filename).resolve(); value = read(path)
        task = value.get("task")
        if (task not in TASKS or task in seen or value.get("status") != "passed"
                or value.get("config_sha256") != sha(config_path)
                or value.get("horizon") != 60 or value.get("predicted_offsets") != 59
                or value.get("batch_size") != 64 or value.get("microbatch_size") != 64
                or value.get("accumulation_steps") != 1
                or value.get("development_payloads_opened") != 0
                or value.get("official_validation_payloads_opened") != 0
                or value.get("source_hashes") != deps or value.get("source_hashes_after") != deps):
            raise ValueError("Profile does not match the complete frozen training recipe")
        rows = value.get("modes", [])
        if len(rows) != 1 or rows[0].get("mode") != MODE or rows[0].get("status") != "passed":
            raise ValueError("Profile did not measure the actual unbounded arm")
        seen.add(task); profile_bindings[str(path.relative_to(ROOT))] = sha(path)
    review_path = Path(review_path).resolve(); review = read(review_path)
    if (review.get("status") != "passed" or review.get("parent_approved_to_register") is not True
            or review.get("config_sha256") != sha(config_path)
            or review.get("source_dependencies") != deps
            or review.get("profile_sha256") != profile_bindings
            or review.get("official_validation_payloads_read") != 0):
        raise ValueError("Exact independent review and parent approval required before freeze")
    baseline_path = ROOT / config["evaluation"]["baseline_finalization"]
    recovery_path = ROOT / "reports/real_video_iws/recovery/common_cpu_v1/completion.json"
    baseline = read(baseline_path); recovery = read(recovery_path)
    validator = module(ROOT / "scripts/real_video_iws/finalize.py", "_unbounded_completed_baseline_gate")
    validator.verify_existing(baseline, base.check_registration())
    if (recovery.get("status") != "passed_full27_common_cpu_finalization"
            or recovery.get("official_validation_payloads_read") != 0
            or recovery["scientific_finalization"]["sha256"] != sha(baseline_path)):
        raise ValueError("The complete v1 comparison must be frozen on the common CPU backend")
    deps[str(baseline_path.relative_to(ROOT))] = sha(baseline_path)
    deps[str(recovery_path.relative_to(ROOT))] = sha(recovery_path)
    deps.update(profile_bindings)
    deps[str(review_path.relative_to(ROOT))] = sha(review_path)
    deps[str(config_path.relative_to(ROOT))] = sha(config_path)
    value = {"schema":"iws_unbounded_training_registration_v1", "status":"registered_before_ablation_training",
             "created_utc":now(), "config_sha256":sha(config_path), "runs":grid(config),
             "expected_runs":9, "epochs_per_run":30, "dependencies":deps,
             "official_validation_payloads_allowed":False, "scope":config["scope"],
             "baseline_registration_sha256":sha(base.REGISTRATION)}
    atomic_json(value, REGISTRATION); check_registration(config_path)
    return value


def run_index(config_path, index):
    reg = check_registration(config_path); config = read(config_path)
    if index not in range(9):
        raise ValueError("Unregistered run index")
    row = reg["runs"][index]; policy = config["resource_policy"]
    cmd = [sys.executable, str(Path(__file__).with_name("train.py")), "train", "--config", str(config_path),
           "--task", row["task"], "--mode", MODE, "--seed", str(row["seed"]),
           "--output", str(ROOT / row["output"]), "--resume-if-present",
           "--max-runtime-seconds", str(policy["epoch_boundary_requeue_after_seconds"])]
    summary_path = ROOT / row["output"] / "training_summary.json"
    already_complete = summary_path.exists() and read(summary_path).get("status") == "completed"
    if already_complete:
        # A completed run is immutable. Re-entering frozen fit(start==30) would
        # rewrite its operational summary/config and invalidate existing receipts.
        helper = trainer(); helper.validate_completed(ROOT/row["output"])
        _,state = helper.read_package(ROOT/row["output"]/"best")
        expected = {"task":row["task"],"mode":row["mode"],"seed":row["seed"],"model":config["model"],
                    "training":config["training"],"task_config":config["tasks"][row["task"]],"study_config_sha256":sha(config_path)}
        if state["config"]["metadata"]["identity"]["scientific_config"] != expected:
            raise ValueError("Completed run belongs to another ablation recipe")
    else:
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if code not in (0,75):
            raise RuntimeError("Ablation training failed with status " + str(code))
    check_registration(config_path)
    summary = read(summary_path) if summary_path.exists() else {}
    if summary.get("status") == "completed" and summary.get("completed_epochs") == 30:
        # CPU FP32 is deliberate: v1 comparison uses the same recovered backend.
        output = REPORT / "evaluations" / (row["name"] + ".json")
        subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate.py")),
            "--config", str(config_path), "--name", row["name"], "--output", str(output)], cwd=ROOT, check=True)
        marker = REPORT / "completions" / (row["name"] + ".json")
        value = {"status":"training_and_cpu_development_evaluation_completed", **row,
                 "summary_sha256":sha(summary_path), "evaluation_sha256":sha(output)}
        if marker.exists():
            if read(marker) != value:raise ValueError("Completed ablation receipt changed")
        else:atomic_json(value,marker)
        subprocess.run([sys.executable, str(Path(__file__).with_name("finalize.py")),
                        "--config", str(config_path), "--if-ready"], cwd=ROOT, check=True)
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or int(os.environ.get("SLURM_RESTART_COUNT", "0")) >= policy["max_restarts"]:
        raise RuntimeError("Incomplete ablation requires scheduler continuation")
    subprocess.run(["scontrol", "requeue", job], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["candidate", "verify", "register", "run-index"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--profile", type=Path, action="append", default=[])
    parser.add_argument("--review", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "candidate": result = candidate(args.config)
    elif args.command == "verify": result = check_registration(args.config)
    elif args.command == "register": result = register(args.config, args.profile, args.review)
    else: result = run_index(args.config, args.index)
    print(json.dumps(result, indent=2, sort_keys=True))
