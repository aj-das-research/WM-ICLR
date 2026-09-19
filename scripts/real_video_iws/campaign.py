#!/usr/bin/env python3
"""Freeze the matched IWS study, launch scheduler jobs, and preserve its identity.

Resource profiling is train-only and precedes registration. This entry point
never opens official-validation payloads and cannot authorize that evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/real_video_iws/training_v1.json"
REGISTRATION = ROOT / "configs/real_video_iws/training_registration_v1.json"
REPORT = ROOT / "reports/real_video_iws"
TASKS = {"pusht": 4, "bimanual_box": 14, "bimanual_rope": 8}
MODES = ("autoregressive", "anchored_additive", "bounded_spatial_mix")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def trainer():
    spec = importlib.util.spec_from_file_location("iws_campaign_trainer", Path(__file__).with_name("train.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_config(config):
    if config.get("schema") != "shiftwm_iws_single_observation_training_v1":
        raise ValueError("Wrong IWS study schema")
    if set(config["tasks"]) != set(TASKS) or config["modes"] != list(MODES) or config["seeds"] != [0, 1, 2]:
        raise ValueError("The complete three-task/three-arm/three-seed grid is required")
    for task, width in TASKS.items():
        if config["tasks"][task]["action_dim"] != width:
            raise ValueError("Native command width changed: " + task)
    training = config["training"]
    fixed = {"epochs": 30, "horizon": 60, "stride": 5, "validation_stride": 5,
             "batch_size": 64, "lr": 1e-4, "min_lr": 1e-6, "weight_decay": .01,
             "grad_clip": 1., "bf16": True, "tf32": False,
             "selector": "internal_dev_equal_trajectory_endpoint_H60_standardized_mse",
             "selector_tie_break": "earliest_strict_minimum", "validation_precision": "float32"}
    if any(training.get(key) != value for key, value in fixed.items()):
        raise ValueError("The declared complete-horizon scientific recipe changed")
    expected_model = {"feature_dim": 6144, "channels": 384, "grid_size": 4, "hidden_dim": 96,
                      "depth": 4, "temporal_slots": 3, "innovation_bound": 1., "identity_bias": 4., "initial_gate_logit": -3.}
    if config["model"] != expected_model or training.get("loss") != "uniform_standardized_mse_all_59_future_offsets_all_coordinates_equal_windows":
        raise ValueError("Fixed architecture or all-offset objective changed")
    micro = training["microbatch_size"]
    accumulation = training["accumulation_steps"]
    if type(micro) is not int or type(accumulation) is not int or micro < 1 or accumulation < 1 or micro * accumulation != 64:
        raise ValueError("Microbatch accumulation must retain effective batch64")
    if config["evaluation"].get("official_validation_allowed_during_training") is not False:
        raise ValueError("Official validation must remain inaccessible during training")
    evaluation = config["evaluation"]
    fixed_evaluation = {"official_handle_horizon": 60, "secondary_prefix_horizons": [15, 30, 45],
                        "primary_metric": "standardized_feature_mse", "cosine_epsilon": 1e-8, "cosine_clamp": [-1, 1],
                        "primary_method": "bounded_spatial_mix", "primary_comparator": "anchored_additive",
                        "primary_aggregation": "equal_trajectory_mean_of_equal_handle_means",
                        "secondary_metrics": ["standardized_feature_mae", "raw_dinov2_feature_l1", "flattened_feature_cosine_distance"]}
    if any(evaluation.get(key) != value for key, value in fixed_evaluation.items()):
        raise ValueError("Fixed evaluator mathematics or comparator differs")
    if config.get("package_kind") != "shiftwm_iws_single_observation_v1":
        raise ValueError("A distinct single-observation package is required")
    if config["split_sha256"] != sha(ROOT / "configs/real_video_iws/split_v1.json"):
        raise ValueError("Frozen IWS split changed")
    return config


def grid(config):
    return [{"name": f"{task}_{mode}_s{seed}", "task": task, "mode": mode, "seed": seed,
             "output": f"runs/real_video_iws/v1/{task}_{mode}_s{seed}"}
            for seed in config["seeds"] for task in TASKS for mode in config["modes"]]


def source_dependencies():
    paths = {
        "scripts/real_video_iws/campaign.py", "scripts/real_video_iws/train.py",
        "scripts/real_video_iws/train.slurm", "src/shiftwm/real_video_iws/model.py",
        "src/shiftwm/real_video_iws/windows.py", "scripts/real_video/train.py",
        "src/shiftwm/checkpoint.py", "src/shiftwm/upstream.py",
        "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/NOTICE.json",
        "src/shiftwm/real_video_iws/__init__.py", "src/shiftwm/real_video_iws/cache.py",
        "src/shiftwm/real_video_iws/data.py", "src/shiftwm/real_video_iws/features.py",
        "src/shiftwm/real_video_iws_tasks/__init__.py", "src/shiftwm/real_video_iws_tasks/cache.py",
        "src/shiftwm/real_video_iws_tasks/data.py", "scripts/real_video_iws/prepare_cache.py",
        "scripts/real_video_iws_tasks/prepare_cache.py",
        "reports/real_video_development/iws_single_observation_design.md",
        "configs/real_video_iws/split_v1.json",
        "scripts/real_video_iws/evaluate.py", "scripts/real_video_iws/finalize.py",
        "tests/test_iws_campaign.py", "tests/test_iws_development_evaluation.py",
        "tests/test_iws_finalization.py", "tests/test_iws_training.py",
        "tests/test_real_video_iws_model.py", "tests/test_real_video_iws_windows.py",
        "paper/scripts/refresh_experiment_alignment.py", "paper/tests/test_experiment_alignment_reporting.py",
    }
    helper = trainer()
    if hasattr(helper, "source_files"):
        for path in helper.source_files():
            path = Path(path)
            paths.add(str(path.relative_to(ROOT)) if path.is_absolute() else str(path))
    return {name: sha(ROOT / name) for name in sorted(paths)}


def check_registration(config_path=CONFIG):
    config_path = Path(config_path).resolve()
    registry = read(REGISTRATION)
    if registry.get("status") != "registered_before_predictor_training" or registry.get("schema") != "shiftwm_iws_training_registration_v1":
        raise ValueError("Full IWS training has no valid immutable registration")
    config = validate_config(read(config_path))
    if registry["config_sha256"] != sha(config_path) or registry["runs"] != grid(config):
        raise ValueError("Executed configuration/grid differs from the registered recipe")
    for name, expected in registry["dependencies"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen IWS dependency changed: " + name)
    if registry["expected_runs"] != 27 or registry["epochs_per_run"] != 30:
        raise ValueError("Incomplete IWS registration")
    return registry


def register(config_path, profile_path, review_path):
    if REGISTRATION.exists():
        raise ValueError("Registration already exists; cannot overwrite")
    config_path, profile_path, review_path = map(Path, (config_path, profile_path, review_path))
    config = validate_config(read(config_path))
    profile, review = read(profile_path), read(review_path)
    tr = config["training"]
    if (profile.get("status") != "passed" or profile.get("horizon") != 60 or profile.get("predicted_offsets") != 59
            or profile.get("batch_size") != 64 or profile.get("microbatch_size") != tr["microbatch_size"]
            or profile.get("accumulation_steps") != tr["accumulation_steps"]
            or profile.get("config_sha256") != sha(config_path)
            or profile.get("development_payloads_opened") != 0 or profile.get("official_validation_payloads_opened") != 0):
        raise ValueError("Successful train-only full-shape resource evidence does not match the recipe")
    modes = profile.get("modes", [])
    if len(modes) != 3 or {row.get("mode") for row in modes} != set(MODES) or any(row.get("status") != "passed" for row in modes):
        raise ValueError("Resource profiling did not complete every matched arm")
    if not profile.get("source_hashes") or profile["source_hashes"] != profile.get("source_hashes_after"):
        raise ValueError("Profile sources are absent or changed during measurement")
    for name, expected in profile["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Profiled implementation changed: " + name)
    if review.get("status") != "passed" or review.get("config_sha256") != sha(config_path) or review.get("profile_sha256") != sha(profile_path):
        raise ValueError("A source-bound successful full-shape resource/correctness review is required")
    if review.get("profiled_modes") != list(MODES) or review.get("official_validation_payloads_read") != 0:
        raise ValueError("Every learned arm must fit without official validation access")
    if review.get("complete_horizon") != 60 or review.get("effective_batch_size") != 64:
        raise ValueError("Resource review used a different horizon or optimizer batch")
    dependencies = source_dependencies()
    for name, expected in review["source_dependencies"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Reviewed source changed: " + name)
        dependencies[name] = expected
    if not {"src/shiftwm/real_video_iws/model.py", "src/shiftwm/real_video_iws/windows.py", "scripts/real_video_iws/train.py"} <= set(review["source_dependencies"]):
        raise ValueError("Correctness/resource review does not bind actual training sources")
    for path in (profile_path, review_path, config_path):
        dependencies[str(path.resolve().relative_to(ROOT))] = sha(path)
    for task, spec in config["tasks"].items():
        cache = ROOT / spec["cache_root"]
        manifest = read(cache / "manifest.json")
        if manifest.get("status") != "complete" or manifest.get("task") != task or manifest.get("command_width") != TASKS[task]:
            raise ValueError("Missing completed native task cache: " + task)
        for name in ("manifest.json", "identity.json", "episode_index.json", "training_statistics.json"):
            path = cache / name
            dependencies[str(path.relative_to(ROOT))] = sha(path)
        registry_path = ROOT / spec["cache_registration"]
        dependencies[str(registry_path.relative_to(ROOT))] = sha(registry_path)
        for name, expected in read(registry_path)["dependencies"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Frozen cache source changed: " + name)
            dependencies[name] = expected
    result = {"schema": "shiftwm_iws_training_registration_v1", "status": "registered_before_predictor_training",
              "created_utc": now(), "config_sha256": sha(config_path), "expected_runs": 27, "epochs_per_run": 30,
              "runs": grid(config), "dependencies": dependencies,
              "scope": "Internal train/development only; this registration cannot unlock official validation",
              "resource_profile_sha256": sha(profile_path), "resource_review_sha256": sha(review_path)}
    atomic_json(result, REGISTRATION)
    check_registration(config_path)
    return {"status": result["status"], "registration_sha256": sha(REGISTRATION), "runs": 27}


def submit(config_path):
    registry = check_registration(config_path)
    config = read(config_path)
    receipt = REPORT / "submission.json"
    if receipt.exists():
        raise ValueError("Campaign submission already recorded; inspect jobs before any retry")
    policy = config["resource_policy"]
    value = {"status": "submitting", "utc": now(), "jobs": [], "array_tasks": 27,
             "maximum_concurrent_gpus": policy["max_concurrent_runs"], "registration_sha256": sha(REGISTRATION)}
    # The account has two ws-ia jobs plus one GPU-partition GPU. Rotate arms
    # across partitions by seed so every arm receives the same partition mix.
    # The first three allocations can cover all three PushT arms at seed0.
    gpu_indices = [i for i, row in enumerate(registry["runs"])
                   if MODES.index(row["mode"]) == (2 + row["seed"]) % 3]
    workstation_indices = [i for i in range(27) if i not in gpu_indices]
    plans = [(policy["partition"], ",".join(map(str, workstation_indices)), policy["workstation_concurrency"]),
             (policy["secondary_partition"], ",".join(map(str, gpu_indices)), policy["gpu_partition_concurrency"])]
    atomic_json(value, receipt)
    for partition, indices, cap in plans:
        command = ["sbatch", "--parsable", "--partition=" + partition, f"--array={indices}%{cap}",
                   str(ROOT / "scripts/real_video_iws/train.slurm"), str(Path(config_path).resolve())]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("Slurm did not return an unambiguous job ID; inspect queue before retry")
        value["jobs"].append({"job_id": job, "partition": partition, "indices": indices,
                              "maximum_concurrent_gpus": cap, "command": command})
        atomic_json(value, receipt)
    value["status"] = "submitted"
    atomic_json(value, receipt)
    return value


def run_index(config_path, index):
    registry = check_registration(config_path)
    if index < 0 or index >= len(registry["runs"]):
        raise ValueError("Unregistered array index")
    row = registry["runs"][index]
    config = read(config_path)
    policy = config["resource_policy"]
    command = [sys.executable, str(Path(__file__).with_name("train.py")), "train", "--config", str(config_path),
               "--task", row["task"], "--mode", row["mode"], "--seed", str(row["seed"]),
               "--output", str(ROOT / row["output"]), "--resume-if-present",
               "--max-runtime-seconds", str(policy["epoch_boundary_requeue_after_seconds"])]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode not in (0, 75):
        raise subprocess.CalledProcessError(completed.returncode, command)
    check_registration(config_path)
    summary_path = ROOT / row["output"] / "training_summary.json"
    summary = read(summary_path) if summary_path.is_file() else {}
    if summary.get("status") == "completed" and summary.get("completed_epochs") == 30:
        # Training is complete before any model-comparison report is exposed.
        # The evaluator independently reloads the selected checkpoint and keeps
        # all development windows. Official validation remains inaccessible.
        evaluation_path = REPORT / "evaluations" / (row["name"] + ".json")
        if not evaluation_path.exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate.py")),
                            "--config", str(config_path), "--name", row["name"],
                            "--output", str(evaluation_path)], cwd=ROOT, check=True)
        atomic_json({"status": "training_and_development_evaluation_completed", "utc": now(), **row,
                     "summary_sha256": sha(summary_path), "evaluation_sha256": sha(evaluation_path)},
                    REPORT / "training_completions" / (row["name"] + ".json"))
        subprocess.run([sys.executable, str(Path(__file__).with_name("finalize.py")),
                        "--config", str(config_path), "--if-ready"], cwd=ROOT, check=True)
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    restarts = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if not job.isdigit() or restarts >= policy["max_restarts"]:
        raise RuntimeError("Full training unfinished; continuation requires a valid scheduler allocation")
    atomic_json({"status": "checkpointed_requeue_requested", "utc": now(), "job_id": job,
                 "completed_epochs": summary.get("completed_epochs"), "restart_count": restarts},
                REPORT / "continuations" / (row["name"] + ".json"))
    subprocess.run(["scontrol", "requeue", job], check=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("register", "verify", "submit", "run-index"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "register":
        if args.profile is None or args.review is None:
            parser.error("register requires --profile and --review")
        result = register(args.config, args.profile, args.review)
    elif args.command == "verify":
        result = {"status": "verified", "runs": len(check_registration(args.config)["runs"])}
    elif args.command == "submit":
        result = submit(args.config)
    else:
        if args.index is None:
            parser.error("run-index requires --index")
        run_index(args.config, args.index)
        result = {"status": "array_task_returned", "index": args.index}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
