#!/usr/bin/env python3
"""Post-hoc development intervention: fixed factorized WM, learned donor goal.

Only goal_embedding changes. The donor is the completed framewise seed0 model;
it sees the same available shifted goal, never a canonical/privileged goal.
This two-package inference hybrid is not a trained primary method.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from shiftwm.checkpoint import load_package
from shiftwm.data import split_combinations, validate_manifest
from shiftwm.evaluate import atomic_json, evaluate_planning
import shiftwm.evaluate as evaluation_source
import shiftwm.model as model_source
import shiftwm.generate as generation_source


PROTOCOL = {"split": "development", "episodes_per_dynamics": 32, "horizon": 5,
            "samples": 300, "iterations": 30, "elites": 30, "native_budget": 50,
            "goal_offset": 5, "seed": 1701, "policy": "world_model"}
SEARCH_COORDINATES = "checkpoint_training_action_z_scores; native actions clipped to environment bounds"
PLANNER_METADATA = {"implementation": "stable_worldmodel.planning.solver.cem.CEMSolver",
                    "samples": 300, "iterations": 30, "elites": 30, "horizon": 5,
                    "native_budget": 50, "action_block": 5, "receding_horizon": 1,
                    "history_steps_charged": 10, "goal_offset_from_end_of_history": 5,
                    "search_coordinates": SEARCH_COORDINATES}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def completed_source(package, required_mode, data_manifest_sha256, environment):
    package = Path(package)
    if package.name != "best":
        raise ValueError("Intervention requires completed validation-best packages")
    config = read(package / "config.json")
    run_config = read(package.parent / "run_config.json")
    summary = read(package.parent / "training_summary.json")
    if config["model_config"]["mode"] != required_mode:
        raise ValueError(f"Expected {required_mode} package")
    if int(run_config["seed"]) != 0:
        raise ValueError("This bounded intervention is fixed to training seed0")
    if summary.get("status") != "completed" or summary.get("completed_epochs") != run_config["epochs"]:
        raise ValueError("Training must be fully completed")
    provenance = config["provenance"]
    if provenance.get("data_manifest_sha256") != data_manifest_sha256:
        raise ValueError("Checkpoint was trained with different dataset provenance")
    if provenance.get("download", {}).get("repo") != "quentinll/lewm-" + environment:
        raise ValueError("Pretrained model family differs from environment")
    stats = Path(provenance["action_stats"])
    if digest(stats) != provenance["action_stats_sha256"]:
        raise ValueError("Official action statistics differ from recorded checksum")
    rows = [json.loads(line) for line in (package.parent / "metrics.jsonl").read_text().splitlines() if line.strip()]
    epochs = int(run_config["epochs"])
    if sorted(row["epoch"] for row in rows) != list(range(1, epochs + 1)):
        raise ValueError("Expected complete unique epoch metric history")
    values = [float(row["val"]["prediction_loss"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite validation prediction loss")
    best = min(values)
    if not math.isclose(best, summary["best_validation_prediction_loss"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Training summary disagrees with validation-best metric")
    return {"path": str(package.resolve()), "config": config, "best_metric": best,
            "best_epochs": [row["epoch"] for row, value in zip(rows, values)
                            if math.isclose(value, best, rel_tol=1e-10, abs_tol=1e-12)],
            "hashes": {"model_sha256": digest(package / "model.pt"),
                       "config_sha256": digest(package / "config.json"),
                       "run_config_sha256": digest(package.parent / "run_config.json"),
                       "training_summary_sha256": digest(package.parent / "training_summary.json"),
                       "metrics_sha256": digest(package.parent / "metrics.jsonl"),
                       "action_stats_sha256": digest(stats)}}


def validate_loaded_state(state, source):
    if state["config"] != source["config"]:
        raise ValueError("Embedded and external checkpoint configurations differ")
    if state["epoch"] not in source["best_epochs"] or not math.isclose(
            state["best_metric"], source["best_metric"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Model is not the recorded validation-best checkpoint")


def equal_modules(left, right, label):
    a, b = left.state_dict(), right.state_dict()
    if a.keys() != b.keys():
        raise ValueError(f"Different {label} state keys")
    for key in a:
        if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape or not torch.equal(a[key], b[key]):
            raise ValueError(f"Different {label} tensor: {key}")


def validate_pair(fixed, donor):
    """All shared observation/target coordinates and action units must match."""
    if fixed.config.mode != "factorized" or donor.config.mode != "framewise":
        raise ValueError("Require factorized recipient and framewise donor")
    if not fixed.config.freeze_visual or not donor.config.freeze_visual:
        raise ValueError("Both visual encoders must be frozen")
    if fixed.base_config != donor.base_config or fixed.config.history_length != donor.config.history_length:
        raise ValueError("Architecture or history interface differs")
    if fixed.action_dim != 10 or fixed.config.history_length != 3:
        raise ValueError("Intervention is fixed to H3 and five native 2D actions")
    for key in ("weights_sha256", "data_manifest_sha256", "action_stats_sha256"):
        if not fixed.provenance.get(key) or fixed.provenance.get(key) != donor.provenance.get(key):
            raise ValueError(f"Different or missing {key}")
    for key in ("action_mean", "action_std", "pixel_mean", "pixel_std"):
        a, b = getattr(fixed, key), getattr(donor, key)
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError(f"Different {key}")
    for name, left, right in (
        ("visual_encoder", fixed.base.encoder, donor.base.encoder),
        ("visual_projector", fixed.base.projector, donor.base.projector),
        ("reference_encoder", fixed.reference_encoder, donor.reference_encoder),
        ("reference_projector", fixed.reference_projector, donor.reference_projector),
        ("fixed_visual_vs_target_encoder", fixed.base.encoder, fixed.reference_encoder),
        ("fixed_visual_vs_target_projector", fixed.base.projector, fixed.reference_projector),
    ):
        if any(p.requires_grad for module in (left, right) for p in module.parameters()):
            raise ValueError(f"Unfrozen {name}")
        equal_modules(left, right, name)


class GoalCalibrationIntervention(nn.Module):
    """Delegate every observed-history/dynamics operation to the fixed model."""
    def __init__(self, fixed, donor):
        super().__init__()
        validate_pair(fixed, donor)
        self.fixed, self.donor = fixed, donor
        self.eval().requires_grad_(False)

    @property
    def config(self):
        return self.fixed.config

    @property
    def action_dim(self):
        return self.fixed.action_dim

    @property
    def action_mean(self):
        return self.fixed.action_mean

    @property
    def action_std(self):
        return self.fixed.action_std

    def encode_images(self, *args, **kwargs):
        return self.fixed.encode_images(*args, **kwargs)

    def infer_context(self, *args, **kwargs):
        return self.fixed.infer_context(*args, **kwargs)

    def correct_observations(self, *args, **kwargs):
        return self.fixed.correct_observations(*args, **kwargs)

    def predict_features(self, *args, **kwargs):
        return self.fixed.predict_features(*args, **kwargs)

    def rollout_features(self, *args, **kwargs):
        return self.fixed.rollout_features(*args, **kwargs)

    def goal_embedding(self, shifted_goal_images, observation_context):
        # The framewise donor ignores context by construction. Its frozen visual
        # encoder/projector are tensor-identical to the recipient's, checked above.
        return self.donor.goal_embedding(shifted_goal_images, observation_context)


def development_keys(manifest):
    validate_manifest(manifest)
    combinations = split_combinations("development")
    selected, counts = [], {}
    for ep in sorted(manifest["episodes"], key=lambda row: (row["seed"], row["dynamics_id"])):
        d = ep["dynamics_id"]
        if ep["split"] != "development" or counts.get(d, 0) >= 32 or not any(c[1] == d for c in combinations):
            continue
        if ep["steps"] < 7:
            raise ValueError("Development episode cannot supply the fixed goal")
        counts[d] = counts.get(d, 0) + 1
        selected.extend((ep["trajectory_id"], o) for o, dynamics in combinations if dynamics == d)
    if any(counts.get(d, 0) != 32 for _, d in combinations) or len(selected) != 32 or len(set(selected)) != 32:
        raise ValueError("Expected the fixed32 unique development keys")
    return selected


def validate_completed_result(record, identity, expected_keys):
    if record.get("status") != "complete" or record.get("run_identity") != identity:
        raise ValueError("Completed hybrid result belongs to different sources")
    planning = record["planning"]
    if planning.get("status") != "complete":
        raise ValueError("Planning result is incomplete")
    if planning.get("split") != PROTOCOL["split"] or planning.get("policy") != PROTOCOL["policy"]:
        raise ValueError("Planning summary split or policy differs from the fixed protocol")
    for key, value in PLANNER_METADATA.items():
        if planning.get("planner", {}).get(key) != value:
            raise ValueError(f"Planning summary metadata differs: {key}")
    protocol = planning["protocol"]
    for key, value in PROTOCOL.items():
        if protocol.get(key) != value:
            raise ValueError(f"Changed development protocol: {key}")
    if protocol.get("search_coordinates") != SEARCH_COORDINATES:
        raise ValueError("Changed development protocol: search_coordinates")
    if protocol.get("run_identity") != identity or protocol.get("evaluator_sha256") != identity["evaluator_sha256"]:
        raise ValueError("Planning identity does not match hybrid sources")
    expected_signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if planning.get("signature") != expected_signature:
        raise ValueError("Planning protocol signature differs")
    keys = [(row["trajectory_id"], row["observation_id"]) for row in planning["records"]]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys):
        raise ValueError("Completed hybrid result has different or duplicate development keys")


@contextmanager
def exclusive_output_lock(directory):
    """Hold one OS lease across reuse checks, model execution and atomic writes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".planning_development.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_environment(args, environment, remaining_seconds=None):
    directory = Path(args.output_root) / f"{environment}_factorized_s0_framewise_goal_s0"
    with exclusive_output_lock(directory):
        return _run_environment_locked(args, environment, remaining_seconds)


def _run_environment_locked(args, environment, remaining_seconds=None):
    data_name = "pusht_relative" if environment == "pusht" else "reacher"
    data = Path(args.data_root) / data_name
    manifest = read(data / "manifest.json")
    if manifest["environment"] != environment or manifest["action_block"] != 5:
        raise ValueError("Dataset environment/action grouping differs")
    if environment == "pusht" and manifest.get("action_interface") != "relative":
        raise ValueError("PushT intervention requires relative controls")
    expected_keys = development_keys(manifest)
    data_sha = digest(data / "manifest.json")
    fixed_path = Path(args.runs_root) / f"{environment}_factorized_s0" / "best"
    donor_path = Path(args.runs_root) / f"{environment}_framewise_s0" / "best"
    fixed_source = completed_source(fixed_path, "factorized", data_sha, environment)
    donor_source = completed_source(donor_path, "framewise", data_sha, environment)
    identity = {"experiment": "posthoc_development_factorized_with_learned_framewise_goal",
                "fixed_checkpoint_sha256": fixed_source["hashes"]["model_sha256"],
                "donor_checkpoint_sha256": donor_source["hashes"]["model_sha256"],
                "fixed_source_hashes": fixed_source["hashes"], "donor_source_hashes": donor_source["hashes"],
                "data_manifest_sha256": data_sha, "intervention_source_sha256": digest(__file__),
                "evaluator_sha256": digest(evaluation_source.__file__),
                "model_source_sha256": digest(model_source.__file__),
                "generation_source_sha256": digest(generation_source.__file__),
                "goal_input": "available_shifted_goal_image_only", "new_training": False}
    run_id = f"{environment}_factorized_s0_framewise_goal_s0"
    directory = Path(args.output_root) / run_id
    output = directory / "planning_development.json"
    if output.exists():
        existing = read(output)
        if existing.get("status") == "complete":
            validate_completed_result(existing, identity, expected_keys)
            return existing
        if existing.get("run_identity") != identity:
            raise ValueError("Interrupted hybrid output belongs to different sources")
    # Validate coordinates on CPU before transferring either model to the device.
    fixed, fixed_state = load_package(fixed_path, device="cpu")
    donor, donor_state = load_package(donor_path, device="cpu")
    validate_loaded_state(fixed_state, fixed_source)
    validate_loaded_state(donor_state, donor_source)
    hybrid = GoalCalibrationIntervention(fixed, donor).to(args.device)
    print(json.dumps({"event": "goal_only_intervention_start", "run_id": run_id,
                      "fixed_epoch": fixed_state["epoch"], "donor_epoch": donor_state["epoch"],
                      "run_identity": identity}), flush=True)
    planning = evaluate_planning(hybrid, data, **PROTOCOL,
                                 progress_path=directory / "planning_development.progress.json",
                                 save_video=args.save_video, max_runtime_seconds=remaining_seconds,
                                 run_identity=identity)
    record = {"status": planning["status"], "kind": "posthoc_learned_goal_calibration_intervention",
              "model_mode": "factorized_with_framewise_goal", "environment": environment,
              "run_identity": identity, "fixed_checkpoint": str(fixed_path.resolve()),
              "donor_checkpoint": str(donor_path.resolve()), "fixed_epoch": fixed_state["epoch"],
              "donor_epoch": donor_state["epoch"], "selected_keys": [list(key) for key in expected_keys],
              "interpretation": "development-only two-package inference hybrid; not a new trained primary method or canonical-goal oracle",
              "planning": planning, "execution_context": {"device": args.device,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "main_efficiency_claim_eligible": False}}
    if record["status"] == "complete":
        validate_completed_result(record, identity, expected_keys)
    atomic_json(record, output)
    print(json.dumps({"event": "goal_only_intervention_" + record["status"], "run_id": run_id}), flush=True)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["pusht", "reacher", "both"], default="both")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/world"))
    parser.add_argument("--data-root", type=Path, default=Path("data/world"))
    parser.add_argument("--output-root", type=Path, default=Path("results/development_goal_intervention"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-runtime-seconds", type=int)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    environments = ("pusht", "reacher") if args.environment == "both" else (args.environment,)
    for environment in environments:
        remaining = None if args.max_runtime_seconds is None else int(args.max_runtime_seconds - (time.monotonic() - started))
        if remaining is not None and remaining < 60:
            return 75
        result = run_environment(args, environment, remaining_seconds=remaining)
        if result["status"] != "complete":
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
