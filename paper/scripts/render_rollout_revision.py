#!/usr/bin/env python3
"""Report the eight-arm development study from validated local evidence only.

Reads JSON/JSONL metadata and streams hashes; never deserializes model tensors.
The evaluator preflight verifies tensor reconstruction and frozen-donor equality.
This reporter verifies its pinned identity and independently recomputes outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


previous = module("rollout_reporting_previous", Path(__file__).with_name("render_dynamics_revision.py"))
contract = module("rollout_reporting_contract", PROJECT / "scripts/evaluate_rollout_revision.py")
training = contract.training
intervention = contract.intervention
from shiftwm.rollout_revision import PACKAGE_KIND, RolloutRevisionConfig
from shiftwm.dynamics_revision import completed_framewise_source

read, fingerprint, rooted = previous.read, previous.fingerprint, previous.rooted
SELECTION = "minimum_validation_recursive_mse_targets_H_to_T-1_over_completed_training_epochs"
VALIDATION_PROTOCOL = "recursive_from_observed_support_all_query_steps"
ARMS = tuple({"environment": env, "objective": objective, "context_mode": context,
              "training_seed": 0, "run_id": f"{env}_{objective}_{context}_s0"}
             for env in ("pusht", "reacher")
             for objective in ("teacher_forced", "recursive")
             for context in ("inferred", "constant"))


def scientific_config(config):
    return {key: value for key, value in config.items() if key not in training.OPERATIONAL}


def arm_config(root, arm):
    path = Path(root) / "configs/rollout_revision" / (arm["run_id"] + ".json")
    config = read(path)
    expected = Path(root) / "runs/rollout_revision" / arm["run_id"]
    revision = RolloutRevisionConfig(**config.get("revision", {}))
    if (type(config.get("seed")) is not int or config["seed"] != 0
            or type(config.get("epochs")) is not int or config["epochs"] != 30
            or config.get("sequence_length") != 8
            or (revision.context_dim, revision.context_hidden) != (32, 128)
            or revision.objective != arm["objective"] or revision.context_mode != arm["context_mode"]
            or rooted(root, config["output_dir"]) != expected.resolve()
            or rooted(root, config["donor_checkpoint"]) != (
                Path(root) / "runs/world" / f"{arm['environment']}_framewise_s0/best").resolve()):
        raise ValueError("Rollout arm configuration differs from the eight-arm development protocol")
    return config, path


def training_metadata(root, arm, planned, selected_epoch=None):
    """Check a complete history and package metadata without tensor loading."""
    output = Path(root) / "runs/rollout_revision" / arm["run_id"]
    package = output / "best"
    config, run = read(package / "config.json"), read(output / "run_config.json")
    summary = read(output / "training_summary.json")
    if (scientific_config(run) != scientific_config(planned)
            or summary.get("status") != "completed" or summary.get("completed_epochs") != 30
            or summary.get("validation_metric") != "recursive_mse_all_query_steps"):
        raise ValueError("Rollout training is incomplete or differs from the declared scientific configuration")
    expected_revision = asdict(RolloutRevisionConfig(**run.get("revision", {})))
    if (config.get("format_version") != 1 or config.get("package_kind") != PACKAGE_KIND
            or config.get("revision_config") != expected_revision
            or config.get("model_config") != expected_revision):
        raise ValueError("Rollout package configuration differs from the training arm")
    provenance = config["provenance"]
    proxy = SimpleNamespace(provenance=provenance, export_config=lambda: config)
    identity = training.training_identity(run, proxy)
    if summary.get("training_identity") != identity:
        raise ValueError("Rollout training identity differs from configuration/provenance")
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if any(type(r.get("epoch")) is not int for r in rows) or sorted(r["epoch"] for r in rows) != list(range(1, 31)):
        raise ValueError("Rollout training requires 30 complete, unique metric epochs")
    values = [float(r["val"]["prediction_loss"]) for r in rows]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Nonfinite or negative recursive validation MSE")
    best = min(values)
    if not math.isclose(best, float(summary["best_validation_prediction_loss"]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Rollout summary differs from its full recursive validation history")
    best_epochs = [r["epoch"] for r, value in zip(rows, values)
                   if math.isclose(value, best, rel_tol=1e-10, abs_tol=1e-12)]
    if selected_epoch is not None and (type(selected_epoch) is not int or selected_epoch not in best_epochs):
        raise ValueError("Rollout evaluation checkpoint is not validation-best")
    final = next(r for r in rows if r["epoch"] == 30)
    if type(summary.get("step")) is not int or summary["step"] != final.get("step"):
        raise ValueError("Rollout final step differs from the full metric history")
    for name in ("best", "last"):
        exported = read(output / name / "config.json")
        meta = exported.get("metadata", {})
        if (exported.get("format_version") != 1 or exported.get("package_kind") != PACKAGE_KIND
                or exported.get("revision_config") != expected_revision
                or exported.get("model_config") != expected_revision
                or exported.get("provenance") != provenance
                or exported.get("donor_config") != config.get("donor_config")
                or meta.get("training_identity") != identity
                or scientific_config(meta.get("config", {})) != scientific_config(run)
                or meta.get("training_objective") != arm["objective"]
                or meta.get("context_mode") != arm["context_mode"]):
            raise ValueError("Rollout best/last metadata differs from the trained arm")
        contract.validate_selection({"config": exported}, summary)
        if not (output / name / "model.pt").is_file():
            raise ValueError("Rollout training lacks a completed model artifact")
    return {"config": config, "run_config": run, "summary": summary, "identity": identity,
            "best_epochs": best_epochs, "best_metric": best, "package": package}


def build_expectations(root, arm, planned, record=None):
    root = Path(root)
    if record is not None and type(record.get("checkpoint_epoch")) is not int:
        raise ValueError("Rollout result requires an explicit selected checkpoint epoch")
    metadata = training_metadata(root, arm, planned, None if record is None else record.get("checkpoint_epoch"))
    package, config, run = metadata["package"], metadata["config"], metadata["run_config"]
    environment, provenance = arm["environment"], config["provenance"]
    if record is not None and rooted(root, record.get("checkpoint", "")) != package.resolve():
        raise ValueError("Rollout evaluation references a different checkpoint path")
    manifest_path = rooted(root, run["data_root"]) / "manifest.json"
    manifest, data_sha = read(manifest_path), fingerprint(manifest_path)
    if (manifest.get("environment") != environment or manifest.get("action_block") != 5
            or (environment == "pusht" and manifest.get("action_interface") != "relative")
            or provenance.get("revision_environment") != environment
            or provenance.get("data_manifest_sha256") != data_sha):
        raise ValueError("Rollout dataset/family/action identity differs")
    keys = intervention.development_keys(manifest)
    episodes = {r["trajectory_id"]: r for r in manifest["episodes"] if r["split"] == "development"}
    tasks = {key: {"seed": episodes[key[0]]["seed"], "dynamics_id": episodes[key[0]]["dynamics_id"]} for key in keys}
    if len({task["seed"] for task in tasks.values()}) != 32:
        raise ValueError("Rollout development requires 32 independent initial-state seeds")
    donor = completed_framewise_source(rooted(root, run["donor_checkpoint"]))
    donor_export = {k: v for k, v in donor["config"].items() if k not in {"format_version", "metadata"}}
    if (provenance.get("revision_donor", {}).get("hashes") != donor["hashes"]
            or donor["config"]["model_config"].get("freeze_visual") is not True
            or config.get("donor_config") != donor_export
            or provenance["revision_donor"].get("epoch") not in donor["best_epochs"]
            or not math.isclose(provenance["revision_donor"].get("best_metric", math.inf),
                                donor["best_metric"], rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("Rollout frozen validation-best donor identity differs")
    for key in ("sequence_length", "stride", "batch_size", "lr", "min_lr", "weight_decay", "grad_clip", "bf16"):
        if run.get(key) != donor["run_config"].get(key):
            raise ValueError(f"Rollout revision changed the donor training setting: {key}")
    if set(run.get("dataset_kwargs", {})) - {"feature_cache", "preload_features"}:
        raise ValueError("Rollout revision changed the fixed dataset protocol")
    cache_path = rooted(root, run["dataset_kwargs"]["feature_cache"]) / "manifest.json"
    for name, path in (("data_manifest", manifest_path), ("cache_manifest", cache_path),
                       ("action_stats", rooted(root, run["action_stats"]))):
        value = fingerprint(path)
        if provenance.get(name + "_sha256") != value or donor["config"]["provenance"].get(name + "_sha256") != value:
            raise ValueError(f"Rollout revision/donor {name} identity differs")
    if (read(cache_path).get("encoder_weights_sha256") != provenance.get("weights_sha256")
            or provenance.get("download", {}).get("repo") != "quentinll/lewm-" + environment):
        raise ValueError("Rollout canonical feature coordinates/pretrained family differ")
    contract.validate_implementation(provenance)
    control_path = root / "results/development_official_budget" / f"{environment}_framewise_s0/planning_development.json"
    control = read(control_path)
    contract.validate_control(control, donor, keys, environment, data_sha)
    variant = {key: arm[key] for key in ("run_id", "objective", "context_mode", "training_seed")}
    variant["model_mode"] = "framewise_rollout_revision"
    identity = {"experiment": "posthoc_development_rollout_objective_context_study", **variant,
        "checkpoint_sha256": fingerprint(package / "model.pt"),
        "checkpoint_epoch": None if record is None else record.get("checkpoint_epoch"),
        "checkpoint_config_sha256": fingerprint(package / "config.json"),
        "training_run_config_sha256": fingerprint(package.parent / "run_config.json"),
        "training_summary_sha256": fingerprint(package.parent / "training_summary.json"),
        "training_metrics_sha256": fingerprint(package.parent / "metrics.jsonl"),
        "training_identity": metadata["identity"],
        "training_implementation_hashes": provenance["revision_implementation_hashes"],
        "donor_source_hashes": donor["hashes"], "control_source_sha256": fingerprint(control_path),
        "data_manifest_sha256": data_sha, "evaluator_sha256": fingerprint(PROJECT / "src/shiftwm/evaluate.py"),
        "generation_source_sha256": fingerprint(PROJECT / "src/shiftwm/generate.py"),
        "revision_source_sha256": fingerprint(PROJECT / "src/shiftwm/rollout_revision.py"),
        "evaluation_script_sha256": fingerprint(PROJECT / "scripts/evaluate_rollout_revision.py"),
        "development_validation_helper_sha256": fingerprint(PROJECT / "scripts/evaluate_dynamics_revision.py"),
        "protocol_helper_sha256": fingerprint(PROJECT / "scripts/evaluate_goal_calibration_intervention.py"),
        "goal_input": "unchanged_frozen_framewise_available_shifted_goal", "new_training": True}
    return {"identity": identity, "variant": variant, "keys": keys, "tasks": tasks, "control": control,
        "training": {"completed_epochs": 30, "selected_epoch": identity["checkpoint_epoch"],
            "best_epoch_candidates_from_history": metadata["best_epochs"],
            "recursive_validation_mse": metadata["best_metric"], "training_identity": metadata["identity"],
            "validation_metric": "recursive_mse_all_query_steps", "validation_precision": "float32",
            "last_config_sha256": fingerprint(package.parent / "last/config.json"),
            "last_model_sha256": fingerprint(package.parent / "last/model.pt"),
            "validation_scope": "Completed external metadata/history and streamed hashes; tensor preflight belongs to evaluator."}}


def validate_completed(record, environment, expected):
    if record.get("status") != "complete" or type(record.get("training_seed")) is not int:
        raise ValueError("Rollout result must complete the seed-zero development protocol")
    contract.validate_result(record, expected["identity"], expected["variant"], expected["keys"],
                             expected["control"]["planning"]["records"], environment)
    revision, counts = previous.validated_rows(record, expected, environment)
    control, control_counts = previous.validated_rows(expected["control"], expected, environment)
    previous.paired.validate_shared_support(revision, control, environment)
    compared = contract.paired_counts(list(revision.values()), list(control.values()))
    if record.get("comparison") != compared:
        raise ValueError("Rollout stored comparison differs from paired outcomes")
    return {"counts": counts, "control_counts": control_counts, "paired": compared,
            "support_pairing": "verified_all_32_tasks", "efficiency_claim_eligible": False}


def collect(root=PROJECT):
    root = Path(root)
    rows, sources = [], {}
    for arm in ARMS:
        planned, config_path = arm_config(root, arm)
        sources[str(config_path)] = fingerprint(config_path)
        output = root / "runs/rollout_revision" / arm["run_id"]
        result_path = root / "results/development_rollout_revision" / arm["run_id"] / "planning_development.json"
        row = {**arm, "status": "planned", "reason": "training_not_complete", "training": None,
               "counts": None, "control_counts": None, "paired": None,
               "config_file": str(config_path), "config_sha256": sources[str(config_path)],
               "source_file": str(result_path)}
        record = read(result_path) if result_path.exists() else None
        if record is not None:
            for split in (record.get("planning", {}).get("split"), record.get("planning", {}).get("protocol", {}).get("split")):
                if split is not None and split != "development":
                    raise ValueError("Rollout reporter refuses non-development results")
            if record.get("status") not in {"complete", "interrupted"}:
                raise ValueError("Unsupported rollout evaluation status")
            row["source_sha256"] = fingerprint(result_path)
        summary_path = output / "training_summary.json"
        summary = read(summary_path) if summary_path.exists() else None
        trained = summary is not None and summary.get("status") == "completed"
        if record is not None and not trained:
            raise ValueError("Rollout planning result exists without completed training")
        if trained:
            expected = build_expectations(root, arm, planned, record)
            row.update(status="trained", reason="planning_not_complete", training=expected["training"],
                       training_and_source_identity=expected["identity"])
            if record is not None:
                contract.validate_result(record, expected["identity"], expected["variant"], expected["keys"],
                                         expected["control"]["planning"]["records"], arm["environment"])
                if record["status"] == "complete":
                    row.update(validate_completed(record, arm["environment"], expected), status="complete", reason=None)
                else:
                    row["reason"] = "planning_interrupted"
        rows.append(row)
    dependencies = [Path(__file__), Path(previous.__file__), PROJECT / "scripts/evaluate_rollout_revision.py",
                    PROJECT / "scripts/evaluate_dynamics_revision.py", PROJECT / "scripts/summarize_goal_intervention.py",
                    PROJECT / "scripts/evaluate_goal_calibration_intervention.py", PROJECT / "scripts/train_rollout_revision.py",
                    PROJECT / "src/shiftwm/rollout_revision.py", PROJECT / "src/shiftwm/dynamics_revision.py",
                    PROJECT / "src/shiftwm/checkpoint.py", PROJECT / "src/shiftwm/model.py",
                    PROJECT / "src/shiftwm/data.py", PROJECT / "src/shiftwm/train.py", PROJECT / "src/shiftwm/upstream.py"]
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(row["status"] == "complete" for row in rows) else "pending",
        "kind": "posthoc_development_rollout_objective_context_study", "rows": rows,
        "expected_records": 8, "trained_records": sum(row["training"] is not None for row in rows),
        "completed_records": sum(row["status"] == "complete" for row in rows),
        "source_configs": sources, "build_sources": {str(path): fingerprint(path) for path in dependencies},
        "protocol": {**intervention.PROTOCOL, "planner": intervention.PLANNER_METADATA},
        "validation": {"metric": "recursive_mse_all_query_steps", "precision": "float32", "targets": [3, 4, 5, 6, 7]},
        "scope": "Eight post-hoc seed-zero development arms; frozen Framewise donor; 30 extra epochs per arm.",
        "limits": "Favorable means are not significance or context-necessity evidence. Nominal parameter matching does not equal effective capacity. No main-test, multi-seed or efficiency claim. Tensor equality is evaluator preflight, not reconstructed by this reporter."}


def render_tex(ledger):
    lines = [r"% Generated only from source-validated rollout study records.",
             r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lllrrrr}\toprule",
             r"Environment & Objective & Context & Status & Val. MSE & Eligible & $\Delta$ (pp) \\\midrule"]
    for row in ledger["rows"]:
        cells = ["PushT" if row["environment"] == "pusht" else "Reacher",
                 "Teacher-forced" if row["objective"] == "teacher_forced" else "Recursive",
                 "Inferred" if row["context_mode"] == "inferred" else "Constant", row["status"].capitalize(),
                 f"{row['training']['recursive_validation_mse']:.5f}" if row["training"] else r"\missing"]
        if row["status"] == "complete":
            c, delta = row["counts"], row["paired"]["eligible_difference_percentage_points"]
            eligible = f"{c['eligible_successes']}/{c['eligible_total']}" if c["eligible_total"] else "n/a"
            difference = "n/a" if delta is None else f"{delta:+.2f}"
            if delta is not None and delta > 0 and float(difference) > 0:
                difference = r"\positivegain{" + difference + "}"
            cells += [eligible, difference]
        else:
            cells += [r"\missing", r"\missing"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule\end{tabular}",
        r"\caption{\textbf{Development objective/context study: protocol and completion status.} Eight seed-zero arms train for 30 extra epochs with a frozen Framewise donor. Planned means completed training is unavailable; trained means validated training is complete but planning is incomplete; complete requires all 32 matched development tasks. Validation MSE is the completed-history minimum of the common FP32 recursive criterion over targets 3--7, not a test score. Eligible success excludes support-only successes; $\Delta$ compares the same eligible tasks against the frozen Framewise donor. Bold green marks strictly positive observed differences, not statistical significance. Missing values are unavailable, never zero. Constant-input contexts can learn nonzero responses; nominal parameter matching does not establish equal effective capacity. Both objectives use the first target after support. Planning uses 300 CEM candidates, 30 iterations, 30 elites and the fixed native-action budget. This post-hoc development study does not establish a main-test, multi-seed, efficiency or context-necessity claim.}",
        r"\label{tab:rollout-revision-plan}\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=PROJECT / "paper/generated")
    args = parser.parse_args()
    ledger = collect(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in (("rollout_revision.json", json.dumps(ledger, indent=2, allow_nan=False) + "\n"),
                       ("rollout_revision.tex", render_tex(ledger))):
        path = args.output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
    print(json.dumps({key: ledger[key] for key in ("status", "trained_records", "completed_records", "expected_records")}))


if __name__ == "__main__":
    main()
