#!/usr/bin/env python3
"""Report the post-hoc dynamics revision from completed development evidence.

Reads JSON/JSONL metadata and streams file hashes only. No model tensors are
loaded and no training, inference, or final-test evaluation is invoked. Tensor
reconstruction/frozen-donor equality are the pinned evaluator's preflight;
this reporter verifies its recorded artifacts and recomputes task outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]


def script(name):
    path = PROJECT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("dynamics_reporting_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = script("evaluate_dynamics_revision")
paired = script("summarize_goal_intervention")
training = contract.training
intervention = contract.intervention
from shiftwm.dynamics_revision import PACKAGE_KIND, RevisionConfig, completed_framewise_source


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with path.open("rb") as handle:
        value = hashlib.file_digest(handle, "sha256").hexdigest()
    if identity(before) != identity(path.stat()):
        raise ValueError(f"Source changed while hashing: {path}")
    return value


def rooted(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def scientific_config(config):
    return {k: v for k, v in config.items() if k not in training.OPERATIONAL}


def training_metadata(root, package, selected_epoch):
    """Validate full training metadata without torch.load or model construction."""
    package = Path(package)
    if package.name != "best":
        raise ValueError("Revision result must use a validation-best package")
    output = package.parent
    config, run = read(package / "config.json"), read(output / "run_config.json")
    summary = read(output / "training_summary.json")
    if (type(run.get("epochs")) is not int or run["epochs"] != 30
            or type(run.get("seed")) is not int or run["seed"] != 0
            or summary.get("status") != "completed" or summary.get("completed_epochs") != 30):
        raise ValueError("Revision requires completed seed-zero 30-epoch training")
    if "model" in run or rooted(root, run["output_dir"]) != output.resolve():
        raise ValueError("Revision training configuration/path differs")
    expected_revision = asdict(RevisionConfig(**run.get("revision", {})))
    if (config.get("format_version") != 1 or config.get("package_kind") != PACKAGE_KIND
            or config.get("revision_config") != expected_revision
            or config.get("model_config") != expected_revision):
        raise ValueError("Revision package configuration differs from the training configuration")
    provenance = config["provenance"]
    proxy = SimpleNamespace(provenance=provenance, export_config=lambda: config)
    identity = training.training_identity(run, proxy)
    if summary.get("training_identity") != identity:
        raise ValueError("Revision training identity differs from actual configuration/provenance")
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if any(type(r.get("epoch")) is not int for r in rows) or sorted(r["epoch"] for r in rows) != list(range(1, 31)):
        raise ValueError("Revision requires all 30 unique metric epochs")
    values = [float(r["val"]["prediction_loss"]) for r in rows]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite revision validation metric")
    best = min(values)
    if not math.isclose(best, float(summary["best_validation_prediction_loss"]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Revision best metric differs from completed history")
    best_epochs = [r["epoch"] for r, value in zip(rows, values) if math.isclose(value, best, rel_tol=1e-10, abs_tol=1e-12)]
    if type(selected_epoch) is not int or selected_epoch not in best_epochs:
        raise ValueError("Revision result did not use the validation-best epoch")
    final = next(r for r in rows if r["epoch"] == 30)
    if type(summary.get("step")) is not int or summary["step"] != final.get("step"):
        raise ValueError("Revision final step differs from completed metric history")
    for name in ("best", "last"):
        exported = read(output / name / "config.json")
        metadata = exported.get("metadata", {})
        if (exported.get("format_version") != 1 or exported.get("package_kind") != PACKAGE_KIND
                or exported.get("revision_config") != expected_revision or exported.get("model_config") != expected_revision
                or exported.get("provenance") != provenance or exported.get("donor_config") != config.get("donor_config")
                or metadata.get("training_identity") != identity
                or scientific_config(metadata.get("config", {})) != scientific_config(run)):
            raise ValueError("Revision best/last metadata differs from the completed training identity")
        if not (output / name / "model.pt").is_file():
            raise ValueError("Revision lacks a required completed model artifact")
    return {"config": config, "run_config": run, "summary": summary, "identity": identity,
            "best_epochs": best_epochs, "best_metric": best,
            "metadata_scope": "External JSON/epoch history and streamed hashes; no tensor deserialization."}


def build_expectations(root, environment, record):
    root = Path(root)
    package = rooted(root, record["checkpoint"])
    metadata = training_metadata(root, package, record.get("checkpoint_epoch"))
    config, run = metadata["config"], metadata["run_config"]
    provenance = config["provenance"]
    data = rooted(root, run["data_root"])
    manifest_path = data / "manifest.json"
    manifest, data_sha = read(manifest_path), fingerprint(manifest_path)
    if (manifest.get("environment") != environment or manifest.get("action_block") != 5
            or (environment == "pusht" and manifest.get("action_interface") != "relative")
            or provenance.get("revision_environment") != environment
            or provenance.get("data_manifest_sha256") != data_sha):
        raise ValueError("Revision dataset/family/action identity differs")
    keys = intervention.development_keys(manifest)
    episodes = {r["trajectory_id"]: r for r in manifest["episodes"] if r["split"] == "development"}
    tasks = {key: {"seed": episodes[key[0]]["seed"], "dynamics_id": episodes[key[0]]["dynamics_id"]} for key in keys}
    if len({task["seed"] for task in tasks.values()}) != 32:
        raise ValueError("Development tasks must have 32 independent seed identities")
    donor = completed_framewise_source(rooted(root, run["donor_checkpoint"]))
    donor_export = {k: v for k, v in donor["config"].items() if k not in {"format_version", "metadata"}}
    if (provenance.get("revision_donor", {}).get("hashes") != donor["hashes"]
            or donor["config"]["model_config"].get("freeze_visual") is not True
            or config.get("donor_config") != donor_export):
        raise ValueError("Revision frozen donor configuration or source hashes changed")
    if provenance["revision_donor"].get("epoch") not in donor["best_epochs"]:
        raise ValueError("Revision donor was not validation-best")
    if not math.isclose(provenance["revision_donor"].get("best_metric", math.inf), donor["best_metric"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Revision donor best metric differs")
    for key in ("sequence_length", "stride", "batch_size", "lr", "min_lr", "weight_decay", "grad_clip", "bf16"):
        if run.get(key) != donor["run_config"].get(key):
            raise ValueError(f"Revision changed the donor training setting: {key}")
    if set(run.get("dataset_kwargs", {})) - {"feature_cache", "preload_features"}:
        raise ValueError("Revision dataset overrides differ from the fixed protocol")
    for name, path in (("data_manifest", manifest_path),
                       ("cache_manifest", rooted(root, run["dataset_kwargs"]["feature_cache"]) / "manifest.json"),
                       ("action_stats", rooted(root, run["action_stats"]))):
        digest = fingerprint(path)
        if provenance.get(name + "_sha256") != digest or donor["config"]["provenance"].get(name + "_sha256") != digest:
            raise ValueError(f"Revision/donor {name} source changed")
    cache = read(rooted(root, run["dataset_kwargs"]["feature_cache"]) / "manifest.json")
    if (cache.get("encoder_weights_sha256") != provenance.get("weights_sha256")
            or provenance.get("download", {}).get("repo") != "quentinll/lewm-" + environment):
        raise ValueError("Revision reference coordinates or pretrained family differ")
    implementations = {name: fingerprint(PROJECT / "src/shiftwm" / name)
                      for name in ("dynamics_revision.py", "model.py", "checkpoint.py", "upstream.py", "data.py", "train.py")}
    implementations["train_dynamics_revision.py"] = fingerprint(PROJECT / "scripts/train_dynamics_revision.py")
    if provenance.get("revision_implementation_hashes") != implementations:
        raise ValueError("Revision implementation sources differ from recorded training")
    control_path = root / "results/development_official_budget" / f"{environment}_framewise_s0/planning_development.json"
    control = read(control_path)
    contract.validate_control(control, donor, keys, environment, data_sha)
    identity = {"experiment": "posthoc_development_frozen_framewise_dynamics_revision",
                "checkpoint_sha256": fingerprint(package / "model.pt"),
                "checkpoint_config_sha256": fingerprint(package / "config.json"),
                "training_summary_sha256": fingerprint(package.parent / "training_summary.json"),
                "training_metrics_sha256": fingerprint(package.parent / "metrics.jsonl"),
                "training_identity": metadata["identity"], "donor_source_hashes": donor["hashes"],
                "control_source_sha256": fingerprint(control_path), "data_manifest_sha256": data_sha,
                "evaluator_sha256": fingerprint(PROJECT / "src/shiftwm/evaluate.py"),
                "generation_source_sha256": fingerprint(PROJECT / "src/shiftwm/generate.py"),
                "revision_source_sha256": fingerprint(PROJECT / "src/shiftwm/dynamics_revision.py"),
                "evaluation_script_sha256": fingerprint(PROJECT / "scripts/evaluate_dynamics_revision.py"),
                "protocol_helper_sha256": fingerprint(PROJECT / "scripts/evaluate_goal_calibration_intervention.py"),
                "goal_input": "unchanged_frozen_framewise_available_shifted_goal", "new_training": True}
    return {"identity": identity, "keys": keys, "tasks": tasks, "control": control,
            "training": {"completed_epochs": 30, "selected_epoch": record["checkpoint_epoch"],
                         "best_metric": metadata["best_metric"], "training_identity": metadata["identity"],
                         "run_config_sha256": fingerprint(package.parent / "run_config.json"),
                         "last_config_sha256": fingerprint(package.parent / "last/config.json"),
                         "last_model_sha256": fingerprint(package.parent / "last/model.pt"),
                         "validation_scope": metadata["metadata_scope"]}}


def strict_binary(value, name):
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError(f"Invalid binary outcome: {name}")


def validated_rows(record, expected, environment):
    # Strengthen the shared helper's membership check: floats 0.0/1.0 are not
    # accepted as Boolean task outcomes, and summaries never replace records.
    for row in record["planning"]["records"]:
        for name in ("success", "success_during_context", "policy_eligible", "final_success"):
            strict_binary(row.get(name), name)
    rows = paired.validate_records(record, expected)
    counts = paired.derived_counts(rows)
    planning = record["planning"]
    for key, numerator, denominator in (
            ("success", counts["raw_successes"], counts["raw_total"]),
            ("success_during_context", counts["support_successes"], counts["raw_total"])):
        check_summary(planning["summary"]["all"][key], numerator, denominator)
    eligible = planning.get("eligible_summary", {}).get("all", {}).get("success")
    if counts["eligible_total"]:
        if eligible is None:
            raise ValueError("Missing eligible-success summary")
        check_summary(eligible, counts["eligible_successes"], counts["eligible_total"])
    elif eligible is not None:
        raise ValueError("No eligible tasks: eligible rate must be undefined")
    return rows, counts


def check_summary(summary, numerator, denominator):
    mean = float(summary["mean"])
    if (summary.get("clusters") != denominator or summary.get("observations") != denominator
            or not math.isfinite(mean) or not math.isclose(mean, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Summary disagrees with independently recomputed outcomes/denominators")


def validate_completed(record, environment, expected):
    if (record.get("status") != "complete" or record.get("environment") != environment
            or record.get("kind") != "posthoc_trainable_dynamics_revision"
            or record.get("model_mode") != "framewise_dynamics_revision"
            or type(record.get("training_seed")) is not int or record["training_seed"] != 0):
        raise ValueError("Mislabelled completed development revision")
    if record.get("selected_keys") != [list(k) for k in expected["keys"]]:
        raise ValueError("Declared revision task selection differs")
    intervention.validate_completed_result(record, expected["identity"], expected["keys"])
    revision, counts = validated_rows(record, expected, environment)
    control, control_counts = validated_rows(expected["control"], expected, environment)
    contract.validate_support(list(revision.values()), list(control.values()), complete=True)
    paired.validate_shared_support(revision, control, environment)
    recomputed = contract.paired_counts(list(revision.values()), list(control.values()))
    if record.get("comparison") != recomputed:
        raise ValueError("Stored revision comparison differs from recomputed paired outcomes")
    return {"counts": counts, "control_counts": control_counts,
            "paired": recomputed, "support_pairing": "verified_all_32_tasks",
            "run_identity": expected["identity"], "training": expected["training"],
            "efficiency_claim_eligible": False}


def collect(root=PROJECT, expectation_factory=build_expectations):
    root = Path(root)
    rows = []
    for environment in ("pusht", "reacher"):
        path = root / "results/development_dynamics_revision" / f"{environment}_s0/planning_development.json"
        row = {"environment": environment, "status": "pending", "reason": "result_missing",
               "source_file": str(path), "counts": None, "control_counts": None, "paired": None}
        if path.is_file():
            raw = path.read_bytes()
            record = json.loads(raw)
            row["source_sha256"] = hashlib.sha256(raw).hexdigest()
            planning = record.get("planning", {})
            for split in (planning.get("split"), planning.get("protocol", {}).get("split")):
                if split is not None and split != "development":
                    raise ValueError("Revision reporter refuses non-development outcomes")
            if record.get("status") == "complete":
                expected = expectation_factory(root, environment, record)
                row.update(validate_completed(record, environment, expected), status="complete", reason=None)
            else:
                row["reason"] = "run_incomplete"
        rows.append(row)
    return {"schema_version": 1, "status": "complete" if all(r["status"] == "complete" for r in rows) else "pending",
            "kind": "posthoc_development_dynamics_revision", "rows": rows,
            "completed_records": sum(r["status"] == "complete" for r in rows), "expected_records": 2,
            "reporter_sha256": fingerprint(__file__),
            "protocol": {**intervention.PROTOCOL, "planner": intervention.PLANNER_METADATA},
            "scope": "Post-hoc seed-zero development; frozen framewise calibrator/predictor; learned dynamics residual; extra optimization, not a matched training-budget comparison.",
            "limits": "No factor-identification, context-necessity, main-test, multi-seed, or efficiency claim. Tensor-state correctness is checked by the pinned evaluator preflight; reporting does not deserialize weights."}


def render_tex(ledger):
    lines = [r"% Generated from completed development dynamics-revision evidence only.",
             r"\begin{table}[H]\centering\small", r"\begin{tabular}{llrrrr}\toprule",
             r"Environment & Model & Raw & Support & Eligible & Wins/losses \\\midrule"]
    for row in ledger["rows"]:
        environment = "PushT" if row["environment"] == "pusht" else "Reacher"
        for key, label in (("control_counts", "Framewise calibration"), ("counts", "+ Dynamics residual (ours)")):
            if row["status"] == "complete":
                c = row[key]
                cells = [f"{c['raw_successes']}/{c['raw_total']}", f"{c['support_successes']}/{c['raw_total']}",
                         f"{c['eligible_successes']}/{c['eligible_total']}" if c["eligible_total"] else r"n/a"]
                cells.append((f"{row['paired']['revision_only_successes']}/{row['paired']['control_only_successes']}"
                              if c["eligible_total"] else r"n/a") if key == "counts" else r"---")
            else:
                cells = [r"\textit{pending}"] * 4
            lines.append(" & ".join([environment, label, *cells]) + r" \\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{\textbf{Post-hoc dynamics-residual development experiment.} Starting from the completed seed-0 framewise model, the visual calibrator and predictor stay frozen while a new dynamics-context/action-embedding residual receives 30 additional training epochs. Both policies use the same 32 development tasks and common support, with 300 CEM candidates, 30 iterations and 30 elites. Raw success includes support-only success; eligible success excludes it. Wins/losses count eligible tasks solved only by the revision/control. Dashes for the control denote no self-comparison; pending cells mean incomplete evidence. This uses extra optimization, not matched training budgets, and does not establish context necessity, physical identification, or a main-test result.}",
              r"\label{tab:dynamics-revision}\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ledger = collect(args.root.resolve())
    output = args.output or args.root / "paper/generated"
    paired.atomic_write(output / "dynamics_revision.json", json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    paired.atomic_write(output / "dynamics_revision.tex", render_tex(ledger))
    print(json.dumps({"status": ledger["status"], "completed_records": ledger["completed_records"]}))


if __name__ == "__main__":
    main()
