#!/usr/bin/env python3
"""Source-validated, three-training-seed paired planning comparisons.

Reads JSON metadata and hashes checkpoints without importing inference code or
unpickling weights. No numerical contrast is released from incomplete cells.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = load_module("paired_main_planning_contract", ROOT / "scripts/aggregate_results.py")
campaign = load_module("paired_digest_cache", ROOT / "scripts/run_evaluation_campaign.py")
SEEDS = (0, 1, 2)
MODES = ("factorized", "framewise", "single", "factorized_unpaired")
COMPARATORS = MODES[1:]
ENVIRONMENTS = ("pusht", "reacher")
BOOTSTRAP_SEED = 1701
BOOTSTRAP_REPETITIONS = 20000
CONDITIONS = {"test": tuple((o, d) for o in range(3) for d in range(3)),
              "extrapolation": ((3, 0), (0, 3), (3, 3))}
SUPPORT_FIELDS = {
    "pusht": ("initial_block_translation_error_px", "initial_block_angle_error_rad",
              "initial_agent_position_error_px"),
    "reacher": ("initial_wrapped_joint_error_rad",),
}
UNCERTAINTY = (
    "Percentile 95% paired bootstrap interval over initial-state seed clusters, "
    "resampling every condition and all three training-seed outcomes together; "
    "20,000 draws, RNG1701. Conditional on these three observed training seeds, "
    "not a confidence interval for a population of training runs. Training-seed "
    "variation is reported separately as sample SD (ddof=1). Exploratory "
    "contrasts have no multiplicity adjustment; a degenerate interval does not "
    "establish equivalence."
)
PAIRING_LIMITATION = (
    "Verified: immutable dataset manifest and evaluator identity, exact task "
    "keys, manifest seed/dynamics identity, goal index, support-success and "
    "eligibility flags, and all recorded initial-distance/error diagnostics "
    "(rtol=1e-9, atol=1e-8), across methods and all training seeds. Primary "
    "records do not log support/goal pixel hashes or full simulator-state "
    "hashes; their equality cannot be retrospectively proven from these rows."
)


def read(path):
    return json.loads(Path(path).read_text())


def manifest_tasks(manifest, split):
    """Reproduce the locked evaluator's manifest selection, then validate it."""
    combinations = CONDITIONS[split]
    dynamics = {d for _, d in combinations}
    selected = {d: [] for d in dynamics}
    ids = [ep["trajectory_id"] for ep in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trajectory IDs in dataset manifest")
    for ep in sorted(manifest["episodes"], key=lambda row: (row["seed"], row["dynamics_id"])):
        d = ep["dynamics_id"]
        if ep["split"] == "test" and d in dynamics and len(selected[d]) < 64:
            if ep["steps"] < 7 or not ep.get("sha256"):
                raise ValueError("Manifest task has insufficient goal horizon or missing episode hash")
            selected[d].append(ep)
    seed_sets = [{ep["seed"] for ep in rows} for rows in selected.values()]
    if any(len(rows) != 64 for rows in selected.values()) or any(len(s) != 64 for s in seed_sets):
        raise ValueError("Expected 64 unique test initial-state seeds per dynamics factor")
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError("Manifest dynamics factors do not share the same initial-state seeds")
    return {(ep["trajectory_id"], o): {"seed": ep["seed"], "dynamics_id": d,
                                      "episode_sha256": ep["sha256"]}
            for o, d in combinations for ep in selected[d]}


def completed_source(root, environment, mode, seed, data_sha, digests):
    package = root / "runs/world" / f"{environment}_{mode}_s{seed}" / "best"
    config = read(package / "config.json")
    run = read(package.parent / "run_config.json")
    summary = read(package.parent / "training_summary.json")
    if config["model_config"]["mode"] != mode or run["model"]["mode"] != mode or run["seed"] != seed:
        raise ValueError("Checkpoint training mode/seed differs from expected source")
    if config.get("metadata", {}).get("config") != run:
        raise ValueError("Saved checkpoint run metadata differs from completed run configuration")
    if config["model_config"].get("history_length") != 3:
        raise ValueError("Checkpoint history differs from the locked planning protocol")
    if summary.get("status") != "completed" or summary.get("completed_epochs") != run["epochs"]:
        raise ValueError("Source training is not complete")
    provenance = config["provenance"]
    if provenance.get("data_manifest_sha256") != data_sha:
        raise ValueError("Checkpoint dataset provenance differs")
    if provenance.get("download", {}).get("repo") != "quentinll/lewm-" + environment:
        raise ValueError("Checkpoint pretrained environment differs")
    stats = Path(provenance["action_stats"])
    if not stats.is_absolute():
        stats = root / stats
    if digests.sha256(stats) != provenance["action_stats_sha256"]:
        raise ValueError("Checkpoint action statistics checksum differs")
    metrics = [json.loads(line) for line in (package.parent / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if sorted(row["epoch"] for row in metrics) != list(range(1, run["epochs"] + 1)):
        raise ValueError("Training metric history is incomplete or duplicated")
    values = [float(row["val"]["prediction_loss"]) for row in metrics]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid validation prediction loss")
    best = min(values)
    if not math.isclose(best, summary["best_validation_prediction_loss"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Validation-best metric disagrees with completed summary")
    filenames = {"checkpoint_sha256": package / "model.pt", "config_sha256": package / "config.json",
                 "run_config_sha256": package.parent / "run_config.json",
                 "training_summary_sha256": package.parent / "training_summary.json",
                 "metrics_sha256": package.parent / "metrics.jsonl", "action_stats_sha256": stats}
    return {"path": str(package.resolve()), "training_seed": seed,
            "best_epochs": [row["epoch"] for row, v in zip(metrics, values)
                            if math.isclose(v, best, rel_tol=1e-10, abs_tol=1e-12)],
            "best_validation_prediction_loss": best,
            "hashes": {key: digests.sha256(path) for key, path in filenames.items()},
            "coordinate_identity": {"weights_sha256": provenance["weights_sha256"],
                                    "action_stats_sha256": provenance["action_stats_sha256"],
                                    "action_mean": config["action_mean"], "action_std": config["action_std"],
                                    "base_config": config["base_config"]}}


def validate_records(record, tasks, environment, complete):
    rows = record["planning"]["records"]
    keys = [(row["trajectory_id"], row["observation_id"]) for row in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= tasks.keys():
        raise ValueError("Duplicate or unexpected planning task keys")
    if complete and set(keys) != tasks.keys():
        raise ValueError("Completed result is missing prescribed planning task keys")
    result = {}
    for key, row in zip(keys, rows):
        if any(row.get(name) != tasks[key][name] for name in ("seed", "dynamics_id")):
            raise ValueError("Planning record identity differs from manifest")
        if row.get("goal_index") != 7 or row.get("policy") != "world_model":
            raise ValueError("Planning record goal or policy differs")
        for field in ("success", "final_success", "success_during_context", "policy_eligible"):
            if row.get(field) not in (0, 1):
                raise ValueError(f"Invalid binary outcome: {field}")
        if row["policy_eligible"] != 1 - row["success_during_context"]:
            raise ValueError("Support-success and policy-eligibility flags disagree")
        if row["success"] < row["success_during_context"] or row["final_success"] > row["success"]:
            raise ValueError("Contradictory success flags")
        for field in ("initial_distance_after_context", *SUPPORT_FIELDS[environment]):
            if not math.isfinite(float(row[field])):
                raise ValueError("Nonfinite initial support diagnostic")
        if not isinstance(row.get("native_steps"), int) or not 1 <= row["native_steps"] <= 50:
            raise ValueError("Invalid paid native interaction count")
        if row["success_during_context"] and (row["native_steps"] > 10 or row.get("num_replans") != 0):
            raise ValueError("Support-only success contains policy planning")
        if row["policy_eligible"] and row["native_steps"] <= 10:
            raise ValueError("Policy-eligible task did not execute policy actions")
        result[key] = row
    return result


def validate_result(record, path, environment, mode, seed, split, source, expected):
    if record.get("status") not in ("complete", "interrupted"):
        raise ValueError("Unrecognized evaluation status")
    complete = record["status"] == "complete"
    if record.get("environment") != environment or record.get("model_mode") != mode or record.get("training_seed") != seed:
        raise ValueError("Evaluation environment/method/training seed differs")
    required = {"checkpoint_sha256": source["hashes"]["checkpoint_sha256"],
                "data_manifest_sha256": expected["data_manifest_sha256"],
                "evaluator_sha256": expected["evaluator_sha256"]}
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("Evaluation checkpoint/data/evaluator source differs")
    if record.get("checkpoint_epoch") not in source["best_epochs"]:
        raise ValueError("Evaluation is not the completed validation-best checkpoint epoch")
    planning = record["planning"]
    if planning.get("status") != record["status"] or planning.get("kind") != "closed_loop_goal_image_planning":
        raise ValueError("Planning completion status/kind differs")
    # Interrupted evaluator returns legitimately omit the derived planner summary.
    # Only supply that known metadata for validation of absent partial summaries;
    # any present summary, and every complete result, must validate unchanged.
    checked = record
    if not complete and "planner" not in planning:
        checked = {**record, "planning": {**planning, "planner": {
            **aggregate.MAIN_PLANNER_BUDGET, **aggregate.MAIN_PLANNER_IDENTITY}}}
    elif not isinstance(planning.get("planner"), dict):
        raise ValueError("Missing or malformed planner metadata")
    if not aggregate.validate_main_planning(checked, path, split):
        raise ValueError("Controls do not belong in paired learned-method comparisons")
    return validate_records(record, expected["tasks"][split], environment, complete)


def validate_shared_support(left, right, environment):
    """Compare every common row, including available interrupted-run records."""
    exact = ("seed", "dynamics_id", "goal_index", "success_during_context", "policy_eligible")
    numeric = ("initial_distance_after_context", *SUPPORT_FIELDS[environment])
    for key in left.keys() & right.keys():
        if any(left[key][name] != right[key][name] for name in exact):
            raise ValueError("Paired task/support identity differs across methods or training seeds")
        if any(not math.isclose(float(left[key][name]), float(right[key][name]), rel_tol=1e-9, abs_tol=1e-8)
               for name in numeric):
            raise ValueError("Paired initial support diagnostics differ across methods or training seeds")


def paired_statistics(pairs, eligible=False, draws=BOOTSTRAP_REPETITIONS):
    """Conditional trajectory-cluster bootstrap, with training seeds kept fixed."""
    if set(pairs) != set(SEEDS):
        raise ValueError("Numerical contrasts require all three training seeds")
    reference = pairs[0][0]
    keys = sorted(key for key, row in reference.items() if not eligible or row["policy_eligible"])
    if not keys:
        return None
    clusters = sorted({reference[key]["seed"] for key in keys})
    positions = {seed: i for i, seed in enumerate(clusters)}
    numerators, denominators = np.zeros(len(clusters)), np.zeros(len(clusters))
    per_seed = []
    for seed in SEEDS:
        left, right = pairs[seed]
        if left.keys() != reference.keys() or right.keys() != reference.keys():
            raise ValueError("Paired populations differ")
        a = np.asarray([left[key]["success"] for key in keys], dtype=np.int64)
        b = np.asarray([right[key]["success"] for key in keys], dtype=np.int64)
        delta = a - b
        for key, difference in zip(keys, delta):
            i = positions[reference[key]["seed"]]
            numerators[i] += difference
            denominators[i] += 1
        per_seed.append({"training_seed": seed, "n": len(keys),
                         "factorized_successes": int(a.sum()), "comparator_successes": int(b.sum()),
                         "wins": int(((a == 1) & (b == 0)).sum()),
                         "losses": int(((a == 0) & (b == 1)).sum()),
                         "both_success": int(((a == 1) & (b == 1)).sum()),
                         "neither_success": int(((a == 0) & (b == 0)).sum()),
                         "paired_difference_pp": float(delta.mean() * 100)})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(clusters), size=(draws, len(clusters)))
    distribution = 100 * numerators[sampled].sum(axis=1) / denominators[sampled].sum(axis=1)
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    differences = [row["paired_difference_pp"] for row in per_seed]
    return {"mean_difference_pp": statistics.mean(differences),
            "training_seed_sd_pp": statistics.stdev(differences),
            "conditional_ci95_pp": [float(lo), float(hi)],
            "degenerate_interval": bool(lo == hi), "n_tasks_per_training_seed": len(keys),
            "n_initial_state_clusters": len(clusters), "per_training_seed": per_seed}


def build_report(root=ROOT):
    root = Path(root).resolve()
    digests = campaign.FileDigestCache()
    loaded, sources, training_sources, data_sources = {}, {}, {}, {}
    owners = {}
    evaluator_sha = digests.sha256(root / "src/shiftwm/evaluate.py")
    for environment in ENVIRONMENTS:
        data_name = "pusht_relative" if environment == "pusht" else environment
        manifest_path = root / "data/world" / data_name / "manifest.json"
        manifest = read(manifest_path)
        if manifest["environment"] != environment or manifest["action_block"] != 5:
            raise ValueError("Dataset environment or native action grouping differs")
        data_sha = digests.sha256(manifest_path)
        expected = {"tasks": {split: manifest_tasks(manifest, split) for split in CONDITIONS},
                    "data_manifest_sha256": data_sha, "evaluator_sha256": evaluator_sha}
        data_sources[environment] = {"path": str(manifest_path), "sha256": data_sha,
                                     "tasks_by_split": {s: len(t) for s, t in expected["tasks"].items()}}
        coordinates = None
        support = {split: {} for split in CONDITIONS}
        for mode in MODES:
            for seed in SEEDS:
                source = None
                run_id = f"{environment}_{mode}_s{seed}"
                for split in CONDITIONS:
                    path = root / "results/world" / run_id / f"planning_{split}.json"
                    if not path.exists():
                        continue
                    if source is None:
                        source = completed_source(root, environment, mode, seed, data_sha, digests)
                        checkpoint_sha = source["hashes"]["checkpoint_sha256"]
                        if checkpoint_sha in owners and owners[checkpoint_sha] != run_id:
                            raise ValueError("Identical checkpoint reused across distinct trained runs")
                        owners[checkpoint_sha] = run_id
                        if coordinates is not None and source["coordinate_identity"] != coordinates:
                            raise ValueError("Checkpoint action or latent coordinate systems differ")
                        coordinates = source["coordinate_identity"]
                        training_sources[run_id] = source
                    raw = path.read_bytes()
                    record = json.loads(raw)
                    rows = validate_result(record, path, environment, mode, seed, split, source, expected)
                    validate_shared_support(support[split], rows, environment)
                    for key, row in rows.items():
                        support[split].setdefault(key, row)
                    loaded[environment, mode, seed, split] = {"status": record["status"], "rows": rows}
                    sources[str(path.relative_to(root))] = {
                        "sha256": hashlib.sha256(raw).hexdigest(), "status": record["status"],
                        "checkpoint_sha256": record["checkpoint_sha256"], "records": len(rows),
                        "data_manifest_sha256": data_sha, "evaluator_sha256": evaluator_sha}
    results = []
    for environment in ENVIRONMENTS:
        for split in CONDITIONS:
            for comparator in COMPARATORS:
                availability = {mode: {str(seed): loaded.get((environment, mode, seed, split), {}).get("status", "missing")
                                       for seed in SEEDS} for mode in ("factorized", comparator)}
                complete = all(status == "complete" for states in availability.values() for status in states.values())
                pairs = {}
                if complete:
                    for seed in SEEDS:
                        def population(mode):
                            rows = loaded[environment, mode, seed, split]["rows"]
                            return {key: row for key, row in rows.items() if split == "extrapolation"
                                    or (row["observation_id"], row["dynamics_id"]) == (2, 2)}
                        pairs[seed] = population("factorized"), population(comparator)
                for eligible in (False, True):
                    metrics = paired_statistics(pairs, eligible=eligible) if complete else None
                    results.append({"environment": environment, "population": "heldout_o2_d2" if split == "test" else "extrapolation_all3",
                                    "split": split, "comparison": f"factorized_minus_{comparator}",
                                    "comparator": comparator, "metric": "policy_eligible_success" if eligible else "raw_success",
                                    "status": "complete" if metrics is not None else "pending",
                                    "pending_reason": None if metrics is not None else (
                                        "no_policy_eligible_tasks" if complete else "requires_all_three_training_seeds_for_both_methods"),
                                    "required_training_seeds": list(SEEDS), "availability": availability, "metrics": metrics})
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "reporter_sha256": digests.sha256(Path(__file__)),
            "validator_sha256": digests.sha256(ROOT / "scripts/aggregate_results.py"),
            "main_planner_budget": aggregate.MAIN_PLANNER_BUDGET,
            "main_planner_identity": aggregate.MAIN_PLANNER_IDENTITY,
            "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS,
                          "cluster": "manifest initial-state seed; all conditions and all fixed training seeds together",
                          "eligible_zero_count_clusters": "omit clusters containing no eligible task"},
            "uncertainty": UNCERTAINTY, "pairing_checks_and_limitations": PAIRING_LIMITATION,
            "selection": "Completed validation-best checkpoints; no test outcome used to select a method or checkpoint.",
            "missing_policy": "All six completed source files are required per comparison/population; no numerical partial contrast.",
            "data_sources": data_sources, "training_sources": training_sources,
            "evaluation_sources": sources, "results": results}


def render_tex(report):
    labels = {"framewise": "Framewise calibration", "single": "Shared context", "factorized_unpaired": "Unpaired contexts"}
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{lllrr}\toprule",
             r"Environment & Population & Comparator & Raw $\Delta$ (pp) & Eligible $\Delta$ (pp)\\\midrule"]
    def cell(row):
        if row["metrics"] is None:
            return "--"
        m = row["metrics"]
        lo, hi = m["conditional_ci95_pp"]
        difference = f"{m['mean_difference_pp']:+.1f}"
        if float(difference) > 0:
            difference = r"\positivegain{" + difference + "}"
        return (r"\shortstack[r]{" + difference + f" $\\pm$ {m['training_seed_sd_pp']:.1f}"
                + r"\\{}" + f"[{lo:+.1f}, {hi:+.1f}]" + "}")
    lookup = {(r["environment"], r["split"], r["comparator"], r["metric"]): r for r in report["results"]}
    for env in ENVIRONMENTS:
        for split in CONDITIONS:
            for comparator in COMPARATORS:
                values = ["PushT" if env == "pusht" else "Reacher", "Held-out" if split == "test" else "Extrap.", labels[comparator]]
                values += [cell(lookup[env, split, comparator, metric]) for metric in ("raw_success", "policy_eligible_success")]
                lines.append(" & ".join(values) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Exploratory paired planning contrasts: ShiftWM (ours) minus comparator, in percentage points. Entries are mean $\pm$ sample SD across the three fixed training seeds, with conditional 95\% initial-state-cluster bootstrap intervals on the second line (20,000 draws). Green bold means indicate positive differences against the named comparator, not statistical significance. Each resampled cluster retains all conditions and all three training seeds. Eligible success excludes tasks solved during paid support acquisition. Dashes require further completed sources; no partial-seed contrasts are shown. Intervals do not capture population-level training uncertainty, are not multiplicity-adjusted, and a degenerate interval does not demonstrate equivalence.}",
              r"\label{tab:paired-planning}\end{table}"]
    return "\n".join(lines) + "\n"


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = build_report(args.root)
    atomic_text(args.root / "reports/evidence/paired_planning_results.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_text(args.root / "paper/generated/paired_planning.tex", render_tex(report))
    print(json.dumps({"source_files": len(report["evaluation_sources"]), "complete_cells": sum(r["status"] == "complete" for r in report["results"]),
                      "total_cells": len(report["results"])}))


if __name__ == "__main__":
    main()
