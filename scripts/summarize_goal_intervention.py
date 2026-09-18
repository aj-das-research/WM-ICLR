#!/usr/bin/env python3
"""Deterministic, strictly development-only reporting for the goal-only hybrid.

No model inference or main-test results are read. Missing/incomplete evaluation
files remain pending. Counts are recomputed from records, and uncertainty is a
paired development-trajectory bootstrap at fixed training seed0.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("goal_intervention_reporting_contract",
                                            ROOT / "scripts/evaluate_goal_calibration_intervention.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
MODES = ("factorized", "single", "framewise", "hybrid")
LABELS = {"factorized": "ShiftWM (ours)", "single": "Shared context",
          "framewise": "Framewise calibration", "hybrid": "Goal-only hybrid (post hoc)"}
BOOTSTRAP_SEED = 1701
BOOTSTRAP_REPETITIONS = 20000


def fingerprint(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_expectations(root, environment):
    """Read training/source metadata and hashes, never unpickle model weights."""
    root = Path(root)
    data_name = "pusht_relative" if environment == "pusht" else "reacher"
    manifest_path = root / "data/world" / data_name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["environment"] != environment or manifest["action_block"] != 5:
        raise ValueError("Dataset environment/action grouping differs")
    keys = contract.development_keys(manifest)
    data_sha = fingerprint(manifest_path)
    sources = {mode: contract.completed_source(root / "runs/world" / f"{environment}_{mode}_s0" / "best",
                                               mode, data_sha, environment)
               for mode in ("factorized", "single", "framewise")}
    code = {"intervention_source_sha256": fingerprint(ROOT / "scripts/evaluate_goal_calibration_intervention.py"),
            "evaluator_sha256": fingerprint(ROOT / "src/shiftwm/evaluate.py"),
            "model_source_sha256": fingerprint(ROOT / "src/shiftwm/model.py"),
            "generation_source_sha256": fingerprint(ROOT / "src/shiftwm/generate.py")}
    identity = {"experiment": "posthoc_development_factorized_with_learned_framewise_goal",
                "fixed_checkpoint_sha256": sources["factorized"]["hashes"]["model_sha256"],
                "donor_checkpoint_sha256": sources["framewise"]["hashes"]["model_sha256"],
                "fixed_source_hashes": sources["factorized"]["hashes"],
                "donor_source_hashes": sources["framewise"]["hashes"],
                "data_manifest_sha256": data_sha, **code,
                "goal_input": "available_shifted_goal_image_only", "new_training": False}
    episodes = {row["trajectory_id"]: row for row in manifest["episodes"] if row["split"] == "development"}
    tasks = {key: {"seed": episodes[key[0]]["seed"], "dynamics_id": episodes[key[0]]["dynamics_id"]} for key in keys}
    return {"keys": keys, "tasks": tasks, "data_manifest_sha256": data_sha, "sources": sources,
            "evaluator_sha256": code["evaluator_sha256"], "hybrid_identity": identity}


def validate_protocol(planning, identity, evaluator_sha256, require_complete):
    if require_complete and planning.get("status") != "complete":
        raise ValueError("Outer completion status differs from planning completion")
    if planning.get("split") != "development" or planning.get("policy") != "world_model":
        raise ValueError("Invalid development planning split/policy")
    # The evaluator's interrupted return contains the complete hashed protocol
    # but no derived planner summary. Validate every present summary, and always
    # require it for completed outputs; partial runs still pass all checks below.
    if require_complete or planning.get("status") != "interrupted" or "planner" in planning:
        if not isinstance(planning.get("planner"), dict):
            raise ValueError("Changed development planner metadata: missing or invalid summary")
        for key, value in contract.PLANNER_METADATA.items():
            if planning["planner"].get(key) != value:
                raise ValueError(f"Changed development planner metadata: {key}")
    protocol = planning["protocol"]
    for key, value in contract.PROTOCOL.items():
        if protocol.get(key) != value:
            raise ValueError(f"Changed development protocol: {key}")
    if protocol.get("search_coordinates") != contract.SEARCH_COORDINATES:
        raise ValueError("Changed development search coordinates")
    if protocol.get("run_identity") != identity or protocol.get("evaluator_sha256") != evaluator_sha256:
        raise ValueError("Development protocol source identity differs")
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if planning.get("signature") != signature:
        raise ValueError("Changed development protocol signature")


def validate_regular(record, environment, mode, expected):
    if record.get("environment") != environment or record.get("model_mode") != mode or record.get("training_seed") != 0:
        raise ValueError("Mislabelled development environment/method/training seed")
    source = expected["sources"][mode]
    checks = {"checkpoint_sha256": source["hashes"]["model_sha256"],
              "data_manifest_sha256": expected["data_manifest_sha256"],
              "evaluator_sha256": expected["evaluator_sha256"]}
    if any(record.get(key) != value for key, value in checks.items()):
        raise ValueError("Changed development checkpoint/data/evaluator source")
    if record.get("checkpoint_epoch") not in source["best_epochs"]:
        raise ValueError("Development result did not use the completed validation-best epoch")
    identity = {key: checks[key] for key in ("checkpoint_sha256", "data_manifest_sha256")}
    validate_protocol(record["planning"], identity, checks["evaluator_sha256"], record.get("status") == "complete")


def validate_records(record, expected):
    rows = record["planning"]["records"]
    keys = [(row["trajectory_id"], row["observation_id"]) for row in rows]
    if len(keys) != 32 or len(keys) != len(set(keys)) or set(keys) != set(expected["keys"]):
        raise ValueError("Different or duplicate development trajectory keys")
    result = {}
    for key, row in zip(keys, rows):
        if row.get("seed") != expected["tasks"][key]["seed"] or row.get("dynamics_id") != expected["tasks"][key]["dynamics_id"]:
            raise ValueError("Development trajectory identity differs from the manifest")
        if row.get("goal_index") != 7 or row.get("policy") != "world_model":
            raise ValueError("Development record goal or policy differs")
        for name in ("success", "success_during_context", "policy_eligible", "final_success"):
            if row.get(name) not in (0, 1):
                raise ValueError(f"Invalid binary outcome: {name}")
        if row["policy_eligible"] != 1 - row["success_during_context"]:
            raise ValueError("Support-success and policy-eligibility classification disagree")
        if row["success"] < row["success_during_context"] or row["final_success"] > row["success"]:
            raise ValueError("Contradictory success classifications")
        if not math.isfinite(float(row["initial_distance_after_context"])):
            raise ValueError("Nonfinite initial support distance")
        if not 0 < row["native_steps"] <= 50:
            raise ValueError("Native action budget violated")
        if row["success_during_context"] and (row["native_steps"] > 10 or row["num_replans"] != 0):
            raise ValueError("Support-only success includes policy execution")
        result[key] = row
    return result


def validate_shared_support(left, right, environment):
    if set(left) != set(right):
        raise ValueError("Paired methods have different development keys")
    errors = ["block_translation_error_px", "block_angle_error_rad", "agent_position_error_px"] if environment == "pusht" else ["wrapped_joint_error_rad"]
    for key in left:
        a, b = left[key], right[key]
        for name in ("seed", "dynamics_id", "goal_index", "success_during_context", "policy_eligible"):
            if a[name] != b[name]:
                raise ValueError(f"Different initial support classification/identity: {name}, {key}")
        for name in ["initial_distance_after_context", *("initial_" + error for error in errors)]:
            values = float(a[name]), float(b[name])
            if not all(map(math.isfinite, values)) or not math.isclose(*values, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"Different initial support state: {name}, {key}")


def derived_counts(rows):
    values = list(rows.values())
    eligible = [row for row in values if row["policy_eligible"]]
    return {"raw_successes": sum(row["success"] for row in values), "raw_total": len(values),
            "support_successes": sum(row["success_during_context"] for row in values),
            "eligible_successes": sum(row["success"] for row in eligible), "eligible_total": len(eligible)}


def paired_outcomes(hybrid, baseline):
    keys = sorted(key for key in hybrid if hybrid[key]["policy_eligible"])
    if not keys:
        return {"status": "pending", "reason": "no_policy_eligible_trajectories", "eligible_total": 0,
                "success_difference": None, "ci95": None}
    pairs = [{"trajectory_id": key[0], "observation_id": key[1], "seed": hybrid[key]["seed"],
              "hybrid_success": hybrid[key]["success"], "baseline_success": baseline[key]["success"],
              "difference": hybrid[key]["success"] - baseline[key]["success"]} for key in keys]
    seeds = sorted({row["seed"] for row in pairs})
    sums = np.array([sum(row["difference"] for row in pairs if row["seed"] == seed) for seed in seeds])
    counts = np.array([sum(row["seed"] == seed for row in pairs) for seed in seeds])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selected = rng.integers(0, len(seeds), size=(BOOTSTRAP_REPETITIONS, len(seeds)))
    draws = sums[selected].sum(1) / counts[selected].sum(1)
    return {"status": "complete", "eligible_total": len(keys), "clusters": len(seeds),
            "hybrid_only_successes": sum(row["difference"] == 1 for row in pairs),
            "baseline_only_successes": sum(row["difference"] == -1 for row in pairs),
            "both_successes": sum(row["hybrid_success"] == row["baseline_success"] == 1 for row in pairs),
            "neither_successes": sum(row["hybrid_success"] == row["baseline_success"] == 0 for row in pairs),
            "success_difference": float(sums.sum() / counts.sum()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "pairs": pairs}


def load_result(path, environment, mode, expected):
    path = Path(path)
    metadata = {"environment": environment, "mode": mode, "label": LABELS[mode], "source": str(path.resolve())}
    if not path.exists():
        return {**metadata, "status": "pending", "reason": "result_missing", "counts": None}, None
    raw = path.read_bytes()
    record = json.loads(raw)
    metadata["source_sha256"] = hashlib.sha256(raw).hexdigest()
    planning = record.get("planning", {})
    for value in (planning.get("split"), planning.get("protocol", {}).get("split")):
        if value is not None and value != "development":
            raise ValueError("Reporter refuses non-development evaluation data")
    if mode == "hybrid":
        if record.get("environment") != environment or record.get("model_mode") != "factorized_with_framewise_goal":
            raise ValueError("Mislabelled hybrid environment/method")
        if record.get("run_identity") != expected["hybrid_identity"]:
            raise ValueError("Changed hybrid sources")
        validate_protocol(record["planning"], expected["hybrid_identity"], expected["evaluator_sha256"], record.get("status") == "complete")
        if record.get("fixed_epoch") not in expected["sources"]["factorized"]["best_epochs"] or record.get("donor_epoch") not in expected["sources"]["framewise"]["best_epochs"]:
            raise ValueError("Hybrid source epochs differ from completed validation-best sources")
        if record.get("status") == "complete":
            contract.validate_completed_result(record, expected["hybrid_identity"], expected["keys"])
    else:
        validate_regular(record, environment, mode, expected)
    if record.get("status") != "complete":
        return {**metadata, "status": "pending", "reason": "run_incomplete", "counts": None}, None
    rows = validate_records(record, expected)
    return {**metadata, "status": "complete", "counts": derived_counts(rows)}, rows


def build_report(root=ROOT, expectation_factory=build_expectations):
    root = Path(root)
    methods, comparisons, provenance = [], [], {}
    for environment in ("pusht", "reacher"):
        expected = expectation_factory(root, environment)
        provenance[environment] = {"data_manifest_sha256": expected["data_manifest_sha256"],
                                   "evaluator_sha256": expected["evaluator_sha256"],
                                   "selected_keys": [list(key) for key in expected["keys"]],
                                   "hybrid_identity": expected["hybrid_identity"],
                                   "regular_source_hashes": {mode: source["hashes"] for mode, source in expected["sources"].items()}}
        loaded = {}
        for mode in MODES:
            if mode == "hybrid":
                path = root / "results/development_goal_intervention" / f"{environment}_factorized_s0_framewise_goal_s0" / "planning_development.json"
            else:
                path = root / "results/development_official_budget" / f"{environment}_{mode}_s0" / "planning_development.json"
            metadata, loaded[mode] = load_result(path, environment, mode, expected)
            methods.append(metadata)
        available = [mode for mode in MODES if loaded[mode] is not None]
        if available:
            anchor = loaded[available[0]]
            for mode in available[1:]:
                validate_shared_support(anchor, loaded[mode], environment)
        for baseline in ("factorized", "framewise"):
            if loaded["hybrid"] is None or loaded[baseline] is None:
                outcome = {"status": "pending", "reason": "paired_run_missing_or_incomplete",
                           "success_difference": None, "ci95": None}
            else:
                outcome = paired_outcomes(loaded["hybrid"], loaded[baseline])
            comparisons.append({"environment": environment, "comparison": "hybrid_minus_" + baseline, **outcome})
    complete = all(row["status"] == "complete" for row in methods + comparisons)
    return {"schema_version": 1, "status": "complete" if complete else "pending",
            "scope": "exploratory post-hoc development comparison at fixed training seed0; no main-test outcomes",
            "reporter_source_sha256": fingerprint(__file__),
            "protocol": {**contract.PROTOCOL, "planner": contract.PLANNER_METADATA},
            "uncertainty": {"method": "paired trajectory/seed-cluster percentile bootstrap",
                            "repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED,
                            "unit": "eligible success-rate difference, hybrid minus baseline",
                            "limitations": "Development trajectories only, one training seed; no confirmatory or training-seed robustness claim; degenerate intervals do not prove equivalence."},
            "provenance": provenance, "methods": methods, "comparisons": comparisons}


def render_tex(report):
    tex = [r"% Generated by scripts/summarize_goal_intervention.py; development only.",
           r"\begin{table}[t]\centering\small", r"\begin{tabular}{llrrr}\toprule",
           r"Environment & Method & Raw & Support & Eligible \\\midrule"]
    for row in report["methods"]:
        environment = "PushT" if row["environment"] == "pusht" else "Reacher"
        if row["status"] == "complete":
            count = row["counts"]
            cells = [f"{count['raw_successes']}/{count['raw_total']}",
                     f"{count['support_successes']}/{count['raw_total']}",
                     f"{count['eligible_successes']}/{count['eligible_total']}"]
        else:
            cells = [r"\textit{pending}"] * 3
        tex.append(" & ".join([environment, row["label"], *cells]) + r" \\")
    tex += [r"\bottomrule\end{tabular}",
            r"\caption{Exploratory development results at training seed 0. The post-hoc hybrid changes only the fixed ShiftWM model's learned goal calibration. Raw success includes support-only successes; eligible success excludes them. All methods use the same 32 development tasks and 300/30/30 CEM budget. Pending runs are not zero-valued outcomes.}",
            r"\label{tab:goal_intervention_counts}\end{table}",
            r"\begin{table}[t]\centering\small", r"\begin{tabular}{llrrr}\toprule",
            r"Environment & Paired comparison & Wins/losses & $\Delta$ (pp) & 95\% interval (pp) \\\midrule"]
    for row in report["comparisons"]:
        environment = "PushT" if row["environment"] == "pusht" else "Reacher"
        baseline = "ShiftWM (ours)" if row["comparison"] == "hybrid_minus_factorized" else "Framewise calibration"
        if row["status"] == "complete":
            difference = f"{100 * row['success_difference']:+.1f}"
            if float(difference) > 0:
                difference = r"\positivegain{" + difference + "}"
            cells = [f"{row['hybrid_only_successes']}/{row['baseline_only_successes']}",
                     difference,
                     f"[{100 * row['ci95'][0]:+.1f}, {100 * row['ci95'][1]:+.1f}]"]
        else:
            cells = [r"\textit{pending}"] * 3
        tex.append(" & ".join([environment, r"Hybrid $-$ " + baseline, *cells]) + r" \\")
    tex += [r"\bottomrule\end{tabular}",
            r"\caption{Paired eligible-success differences for the post-hoc hybrid. Green bold means indicate positive differences against the named comparator, not statistical significance. Wins/losses count discordant task outcomes. Intervals resample paired development trajectories (20,000 draws, seed 1701); they do not measure variation across training seeds and are exploratory, not confirmatory tests.}",
            r"\label{tab:goal_intervention_paired}\end{table}"]
    return "\n".join(tex) + "\n"


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tex", type=Path)
    args = parser.parse_args()
    report = build_report(args.root)
    atomic_write(args.output_json or args.root / "reports/evidence/goal_intervention_results.json",
                 json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_write(args.output_tex or args.root / "paper/generated/goal_intervention.tex", render_tex(report))
    print(json.dumps({"status": report["status"], "completed_methods": sum(row["status"] == "complete" for row in report["methods"]),
                      "completed_comparisons": sum(row["status"] == "complete" for row in report["comparisons"])}))


if __name__ == "__main__":
    main()
