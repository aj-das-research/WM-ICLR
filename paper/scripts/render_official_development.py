#!/usr/bin/env python3
"""Render completed 32-task official-budget development rows, never final-test data."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
from run_evaluation_campaign import OFFICIAL_PLANNER_BUDGET, validate_completed_planning

MODES = (("frozen", "Frozen LeWM"), ("single", "Shared context"),
         ("factorized", "ShiftWM (ours)"), ("framewise", "Framewise calibration"))
ENVIRONMENTS = (("pusht", "pusht_relative"), ("reacher", "reacher"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def binary(value, name):
    if type(value) not in (bool, int) or value not in (0, 1):
        raise ValueError(f"{name} must be binary")
    return int(value)


def check_summary(item, numerator, denominator, name):
    if item.get("observations") != denominator or item.get("clusters") != denominator:
        raise ValueError(f"{name} summary denominator differs from independent task count")
    mean = float(item["mean"])
    if not math.isfinite(mean) or not math.isclose(mean, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} summary differs from per-task outcomes")


def validate_record(record, environment, mode, expected_tasks, manifest_sha256):
    if record.get("status") != "complete":
        return None
    if record.get("environment") != environment or record.get("model_mode") != mode:
        raise ValueError("Development environment/model label mismatch")
    if record.get("training_seed") != (None if mode == "frozen" else 0):
        raise ValueError("Development comparison requires the prespecified seed-zero models")
    if record.get("data_manifest_sha256") != manifest_sha256:
        raise ValueError("Development data manifest mismatch")
    for key in ("checkpoint_sha256", "evaluator_sha256"):
        if len(record.get(key, "")) != 64:
            raise ValueError(f"Missing or invalid {key}")
    task = {"policy": "world_model", "split": "development", "planner_budget": OFFICIAL_PLANNER_BUDGET}
    validate_completed_planning(record, task, 32)
    planning = record["planning"]
    if planning.get("status") != "complete":
        raise ValueError("Top-level completion differs from planning completion")
    records = planning["records"]
    if len(records) != 32 or len(expected_tasks) != 32:
        raise ValueError("Development comparison requires exactly 32 prespecified tasks")
    actual_tasks = {r["trajectory_id"]: r["seed"] for r in records}
    if len(actual_tasks) != 32 or actual_tasks != expected_tasks:
        raise ValueError("Development record does not contain the exact 32 manifest seed/trajectory identities")
    if len(set(actual_tasks.values())) != 32:
        raise ValueError("Duplicate development seeds are not independent tasks")
    paired, raw_successes, support_successes, eligible_successes = {}, 0, 0, 0
    for row in records:
        if (row["observation_id"], row["dynamics_id"], row["goal_index"], row["policy"]) != (1, 1, 7, "world_model"):
            raise ValueError("Development factor, goal or policy identity differs")
        success = binary(row["success"], "success")
        support = binary(row["success_during_context"], "success_during_context")
        eligible = binary(row["policy_eligible"], "policy_eligible")
        if eligible != 1 - support or support > success:
            raise ValueError("Policy-eligible/support masks or success denominator are inconsistent")
        distance = float(row["initial_distance_after_context"])
        if not math.isfinite(distance): raise ValueError("Nonfinite post-support distance")
        paired[row["trajectory_id"]] = {"seed": row["seed"], "goal_index": row["goal_index"],
            "support_success": support, "policy_eligible": eligible, "initial_distance_after_context": distance}
        raw_successes += success
        support_successes += support
        eligible_successes += success * eligible
    eligible_tasks = 32 - support_successes
    check_summary(planning["summary"]["all"]["success"], raw_successes, 32, "Raw success")
    check_summary(planning["summary"]["all"]["success_during_context"], support_successes, 32, "Support success")
    eligible_summary = planning.get("eligible_summary", {}).get("all", {}).get("success")
    if eligible_tasks:
        if eligible_summary is None: raise ValueError("Missing policy-eligible summary")
        check_summary(eligible_summary, eligible_successes, eligible_tasks, "Policy-eligible success")
    elif eligible_summary is not None:
        raise ValueError("No eligible tasks: a numerical eligible rate is undefined")
    return {"environment": environment, "mode": mode, "status": "complete", "tasks": 32,
        "raw_successes": raw_successes, "support_successes": support_successes,
        "eligible_successes": eligible_successes, "eligible_tasks": eligible_tasks,
        "training_seed": record["training_seed"], "checkpoint_epoch": record["checkpoint_epoch"],
        "checkpoint_sha256": record["checkpoint_sha256"], "data_manifest_sha256": manifest_sha256,
        "evaluator_sha256": record["evaluator_sha256"], "paired_tasks": paired,
        "execution_context": record.get("execution_context"), "efficiency_claim_eligible": False}


def verify_pairing(rows):
    audit = {}
    for environment, _ in ENVIRONMENTS:
        available = [r for r in rows if r["environment"] == environment and r["status"] == "complete"]
        if not available:
            audit[environment] = {"status": "pending", "completed_modes": []}
            continue
        reference = available[0]["paired_tasks"]
        for row in available[1:]:
            if row["paired_tasks"].keys() != reference.keys(): raise ValueError("Paired development task identities differ")
            for identity, expected in reference.items():
                actual = row["paired_tasks"][identity]
                for key in ("seed", "goal_index", "support_success", "policy_eligible"):
                    if actual[key] != expected[key]: raise ValueError(f"Common support/goal mask differs: {environment}/{identity}/{key}")
                if not math.isclose(actual["initial_distance_after_context"], expected["initial_distance_after_context"], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError(f"Post-support state distance differs: {environment}/{identity}")
        audit[environment] = {"status": "verified_available_modes" if len(available) >= 2 else "awaiting_second_mode",
            "completed_modes": [r["mode"] for r in available], "tasks": 32,
            "all_four_modes_available": len(available) == len(MODES),
            "common_support_mask_sha256": sha(json.dumps(reference, sort_keys=True).encode())}
    return audit


def collect(root):
    rows, manifests = [], {}
    for environment, data_name in ENVIRONMENTS:
        manifest_path = root / f"data/world/{data_name}/manifest.json"
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        expected = {ep["trajectory_id"]: ep["seed"] for ep in manifest["episodes"]
                    if ep["split"] == "development" and ep["dynamics_id"] == 1}
        if len(expected) != 32: raise ValueError("Manifest must define exactly 32 development tasks at dynamics 1")
        manifests[environment] = {"file": str(manifest_path.relative_to(root)), "sha256": sha(raw)}
        for mode, label in MODES:
            path = root / f"results/development_official_budget/{environment}_{mode}_s0/planning_development.json"
            row = {"environment": environment, "mode": mode, "label": label, "status": "pending", "tasks": 32}
            if path.is_file():
                raw = path.read_bytes()
                validated = validate_record(json.loads(raw), environment, mode, expected, manifests[environment]["sha256"])
                if validated is not None:
                    row.update(validated)
                    row.update(source_file=str(path.relative_to(root)), source_sha256=sha(raw))
            rows.append(row)
    versions = {r["evaluator_sha256"] for r in rows if r["status"] == "complete"}
    if len(versions) > 1: raise ValueError("Mixed evaluator versions in official development comparison")
    pairs = verify_pairing(rows)
    return {"schema_version": 1, "kind": "official_budget_development_comparison", "rows": rows,
        "completed_records": sum(r["status"] == "complete" for r in rows), "expected_records": 8,
        "planner": OFFICIAL_PLANNER_BUDGET, "manifests": manifests, "paired_support_verification": pairs,
        "conclusions": "pending_matched_comparisons; one trained seed does not establish a factorization benefit",
        "timing_policy": "No development timing is promoted to main efficiency results",
        "scope": "Only development_official_budget planning records and development manifest identities are read; no final-test results"}


def render_tex(ledger):
    text = [r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{llccc}\toprule",
        r"Environment & Model & Raw success & During support & Policy-eligible\\\midrule"]
    for row in ledger["rows"]:
        counts = ([f"{row['raw_successes']}/32", f"{row['support_successes']}/32",
                   f"{row['eligible_successes']}/{row['eligible_tasks']}" if row["eligible_tasks"] else r"n/a (0 eligible)"]
                  if row["status"] == "complete" else [r"\missing"] * 3)
        text.append(" & ".join(["PushT" if row["environment"] == "pusht" else "Reacher", row["label"], *counts]) + r"\\")
    text += [r"\bottomrule\end{tabular}",
        r"\caption{\textbf{Official-budget development comparison, separate from the main test.} Each completed row uses all 32 prespecified development tasks at $(v_1,p_1)$, with 300 CEM candidates, 30 iterations and 30 elites. Raw and support rates use 32 tasks; policy-eligible rates exclude tasks already solved during common support. Trained models use seed 0; frozen LeWM has no new training seed. Dashes denote unfinished rows. These counts do not establish a multi-seed or factorization benefit; changing the CEM budget is not itself evidence of improvement.}",
        r"\label{tab:official-development}\end{table}"]
    notes = []
    for environment, audit in ledger["paired_support_verification"].items():
        name = "PushT" if environment == "pusht" else "Reacher"
        count = len(audit["completed_modes"])
        if count >= 2:
            notes.append(f"{name}: the same 32 seed/trajectory identities, goal indices, support-success masks, eligibility masks and post-support distances agree across {count} completed modes.")
        else: notes.append(f"{name}: cross-model support verification is pending ({count} completed modes).")
    text += [r"\noindent\textbf{Paired-support audit.} " + " ".join(notes),
        f"The comparison currently contains {ledger['completed_records']}/8 completed model/environment records. "
        "Unfinished records do not contribute outcomes. Comparative conclusions remain pending; "
        "the main multi-seed test and its reporting gate are unchanged.\n"]
    return "\n".join(text) + "\n"


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=PROJECT / "paper/generated")
    args = parser.parse_args()
    ledger = collect(args.root.resolve())
    atomic_text(args.output / "official_development.json", json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    atomic_text(args.output / "official_development.tex", render_tex(ledger))
    print(json.dumps({"completed_development_records": ledger["completed_records"], "expected_records": 8,
                      "support_pairing": ledger["paired_support_verification"]}))


if __name__ == "__main__": main()
