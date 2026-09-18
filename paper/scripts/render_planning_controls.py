#!/usr/bin/env python3
"""Report random-action and privileged-replay controls with existing validators."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from shiftwm.data import validate_manifest

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired = script("summarize_paired_planning")
campaign = paired.campaign
POLICIES = (("random", "Random actions"), ("replay_oracle", "Privileged replay"))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_summary(rows, summary, metrics):
    """Check every displayed count/denominator against the evaluator's summaries."""
    groups = defaultdict(list)
    for row in rows:
        groups[f"o{row['observation_id']}_d{row['dynamics_id']}"].append(row)
    if rows:
        groups["all"] = rows
    if set(summary) != set(groups):
        raise ValueError("Control summary populations differ from complete task records")
    for condition, selected in groups.items():
        for metric in metrics:
            item = summary[condition][metric]
            n = len(selected)
            if (item.get("observations") != n
                    or item.get("clusters") != len({row["seed"] for row in selected})
                    or not math.isclose(float(item["mean"]), sum(row[metric] for row in selected) / n,
                                        rel_tol=1e-11, abs_tol=1e-12)):
                raise ValueError("Control summary count, mean or denominator disagrees with records")


def validate_record(record, environment, split, policy, tasks, identities):
    if record.get("status") != "complete":
        return None
    expected = {"environment": environment, "model_mode": "frozen", "training_seed": None,
                "checkpoint_epoch": 0, **identities}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Control checkpoint/data/evaluator or frozen export identity differs")
    planning = record["planning"]
    if planning.get("status") != "complete" or planning.get("kind") != "closed_loop_goal_image_planning":
        raise ValueError("Control planning kind or completion differs")
    task = {"policy": policy, "split": split,
            "planner_budget": (campaign.OFFICIAL_PLANNER_BUDGET if policy == "world_model"
                               else campaign.CONTROL_METADATA_BUDGET)}
    campaign.validate_completed_planning(record, task, 64)
    if any(row.get("policy") != policy for row in planning["records"]):
        raise ValueError("Original per-task control policy differs from requested policy")
    if policy not in {"world_model", "random", "replay_oracle"}:
        raise ValueError("Unexpected control policy")
    if policy != "world_model" and any(row.get("num_replans") != 0 or row.get("total_solve_seconds") != 0
                                       for row in planning["records"]):
        raise ValueError("Random/replay control incorrectly records a model planning solve")
    # The established record validator hardcodes world_model, but its remaining
    # checks concern common task identity, goal, support and interaction budget.
    # Authenticate original control policies/protocol above, normalize ONLY the
    # policy in a private copy for those checks, and retain original records.
    checked = copy.deepcopy(record)
    for row in checked["planning"]["records"]:
        row["policy"] = "world_model"
    paired.validate_records(checked, tasks, environment, complete=True)
    rows = planning["records"]
    verify_summary(rows, planning["summary"], ("success", "final_success", "success_during_context"))
    verify_summary([row for row in rows if row["policy_eligible"]], planning["eligible_summary"],
                   ("success", "final_success"))
    return {(row["trajectory_id"], row["observation_id"]): row for row in rows}


def population_counts(rows, split):
    selected = [row for row in rows.values()
                if split == "extrapolation" or (row["observation_id"], row["dynamics_id"]) == (2, 2)]
    required = 192 if split == "extrapolation" else 64
    if len(selected) != required:
        raise ValueError("Control displayed population differs from main held-out/extrapolation definition")
    eligible = [row for row in selected if row["policy_eligible"]]
    raw = sum(row["success"] for row in selected)
    eligible_success = sum(row["success"] for row in eligible)
    return {"raw_successes": raw, "raw_total": len(selected), "raw_percent": 100 * raw / len(selected),
            "support_successes": sum(row["success_during_context"] for row in selected),
            "eligible_successes": eligible_success, "eligible_total": len(eligible),
            "eligible_percent": 100 * eligible_success / len(eligible) if eligible else None,
            "initial_state_clusters": len({row["seed"] for row in selected})}


def build_report(root=ROOT):
    root = Path(root)
    hashes = campaign.FileDigestCache()
    evaluator_sha = hashes.sha256(root / "src/shiftwm/evaluate.py")
    sources, rows, support_audits = {}, [], []
    for environment, data_name in (("pusht", "pusht_relative"), ("reacher", "reacher")):
        manifest_path = root / "data/world" / data_name / "manifest.json"
        manifest = read(manifest_path)
        validate_manifest(manifest)
        if manifest["environment"] != environment or manifest["action_block"] != 5:
            raise ValueError("Unexpected control dataset family or action grouping")
        if environment == "pusht" and manifest.get("action_interface") != "relative":
            raise ValueError("PushT controls require the locked relative action interface")
        package = root / "runs/world" / f"{environment}_frozen_s0/best"
        config = read(package / "config.json")
        if (config["model_config"]["mode"] != "frozen" or config["model_config"]["history_length"] != 3
                or config["metadata"].get("environment") != environment
                or config["metadata"].get("training_status") != "released_upstream_weights_not_finetuned"):
            raise ValueError("Control carrier is not the locked frozen export")
        identities = {"checkpoint_sha256": hashes.sha256(package / "model.pt"),
                      "data_manifest_sha256": hashes.sha256(manifest_path), "evaluator_sha256": evaluator_sha}
        sources[str(manifest_path.relative_to(root))] = {"sha256": identities["data_manifest_sha256"], "kind": "data_manifest"}
        sources[str((package / "config.json").relative_to(root))] = {"sha256": hashes.sha256(package / "config.json"), "kind": "frozen_export_config"}
        sources[str((package / "model.pt").relative_to(root))] = {"sha256": identities["checkpoint_sha256"], "kind": "frozen_export_weights"}
        for split in ("test", "extrapolation"):
            tasks = paired.manifest_tasks(manifest, split)
            directory = root / "results/world" / f"{environment}_frozen_s0"
            reference_path = directory / f"planning_{split}.json"
            reference = validate_record(read(reference_path), environment, split, "world_model", tasks, identities)
            if reference is None:
                raise ValueError("Completed frozen planning reference required for common-support audit")
            sources[str(reference_path.relative_to(root))] = {"sha256": hashes.sha256(reference_path), "kind": "common_support_reference"}
            for policy, label in POLICIES:
                path = directory / f"planning_{split}_{policy}.json"
                result = {"environment": environment, "split": split, "policy": policy, "label": label,
                          "status": "pending", "counts": None, "source": str(path.relative_to(root))}
                if path.is_file():
                    original = read(path)
                    checked = validate_record(original, environment, split, policy, tasks, identities)
                    if checked is not None:
                        if checked.keys() != reference.keys():
                            raise ValueError("Control and reference task populations differ")
                        paired.validate_shared_support(reference, checked, environment)
                        result.update(status="complete", counts=population_counts(checked, split),
                                      source_sha256=hashes.sha256(path))
                        sources[str(path.relative_to(root))] = {"sha256": result["source_sha256"],
                            "kind": "control_evaluation", **identities, "signature": original["planning"]["signature"]}
                        support_audits.append({"environment": environment, "split": split, "policy": policy,
                            "reference": str(reference_path.relative_to(root)), "all_prescribed_tasks_verified": len(checked),
                            "checks": "Exact task/goal/support masks and recorded initial-state diagnostics against frozen planning reference.",
                            "limitations": "Support/goal pixel hashes and full simulator-state hashes are not recorded; their equality is not retrospectively claimed."})
                rows.append(result)
    return {"schema_version": 1, "status": "complete" if all(row["status"] == "complete" for row in rows) else "pending",
        "reporter_sha256": digest(__file__), "validators": {
            "campaign": {"path": "scripts/run_evaluation_campaign.py", "sha256": hashes.sha256(root / "scripts/run_evaluation_campaign.py")},
            "paired_records": {"path": "scripts/summarize_paired_planning.py", "sha256": hashes.sha256(root / "scripts/summarize_paired_planning.py")}},
        "sources": sources, "rows": rows, "support_audits": support_audits,
        "control_runs_complete": sum(row["status"] == "complete" for row in rows), "control_runs_required": 8,
        "protocol": {"initial_states_per_dynamics": 64, "native_budget": 50, "paid_support_max_steps": 10,
                     "action_block": 5, "goal_index": 7, "policy_rng_rule": "numpy.default_rng(1701 + manifest_initial_state_seed), reused across appearances"},
        "scope": "Random and privileged replay are controls, not trained world-model comparisons. No training-seed SD, independent random-policy replicate uncertainty, latency or superiority claim.",
        "policy_roles": {"random": "Uniform native actions within environment bounds; one fixed RNG rule; no model-based action selection.",
                         "replay_oracle": "Privileged recorded future actions from the target trajectory; feasibility/protocol check, not a fair learned-policy inference baseline."},
        "adapter": "Original control policy/protocol authenticated before a private copy normalizes only policy for existing policy-independent per-record checks; originals and source hashes retained."}


def render_tex(report):
    lines = [r"% Validated control policies, distinct from trained model comparisons.",
             r"\begin{table}[t]\centering\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.1}",
             r"\begin{tabular}{lllrrr}\toprule",
             r"Environment & Population & Control & Raw (\%) & Support & Eligible (\%)\\\midrule"]
    for row in report["rows"]:
        if row["status"] == "complete":
            c = row["counts"]
            cells = [f"{c['raw_percent']:.2f} [{c['raw_successes']}/{c['raw_total']}]",
                     f"{c['support_successes']}/{c['raw_total']}",
                     f"{c['eligible_percent']:.2f} [{c['eligible_successes']}/{c['eligible_total']}]" if c["eligible_total"] else "n/a"]
        else:
            cells = [r"\missing"] * 3
        lines.append(" & ".join(["PushT" if row["environment"] == "pusht" else "Reacher",
            "Held-out" if row["split"] == "test" else "Extrap.", row["label"], *cells]) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
        r"\caption{\textbf{Random-action and privileged-replay controls.} Raw and eligible columns show success percentages with successes/tasks in brackets; support counts are solved during paid history acquisition. Held-out is $(v_2,p_2)$ (64 tasks); extrapolation pools the three prespecified conditions (192 tasks, 64 initial-state seeds). The task identities, goal index, support-success masks and recorded post-support diagnostics match the frozen-model reference. All policies share a 50-step native budget, including up to 10 support steps. Random actions use one fixed RNG rule (1701 plus initial-state seed), reused across appearances; these are not three training-seed estimates. Privileged replay uses recorded future actions and checks goal feasibility, not fair learned-policy inference. Neither control selects actions with model predictions, so CEM search metadata is unused. No independent-policy-seed uncertainty or significance claim is made.}",
        r"\label{tab:planning-controls}\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = build_report(args.root)
    directory = args.root / "paper/generated"
    (directory / "planning_controls.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "planning_controls.tex").write_text(render_tex(report))
    print(json.dumps({"control_runs_complete": report["control_runs_complete"], "required": 8,
                      "common_support_audits": len(report["support_audits"])}))


if __name__ == "__main__":
    main()
