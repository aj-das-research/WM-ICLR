#!/usr/bin/env python3
"""Run eight fixed full-development diagnostics after their planning comparisons."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from diagnose_goal_calibration import sha256, planning_agreement, verify_completed_source
from run_evaluation_campaign import OFFICIAL_PLANNER_BUDGET, validate_completed_planning


def validate_completed_diagnostic(result, planning, identity, planning_path):
    """Validate existing evidence without altering records or recomputing metrics."""
    if result.get("status") != "complete" or any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("Diagnostic provenance changed")
    if result.get("kind") != "development_goal_calibration_diagnostic" or result.get("split") != "development":
        raise ValueError("Diagnostic kind or split changed")
    if planning.get("status") != "complete":
        raise ValueError("Development planning is not complete")
    task = {"policy": "world_model", "split": "development", "planner_budget": OFFICIAL_PLANNER_BUDGET}
    validate_completed_planning(planning, task, 32)
    linkage = result.get("planning_setup_agreement", {})
    if (linkage.get("status") != "passed" or linkage.get("compared_records") != 32
            or linkage.get("source_sha256") != sha256(planning_path)
            or linkage.get("source") != str(Path(planning_path).resolve())):
        raise ValueError("Planning source linkage differs from the completed diagnostic")
    records = result.get("records", [])
    keys = [(row["trajectory_id"], row["observation_id"]) for row in records]
    if len(records) != 32 or len(set(keys)) != 32:
        raise ValueError("Diagnostic must contain exactly32 unique development records")
    if result.get("selected_keys") != [list(key) for key in keys]:
        raise ValueError("Diagnostic selected-key metadata differs from its records")
    measured = support_success = 0
    for row in records:
        if row.get("diagnostic_status") == "measured" and row.get("policy_eligible") == 1 and row.get("success_during_context") == 0:
            measured += 1
        elif row.get("diagnostic_status") == "excluded_support_success" and row.get("policy_eligible") == 0 and row.get("success_during_context") == 1:
            support_success += 1
        else:
            raise ValueError("Diagnostic record classification disagrees with its support-success mask")
    expected_counts = {"selected": len(records), "measured": measured, "support_success": support_success}
    if result.get("counts") != expected_counts:
        raise ValueError("Diagnostic counts differ from its records")
    return planning_agreement(result, planning, identity)


def main():
    root = Path("results/diagnostics")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".goal_calibration.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for environment, data_name in (("pusht", "pusht_relative"), ("reacher", "reacher")):
            for mode in ("frozen", "single", "factorized", "framewise"):
                run_id = f"{environment}_{mode}_s0"
                checkpoint = Path("runs/world") / run_id / "best"
                model_file, _ = verify_completed_source(checkpoint)
                data = Path("data/world") / data_name
                planning_path = Path("results/development_official_budget") / run_id / "planning_development.json"
                planning = json.loads(planning_path.read_text())
                if planning.get("status") != "complete":
                    raise ValueError(f"Development planning is not complete: {run_id}")
                validate_completed_planning(planning, {"policy": "world_model", "split": "development",
                                            "planner_budget": OFFICIAL_PLANNER_BUDGET}, 32)
                output = root / run_id / "goal_calibration.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                identity = {"checkpoint_sha256": sha256(model_file), "config_sha256": sha256(checkpoint / "config.json"),
                            "data_manifest_sha256": sha256(data / "manifest.json"),
                            "evaluator_sha256": sha256("src/shiftwm/evaluate.py"),
                            "model_source_sha256": sha256("src/shiftwm/model.py"),
                            "generation_source_sha256": sha256("src/shiftwm/generate.py"),
                            "diagnostic_source_sha256": sha256("scripts/diagnose_goal_calibration.py")}
                if not output.exists():
                    command = [sys.executable, "scripts/diagnose_goal_calibration.py", "--checkpoint", str(checkpoint),
                               "--data", str(data), "--device", "cuda", "--planning-result", str(planning_path),
                               "--output", str(output)]
                    print(json.dumps({"event": "diagnostic_start", "run_id": run_id}), flush=True)
                    with output.with_suffix(".log").open("a") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                       pass_fds=(lock.fileno(),))
                result = json.loads(output.read_text())
                validate_completed_diagnostic(result, planning, identity, planning_path)
                rows.append({"run_id": run_id, "source": str(output), "source_sha256": sha256(output),
                             "counts": result["counts"], "summary": result["summary"]})
                print(json.dumps({"event": "diagnostic_complete", "run_id": run_id, "counts": result["counts"]}), flush=True)
        destination = root / "goal_calibration_summary.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps({"status": "complete", "scope": "development_only; calibration diagnostics, not planning efficacy",
                                         "runs": rows}, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)


if __name__ == "__main__":
    main()
