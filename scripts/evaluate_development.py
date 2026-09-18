#!/usr/bin/env python3
"""Evaluate every development episode with actual trained models and full CEM.

Development results are separate from the locked main test grid. This command
records that its allocation may be shared, so its timings are not used for a
main efficiency claim. No model selection or hyperparameter update occurs here.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

for environment, data_name in (("pusht", "pusht_relative"), ("reacher", "reacher")):
    name = f"{environment}_factorized_s0"
    run = Path("runs/world") / name
    summary = json.loads((run / "training_summary.json").read_text())
    if summary.get("status") != "completed":
        raise SystemExit(f"Full training is not complete: {run}")
    output = Path("results/development") / name / "both_development.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "shiftwm.evaluate", "--checkpoint", str(run / "best"),
               "--data", f"data/world/{data_name}", "--feature-cache", f"data/features/{data_name}",
               "--kind", "both", "--split", "development", "--episodes", "32", "--workers", "0",
               "--save-video", "--output", str(output)]
    print(json.dumps({"event": "full_development_evaluation", "run": name, "command": command}), flush=True)
    with output.with_suffix(".log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = json.loads(output.read_text())
    record["execution_context"] = {"job_id": os.getenv("SLURM_JOB_ID"),
                                    "shared_gpu_with_training": True,
                                    "main_efficiency_claim_eligible": False,
                                    "scope": "all_32_prespecified_development_tasks; no hyperparameter tuning"}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"event": "development_complete", "run": name,
                      "success": record["planning"]["summary"]["all"]["success"],
                      "episodes": len(record["planning"]["records"])}), flush=True)
