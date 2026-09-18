#!/usr/bin/env python3
"""Complete development-set comparison at the upstream CEM search budget.

Uses all 32 prespecified development tasks per environment. Outputs are separate
from both the historical low-search-budget diagnostic and the locked test grid.
Interrupted evaluations resume through the evaluator's per-episode journal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_evaluation_campaign import OFFICIAL_PLANNER_BUDGET, validate_completed_planning


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def training_ready(run, mode):
    if mode != "frozen":
        try:
            summary = json.loads((run / "training_summary.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if summary.get("status") != "completed":
            return False
    return (run / "best/model.pt").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, default=27000)
    args = parser.parse_args()
    started = time.monotonic()
    pending = [(environment, data_name, mode)
               for environment, data_name in (("pusht", "pusht_relative"), ("reacher", "reacher"))
               for mode in ("frozen", "single", "factorized", "framewise")]
    while pending:
        for environment, data_name, mode in pending:
            name = f"{environment}_{mode}_s0"
            run = Path("runs/world") / name
            if not training_ready(run, mode):
                continue
            # Preserve the fixed task order among ready checkpoints. A slow
            # earlier training run must not block another environment's work.
            break
        else:
            if time.monotonic() - started >= args.max_seconds - 180:
                return 75
            time.sleep(15)
            continue
        pending.remove((environment, data_name, mode))
        output = Path("results/development_official_budget") / name / "planning_development.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        identity = {"checkpoint_sha256": digest(run / "best/model.pt"),
                    "data_manifest_sha256": digest(f"data/world/{data_name}/manifest.json"),
                    "evaluator_sha256": digest("src/shiftwm/evaluate.py")}
        task = {"policy": "world_model", "split": "development",
                "planner_budget": OFFICIAL_PLANNER_BUDGET}
        if output.exists():
            record = json.loads(output.read_text())
            if record.get("status") == "complete":
                if any(record.get(key) != value for key, value in identity.items()):
                    raise RuntimeError(f"Completed development provenance changed: {output}")
                validate_completed_planning(record, task, 32)
                continue
        remaining = int(args.max_seconds - (time.monotonic() - started)) - 90
        if remaining < 180:
            return 75
        command = [sys.executable, "-m", "shiftwm.evaluate", "--checkpoint", str(run / "best"),
                   "--data", f"data/world/{data_name}", "--kind", "planning", "--split", "development",
                   "--episodes", "32", "--workers", "0", "--samples", "300", "--iterations", "30",
                   "--elites", "30", "--horizon", "5", "--native-budget", "50",
                   "--max-runtime-seconds", str(remaining), "--save-video", "--output", str(output)]
        print(json.dumps({"event": "development_comparison_start", "run": name,
                          "command": command}), flush=True)
        with output.with_suffix(".log").open("a") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            return process.returncode
        record = json.loads(output.read_text())
        if record.get("status") != "complete":
            return 75
        validate_completed_planning(record, task, 32)
        record["execution_context"] = {
            "job_id": os.getenv("SLURM_JOB_ID"), "shared_gpu_with_training": False,
            "main_efficiency_claim_eligible": False,
            "scope": "dedicated development comparison; all 32 prespecified tasks; seed-0 models only"}
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output)
        print(json.dumps({"event": "development_comparison_complete", "run": name,
                          "success": record["planning"]["summary"]["all"]["success"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
