#!/usr/bin/env python3
"""Run the two required official clean controls and a time-bounded blur pair."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/baselines/adajepa"


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "protocol": "official CEM50, seed100",
              "required": ["clean-frozen", "clean-adaptive"], "completed": [],
              "comparison_scope": "published RGB+proprio protocol, not matched to ShiftWM",
              "optional_blur": "pending measured clean runtime"}
    report = OUT / "campaign.json"

    def save():
        record["elapsed_seconds"] = time.monotonic() - started
        temporary = report.with_suffix(".partial.json")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(report)

    def run(condition, method):
        save()
        before = time.monotonic()
        result = subprocess.run([sys.executable, str(ROOT / "scripts/baselines/adajepa_run.py"),
                                 "--condition", condition, "--method", method, "--episodes", "50", "--seed", "100"],
                                cwd=ROOT)
        if result.returncode:
            record["status"] = "failed"
            record["failed_stage"] = f"{condition}-{method}"
            save()
            raise SystemExit(result.returncode)
        record["completed"].append(f"{condition}-{method}")
        save()
        return time.monotonic() - before

    runtimes = [run("clean", "frozen"), run("clean", "adaptive")]
    remaining = 5.5 * 3600 - (time.monotonic() - started)
    if remaining > 3 * max(runtimes) + 600:
        record["optional_blur"] = "running"
        run("blur", "frozen")
        run("blur", "adaptive")
        record["optional_blur"] = "complete"
    else:
        record["optional_blur"] = "deferred: insufficient wall-time margin from measured clean runtime"
    record["status"] = "complete_required_protocol"
    save()


if __name__ == "__main__":
    main()
