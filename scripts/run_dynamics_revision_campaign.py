#!/usr/bin/env python3
"""Resume the two development-only dynamics revisions within one GPU allocation.

Training and evaluation retain separate completion states. A saved partial run
is pending, not a failed or completed experiment. The next allocation resumes it.
The original full-test campaign is independent of this development pipeline.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ("pusht", "reacher")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def pipeline(max_seconds=6600, root=ROOT, runner=subprocess.run, clock=time.monotonic):
    if max_seconds < 360:
        raise ValueError("At least 360 seconds are required for a resumable allocation")
    root = Path(root).resolve()
    state_dir = root / "runs/dynamics_revision/campaign_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    started = clock()
    with (state_dir / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"event": "revision_worker_already_active"}), flush=True)
            return 0
        state = {
            "scope": "posthoc development only; fixed framewise donor; two seed-zero revisions",
            "status": "running", "job_id": os.getenv("SLURM_JOB_ID"),
            "node": os.uname().nodename,
            "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "max_runtime_seconds": max_seconds, "tasks": {},
            "source_hashes": {},
        }
        for relative in ("scripts/run_dynamics_revision_campaign.py",
                         "scripts/train_dynamics_revision.py",
                         "scripts/evaluate_dynamics_revision.py",
                         "src/shiftwm/dynamics_revision.py"):
            source = root / relative
            state["source_hashes"][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        state_path = state_dir / "status.json"

        def save(status=None):
            if status:
                state["status"] = status
            state["elapsed_seconds"] = clock() - started
            atomic_json(state_path, state)

        def run_child(command, log):
            print(json.dumps({"event": "revision_stage_start", "command": command}), flush=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", buffering=1) as handle:
                result = runner(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                                pass_fds=(lock.fileno(),))
            if result.returncode not in (0, 75):
                state["failed_command"] = command
                state["returncode"] = result.returncode
                save("failed")
            return result.returncode

        save()
        for environment in ENVIRONMENTS:
            config_path = root / "configs/dynamics_revision" / f"{environment}_s0.json"
            config = load_json(config_path)
            if config.get("seed") != 0 or config.get("epochs") != 30:
                raise ValueError("Revision protocol requires seed zero and all 30 epochs")
            output = root / config["output_dir"]
            expected_output = root / f"runs/dynamics_revision/{environment}_s0"
            if output.resolve() != expected_output.resolve():
                raise ValueError("Revision output must be isolated from original model runs")
            state["tasks"][environment] = item = {
                "config": str(config_path.relative_to(root)),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "training": "pending", "planning": "pending",
            }
            remaining = int(max_seconds - (clock() - started))
            if remaining < 360:
                save("pending")
                return 0
            train_budget = min(int(config.get("max_runtime_seconds", 6000)), remaining - 240)
            command = [sys.executable, str(root / "scripts/train_dynamics_revision.py"),
                       "--config", str(config_path), "--resume-if-present",
                       "--max-runtime-seconds", str(train_budget)]
            item["training"] = "running"
            save()
            returncode = run_child(command, output / "campaign_training.log")
            if returncode not in (0, 75):
                return 1
            summary = load_json(output / "training_summary.json")
            item["training_summary"] = summary
            if summary.get("status") != "completed":
                if returncode != 75 or summary.get("status") != "interrupted_checkpoint_saved":
                    raise ValueError("Training exited without a valid resumable or completed result")
                item["training"] = "pending"
                save("pending")
                return 0
            if returncode != 0:
                raise ValueError("Training interruption code contradicts completed summary")
            if summary.get("completed_epochs") != 30 or not (output / "best/model.pt").is_file():
                raise ValueError("Completed revision is missing epochs or its best package")
            item["training"] = "complete"
            remaining = int(max_seconds - (clock() - started))
            if remaining < 180:
                save("pending")
                return 0
            command = [sys.executable, str(root / "scripts/evaluate_dynamics_revision.py"),
                       "--environment", environment, "--checkpoint", str(output / "best"),
                       "--device", "cuda", "--save-video",
                       "--max-runtime-seconds", str(remaining - 120)]
            result_path = root / f"results/development_dynamics_revision/{environment}_s0/planning_development.json"
            item.update(planning="running", result=str(result_path.relative_to(root)))
            save()
            returncode = run_child(command, output / "campaign_planning.log")
            if returncode not in (0, 75):
                return 1
            result = load_json(result_path)
            if result.get("status") != "complete":
                if returncode != 75 or result.get("status") != "interrupted":
                    raise ValueError("Planning exited without a valid resumable or completed result")
                item["planning"] = "pending"
                save("pending")
                return 0
            if returncode != 0:
                raise ValueError("Planning interruption code contradicts completed result")
            # The dedicated CLI validates source identities and complete task
            # pairing before writing/reusing this result; don't count a header alone.
            planning = result.get("planning", {})
            if (planning.get("status") != "complete" or planning.get("split") != "development"
                    or len(planning.get("records", [])) != 32):
                raise ValueError("Revision planning output lacks complete development records")
            item["planning"] = "complete"
            item["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            save()
        save("complete")
        print(json.dumps({"event": "revision_campaign_complete", "state": str(state_path)}), flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, default=6600)
    args = parser.parse_args()
    return pipeline(args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
