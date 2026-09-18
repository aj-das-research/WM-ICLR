#!/usr/bin/env python3
"""Coordinate independent full training runs across entitled Slurm allocations.

An inherited per-task lock prevents duplicate runs on the shared cluster.
Legacy workers without task locks remain protected by their Slurm ownership.
Each run saves its own full logs, validation-selected weights, and RNG state.
Interrupted jobs can resume from the last completed minibatch.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class SlurmActivity:
    """Cache one read-only scheduler query for all recorded job owners."""
    def __init__(self, refresh_seconds=15):
        self.refresh_seconds = refresh_seconds
        self.updated = float("-inf")
        self.active = None

    def _refresh(self):
        try:
            result = subprocess.run(["squeue", "--noheader", "--user", str(os.getuid()),
                                     "--format", "%i"], capture_output=True, text=True, timeout=5)
            self.active = set(result.stdout.split()) if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            self.active = None
        self.updated = time.monotonic()

    def __call__(self, job_id):
        refreshed = time.monotonic() - self.updated >= self.refresh_seconds
        if refreshed:
            self._refresh()
        if self.active is None:
            return None  # Unknown scheduler state must never steal an active run.
        job_id = str(job_id)
        contains = lambda: any(value == job_id or value.startswith(job_id + "_") or value.startswith(job_id + ".")
                               for value in self.active)
        if contains():
            return True
        # A newly submitted legacy job may be newer than the cached snapshot.
        # Cached absence is never sufficient evidence to reclaim its lease.
        if not refreshed:
            self._refresh()
        return None if self.active is None else contains()


def pid_exists(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner_is_active(state, job_activity, *, node=None, process_activity=pid_exists):
    """Protect live and unverifiable legacy owners; reclaim confirmed dead ones.

    A kernel-held task lock is checked before this helper. This additional
    ownership check is essential while old Slurm workers use only JSON leases.
    """
    node = node or os.uname().nodename
    job_id = state.get("job_id")
    if job_id:
        job_live = job_activity(str(job_id))
        if job_live is not False:
            return True
    if state.get("pid") and state.get("node") == node:
        return process_activity(state["pid"])
    if job_id:
        return False  # Confirmed absent Slurm owner, with no live local process.
    # Non-Slurm remote/legacy states without verifiable PID ownership are kept.
    return True


def claim_task(task, config, state_dir, job_activity):
    """Claim under a task lock, retaining the global lock for legacy interop."""
    state_dir = Path(state_dir)
    task_lock = (state_dir / (task["id"] + ".lock")).open("a+")
    try:
        fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        task_lock.close()
        return None, "busy"
    state_path = state_dir / (task["id"] + ".json")
    try:
        with (state_dir / ".lock").open("a+") as global_lock:
            fcntl.flock(global_lock, fcntl.LOCK_EX)
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            status = state.get("status")
            if status in ("complete", "failed"):
                task_lock.close()
                return None, status
            if status == "running" and owner_is_active(state, job_activity):
                task_lock.close()
                return None, "active_owner"
            required = [Path(config["action_stats"]),
                        Path(config["dataset_kwargs"]["feature_cache"]) / "manifest.json"]
            if not all(path.is_file() for path in required):
                task_lock.close()
                return None, "not_ready"
            new_state = {"status": "running", "lease_version": 2,
                         "job_id": os.getenv("SLURM_JOB_ID"), "pid": os.getpid(),
                         "node": os.uname().nodename, "start_unix": time.time(),
                         "config": task["config"], "recovery_count": state.get("recovery_count", 0)}
            if status == "running":
                new_state["recovered_owner"] = {key: state.get(key) for key in ("job_id", "pid", "node", "start_unix")}
                new_state["recovery_count"] += 1
            write_json(state_path, new_state)
        return task_lock, "claimed"
    except BaseException:
        task_lock.close()
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign", default="configs/world/training_campaign.json")
    parser.add_argument("--state", default="runs/world/campaign_state")
    parser.add_argument("--max-seconds", type=int, default=27000)
    parser.add_argument("--exit-when-not-ready", action="store_true",
                        help="Exit when no task is claimable instead of waiting for another owner or missing inputs")
    args = parser.parse_args()
    tasks = json.loads(Path(args.campaign).read_text())["tasks"]
    state_dir = Path(args.state)
    state_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failed = 0
    job_activity = SlurmActivity()
    while time.monotonic() - started < args.max_seconds - 180:
        chosen = None
        waiting = False
        for task in tasks:
            config = json.loads(Path(task["config"]).read_text())
            task_lock, reason = claim_task(task, config, state_dir, job_activity)
            if task_lock is not None:
                chosen = task
                break
            if reason in ("busy", "active_owner", "not_ready"):
                waiting = True
        if chosen is None:
            if waiting and not args.exit_when_not_ready:
                time.sleep(15)
                continue
            break
        config = json.loads(Path(chosen["config"]).read_text())
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if (output / "last" / "training_state.pt").is_file():
            config["resume"] = str(output / "last")
        config["max_runtime_seconds"] = int(args.max_seconds - (time.monotonic() - started) - 120)
        runtime = output / f"runtime_config_{os.getenv('SLURM_JOB_ID', os.getpid())}.json"
        write_json(runtime, config)
        log = output / f"train_{os.getenv('SLURM_JOB_ID', os.getpid())}.log"
        print(json.dumps({"event": "training_start", "id": chosen["id"], "log": str(log)}), flush=True)
        with log.open("a", buffering=1) as handle:
            result = subprocess.run([sys.executable, "-m", "shiftwm.train", "--config", str(runtime)],
                                    stdout=handle, stderr=subprocess.STDOUT,
                                    pass_fds=(task_lock.fileno(),))
        summary_path = output / "training_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        status = ("complete" if result.returncode == 0 and summary.get("status") == "completed" else
                  "pending" if result.returncode == 0 and summary.get("status") == "interrupted_checkpoint_saved" else "failed")
        state_path = state_dir / (chosen["id"] + ".json")
        with (state_dir / ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(state_path.read_text())
            state.update(status=status, end_unix=time.time(), returncode=result.returncode,
                         summary=summary, log=str(log))
            write_json(state_path, state)
        task_lock.close()
        print(json.dumps({"event": "training_end", "id": chosen["id"], "status": status,
                          "summary": summary}), flush=True)
        failed += status == "failed"
        if status == "pending":
            break
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
