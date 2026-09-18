#!/usr/bin/env python3
"""Build local inference releases as full training runs finish, for at most120h.

This watcher never trains, evaluates benchmarks, aggregates paper results,
compiles TeX, or uploads artifacts. The paper watcher owns paper generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metadata_signature(paths):
    """Cheap change detection; the release builder verifies full file hashes."""
    entries = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def release_signature(directory):
    """Detect deleted/changed published files without rehashing weights per scan."""
    directory = Path(directory)
    manifest = directory / "release_manifest.json"
    if not manifest.is_file():
        return None
    try:
        record = json.loads(manifest.read_text())
        files = record.get("package_files", {})
        if record.get("status") != "ready" or not {"model.pt", "config.json", "MODEL_CARD.md", "verification.json"} <= files.keys():
            return None
        return metadata_signature([manifest, *[directory / name for name in sorted(files)]])
    except (OSError, ValueError, TypeError):
        return None


def invoke_builder(command, *, log, root, timeout, inherited_fds):
    environment = os.environ.copy()
    project_source = str(root / "src")
    environment["PYTHONPATH"] = project_source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    with log.open("a", buffering=1) as handle:
        try:
            result = subprocess.run(command, cwd=root, env=environment, stdout=handle,
                                    stderr=subprocess.STDOUT, timeout=timeout,
                                    pass_fds=inherited_fds)
            return result.returncode
        except subprocess.TimeoutExpired:
            handle.write("Release builder exceeded the watcher deadline.\n")
            return 124
        except OSError as error:
            handle.write(f"Could not execute release builder: {error}\n")
            return 127


class ReleaseWatcher:
    def __init__(self, root, campaign, releases, status_path, *, source_wheel=None,
                 invoke=invoke_builder, retry_seconds=1800, max_attempts=3, inherited_fds=()):
        self.root = Path(root).resolve()
        resolve = lambda path: Path(path) if Path(path).is_absolute() else self.root / path
        self.campaign = resolve(campaign)
        self.releases = resolve(releases)
        self.status_path = resolve(status_path)
        self.source_wheel = resolve(source_wheel) if source_wheel else None
        self.invoke = invoke
        self.retry_seconds, self.max_attempts = retry_seconds, max_attempts
        self.inherited_fds = inherited_fds
        self.builder = self.root / "scripts" / "build_checkpoint_release.py"
        self.state = json.loads(self.status_path.read_text()) if self.status_path.exists() else {"runs": {}}
        self.state.update(schema_version=1, status="watching", pid=os.getpid(), node=os.uname().nodename,
                          started_utc=utc_now(), campaign=str(self.campaign),
                          scope="completed_training_to_local_release_only")

    def save(self):
        self.state["updated_utc"] = utc_now()
        self.state["counts"] = dict(Counter(run.get("status", "pending") for run in self.state["runs"].values()))
        atomic_json(self.status_path, self.state)

    def resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def scan_once(self, deadline=None):
        tasks = json.loads(self.campaign.read_text())["tasks"]
        if len({task["id"] for task in tasks}) != len(tasks):
            raise ValueError("Campaign contains duplicate run identifiers")
        self.state["campaign_runs"] = len(tasks)
        for task in tasks:
            run_id = task["id"]
            config_path = self.resolve(task["config"])
            config = json.loads(config_path.read_text())
            run_dir = self.resolve(config["output_dir"])
            destination = self.releases / run_id
            previous = self.state["runs"].get(run_id, {})
            summary_path = run_dir / "training_summary.json"
            try:
                summary = json.loads(summary_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                self.state["runs"][run_id] = {**previous, "status": "pending", "reason": "training_not_completed"}
                continue
            if summary.get("status") != "completed":
                self.state["runs"][run_id] = {**previous, "status": "pending", "reason": "training_not_completed"}
                continue
            if summary.get("completed_epochs") != config["epochs"]:
                self.state["runs"][run_id] = {**previous, "status": "failed", "reason": "completed_epoch_count_mismatch"}
                continue
            source_paths = [config_path, summary_path, run_dir / "run_config.json", run_dir / "metrics.jsonl",
                            run_dir / "best" / "model.pt", run_dir / "best" / "config.json", self.builder,
                            self.resolve(config["data_root"]) / "manifest.json",
                            self.resolve(config["dataset_kwargs"]["feature_cache"]) / "manifest.json"]
            if self.source_wheel:
                source_paths.append(self.source_wheel)
            missing = [str(path) for path in source_paths if not path.is_file()]
            if missing:
                self.state["runs"][run_id] = {**previous, "status": "pending", "reason": "release_inputs_missing", "missing": missing}
                continue
            source_signature = metadata_signature(source_paths)
            current_release = release_signature(destination)
            if (previous.get("status") == "ready" and previous.get("source_signature") == source_signature
                    and previous.get("release_signature") == current_release and current_release is not None):
                continue
            same_source = previous.get("source_signature") == source_signature
            attempts = previous.get("attempts", 0) if same_source else 0
            if same_source and previous.get("status") == "failed":
                if attempts >= self.max_attempts or time.time() - previous.get("last_attempt_unix", 0) < self.retry_seconds:
                    continue
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining < 60:
                self.state["runs"][run_id] = {**previous, "status": "pending", "reason": "watcher_deadline_near"}
                break
            attempts += 1
            log = self.status_path.parent / "logs" / run_id / f"{source_signature[:12]}-attempt{attempts}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(self.builder), "--run-dir", str(run_dir), "--output", str(destination)]
            if self.source_wheel:
                command += ["--source-wheel", str(self.source_wheel)]
            record = {"status": "running", "source_signature": source_signature, "attempts": attempts,
                      "last_attempt_unix": time.time(), "started_utc": utc_now(), "log": str(log),
                      "release_directory": str(destination), "command": command}
            self.state["runs"][run_id] = record
            self.save()
            print(json.dumps({"event": "release_start", "run_id": run_id, "log": str(log)}), flush=True)
            returncode = self.invoke(command, log=log, root=self.root, timeout=remaining, inherited_fds=self.inherited_fds)
            signature = release_signature(destination)
            record.update(returncode=returncode, finished_utc=utc_now(), release_signature=signature,
                          status="ready" if returncode == 0 and signature is not None else "failed")
            if record["status"] == "failed":
                record["reason"] = "release_builder_failed_or_incomplete_package"
            self.save()
            print(json.dumps({"event": "release_end", "run_id": run_id, "status": record["status"]}), flush=True)
        self.state["last_scan_utc"] = utc_now()
        complete = bool(tasks) and all(self.state["runs"].get(task["id"], {}).get("status") == "ready" for task in tasks)
        self.state["status"] = "complete" if complete else "watching"
        self.save()
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--campaign", default="configs/world/full_campaign.json")
    parser.add_argument("--releases", default="artifacts/releases")
    parser.add_argument("--status", default="runs/world/artifact_watcher/status.json")
    parser.add_argument("--source-wheel")
    parser.add_argument("--interval", type=float, default=120)
    parser.add_argument("--max-hours", type=float, default=120)
    parser.add_argument("--retry-seconds", type=float, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_hours <= 120 or args.interval <= 0 or args.max_attempts < 1:
        parser.error("Require0<max-hours<=120, interval>0, and max-attempts>=1")
    root = args.root.resolve()
    status_path = Path(args.status) if Path(args.status).is_absolute() else root / args.status
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with (status_path.parent / ".watch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "already_running", "lock": str(status_path.parent / ".watch.lock")}))
            return 0
        watcher = ReleaseWatcher(root, args.campaign, args.releases, status_path, source_wheel=args.source_wheel,
                                 retry_seconds=args.retry_seconds, max_attempts=args.max_attempts,
                                 inherited_fds=(lock.fileno(),))
        deadline = time.monotonic() + args.max_hours * 3600
        stopping = [False]
        def stop(signum, frame):
            stopping[0] = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not stopping[0] and time.monotonic() < deadline:
            if watcher.scan_once(deadline=deadline):
                return 0
            if args.once:
                break
            until = min(time.monotonic() + args.interval, deadline)
            while not stopping[0] and time.monotonic() < until:
                time.sleep(max(0, min(1, until - time.monotonic())))
        watcher.state["status"] = "stopped" if stopping[0] or args.once else "expired"
        watcher.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
