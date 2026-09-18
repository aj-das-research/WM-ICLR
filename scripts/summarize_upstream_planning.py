#!/usr/bin/env python3
"""Collect verified upstream outcomes; missing, failed, or ambiguous runs have no score."""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import yaml
from run_evaluation_campaign import FileDigestCache

ROOT = Path(__file__).resolve().parents[1]
BLOCK_MARKER = "==== CONFIG ===="
RESULT_MARKER = "==== RESULTS ===="
TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


class InvalidOutput(ValueError): pass
class IncompleteOutput(InvalidOutput): pass


class UniqueLoader(yaml.SafeLoader):
    """Safe YAML with duplicate mapping keys rejected, including nested keys."""
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in mapping:
                raise InvalidOutput("Config contains a non-string or duplicate mapping key")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def literal(node):
    """Interpret the small upstream metrics grammar; never execute expressions."""
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if isinstance(node, ast.List): return [literal(value) for value in node.elts]
    if isinstance(node, ast.Dict):
        keys = [literal(key) for key in node.keys]
        if not all(type(key) is str for key in keys) or len(set(keys)) != len(keys):
            raise InvalidOutput("Metrics have invalid or duplicate keys")
        return dict(zip(keys, (literal(value) for value in node.values)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = literal(node.operand)
        if type(value) not in (int, float): raise InvalidOutput("Invalid signed metric")
        return -value if isinstance(node.op, ast.USub) else value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "array"
            and len(node.args) == 1 and not node.keywords):
        value = literal(node.args[0])
        if not isinstance(value, list): raise InvalidOutput("Only array([...]) is accepted")
        return value
    raise InvalidOutput("Unsupported expression in metrics")


def resolve_config(config):
    """Resolve only the three literal references present in official YAML."""
    allowed = {"${seed}": config.get("seed"),
               "${eval.num_eval}": config.get("eval", {}).get("num_eval"),
               "${eval.dataset_name}": config.get("eval", {}).get("dataset_name")}
    def walk(value):
        if isinstance(value, dict): return {key: walk(item) for key,item in value.items()}
        if isinstance(value, list): return [walk(item) for item in value]
        if isinstance(value, str) and "${" in value:
            if value not in allowed or allowed[value] is None:
                raise InvalidOutput("Unexpected config interpolation")
            if not isinstance(allowed[value], (str,int,float,bool)) or "${" in str(allowed[value]):
                raise InvalidOutput("Nonliteral config reference")
            return allowed[value]
        return value
    return walk(config)


def parse_output(text):
    if len(text) > 2_000_000: raise InvalidOutput("Result file exceeds expected size")
    count = text.count(BLOCK_MARKER)
    if count == 0: raise IncompleteOutput("No completed configuration/results block")
    if count != 1: raise InvalidOutput("Multiple appended run blocks are ambiguous")
    prefix, block = text.split(BLOCK_MARKER, 1)
    if prefix.strip(): raise InvalidOutput("Unexpected content before run block")
    if block.count(RESULT_MARKER) != 1: raise IncompleteOutput("Missing or ambiguous results delimiter")
    config_text, results = block.split(RESULT_MARKER, 1)
    try:
        config = yaml.load(config_text, Loader=UniqueLoader)
    except yaml.YAMLError as error:
        raise InvalidOutput("Invalid or unsafe configuration YAML") from error
    if not isinstance(config, dict): raise InvalidOutput("Configuration is not a mapping")
    match = re.fullmatch(r"\s*metrics:\s*(.*?)\s*evaluation_time:\s*([0-9.eE+\-]+) seconds\s*", results, re.S)
    if match is None: raise IncompleteOutput("Missing complete metrics/runtime footer")
    try:
        metrics = literal(ast.parse(match.group(1), mode="eval").body)
        seconds = float(match.group(2))
    except (SyntaxError, TypeError, ValueError, OverflowError, RecursionError) as error:
        raise InvalidOutput("Malformed metrics expression") from error
    if not isinstance(metrics, dict) or set(metrics) != {"success_rate", "episode_successes", "seeds"}:
        raise InvalidOutput("Unexpected upstream metric schema")
    successes = metrics["episode_successes"]
    if not isinstance(successes, list) or len(successes) != 50 or any(type(x) is not bool for x in successes):
        raise InvalidOutput("Exactly 50 Boolean episode successes are required")
    if metrics["seeds"] is not None:
        raise InvalidOutput("Official input files contain no seed column; unexpected reset seeds")
    rate = metrics["success_rate"]
    if type(rate) not in (float,int) or not math.isfinite(rate) or not math.isclose(rate, 100*sum(successes)/50, abs_tol=1e-9, rel_tol=1e-12):
        raise InvalidOutput("Printed success percentage disagrees with the 50 outcomes")
    if not math.isfinite(seconds) or seconds < 0: raise InvalidOutput("Invalid elapsed evaluation time")
    return {"config": resolve_config(config), "successes": successes, "success_count": sum(successes),
            "tasks": 50, "success_rate_percent": float(rate), "evaluation_seconds_including_video": seconds}


def validate_config(actual, expected, environment):
    fixed = {("seed",): 42, ("eval","num_eval"): 50, ("eval","goal_offset_steps"): 25,
             ("eval","eval_budget"): 50, ("solver","num_samples"): 300,
             ("solver","n_steps"): 30, ("solver","topk"): 30,
             ("plan_config","horizon"): 5, ("plan_config","receding_horizon"): 5,
             ("plan_config","action_block"): 5}
    for keys, value in fixed.items():
        current = actual
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise InvalidOutput("Missing official protocol identity")
            current = current[key]
        if type(current) is not int or current != value:
            raise InvalidOutput("Configuration changed the fixed official protocol")
    expected = json.loads(json.dumps(expected))
    expected["output"]["filename"] = f"{environment}_results.txt"  # explicit launch-wrapper override
    if actual != expected: raise InvalidOutput("Effective full configuration differs from official preflight/launch identity")


def validate_receipt(record, job, root):
    if str(record.get("job_id")) != str(job["job_id"]): raise InvalidOutput("Job receipt identity differs from schedule")
    command = record.get("command", [])
    if (len(command) != 3 or Path(command[0]).name != "bash" or command[2] != job["environment"]
            or (root / command[1]).resolve() != root / "scripts/run_upstream_planning_reproduction.sh"):
        raise InvalidOutput("Job command is not the scheduled full upstream evaluation")
    if record.get("status") != "complete" or record.get("exit_code") != 0:
        raise InvalidOutput("A successful completed job receipt is required")
    try:
        start = dt.datetime.fromisoformat(record["start_utc"])
        end = dt.datetime.fromisoformat(record["end_utc"])
    except (KeyError, ValueError, TypeError) as error:
        raise InvalidOutput("Missing valid job timing receipt") from error
    if start.tzinfo is None or end.tzinfo is None or end < start:
        raise InvalidOutput("Job timing receipt is not a valid UTC interval")
    return start.timestamp(), end.timestamp()


def verify_selection(log_text, tasks):
    expected = [task["row"] for task in tasks]
    if len(expected) != 50 or len(set(expected)) != 50 or expected != sorted(expected):
        raise InvalidOutput("Preflight must identify 50 sorted unique sampled rows")
    arrays = [[int(x) for x in block.split()] for block in re.findall(r"(?m)^\s*\[([\d\s]+)\]\s*$", log_text)]
    if expected not in arrays: raise InvalidOutput("Scheduled stdout does not confirm the exact preflight row selection")


class SourceVerifier:
    """Check archived Python/config sources against actual pinned Git blobs."""
    def __init__(self, root, digests):
        self.root, self.digests, self.expected = root, digests, {}

    def verify(self, preflight):
        groups = []
        for name, key in (("le-wm", "lewm"), ("stable-worldmodel", "historical_stable_worldmodel")):
            revision = preflight["source_revisions"][key]
            tree = subprocess.run(["git", "-C", str(self.root / "external" / name), "ls-tree", "-r", revision],
                                  check=True, text=True, capture_output=True).stdout
            files = {}
            for line in tree.splitlines():
                metadata, relative = line.split("\t", 1)
                if not relative.endswith((".py", ".yaml", ".yml", ".toml", ".json")): continue
                path = self.root / "artifacts/upstream_planning_reproduction" / name / relative
                digest = self.digests.sha256(path)
                git_blob = metadata.split()[2]
                cache_key = (str(path), git_blob, digest)
                if cache_key not in self.expected:
                    raw = path.read_bytes()
                    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != git_blob:
                        raise InvalidOutput(f"Staged source differs from pin: {name}/{relative}")
                    self.expected[cache_key] = True
                files[relative] = {"sha256": digest, "git_blob": git_blob}
            if not files: raise InvalidOutput("Pinned snapshot has no verified source/config files")
            groups.append({"name": name, "revision": revision, "files_checked": len(files), "files": files})
        return groups


def scheduler_state(job_id):
    try:
        command = subprocess.run(["scontrol", "show", "job", str(job_id), "-o"], check=True,
                                 text=True, capture_output=True, timeout=10)
        fields = dict(re.findall(r"(?:^|\s)(\w+)=([^ ]*)", command.stdout))
        return {key: fields[key] for key in ("JobState","Reason","Dependency","ExitCode","NodeList") if key in fields}
    except (OSError, subprocess.SubprocessError) as error:
        return {"JobState": "UNKNOWN", "query_error": str(error)}


def summarize_job(root, job, preflight, digests, query_scheduler=scheduler_state):
    environment = job["environment"]
    result = {"environment": environment, "job_id": job["job_id"], "status": "pending",
              "success_count": None, "tasks": 50, "success_rate_percent": None,
              "evaluation_seconds_including_video": None, "scheduler": query_scheduler(job["job_id"])}
    receipt_path = root / "runs/jobs" / str(job["job_id"]) / "record.json"
    output_path = root / "artifacts/upstream_planning_reproduction/cache" / environment / f"{environment}_results.txt"
    log_path = root / job["log"]
    state = result["scheduler"].get("JobState", "UNKNOWN").split("+")[0]
    result["status"] = "failed" if state in TERMINAL_FAILURES else ("queued" if state == "PENDING" else "running" if state in {"RUNNING","COMPLETING"} else "completion_unverified")
    if not receipt_path.is_file(): return result
    try:
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError:
        result.update(status="receipt_write_in_progress")
        return result
    result["receipt"] = {"file": str(receipt_path.relative_to(root)), "sha256": digests.sha256(receipt_path), "record": receipt}
    if receipt.get("status") == "failed":
        result.update(status="failed", error="Job receipt reports failure; no success rate inferred")
        return result
    if receipt.get("status") != "complete":
        if state not in TERMINAL_FAILURES: result["status"] = "running" if state != "UNKNOWN" else "completion_unverified"
        return result
    try:
        start, end = validate_receipt(receipt, job, root)
        if state in TERMINAL_FAILURES: raise InvalidOutput("Scheduler failure conflicts with successful receipt")
        if state in {"PENDING", "RUNNING", "COMPLETING"}:
            result["status"] = "awaiting_scheduler_completion"
            return result
        if not output_path.is_file(): raise IncompleteOutput("Completed job has no official results file")
        if not start - 3 <= output_path.stat().st_mtime <= end + 3:
            raise InvalidOutput("Result file modification time is outside this job's recorded execution")
        parsed = parse_output(output_path.read_text())
        validate_config(parsed["config"], preflight["environments"][environment]["config"], environment)
        if not log_path.is_file(): raise InvalidOutput("Missing scheduled stdout provenance")
        tasks = preflight["environments"][environment]["official_data"]["selected_tasks"]
        verify_selection(log_path.read_text(), tasks)
        result.update(parsed, status="complete", output_file=str(output_path.relative_to(root)),
                      output_sha256=digests.sha256(output_path), log_file=job["log"], log_sha256=digests.sha256(log_path),
                      task_outcomes=[{**task, "success": success} for task,success in zip(tasks, parsed["successes"])])
        result.pop("successes")
    except IncompleteOutput as error: result.update(status="incomplete_output", error=str(error))
    except (InvalidOutput, KeyError, OSError) as error: result.update(status="invalid_output", error=str(error))
    return result


def collect(root, digests, source_verifier):
    preflight_path = root / "reports/evidence/upstream_planning_preflight.json"
    schedule_path = root / "reports/evidence/upstream_reproduction_schedule.json"
    preflight = json.loads(preflight_path.read_text())
    allocations_path = root / "reports/evidence/full_evaluation_allocations.json"
    allocations = json.loads(allocations_path.read_text())
    jobs = []
    for allocation in allocations["upstream_reproduction_jobs"]["jobs"]:
        output_args = [arg.split("=",1)[1] for arg in allocation["command"] if arg.startswith("--output=")]
        if len(output_args) != 1 or allocation.get("full_official_tasks") != 50:
            raise InvalidOutput("Authoritative allocation has no unique log path/full 50-task identity")
        original_dependency = next((arg.split("=",1)[1] for arg in allocation["command"] if arg.startswith("--dependency=")), None)
        jobs.append({"job_id": int(allocation["job_id"]), "environment": allocation["environment"],
                     "log": output_args[0].replace("%j", str(allocation["job_id"])),
                     "dependency": allocation.get("current_dependency", original_dependency)})
    if sorted(job["environment"] for job in jobs) != ["pusht", "reacher"]:
        raise InvalidOutput("Schedule must identify exactly one job per official environment")
    sources = source_verifier.verify(preflight)
    artifact_manifest = json.loads((root / "references/world_artifact_sources.json").read_text())
    weights, datasets = {}, {}
    for job in jobs:
        environment = job["environment"]
        weights[environment] = {}
        for entry in next(x for x in artifact_manifest["models"] if x["environment"] == environment)["files"]:
            digest = digests.sha256(root / entry["path"])
            if digest != entry["sha256"] or digest != preflight["environments"][environment]["weights"][entry["name"]]:
                raise InvalidOutput("Released checkpoint/config does not match recorded source identity")
            linked = root / "artifacts/upstream_planning_reproduction/cache/checkpoints" / environment / "lewm" / entry["name"]
            if linked.resolve() != (root / entry["path"]).resolve(): raise InvalidOutput("Runtime checkpoint link changed")
            weights[environment][entry["name"]] = digest
        data = preflight["environments"][environment]["official_data"]
        original = root / data["file"]
        alias = "pusht_expert_train.h5" if environment == "pusht" else "dmc/reacher_random.h5"
        linked = root / "artifacts/upstream_planning_reproduction/cache/datasets" / alias
        if linked.resolve() != original.resolve() or original.stat().st_size != data["bytes"]:
            raise InvalidOutput("Runtime official data path/size changed")
        datasets[environment] = {"file": data["file"], "bytes": data["bytes"], "selection_sha256": data["selection_sha256"],
                                 "full_extracted_hdf5_rehashed": False}
    provenance = {"preflight_sha256": digests.sha256(preflight_path), "schedule_sha256": digests.sha256(schedule_path),
                  "authoritative_allocations_sha256": digests.sha256(allocations_path), "scheduled_jobs": jobs,
                  "artifact_manifest_sha256": digests.sha256(root / "references/world_artifact_sources.json"),
                  "dependency_lock_sha256": digests.sha256(root / "reports/evidence/upstream_reproduction_requirements.lock.txt"),
                  "launcher_sha256": digests.sha256(root / "scripts/run_upstream_planning_reproduction.sh"),
                  "collector_sha256": digests.sha256(Path(__file__).resolve()),
                  "source_snapshots": sources, "weights": weights, "official_data": datasets}
    return {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": "Original released data/protocol on a compatible historical snapshot; not an author-established training runtime pin.",
            "runtime_units": "world.evaluate elapsed seconds, including simulator execution and video encoding; not solver-only latency",
            "provenance_scope": "Current staged code/config checked against Git pins; weights freshly hash-verified with metadata cache. Extracted HDF5 files checked by path/size and previously verified selected-frame identity, not fully rehashed.",
            "provenance": provenance, "runs": [summarize_job(root,job,preflight,digests) for job in jobs]}


def markdown(ledger):
    lines = ["# Official-protocol LeWM reproduction results", "", f"Updated {ledger['generated_at_utc']}.", "",
             ledger["scope"], "", "| Environment | Job | Status | Successes / tasks | Success (%) | Evaluation seconds* |",
             "|---|---:|---|---:|---:|---:|"]
    for row in ledger["runs"]:
        values = [f"{row['success_count']}/50", f"{row['success_rate_percent']:.2f}", f"{row['evaluation_seconds_including_video']:.2f}"] if row["status"] == "complete" else ["—"]*3
        lines.append("| " + " | ".join([row["environment"], str(row["job_id"]), row["status"], *values]) + " |")
    lines += ["", "Current scheduled dependencies (read from the authoritative allocation ledger):"]
    for job in ledger.get("provenance", {}).get("scheduled_jobs", []):
        lines.append(f"- {job['environment']} job {job['job_id']}: `{job['dependency']}`.")
    lines += ["", "*Runtime includes simulator execution and video encoding inside `world.evaluate`; it is not solver-only GPU latency.",
              "", "Missing, failed, truncated, ambiguous, or unverified runs have no numerical score. Completion requires a successful matching job receipt, complete output, the exact full official configuration and 50 Boolean outcomes, percentage consistency, and logged task identities. No published score is assumed. These results are separate from adaptation-benchmark comparisons.",
              "", "Full configurations, per-task outcomes when available, receipt/log identities, and source/checkpoint/preflight/schedule hashes are in `reports/evidence/upstream_planning_results.json`."]
    for row in ledger["runs"]:
        if row.get("error"): lines += ["", f"{row['environment']}: {row['error']}."]
    return "\n".join(lines) + "\n"


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(value)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--max-hours", type=float, default=120)
    args = parser.parse_args()
    if args.interval < 60 or not 0 < args.max_hours <= 120: parser.error("Require interval >=60 seconds and 0<max-hours<=120")
    root = args.root.resolve()
    directory = root / "reports/evidence"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".upstream_results_watch.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: parser.error("An upstream result collector is already running")
    digests = FileDigestCache()
    verifier = SourceVerifier(root, digests)
    deadline = time.monotonic() + args.max_hours*3600
    while True:
        try:
            ledger = collect(root, digests, verifier)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            ledger = {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "scope": "Collection failed validation; no reproduction result is reported.", "collection_error": str(error),
                      "runs": [{"environment": env, "job_id": "unknown", "status": "collection_error",
                                "success_count": None, "success_rate_percent": None,
                                "evaluation_seconds_including_video": None, "error": str(error)} for env in ("pusht","reacher")]}
        atomic_write(directory / "upstream_planning_results.json", json.dumps(ledger, indent=2, allow_nan=False) + "\n")
        atomic_write(root / "reports/upstream_planning_results.md", markdown(ledger))
        print(json.dumps({"time": ledger["generated_at_utc"], "runs": [{"environment": r["environment"], "status": r["status"]} for r in ledger["runs"]]}), flush=True)
        if not args.watch or all(r["status"] == "complete" for r in ledger["runs"]) or time.monotonic() >= deadline: break
        time.sleep(args.interval)


if __name__ == "__main__": main()
