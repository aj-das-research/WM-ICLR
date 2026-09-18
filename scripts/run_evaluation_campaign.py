#!/usr/bin/env python3
"""Run the full preregistered evaluation grid on independent GPU workers.

Each task has a held advisory lock inherited by its evaluation process. A
terminated Slurm allocation therefore leaves no permanent running lease. Only
completed training runs supply immutable validation-selected checkpoints.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


OFFICIAL_PLANNER_BUDGET = {"samples": 300, "iterations": 30, "elites": 30,
                           "horizon": 5, "native_budget": 50, "action_block": 5,
                           "receding_horizon": 1, "history_steps_charged": 10}
CONTROL_METADATA_BUDGET = {**OFFICIAL_PLANNER_BUDGET, "samples": 128, "iterations": 5, "elites": 16}
PLANNER_IDENTITY = {"implementation": "stable_worldmodel.planning.solver.cem.CEMSolver",
                    "search_coordinates": "checkpoint_training_action_z_scores; native actions clipped to environment bounds",
                    "goal_offset_from_end_of_history": 5}


class FileDigestCache:
    """Avoid rereading immutable large files on every worker task scan.

    A cache entry is valid only for the same resolved path, file size and
    nanosecond modification time. Metadata changes force a new streaming hash;
    a file changing while being hashed is rejected rather than cached.
    """
    def __init__(self):
        self._entries = {}

    def sha256(self, path):
        resolved = Path(path).resolve()
        before = resolved.stat()
        identity = (before.st_size, before.st_mtime_ns)
        cached = self._entries.get(resolved)
        if cached is not None and cached[:2] == identity:
            return cached[2]
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = resolved.stat()
        if (after.st_size, after.st_mtime_ns) != identity:
            raise RuntimeError(f"File changed while computing SHA256: {resolved}")
        value = digest.hexdigest()
        self._entries[resolved] = (*identity, value)
        return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def evaluation_tasks(campaign, *, samples=300, iterations=30, elites=30):
    if samples < 2 or iterations < 1 or not 2 <= elites <= samples:
        raise ValueError("Require samples>=2, iterations>=1 and2<=elites<=samples")
    main_budget = {**OFFICIAL_PLANNER_BUDGET, "samples": samples, "iterations": iterations, "elites": elites}
    training = json.loads(Path(campaign).read_text())["tasks"]
    configs = [(task, json.loads(Path(task["config"]).read_text())) for task in training]
    envs = {}
    model_runs = []
    for task, cfg in configs:
        environment = Path(cfg["pretrained_dir"]).name
        envs.setdefault(environment, cfg)
        model_runs.append({"id": task["id"], "environment": environment, "config": cfg,
                           "checkpoint": str(Path(cfg["output_dir"]) / "best"),
                           "training_summary": str(Path(cfg["output_dir"]) / "training_summary.json")})
    frozen = [{"id": f"{env}_frozen_s0", "environment": env, "config": cfg,
               "checkpoint": f"runs/world/{env}_frozen_s0/best", "training_summary": None}
              for env, cfg in envs.items()]
    tasks = []
    # Ordering is scientific priority, independent of which GPU becomes free.
    for run in frozen + model_runs:
        for split in ("test", "extrapolation"):
            tasks.append({**run, "kind": "forecast", "split": split, "policy": "world_model"})
    for run in frozen:
        for policy in ("random", "replay_oracle"):
            for split in ("test", "extrapolation"):
                tasks.append({**run, "kind": "planning", "split": split, "policy": policy})
    for run in frozen + model_runs:
        for split in ("test", "extrapolation"):
            tasks.append({**run, "kind": "planning", "split": split, "policy": "world_model"})
    for task in tasks:
        suffix = "" if task["policy"] == "world_model" else "_" + task["policy"]
        task["task_id"] = f"{task['id']}_{task['kind']}_{task['split']}{suffix}"
        task["output"] = f"results/world/{task['id']}/{task['kind']}_{task['split']}{suffix}.json"
        if task["kind"] == "planning":
            task["uses_model_planner"] = task["policy"] == "world_model"
            task["planner_budget"] = main_budget.copy() if task["uses_model_planner"] else CONTROL_METADATA_BUDGET.copy()
            task["planner_metadata_interpretation"] = ("active_CEM_search" if task["uses_model_planner"]
                                                       else "sampling_fields_unused_by_random_or_replay_policy")
    return tasks


def validate_completed_planning(result, task, episodes):
    """Reject mismatched or relabelled main budgets before reusing an output."""
    planning = result["planning"]
    protocol = planning["protocol"]
    if planning.get("policy") != task["policy"] or planning.get("split") != task["split"]:
        raise RuntimeError("Completed planning policy/split differs from requested task")
    if protocol.get("episodes_per_dynamics") != episodes:
        raise RuntimeError("Completed evaluation used a different episode budget")
    expected = task["planner_budget"]
    # Random/replay never invoke CEM: its sampling metadata is scientifically
    # irrelevant and does not require recomputing those control trajectories.
    keys = tuple(expected) if task["policy"] == "world_model" else ("native_budget", "action_block", "history_steps_charged")
    for key in keys:
        if planning.get("planner", {}).get(key) != expected[key]:
            raise RuntimeError(f"Completed planning budget differs for {key}")
        if key in ("samples", "iterations", "elites", "horizon", "native_budget") and protocol.get(key) != expected[key]:
            raise RuntimeError(f"Completed hashed protocol budget differs for {key}")
    if protocol.get("policy") != task["policy"] or protocol.get("split") != task["split"]:
        raise RuntimeError("Completed protocol policy/split differs from requested task")
    if protocol.get("goal_offset") != 5 or protocol.get("seed") != 1701:
        raise RuntimeError("Completed protocol changed the goal offset or evaluation seed")
    if task["policy"] == "world_model":
        for key, expected_value in PLANNER_IDENTITY.items():
            if planning.get("planner", {}).get(key) != expected_value:
                raise RuntimeError(f"Completed planner identity differs for {key}")
        if protocol.get("search_coordinates") != PLANNER_IDENTITY["search_coordinates"]:
            raise RuntimeError("Completed protocol changed search coordinates")
    if protocol.get("evaluator_sha256") != result.get("evaluator_sha256"):
        raise RuntimeError("Completed protocol changed evaluator identity")
    if any(protocol.get("run_identity", {}).get(key) != result.get(key)
           for key in ("checkpoint_sha256", "data_manifest_sha256")):
        raise RuntimeError("Completed protocol changed checkpoint/data identity")
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if planning.get("signature") != signature:
        raise RuntimeError("Completed planning protocol signature is inconsistent")


class CompletedEvaluationCache:
    """Cache validation, never the large per-frame/per-episode records.

    Reuse requires the same resolved file, device/inode, size, modification and
    change times, and requested scientific identities. Changed requests and
    files undergo full validation again. Partial or concurrently changed files
    are never cached. This is a per-process optimization for immutable outputs,
    not a replacement for the campaign's locks or provenance validation.
    """
    def __init__(self):
        self._entries = {}

    @staticmethod
    def _file_identity(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def is_complete(self, path, task, episodes, checkpoint_sha, dataset_sha, evaluator_sha):
        resolved = Path(path).resolve()
        before = self._file_identity(resolved)
        request = json.dumps({
            "checkpoint_sha256": checkpoint_sha, "data_manifest_sha256": dataset_sha,
            "evaluator_sha256": evaluator_sha, "kind": task["kind"],
            "split": task["split"], "policy": task["policy"],
            "environment": task.get("environment"), "episodes": episodes,
            "planner_budget": task.get("planner_budget"),
            "planner_identity": PLANNER_IDENTITY,
        }, sort_keys=True)
        if self._entries.get(resolved) == (before, request):
            return True
        # Remove stale validation before reading: a failed revalidation must not
        # leave a successful cache entry behind for this path.
        self._entries.pop(resolved, None)
        result = json.loads(resolved.read_text())
        complete = result.get("status") == "complete"
        if complete:
            if result.get("checkpoint_sha256") != checkpoint_sha or result.get("data_manifest_sha256") != dataset_sha:
                raise RuntimeError(f"Immutable completed evaluation changed: {resolved}")
            if result.get("evaluator_sha256") != evaluator_sha:
                raise RuntimeError(f"Completed evaluation used different evaluator code: {resolved}")
            if task.get("environment") is not None and result.get("environment") != task["environment"]:
                raise RuntimeError(f"Completed evaluation used different environment: {resolved}")
            if not isinstance(result.get(task["kind"]), dict):
                raise RuntimeError(f"Completed evaluation lacks requested {task['kind']} record: {resolved}")
            if task["kind"] == "planning":
                validate_completed_planning(result, task, episodes)
            elif task["kind"] == "forecast":
                if task["policy"] != "world_model" or result["forecast"].get("split") != task["split"]:
                    raise RuntimeError(f"Completed forecast policy/split differs from requested task: {resolved}")
                if result["forecast"].get("kind") != "fixed_reference_forecasting":
                    raise RuntimeError(f"Completed forecast uses an unknown protocol: {resolved}")
            else:
                raise RuntimeError(f"Unknown evaluation kind: {task['kind']}")
        if self._file_identity(resolved) != before:
            raise RuntimeError(f"File changed while validating completed evaluation: {resolved}")
        if complete:
            self._entries[resolved] = (before, request)
        return complete


def mark_completed_reused(state_path, state, output, checkpoint_sha):
    """Preserve original job/timing provenance and avoid unchanged state writes."""
    updated = {**state, "status": "complete", "output": str(output),
               "checkpoint_sha256": checkpoint_sha, "reused": True}
    if updated != state:
        write_json(state_path, updated)


def evaluation_command(task, episodes, max_seconds):
    command = [sys.executable, "-m", "shiftwm.evaluate", "--checkpoint", task["checkpoint"],
               "--data", task["config"]["data_root"], "--output", task["output"],
               "--kind", task["kind"], "--split", task["split"], "--policy", task["policy"],
               "--episodes", str(episodes), "--max-runtime-seconds", str(max_seconds)]
    if task["kind"] == "forecast":
        command += ["--feature-cache", task["config"]["dataset_kwargs"]["feature_cache"]]
    elif task["policy"] == "world_model":
        budget = task["planner_budget"]
        command += ["--samples", str(budget["samples"]), "--iterations", str(budget["iterations"]),
                    "--elites", str(budget["elites"]), "--horizon", str(budget["horizon"]),
                    "--native-budget", str(budget["native_budget"])]
    return command


def execution_context(task, scope, prior_progress, runner_sha):
    """Timing claims need explicit resource provenance for every resumed segment."""
    resumed = len(prior_progress.get("records", []))
    prior = prior_progress.get("execution_context", {})
    dedicated = scope == "dedicated_gpu_campaign"
    known_dedicated_history = not resumed or (
        prior.get("main_efficiency_claim_eligible") is True
        and prior.get("shared_gpu_with_training") is False)
    shared = False if dedicated and known_dedicated_history else None
    if prior.get("shared_gpu_with_training") is True and resumed:
        shared = True
    eligible = (dedicated and known_dedicated_history and task["kind"] == "planning"
                and task["policy"] == "world_model" and task["split"] in ("test", "extrapolation"))
    return {"execution_scope": scope, "main_efficiency_claim_eligible": eligible,
            "shared_gpu_with_training": shared, "resumed_episode_count": resumed,
            "all_timed_segments_explicitly_dedicated": dedicated and known_dedicated_history,
            "job_id": os.getenv("SLURM_JOB_ID"), "node": os.uname().nodename,
            "runner_sha256": runner_sha,
            "provenance_basis": "Explicit runner flag; Slurm job membership alone does not establish exclusivity."}


def ready(task):
    config = task["config"]
    if task["training_summary"]:
        summary = Path(task["training_summary"])
        if not summary.exists():
            return False
        try:
            if json.loads(summary.read_text()).get("status") != "completed":
                return False
        except json.JSONDecodeError:
            return False  # Summary writer has not completed its final write yet.
    if not (Path(task["checkpoint"]) / "model.pt").exists():
        return False
    if not (Path(config["data_root"]) / "manifest.json").exists():
        return False
    if task["kind"] == "forecast" and not (Path(config["dataset_kwargs"]["feature_cache"]) / "manifest.json").exists():
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="configs/world/full_campaign.json")
    parser.add_argument("--state", default="runs/world/evaluation_campaign_state")
    parser.add_argument("--max-seconds", type=int, default=27000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--execution-scope", choices=("unspecified", "dedicated_gpu_campaign"),
                        default="unspecified", help="Explicit resource provenance for main timing claims")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--exit-when-not-ready", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frozen-controls-only", action="store_true",
                           help="Full frozen forecasts plus random/replay controls; no timed learned planning")
    selection.add_argument("--forecasts-only", action="store_true",
                           help="All frozen and trained forecast tasks; no CEM or control-policy planning")
    args = parser.parse_args()
    all_tasks = evaluation_tasks(args.campaign, samples=args.samples, iterations=args.iterations, elites=args.elites)
    tasks = all_tasks
    if args.frozen_controls_only:
        tasks = [task for task in tasks if task["training_summary"] is None and
                 (task["kind"] == "forecast" or task["policy"] != "world_model")]
    elif args.forecasts_only:
        tasks = [task for task in tasks if task["kind"] == "forecast"]
    worker_selection = ("forecasts_only" if args.forecasts_only else
                        "frozen_controls_only" if args.frozen_controls_only else "all")
    state_dir = Path(args.state)
    state_dir.mkdir(parents=True, exist_ok=True)
    write_json(state_dir / "task_grid.json", {"tasks": all_tasks, "episodes_per_dynamics": args.episodes,
               "worker_selection": worker_selection, "selected_task_ids": [task["task_id"] for task in tasks],
               "main_planner_budget": {**OFFICIAL_PLANNER_BUDGET, "samples": args.samples,
                                        "iterations": args.iterations, "elites": args.elites},
               "main_planner_identity": PLANNER_IDENTITY,
               "budget_reference": "external/le-wm/config/eval/solver/cem.yaml",
               "reference_sampling_budget_used": (args.samples, args.iterations, args.elites) == (300, 30, 30),
               "control_sampling_metadata_unused": True,
               "execution_scope": args.execution_scope})
    started = time.monotonic()
    failed = 0
    failed_tasks = set()
    digests = FileDigestCache()
    completed = CompletedEvaluationCache()
    evaluator_sha = digests.sha256("src/shiftwm/evaluate.py")
    runner_sha = digests.sha256(__file__)
    while time.monotonic() - started < args.max_seconds - 180:
        chosen = None
        waiting = False
        for task in tasks:
            if task["task_id"] in failed_tasks:
                continue  # A later worker wave may retry; this process never spins on one failure.
            state_path = state_dir / (task["task_id"] + ".json")
            task_lock = (state_dir / (task["task_id"] + ".lock")).open("a+")
            try:
                fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                task_lock.close()
                waiting = True
                continue
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("status") == "failed" and not args.retry_failed:
                task_lock.close()
                continue
            if not ready(task):
                task_lock.close()
                waiting = True
                continue
            model_file = Path(task["checkpoint"]) / "model.pt"
            checkpoint_sha = digests.sha256(model_file)
            dataset_sha = digests.sha256(Path(task["config"]["data_root"]) / "manifest.json")
            output = Path(task["output"])
            if output.exists() and completed.is_complete(
                    output, task, args.episodes, checkpoint_sha, dataset_sha, evaluator_sha):
                if chosen is not None:
                    raise RuntimeError("Task selection state corrupted")
                mark_completed_reused(state_path, state, output, checkpoint_sha)
                task_lock.close()
                continue
            chosen = task
            write_json(state_path, {"status": "running", "job_id": os.getenv("SLURM_JOB_ID"),
                                    "node": os.uname().nodename, "start_unix": time.time(),
                                    "checkpoint_sha256": checkpoint_sha, "output": str(output)})
            break
        if chosen is None:
            if waiting and not args.exit_when_not_ready:
                time.sleep(15)
                continue
            break
        output = Path(chosen["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        remaining = int(args.max_seconds - (time.monotonic() - started) - 120)
        command = evaluation_command(chosen, args.episodes, remaining)
        progress_path = output.with_suffix(".progress.json")
        prior_progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        context = execution_context(chosen, args.execution_scope, prior_progress, runner_sha)
        log = output.with_suffix(".log")
        print(json.dumps({"event": "evaluation_start", "task": chosen["task_id"], "command": command}), flush=True)
        with log.open("a", buffering=1) as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    pass_fds=(task_lock.fileno(),))
        evaluation = json.loads(output.read_text()) if output.exists() else {}
        if result.returncode == 0 and evaluation.get("status") == "complete" and chosen["kind"] == "planning":
            validate_completed_planning(evaluation, chosen, args.episodes)
        if result.returncode == 0 and evaluation:
            # Existing completed files take the reuse branch above and are never
            # relabelled. Unknown provenance in any resumed segment suppresses
            # timing claims even when the current segment is dedicated.
            evaluation["execution_context"] = context
            write_json(output, evaluation)
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                progress["execution_context"] = context
                write_json(progress_path, progress)
        status = ("complete" if result.returncode == 0 and evaluation.get("status") == "complete"
                  else "pending" if result.returncode == 0 and evaluation.get("status") == "interrupted"
                  else "failed")
        state = json.loads(state_path.read_text())
        state.update(status=status, end_unix=time.time(), returncode=result.returncode, log=str(log))
        write_json(state_path, state)
        task_lock.close()
        print(json.dumps({"event": "evaluation_end", "task": chosen["task_id"], "status": status}), flush=True)
        failed += status == "failed"
        if status == "failed":
            failed_tasks.add(chosen["task_id"])
        if status == "pending":
            break
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
