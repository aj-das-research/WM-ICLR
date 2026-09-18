"""Readiness scheduling tests use tiny temporary artifacts and no evaluator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load_worker(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("development_comparison", scripts / "run_development_comparison.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def make_result(worker, name, output):
    environment = name.split("_", 1)[0]
    data_name = "pusht_relative" if environment == "pusht" else environment
    record = {"status": "complete", "checkpoint_sha256": worker.digest(f"runs/world/{name}/best/model.pt"),
              "data_manifest_sha256": worker.digest(f"data/world/{data_name}/manifest.json"),
              "evaluator_sha256": worker.digest("src/shiftwm/evaluate.py")}
    coordinates = "checkpoint_training_action_z_scores; native actions clipped to environment bounds"
    protocol = {key: worker.OFFICIAL_PLANNER_BUDGET[key] for key in ("samples", "iterations", "elites", "horizon", "native_budget")}
    protocol.update(policy="world_model", split="development", episodes_per_dynamics=32, goal_offset=5, seed=1701,
                    search_coordinates=coordinates, evaluator_sha256=record["evaluator_sha256"],
                    run_identity={key: record[key] for key in ("checkpoint_sha256", "data_manifest_sha256")})
    record["planning"] = {"policy": "world_model", "split": "development", "protocol": protocol,
        "signature": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "planner": {**worker.OFFICIAL_PLANNER_BUDGET, "implementation": "stable_worldmodel.planning.solver.cem.CEMSolver",
                    "search_coordinates": coordinates, "goal_offset_from_end_of_history": 5},
        "summary": {"all": {"success": {"mean": .5}}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record))


def prepare_artifacts(worker):
    evaluator = Path("src/shiftwm/evaluate.py")
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("fixture evaluator identity")
    outputs = {}
    for environment, data_name in (("pusht", "pusht_relative"), ("reacher", "reacher")):
        manifest = Path(f"data/world/{data_name}/manifest.json")
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"environment": environment}))
        for mode in ("frozen", "single", "factorized", "framewise"):
            name = f"{environment}_{mode}_s0"
            run = Path("runs/world") / name
            (run / "best").mkdir(parents=True)
            (run / "best/model.pt").write_bytes(name.encode())
            (run / "training_summary.json").write_text(json.dumps({"status": "completed"}))
            output = Path("results/development_official_budget") / name / "planning_development.json"
            make_result(worker, name, output)
            outputs[name] = output
    return outputs


def test_ready_reacher_runs_before_pending_pusht_and_completed_outputs_are_reused(tmp_path, monkeypatch):
    worker = load_worker(monkeypatch)
    monkeypatch.chdir(tmp_path)
    outputs = prepare_artifacts(worker)
    pending_run = Path("runs/world/pusht_framewise_s0")
    (pending_run / "training_summary.json").write_text(json.dumps({"status": "running"}))
    for name in ("pusht_framewise_s0", "reacher_factorized_s0"):
        outputs[name].unlink()
    completed_bytes = {name: path.read_bytes() for name, path in outputs.items() if path.exists()}
    evaluated, waits = [], []

    def evaluate(command, **kwargs):
        name = Path(command[command.index("--checkpoint") + 1]).parent.name
        evaluated.append(name)
        make_result(worker, name, Path(command[command.index("--output") + 1]))
        return SimpleNamespace(returncode=0)

    def finish_training_on_wait(seconds):
        assert evaluated == ["reacher_factorized_s0"]  # The GPU was not idle while ready work existed.
        waits.append(seconds)
        (pending_run / "training_summary.json").write_text(json.dumps({"status": "completed"}))

    monkeypatch.setattr(worker.subprocess, "run", evaluate)
    monkeypatch.setattr(worker.time, "sleep", finish_training_on_wait)
    monkeypatch.setattr(sys, "argv", ["development", "--max-seconds", "1000"])
    assert worker.main() == 0
    assert evaluated == ["reacher_factorized_s0", "pusht_framewise_s0"]
    assert waits == [15]
    assert all(outputs[name].read_bytes() == raw for name, raw in completed_bytes.items())


def test_completed_output_with_changed_checkpoint_is_rejected(tmp_path, monkeypatch):
    worker = load_worker(monkeypatch)
    monkeypatch.chdir(tmp_path)
    prepare_artifacts(worker)
    Path("runs/world/pusht_frozen_s0/best/model.pt").write_bytes(b"changed weights")
    monkeypatch.setattr(sys, "argv", ["development", "--max-seconds", "1000"])
    def cannot_evaluate(*args, **kwargs):
        pytest.fail("Invalid completed output must be rejected before running another evaluation")
    monkeypatch.setattr(worker.subprocess, "run", cannot_evaluate)
    with pytest.raises(RuntimeError, match="Completed development provenance changed"):
        worker.main()
