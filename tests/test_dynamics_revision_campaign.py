"""Allocation-stage invariants; fixtures are not research measurements."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "revision_campaign", Path(__file__).parents[1] / "scripts/run_dynamics_revision_campaign.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup(root):
    for name in ("scripts/run_dynamics_revision_campaign.py", "scripts/train_dynamics_revision.py",
                 "scripts/evaluate_dynamics_revision.py", "src/shiftwm/dynamics_revision.py"):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# isolated fixture\n")
    for env in module.ENVIRONMENTS:
        module.atomic_json(root / f"configs/dynamics_revision/{env}_s0.json", {
            "seed": 0, "epochs": 30, "output_dir": f"runs/dynamics_revision/{env}_s0",
            "max_runtime_seconds": 6000})


def state(root):
    return module.load_json(root / "runs/dynamics_revision/campaign_state/status.json")


def test_interrupted_training_stays_pending_and_never_starts_evaluation(tmp_path):
    setup(tmp_path)
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        module.atomic_json(tmp_path / "runs/dynamics_revision/pusht_s0/training_summary.json",
                           {"status": "interrupted_checkpoint_saved", "completed_epochs": 12})
        return SimpleNamespace(returncode=75)
    assert module.pipeline(root=tmp_path, runner=runner) == 0
    assert len(commands) == 1
    assert "--resume-if-present" in commands[0]
    assert state(tmp_path)["status"] == "pending"
    assert state(tmp_path)["tasks"]["pusht"]["planning"] == "pending"


def test_failure_is_not_relabelled_pending_or_complete(tmp_path):
    setup(tmp_path)
    assert module.pipeline(root=tmp_path, runner=lambda *a, **k: SimpleNamespace(returncode=7)) == 1
    assert state(tmp_path)["status"] == "failed"
    assert state(tmp_path)["returncode"] == 7


def test_full_pipeline_keeps_development_only_and_checks_all_stages(tmp_path):
    setup(tmp_path)
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        if "--config" in command:
            config = module.load_json(command[command.index("--config") + 1])
            out = tmp_path / config["output_dir"]
            module.atomic_json(out / "training_summary.json", {"status": "completed", "completed_epochs": 30})
            (out / "best").mkdir()
            (out / "best/model.pt").write_bytes(b"fixture")
        else:
            env = command[command.index("--environment") + 1]
            module.atomic_json(tmp_path / f"results/development_dynamics_revision/{env}_s0/planning_development.json",
                               {"status": "complete", "planning": {"status": "complete",
                                "split": "development", "records": [{}] * 32}})
        return SimpleNamespace(returncode=0)
    assert module.pipeline(root=tmp_path, runner=runner) == 0
    assert len(commands) == 4
    assert state(tmp_path)["status"] == "complete"
    assert all("--split" not in command for command in commands)
    assert all(t["training"] == t["planning"] == "complete" for t in state(tmp_path)["tasks"].values())


def test_completed_header_with_wrong_epoch_is_rejected(tmp_path):
    setup(tmp_path)
    def runner(*args, **kwargs):
        module.atomic_json(tmp_path / "runs/dynamics_revision/pusht_s0/training_summary.json",
                           {"status": "completed", "completed_epochs": 1})
        return SimpleNamespace(returncode=0)
    with pytest.raises(ValueError, match="missing epochs"):
        module.pipeline(root=tmp_path, runner=runner)


def test_no_new_child_is_started_without_checkpoint_headroom(tmp_path):
    setup(tmp_path)
    ticks = iter([0, 6500, 6500, 6500])
    def forbidden(*args, **kwargs):
        pytest.fail("Allocation is too close to its limit to start training")
    assert module.pipeline(root=tmp_path, runner=forbidden, clock=lambda: next(ticks)) == 0
    assert state(tmp_path)["status"] == "pending"
