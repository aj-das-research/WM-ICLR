"""Allocation-stage invariants; fixtures are not research measurements."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "revision_campaign", Path(__file__).parents[1] / "scripts/run_rollout_revision_campaign.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup(root):
    sources = ("scripts/run_rollout_revision_campaign.py", "scripts/train_rollout_revision.py",
               "scripts/evaluate_rollout_revision.py", "src/shiftwm/rollout_revision.py",
               "src/shiftwm/dynamics_revision.py", "src/shiftwm/model.py", "src/shiftwm/checkpoint.py",
               "src/shiftwm/data.py", "src/shiftwm/train.py", "src/shiftwm/upstream.py",
               "src/shiftwm/evaluate.py", "src/shiftwm/generate.py",
               "scripts/evaluate_dynamics_revision.py", "scripts/evaluate_goal_calibration_intervention.py",
               "reports/rollout_revision_protocol.md")
    for name in sources:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# isolated fixture\n")
    for env in module.ENVIRONMENTS:
        for obj in module.OBJECTIVES:
            for ctx in module.CONTEXT_MODES:
                run_id = f"{env}_{obj}_{ctx}_s0"
                module.atomic_json(root / f"configs/rollout_revision/{run_id}.json", {
                    "seed": 0, "epochs": 30, "output_dir": f"runs/rollout_revision/{run_id}",
                    "revision": {"objective": obj, "context_mode": ctx}, "max_runtime_seconds": 14400})


def state(root):
    return module.load_json(root / "runs/rollout_revision/campaign_state/pusht.json")


def test_interrupted_training_stays_pending_and_never_starts_evaluation(tmp_path):
    setup(tmp_path)
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        module.atomic_json(tmp_path / "runs/rollout_revision/pusht_teacher_forced_inferred_s0/training_summary.json",
                           {"status": "interrupted_checkpoint_saved", "completed_epochs": 12})
        return SimpleNamespace(returncode=75)
    assert module.pipeline("pusht", root=tmp_path, runner=runner) == 0
    assert len(commands) == 1
    assert "--resume-if-present" in commands[0]
    assert state(tmp_path)["status"] == "pending"
    assert state(tmp_path)["tasks"]["pusht_teacher_forced_inferred_s0"]["planning"] == "pending"


def test_failure_is_not_relabelled_pending_or_complete(tmp_path):
    setup(tmp_path)
    assert module.pipeline("pusht", root=tmp_path, runner=lambda *a, **k: SimpleNamespace(returncode=7)) == 1
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
            run_id = Path(command[command.index("--checkpoint") + 1]).parent.name
            module.atomic_json(tmp_path / f"results/development_rollout_revision/{run_id}/planning_development.json",
                               {"status": "complete", "planning": {"status": "complete",
                                "split": "development", "records": [{}] * 32}})
        return SimpleNamespace(returncode=0)
    assert module.pipeline("pusht", root=tmp_path, runner=runner) == 0
    assert len(commands) == 8
    assert state(tmp_path)["status"] == "complete"
    assert all("--split" not in command for command in commands)
    assert all(t["training"] == t["planning"] == "complete" for t in state(tmp_path)["tasks"].values())


def test_completed_header_with_wrong_epoch_is_rejected(tmp_path):
    setup(tmp_path)
    def runner(*args, **kwargs):
        module.atomic_json(tmp_path / "runs/rollout_revision/pusht_teacher_forced_inferred_s0/training_summary.json",
                           {"status": "completed", "completed_epochs": 1})
        return SimpleNamespace(returncode=0)
    with pytest.raises(ValueError, match="missing epochs"):
        module.pipeline("pusht", root=tmp_path, runner=runner)


def test_no_new_child_is_started_without_checkpoint_headroom(tmp_path):
    setup(tmp_path)
    ticks = iter([0, 26900, 26900, 26900])
    def forbidden(*args, **kwargs):
        pytest.fail("Allocation is too close to its limit to start training")
    assert module.pipeline("pusht", root=tmp_path, runner=forbidden, clock=lambda: next(ticks)) == 0
    assert state(tmp_path)["status"] == "pending"


def test_protocol_changes_rejected_across_continuations(tmp_path):
    setup(tmp_path)
    module.pipeline("pusht", root=tmp_path, runner=lambda *a, **k: SimpleNamespace(returncode=7))
    config = tmp_path / "configs/rollout_revision/reacher_recursive_constant_s0.json"
    item = module.load_json(config)
    item["lr"] = 9
    module.atomic_json(config, item)
    with pytest.raises(ValueError, match="changed after launch"):
        module.pipeline("reacher", root=tmp_path)


def test_environment_locks_are_independent_and_duplicate_worker_does_not_run(tmp_path):
    import fcntl
    setup(tmp_path)
    state_dir = tmp_path / "runs/rollout_revision/campaign_state"
    state_dir.mkdir(parents=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Duplicate environment worker started a child")
    with (state_dir / "pusht.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.pipeline("pusht", root=tmp_path, runner=forbidden) == 0
        assert module.pipeline("reacher", root=tmp_path,
                               runner=lambda *a, **k: SimpleNamespace(returncode=7)) == 1


def test_wrong_arm_is_rejected_before_training(tmp_path):
    setup(tmp_path)
    path = tmp_path / "configs/rollout_revision/pusht_teacher_forced_inferred_s0.json"
    item = module.load_json(path)
    item["revision"]["context_mode"] = "constant"
    module.atomic_json(path, item)
    with pytest.raises(ValueError, match="prespecified arm"):
        module.pipeline("pusht", root=tmp_path)
