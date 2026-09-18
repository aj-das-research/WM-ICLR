"""Guard variant identity and honest resumability of the development study."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = script("evaluate_rollout_revision")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows():
    return [{"trajectory_id": f"development-s{2031000+n}-d1", "observation_id": 1,
             "seed": 2031000+n, "dynamics_id": 1, "goal_index": 7,
             "success_during_context": int(n == 31), "policy_eligible": int(n != 31),
             "policy": "world_model", "initial_distance_after_context": .5,
             "success": int(n in {0, 31})} for n in range(32)]


class FakeModel(nn.Module):
    def __init__(self, donor, provenance):
        super().__init__()
        self.donor = deepcopy(donor).requires_grad_(False).eval()
        self.provenance = provenance
        self.config = SimpleNamespace(mode="framewise_rollout_revision", objective="recursive",
                                      context_mode="inferred")
        self.requested_devices = []

    def to(self, device):
        # Real tensor/GPU and training correctness is covered by model tests.
        self.requested_devices.append(str(device))
        return self


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    run_id = "reacher_recursive_inferred_s0"
    run = tmp_path / "runs" / run_id
    package = run / "best"
    data = tmp_path / "data"
    stats = tmp_path / "action_stats.json"
    write_json(data / "manifest.json", {"environment": "reacher", "action_block": 5})
    write_json(stats, {"mean": [0], "std": [1]})
    config = {"seed": 0, "revision": {"objective": "recursive", "context_mode": "inferred"},
              "output_dir": str(run), "data_root": str(data),
              "action_stats": str(stats), "donor_checkpoint": str(tmp_path / "donor/best")}
    write_json(run / "run_config.json", config)
    summary = {"status": "completed", "completed_epochs": 30,
               "validation_metric": "recursive_mse_all_query_steps"}
    write_json(run / "training_summary.json", summary)
    (run / "metrics.jsonl").write_text('{}\n')
    write_json(package / "config.json", {})
    (package / "model.pt").write_bytes(b"fixture-checkpoint")
    hashes = {}
    for name in evaluation.IMPLEMENTATION_FILES:
        path = (Path(evaluation.__file__).with_name(name) if name == "train_rollout_revision.py"
                else Path(evaluation.revision_source.__file__).with_name(name))
        hashes[name] = evaluation.file_sha256(path)
    donor_source = {"hashes": {"model_sha256": "donor"}, "best_epochs": [4]}
    provenance = {"revision_environment": "reacher", "revision_implementation_hashes": hashes,
                  "revision_donor": {"hashes": donor_source["hashes"]},
                  "data_manifest_sha256": evaluation.file_sha256(data / "manifest.json"),
                  "action_stats_sha256": evaluation.file_sha256(stats)}
    donor = nn.Linear(3, 3).requires_grad_(False).eval()
    model = FakeModel(donor, provenance)
    state = {"epoch": 3, "config": {"metadata": {
        "selection": "minimum_validation_recursive_mse_targets_H_to_T-1_over_completed_training_epochs",
        "validation_protocol": "recursive_from_observed_support_all_query_steps",
        "validation_precision": "float32"}}}
    evaluator_sha = evaluation.file_sha256(evaluation.evaluation_source.__file__)
    control_identity = {"checkpoint_sha256": "donor", "data_manifest_sha256": provenance["data_manifest_sha256"]}
    protocol = {**evaluation.PROTOCOL, "run_identity": control_identity, "evaluator_sha256": evaluator_sha,
                "search_coordinates": evaluation.intervention.SEARCH_COORDINATES}
    control_rows = rows()
    planning = {"status": "complete", "split": "development", "policy": "world_model",
                "protocol": protocol, "signature": digest_json(protocol), "records": control_rows,
                "planner": deepcopy(evaluation.intervention.PLANNER_METADATA)}
    control = {"status": "complete", "environment": "reacher", "model_mode": "framewise",
               "training_seed": 0, "checkpoint_sha256": "donor", "checkpoint_epoch": 4,
               "data_manifest_sha256": provenance["data_manifest_sha256"], "evaluator_sha256": evaluator_sha,
               "planning": planning}
    control_root = tmp_path / "controls"
    write_json(control_root / "reacher_framewise_s0/planning_development.json", control)
    keys = [(r["trajectory_id"], r["observation_id"]) for r in control_rows]
    monkeypatch.setattr(evaluation, "load_rollout_package", lambda *a, **k: (model, state))
    monkeypatch.setattr(evaluation, "completed_framewise_source", lambda *a: donor_source)
    monkeypatch.setattr(evaluation, "load_package", lambda *a, **k: (donor, {}))
    monkeypatch.setattr(evaluation, "validate_donor_state", lambda *a: None)
    monkeypatch.setattr(evaluation.training, "training_identity", lambda *a: "training-identity")
    monkeypatch.setattr(evaluation.training, "validate_completed", lambda *a: summary)
    monkeypatch.setattr(evaluation.intervention, "development_keys", lambda *a: keys)
    calls = []
    statuses = ["complete"]

    def planner(m, d, **kwargs):
        calls.append(kwargs)
        assert m is model and d == data
        assert all(kwargs[k] == v for k, v in evaluation.PROTOCOL.items())
        status = statuses.pop(0)
        p = deepcopy(planning)
        p["status"] = status
        p["protocol"]["run_identity"] = kwargs["run_identity"]
        p["signature"] = digest_json(p["protocol"])
        p["records"][0]["success"] = 0
        p["records"][1]["success"] = 1
        if status == "interrupted":
            p["records"] = p["records"][:9]
        write_json(kwargs["progress_path"], p)
        return p

    monkeypatch.setattr(evaluation, "evaluate_planning", planner)
    args = SimpleNamespace(environment="reacher", checkpoint=package, device="cuda", cpu_threads=1,
                           output_root=tmp_path / "results", control_root=control_root,
                           save_video=False, max_runtime_seconds=120)
    return SimpleNamespace(args=args, model=model, state=state, run=run, config=config,
                           summary=summary, statuses=statuses, calls=calls, control_rows=control_rows,
                           output=args.output_root / run_id / "planning_development.json",
                           progress=args.output_root / run_id / "planning_development.progress.json")


def test_complete_validated_result_reuses_without_planning_and_preserves_pair_counts(setup_run):
    f = setup_run
    first = evaluation.run(f.args)
    second = evaluation.run(f.args)
    assert first == second
    assert len(f.calls) == 1 and f.model.requested_devices == ["cuda"]
    assert first["comparison"]["revision_only_successes"] == 1
    assert first["comparison"]["control_only_successes"] == 1
    assert first["comparison"]["eligible_tasks"] == 31
    assert first["comparison"]["eligible_difference_percentage_points"] == 0
    assert first["run_identity"]["context_mode"] == "inferred"
    assert first["run_identity"]["training_implementation_hashes"] == f.model.provenance["revision_implementation_hashes"]


def test_interruption_resumes_same_variant_without_treating_partial_as_complete(setup_run):
    f = setup_run
    f.statuses[:] = ["interrupted", "complete"]
    first = evaluation.run(f.args)
    assert first["status"] == "interrupted" and first["comparison"] == {"status": "pending"}
    assert len(first["planning"]["records"]) == 9
    second = evaluation.run(f.args)
    assert second["status"] == "complete" and len(f.calls) == 2
    assert f.calls[0]["run_identity"] == f.calls[1]["run_identity"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(context_mode="constant"),
    lambda r: r.update(checkpoint_epoch=29),
    lambda r: r["comparison"].update(revision_only_successes=9),
    lambda r: r["planning"]["records"][0].update(initial_distance_after_context=.8),
    lambda r: r["planning"]["protocol"].update(samples=1000),
    lambda r: r["planning"]["records"].pop(),
])
def test_reuse_rejects_mislabeling_changed_counts_support_protocol_or_missing_tasks(setup_run, mutation):
    f = setup_run
    result = evaluation.run(f.args)
    mutation(result)
    write_json(f.output, result)
    with pytest.raises(ValueError):
        evaluation.run(f.args)
    assert len(f.calls) == 1


def test_partial_resume_rejects_other_arm_or_changed_initial_support(setup_run):
    f = setup_run
    f.statuses[:] = ["interrupted"]
    evaluation.run(f.args)
    f.output.unlink()
    p = json.loads(f.progress.read_text())
    p["protocol"]["run_identity"]["context_mode"] = "constant"
    p["signature"] = digest_json(p["protocol"])
    write_json(f.progress, p)
    with pytest.raises(ValueError, match="another run or protocol"):
        evaluation.run(f.args)
    assert len(f.calls) == 1


def test_variant_directory_and_loaded_objective_must_match_training_config(setup_run):
    f = setup_run
    f.model.config.context_mode = "constant"
    with pytest.raises(ValueError, match="variant differ"):
        evaluation.run(f.args)
    f.model.config.context_mode = "inferred"
    f.config["output_dir"] = str(f.run.with_name("reacher_recursive_constant_s0"))
    write_json(f.run / "run_config.json", f.config)
    with pytest.raises(ValueError, match="directory differs"):
        evaluation.run(f.args)
    assert not f.calls


@pytest.mark.parametrize("mutation", [
    lambda f: f.state["config"]["metadata"].update(validation_precision="bfloat16"),
    lambda f: f.summary.update(validation_metric="one_step"),
    lambda f: f.model.provenance["revision_implementation_hashes"].pop("model.py"),
    lambda f: f.model.provenance["revision_implementation_hashes"].update({"../model.py": "bad"}),
    lambda f: f.model.provenance["revision_implementation_hashes"].update({"rollout_revision.py": "changed"}),
])
def test_selection_and_complete_source_identity_required_before_evaluation(setup_run, mutation):
    mutation(setup_run)
    with pytest.raises(ValueError):
        evaluation.run(setup_run.args)
    assert not setup_run.calls


def test_modified_frozen_donor_is_rejected_before_gpu_evaluation(setup_run):
    f = setup_run
    with torch.no_grad():
        f.model.donor.weight.add_(.1)
    with pytest.raises(ValueError, match="Different complete frozen framewise donor tensor"):
        evaluation.run(f.args)
    assert not f.calls and not f.model.requested_devices


@pytest.mark.parametrize("extra", [["--split", "test"], ["--device", "cpu"],
                                  ["--max-runtime-seconds", "nan"], ["--cpu-threads", "0"]])
def test_cli_preserves_development_gpu_protocol(extra):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--environment", "reacher", "--checkpoint", "best", *extra])


@pytest.mark.parametrize("status,expected", [("complete", 0), ("interrupted", 75)])
def test_cli_returns_resumable_interruption_status(monkeypatch, status, expected):
    monkeypatch.setattr(evaluation, "parse_args", lambda: SimpleNamespace(cpu_threads=1))
    monkeypatch.setattr(evaluation, "run", lambda args: {"status": status, "run_id": "fixture"})
    assert evaluation.main() == expected
