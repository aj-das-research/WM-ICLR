"""Revision correctness fixtures; these are not research benchmark results."""
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from shiftwm.checkpoint import resume_training, save_package
from shiftwm.dynamics_revision import (DynamicsRevision, completed_framewise_source,
                                      load_revision_package, validate_donor_state)
from shiftwm.model import ModelConfig, ShiftWorldModel


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = script("train_dynamics_revision")
evaluation = script("evaluate_dynamics_revision")


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, images, **kwargs):
        return SimpleNamespace(last_hidden_state=self.linear(images.mean((-1, -2)))[:, None])


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.projector = nn.BatchNorm1d(8)
        self.action_encoder = nn.Linear(4, 8)
        self.predictor = nn.Linear(8, 8)
        self.pred_proj = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(.5)

    def predict(self, observations, actions):
        values = self.predictor(self.dropout(observations + actions))
        return self.pred_proj(values.flatten(0, 1)).reshape_as(values)


BASE = {"predictor": {"input_dim": 8, "num_frames": 3},
        "action_encoder": {"input_dim": 4}, "encoder": {"image_size": 8}}


def donor():
    model = ShiftWorldModel(TinyBase(), BASE, ModelConfig(mode="framewise", context_dim=4,
                                                         context_hidden=16),
                            [.1, -.2, .1, -.2], [.3, .4, .3, .4])
    with torch.no_grad():
        model.framewise_adapter[-1].weight.normal_(std=.1)
        model.framewise_adapter[-1].bias.fill_(.2)
    return model.eval()


def revision():
    return DynamicsRevision(donor(), {"context_dim": 4, "context_hidden": 16})


def batch():
    g = torch.Generator().manual_seed(123)
    return {"features": torch.randn(4, 8, 8, generator=g),
            "reference_features": torch.randn(4, 8, 8, generator=g),
            "paired_features": torch.randn(4, 8, 8, generator=g),
            "actions": torch.randn(4, 7, 4, generator=g),
            "observation_id": torch.tensor([0, 0, 1, 1])}


def test_zero_initialized_revision_matches_actual_donor_predictions_rollout_goal():
    base = donor()
    model = DynamicsRevision(base, {"context_dim": 4, "context_hidden": 16})
    b = batch()
    for mode in (False, True):
        model.train(mode)
        torch.testing.assert_close(model(b)["predictions"], base(b)["predictions"], rtol=0, atol=0)
        torch.testing.assert_close(model.rollout_features(b["features"][:, :3], b["actions"][:, :2],
                                                        b["actions"][:, 2:]),
                                   base.rollout_features(b["features"][:, :3], b["actions"][:, :2],
                                                         b["actions"][:, 2:]), rtol=0, atol=0)
    goals = torch.rand(4, 3, 8, 8)
    c = torch.randn(4, 4)
    torch.testing.assert_close(model.goal_embedding(goals, c), base.goal_embedding(goals, c), rtol=0, atol=0)


def test_only_new_modules_train_and_frozen_dropout_bn_stay_in_eval():
    model = revision().train()
    expected = {name for name, _ in model.named_parameters() if name.startswith(("dynamics_context.", "dynamics_adapter."))}
    assert {name for name, p in model.named_parameters() if p.requires_grad} == expected
    assert not any(module.training for module in model.donor.modules())
    before = deepcopy(model.donor.state_dict())
    context_before = deepcopy(model.dynamics_context.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
    b = batch()
    torch.testing.assert_close(model(b)["predictions"], model(b)["predictions"], rtol=0, atol=0)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(b)
        assert out["loss"] == out["prediction_loss"]
        out["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.donor.parameters())
        optimizer.step()
    assert any(not torch.equal(v, model.dynamics_context.state_dict()[k]) for k, v in context_before.items())
    assert model.dynamics_adapter.affine.weight.count_nonzero() > 0
    for key, value in before.items():
        torch.testing.assert_close(model.donor.state_dict()[key], value, rtol=0, atol=0)


def test_context_reads_corrected_observed_support_and_executed_actions_only():
    model = revision().eval()
    b = batch()
    captured = []
    hook = model.dynamics_context.register_forward_pre_hook(lambda module, args: captured.append(args))
    first = model(b)
    hook.remove()
    expected = model.donor.correct_observations(b["features"][:, :3], torch.zeros(4, 4))
    torch.testing.assert_close(captured[0][0], expected, rtol=0, atol=0)
    torch.testing.assert_close(captured[0][1], model.normalize_actions(b["actions"][:, :2]), rtol=0, atol=0)
    other = deepcopy(b)
    other["features"][:, 3:] += 100
    other["actions"][:, 2:] += 100
    other["reference_features"] += 1000
    other["paired_features"] += 1000
    other["observation_id"][:] = 9
    second = model(other)
    torch.testing.assert_close(first["dynamics_context"], second["dynamics_context"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="executed action"):
        model.infer_context(b["features"][:, :3], b["actions"][:, :3])


def test_paired_features_ids_do_not_change_objective_and_goal_is_fixed():
    model = revision().train()
    b = batch()
    before = model(b)
    other = deepcopy(b)
    other["paired_features"] *= 50
    other["observation_id"][:] = 12
    torch.testing.assert_close(before["loss"], model(other)["loss"], rtol=0, atol=0)
    images = torch.rand(4, 3, 8, 8)
    goal = model.goal_embedding(images, torch.randn(4, 4)).detach()
    with torch.no_grad():
        for p in model.dynamics_context.parameters():
            p.add_(.1)
        model.dynamics_adapter.affine.bias.add_(.2)
    torch.testing.assert_close(goal, model.goal_embedding(images, torch.randn(4, 4) * 100), rtol=0, atol=0)


def test_portable_weights_only_full_package_and_optimizer_resume(tmp_path, monkeypatch):
    import shiftwm.upstream
    monkeypatch.setattr(shiftwm.upstream, "create_base", lambda config: TinyBase())
    model = revision().train()
    model.provenance["revision_donor"] = {"path": "/nonexistent/original/donor"}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4)
    generator = torch.Generator().manual_seed(9)
    b = batch()
    model(b)["loss"].backward(); optimizer.step(); optimizer.zero_grad(); scheduler.step()
    save_package(model, tmp_path, optimizer=optimizer, scheduler=scheduler, loader_generator=generator,
                 epoch=1, step=1)
    expected_random, expected_numpy = torch.rand(3), np.random.rand(3)
    model(b)["loss"].backward(); optimizer.step(); scheduler.step()
    expected = deepcopy(model.state_dict())
    loaded, _ = load_revision_package(tmp_path)
    optim2 = torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad], lr=.001)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, 4)
    resume_training(tmp_path, loaded, optim2, sched2, generator)
    torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
    loaded.train(); loaded(b)["loss"].backward(); optim2.step(); sched2.step()
    for key, value in expected.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert scheduler.state_dict() == sched2.state_dict()


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        b = batch()
        self.items = [{key: value[i] for key, value in b.items()} for i in range(4)] * 2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_exact_resume_after_first_batch_matches_full_optimizer_path(tmp_path):
    torch.set_num_threads(1)
    config = {"seed": 0, "epochs": 2, "batch_size": 3, "num_workers": 0, "device": "cpu",
              "lr": .001, "min_lr": .00001, "weight_decay": .001,
              "output_dir": str(tmp_path / "complete")}
    torch.manual_seed(42)
    trainer.fit(revision(), config, TinyDataset(), TinyDataset())
    complete = torch.load(tmp_path / "complete/last/model.pt", weights_only=True)
    config["output_dir"] = str(tmp_path / "interrupted")
    config["max_runtime_seconds"] = 1e-12
    torch.manual_seed(42)
    result = trainer.fit(revision(), config, TinyDataset(), TinyDataset())
    assert result["batches_completed_in_epoch"] == 1
    config.pop("max_runtime_seconds")
    config["resume"] = str(tmp_path / "interrupted/last")
    trainer.fit(revision(), config, TinyDataset(), TinyDataset())
    resumed = torch.load(tmp_path / "interrupted/last/model.pt", weights_only=True)
    assert complete["step"] == resumed["step"] == 6
    for key, value in complete["state_dict"].items():
        torch.testing.assert_close(resumed["state_dict"][key], value, rtol=0, atol=0)
    a = [json.loads(line) for line in (tmp_path / "complete/metrics.jsonl").read_text().splitlines()]
    b = [json.loads(line) for line in (tmp_path / "interrupted/metrics.jsonl").read_text().splitlines()]
    assert [(r["train"], r["val"]) for r in a] == [(r["train"], r["val"]) for r in b]
    config["lr"] = .5
    with pytest.raises(ValueError, match="Resume sources"):
        trainer.fit(revision(), config, TinyDataset(), TinyDataset())


def test_completed_donor_checks_full_training_history_and_best_epoch(tmp_path):
    root = tmp_path / "run"
    save_package(donor(), root / "best", epoch=4, step=4, best_metric=.01)
    (root / "run_config.json").write_text(json.dumps({"seed": 0, "epochs": 30}))
    (root / "training_summary.json").write_text(json.dumps({"status": "completed", "completed_epochs": 30,
                                                           "best_validation_prediction_loss": .01}))
    rows = [{"epoch": n, "val": {"prediction_loss": .01 if n == 4 else .1}} for n in range(1, 31)]
    (root / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    source = completed_framewise_source(root / "best")
    state = torch.load(root / "best/model.pt", weights_only=True)
    validate_donor_state(state, source)
    state["epoch"] = 5
    with pytest.raises(ValueError, match="validation-best"):
        validate_donor_state(state, source)
    (root / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    with pytest.raises(ValueError, match="complete and unique"):
        completed_framewise_source(root / "best")


def support_rows():
    return [{"trajectory_id": f"development-{n}", "observation_id": 1, "seed": 2031000 + n,
             "dynamics_id": 1, "goal_index": 7, "success_during_context": int(n == 2),
             "policy_eligible": int(n != 2), "policy": "world_model",
             "initial_distance_after_context": .5, "initial_wrapped_joint_error_rad": .5,
             "success": int(n in {0, 2})} for n in range(3)]


def test_eval_only_accepts_development_and_unchanged_support():
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--environment", "reacher", "--checkpoint", "best", "--split", "test"])
    rows = support_rows()
    changed = deepcopy(rows)
    changed[0]["success"] = 0
    changed[1]["success"] = 1
    evaluation.validate_support(changed, rows, complete=True)
    counts = evaluation.paired_counts(changed, rows)
    assert counts["revision_only_successes"] == counts["control_only_successes"] == 1
    assert counts["eligible_difference_percentage_points"] == 0
    assert counts["support_successes"] == 1 and counts["eligible_tasks"] == 2
    for field, value in (("goal_index", 8), ("success_during_context", 1),
                         ("initial_distance_after_context", .8), ("trajectory_id", "test-other")):
        bad = deepcopy(rows); bad[0][field] = value
        with pytest.raises(ValueError):
            evaluation.validate_support(bad, rows, complete=True)
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.validate_support(rows + [rows[0]], rows)


def test_revision_config_rejects_nonisolated_settings():
    for config in ({"alignment_weight": 1}, {"freeze_visual": False}, {"history_length": 2}):
        with pytest.raises(ValueError):
            DynamicsRevision(donor(), config)
    with pytest.raises(ValueError, match="framewise"):
        baseline = donor(); baseline.config.mode = "single"
        DynamicsRevision(baseline)


def test_development_control_sources_protocol_and_partial_signature_are_verified():
    rows = support_rows()
    donor_source = {"hashes": {"model_sha256": "donor"}, "best_epochs": [4]}
    evaluator_sha = evaluation.file_sha256(evaluation.evaluation_source.__file__)
    identity = {"checkpoint_sha256": "donor", "data_manifest_sha256": "data"}
    protocol = {**evaluation.PROTOCOL, "run_identity": identity,
                "evaluator_sha256": evaluator_sha,
                "search_coordinates": evaluation.intervention.SEARCH_COORDINATES}
    planning = {"status": "complete", "split": "development", "policy": "world_model",
                "protocol": protocol, "records": rows,
                "planner": deepcopy(evaluation.intervention.PLANNER_METADATA),
                "signature": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}
    record = {"status": "complete", "environment": "reacher", "model_mode": "framewise",
              "training_seed": 0, "checkpoint_sha256": "donor", "checkpoint_epoch": 4,
              "data_manifest_sha256": "data", "evaluator_sha256": evaluator_sha, "planning": planning}
    keys = [(r["trajectory_id"], r["observation_id"]) for r in rows]
    evaluation.validate_control(record, donor_source, keys, "reacher", "data")
    for mutation in (lambda x: x.update(checkpoint_sha256="changed"),
                     lambda x: x["planning"]["protocol"].update(samples=32),
                     lambda x: x["planning"]["planner"].update(search_coordinates="native"),
                     lambda x: x["planning"]["records"].append(deepcopy(rows[0]))):
        bad = deepcopy(record); mutation(bad)
        with pytest.raises(ValueError):
            evaluation.validate_control(bad, donor_source, keys, "reacher", "data")
    revision_identity = {"checkpoint_sha256": "revision", "evaluator_sha256": evaluator_sha}
    partial = deepcopy(planning)
    partial.pop("planner")
    partial["status"] = "interrupted"
    partial["protocol"]["run_identity"] = revision_identity
    partial["signature"] = hashlib.sha256(json.dumps(partial["protocol"], sort_keys=True).encode()).hexdigest()
    evaluation.validate_progress(partial, revision_identity, rows)
    partial["protocol"]["samples"] = 32
    with pytest.raises(ValueError, match="protocol"):
        evaluation.validate_progress(partial, revision_identity, rows)
