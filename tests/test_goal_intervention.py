"""The post-hoc hybrid must change only learned goal calibration."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shiftwm.model import ModelConfig, ShiftWorldModel


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_goal_calibration_intervention.py"
spec = importlib.util.spec_from_file_location("goal_intervention_test", SCRIPT)
intervention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intervention)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, pixels, **kwargs):
        return SimpleNamespace(last_hidden_state=self.linear(pixels.mean((-1, -2)))[:, None])


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.projector = nn.Identity()
        self.action_encoder = nn.Linear(10, 8)
        self.predictor = nn.Linear(8, 8)
        self.pred_proj = nn.Identity()

    def predict(self, features, actions):
        return self.predictor(features + actions)


def model_pair():
    torch.manual_seed(991)
    base = TinyBase()
    config = {"predictor": {"input_dim": 8, "num_frames": 3},
              "action_encoder": {"input_dim": 10}, "encoder": {"image_size": 8}}
    provenance = {"weights_sha256": "frozen_fixture", "data_manifest_sha256": "dataset_fixture",
                  "action_stats_sha256": "stats_fixture"}
    models = [ShiftWorldModel(deepcopy(base), config, ModelConfig(mode=mode, context_dim=4, context_hidden=8),
                              [0.] * 10, [1.] * 10, provenance=deepcopy(provenance)).eval()
              for mode in ("factorized", "framewise")]
    with torch.no_grad():
        models[1].framewise_adapter[-1].bias.fill_(.4)
    return models


def test_goal_changes_but_history_context_predictor_and_rollout_are_identical():
    fixed, donor = model_pair()
    before = {key: value.clone() for key, value in fixed.state_dict().items()}
    hybrid = intervention.GoalCalibrationIntervention(fixed, donor)
    pixels = torch.rand(1, 3, 3, 8, 8)
    goal = torch.rand(1, 3, 8, 8)
    history = fixed.encode_images(pixels)
    past, future = torch.rand(1, 2, 10), torch.rand(1, 5, 10)
    contexts = fixed.infer_context(history, past)
    torch.testing.assert_close(hybrid.encode_images(pixels), history, rtol=0, atol=0)
    for actual, expected in zip(hybrid.infer_context(history, past), contexts):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hybrid.correct_observations(history, contexts[0]),
                               fixed.correct_observations(history, contexts[0]), rtol=0, atol=0)
    incoming_actions = torch.cat((past, future[:, :1]), 1)
    torch.testing.assert_close(hybrid.predict_features(history, incoming_actions, contexts[1]),
                               fixed.predict_features(history, incoming_actions, contexts[1]), rtol=0, atol=0)
    torch.testing.assert_close(hybrid.rollout_features(history, past, future, contexts=contexts),
                               fixed.rollout_features(history, past, future, contexts=contexts), rtol=0, atol=0)
    torch.testing.assert_close(hybrid.goal_embedding(goal, contexts[0]), donor.goal_embedding(goal, contexts[0]), rtol=0, atol=0)
    assert not torch.equal(hybrid.goal_embedding(goal, contexts[0]), fixed.goal_embedding(goal, contexts[0]))
    # A donor framewise goal does not read context/future states or canonical RGB.
    torch.testing.assert_close(hybrid.goal_embedding(goal, contexts[0] + 100),
                               hybrid.goal_embedding(goal, contexts[0]), rtol=0, atol=0)
    for key, value in fixed.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in hybrid.parameters())


@pytest.mark.parametrize("change,match", [("visual", "visual_encoder"), ("reference", "reference_encoder"),
                                         ("action", "action_std"), ("provenance", "data_manifest")])
def test_donor_coordinate_and_unit_mismatches_rejected(change, match):
    fixed, donor = model_pair()
    with torch.no_grad():
        if change == "visual":
            donor.base.encoder.linear.weight.add_(.01)
        elif change == "reference":
            donor.reference_encoder.linear.weight.add_(.01)
        elif change == "action":
            donor.action_std[0] *= 2
        else:
            donor.provenance["data_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match=match):
        intervention.GoalCalibrationIntervention(fixed, donor)


def test_parser_cannot_request_a_test_split_or_different_budget():
    for args in (["--split", "test"], ["--samples", "20"], ["--seed", "2"]):
        with pytest.raises(SystemExit):
            intervention.parse_args(args)
    args = intervention.parse_args([])
    assert args.environment == "both"
    assert str(args.output_root) == "results/development_goal_intervention"
    assert intervention.PROTOCOL["split"] == "development"
    assert intervention.PROTOCOL["samples"] == 300


def complete_record():
    identity = {"fixed_checkpoint_sha256": "fixed", "donor_checkpoint_sha256": "donor",
                "intervention_source_sha256": "source", "evaluator_sha256": "evaluator"}
    protocol = {**intervention.PROTOCOL, "run_identity": identity, "evaluator_sha256": "evaluator",
                "search_coordinates": intervention.SEARCH_COORDINATES}
    keys = [("dev1", 1), ("dev2", 1)]
    record = {"status": "complete", "run_identity": identity,
              "planning": {"status": "complete", "protocol": protocol, "split": "development", "policy": "world_model",
                           "planner": intervention.PLANNER_METADATA.copy(),
                           "signature": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                           "records": [{"trajectory_id": key[0], "observation_id": key[1]} for key in keys]}}
    return identity, keys, record


@pytest.mark.parametrize("change", ["donor", "source", "split", "signature", "duplicate"])
def test_resume_rejects_changed_donor_source_protocol_or_duplicate_records(change):
    identity, keys, record = complete_record()
    intervention.validate_completed_result(record, identity, keys)
    changed = deepcopy(record)
    if change == "donor":
        changed["run_identity"]["donor_checkpoint_sha256"] = "new"
    elif change == "source":
        changed["run_identity"]["intervention_source_sha256"] = "new"
    elif change == "split":
        changed["planning"]["protocol"]["split"] = "test"
    elif change == "signature":
        changed["planning"]["signature"] = "bad"
    else:
        changed["planning"]["records"].append(changed["planning"]["records"][0])
    with pytest.raises(ValueError):
        intervention.validate_completed_result(changed, identity, keys)


def test_source_requires_completed_validation_best_with_matching_data_and_stats(tmp_path):
    package = tmp_path / "best"
    package.mkdir()
    stats = tmp_path / "action_stats.json"
    stats.write_text('{"action":{"mean":[0,0],"std":[1,1]}}')
    config = {"model_config": {"mode": "factorized"}, "provenance": {
        "data_manifest_sha256": "data", "download": {"repo": "quentinll/lewm-pusht"},
        "action_stats": str(stats), "action_stats_sha256": intervention.digest(stats)}}
    (package / "config.json").write_text(json.dumps(config))
    (package / "model.pt").write_bytes(b"not loaded by metadata-only source validation")
    (tmp_path / "run_config.json").write_text(json.dumps({"seed": 0, "epochs": 2}))
    summary = {"status": "completed", "completed_epochs": 2, "best_validation_prediction_loss": .1}
    (tmp_path / "training_summary.json").write_text(json.dumps(summary))
    (tmp_path / "metrics.jsonl").write_text('\n'.join(json.dumps({"epoch": e, "val": {"prediction_loss": v}})
                                                     for e, v in [(1, .2), (2, .1)]))
    source = intervention.completed_source(package, "factorized", "data", "pusht")
    assert source["best_epochs"] == [2]
    intervention.validate_loaded_state({"config": config, "epoch": 2, "best_metric": .1}, source)
    with pytest.raises(ValueError, match="validation-best"):
        intervention.validate_loaded_state({"config": config, "epoch": 1, "best_metric": .1}, source)
    with pytest.raises(ValueError, match="dataset provenance"):
        intervention.completed_source(package, "factorized", "wrong", "pusht")
    stats.write_text('{}')
    with pytest.raises(ValueError, match="action statistics"):
        intervention.completed_source(package, "factorized", "data", "pusht")


def test_development_selection_cannot_substitute_test_episodes():
    from shiftwm.data import TRAIN_COMBINATIONS
    episodes = [{"trajectory_id": f"dev{seed}", "seed": seed, "split": "development", "steps": 8,
                 "dynamics_id": 1} for seed in range(32)]
    manifest = {"episodes": episodes, "train_combinations": TRAIN_COMBINATIONS}
    assert len(intervention.development_keys(manifest)) == 32
    manifest["episodes"][0]["split"] = "test"
    with pytest.raises(ValueError, match="fixed32"):
        intervention.development_keys(manifest)


@pytest.mark.parametrize("change", ["search_coordinates", "split", "policy", "samples", "action_block", "implementation"])
def test_reuse_rejects_resigned_coordinate_change_and_contradictory_planner_summary(change):
    identity, keys, record = complete_record()
    if change == "search_coordinates":
        record["planning"]["protocol"]["search_coordinates"] = "raw_native_actions"
        record["planning"]["signature"] = hashlib.sha256(json.dumps(record["planning"]["protocol"], sort_keys=True).encode()).hexdigest()
    elif change in ("split", "policy"):
        record["planning"][change] = "test" if change == "split" else "random"
    else:
        record["planning"]["planner"][change] = "other_solver" if change == "implementation" else 2
    with pytest.raises(ValueError, match="search_coordinates|Planning summary"):
        intervention.validate_completed_result(record, identity, keys)


def test_output_lock_blocks_overlapping_writer_and_releases_after_exception(tmp_path):
    directory = tmp_path / "intervention"
    with pytest.raises(RuntimeError, match="fixture failure"):
        with intervention.exclusive_output_lock(directory):
            with pytest.raises(BlockingIOError):
                with intervention.exclusive_output_lock(directory):
                    pytest.fail("A second writer entered the same output directory")
            raise RuntimeError("fixture failure")
    with intervention.exclusive_output_lock(directory):
        pass  # A stale on-disk lock filename must not leave an active lease.


def test_environment_lease_covers_the_entire_computation(tmp_path, monkeypatch):
    args = SimpleNamespace(output_root=tmp_path)
    directory = tmp_path / "reacher_factorized_s0_framewise_goal_s0"
    calls = []
    def run_under_lock(actual_args, environment, remaining_seconds):
        assert actual_args is args and environment == "reacher" and remaining_seconds == 99
        with pytest.raises(BlockingIOError):
            with intervention.exclusive_output_lock(directory):
                pytest.fail("Execution did not hold its output lease")
        calls.append(environment)
        return {"status": "fixture_only"}
    monkeypatch.setattr(intervention, "_run_environment_locked", run_under_lock)
    assert intervention.run_environment(args, "reacher", 99) == {"status": "fixture_only"}
    assert calls == ["reacher"]
    with intervention.exclusive_output_lock(directory):
        pass
