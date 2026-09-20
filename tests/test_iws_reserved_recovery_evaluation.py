"""Synthetic-only reserved evaluation access, index, metric and prefix checks."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reserved_recovery_evaluation_tests", ROOT / "scripts/real_video_iws_reserved_recovery_v2/evaluate.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)


def population(task="pusht"):
    ids = [f"synthetic-{i:02d}" for i in range(10)]
    counts = [10, 30] + [20] * 8
    handles = [{"episode_id": eid, "start": start} for eid, count in zip(ids, counts) for start in range(count)]
    return ev.population_contract(task, handles, ids)


def arrays(pop):
    result = {"episode_index": np.array([pop["episode_ids"].index(h["episode_id"]) for h in pop["handles"]], dtype=np.int64),
              "window_start": np.array([h["start"] for h in pop["handles"]], dtype=np.int64)}
    for key in ev.metric_keys():
        width = 3 if key.startswith("prefix_") else 59
        result[key] = np.repeat((result["episode_index"].astype(float) / 10 + .1)[:, None], width, axis=1)
    return result


class Cache:
    def __init__(self, frames=80):
        self.frames = frames
        self.calls = []
    def episode(self, eid):
        self.calls.append(eid); n = self.frames
        return {"features": np.repeat(np.arange(n, dtype=np.float32)[:, None], 6144, 1),
                "commands": np.repeat(np.arange(n, dtype=np.float32)[:, None], 4, 1),
                "frame_indices": np.arange(n, dtype=np.int64), "command_row_indices": np.arange(n, dtype=np.int64)}


class Predictor:
    feature_std = torch.ones(6144)
    def __init__(self): self.calls = []
    def predict(self, initial, commands):
        self.calls.append((initial.clone(), commands.clone()))
        offsets = commands[:, 1:, :1].cumsum(1)
        return initial[:, None] + offsets


def test_native_indices_include_exact60_command_rows_and59_future_targets():
    cache = Cache(); store = {}
    batch = ev.make_batch(cache, [{"episode_id": "a", "start": 3}, {"episode_id": "a", "start": 4}], "pusht", ["a"], store)
    assert cache.calls == ["a"]
    assert batch["initial_features"][:, 0].tolist() == [3, 4]
    assert batch["commands"][0, :, 0].tolist() == list(range(3, 63))
    assert batch["targets"][0, :, 0].tolist() == list(range(4, 63))
    assert batch["targets"].shape == (2, 59, 6144)


def test_strict_unused_tail_and_row_alignment_rejected():
    with pytest.raises(ValueError, match="eligibility"):
        ev.make_batch(Cache(60), [{"episode_id": "a", "start": 0}], "pusht", ["a"], {})
    cache = Cache(); original = cache.episode
    def broken(eid):
        episode = original(eid); episode["command_row_indices"][1] = 0; return episode
    cache.episode = broken
    with pytest.raises(ValueError, match="indexing"):
        ev.make_batch(cache, [{"episode_id": "a", "start": 0}], "pusht", ["a"], {})


def test_targets_never_enter_predict_and_prefixes_are_real_invocations():
    batch = ev.make_batch(Cache(), [{"episode_id": "a", "start": 0}], "pusht", ["a"], {})
    model = Predictor()
    with torch.inference_mode():
        metrics, calls = ev.score_batch(model, batch, 0)
        modified = dict(batch); modified["targets"] = batch["targets"] + 5
        changed, calls_changed = ev.score_batch(Predictor(), modified, 0)
    assert [call[1].shape[1] for call in model.calls] == [60, 15, 30, 45]
    assert [v["prediction_sha256"] for v in calls] == [v["prediction_sha256"] for v in calls_changed]
    assert not np.array_equal(metrics["standardized_mse"], changed["standardized_mse"])
    ev.validate_invocations(calls, n=1)
    # Original full-H60 endpoints remain separately recorded.
    assert metrics["standardized_mse"].shape == (1, 59)
    assert metrics["prefix_standardized_mse"].shape == (1, 3)
    np.testing.assert_array_equal(metrics["prefix_standardized_mse"], metrics["standardized_mse"][:, [13, 28, 43]])


def test_horizon_dependent_predictor_fails_prefix_check():
    class Broken(Predictor):
        def predict(self, initial, commands): return super().predict(initial, commands) + commands.shape[1]
    batch = ev.make_batch(Cache(), [{"episode_id": "a", "start": 0}], "pusht", ["a"], {})
    with torch.inference_mode(), pytest.raises(ValueError, match="prefix consistency"):
        ev.score_batch(Broken(), batch, 0)


@pytest.mark.parametrize("field,value", [("maximum_absolute_difference", 1e9), ("maximum_tolerance_ratio", 1.1),
                                         ("target_native_offset", 15), ("allclose", False)])
def test_prefix_claim_cannot_hide_large_gap_or_wrong_target(field, value):
    batch = ev.make_batch(Cache(), [{"episode_id": "a", "start": 0}], "pusht", ["a"], {})
    with torch.inference_mode(): _, calls = ev.score_batch(Predictor(), batch, 0)
    calls[1][field] = value
    with pytest.raises(ValueError, match="prefix invocation"):
        ev.validate_invocations(calls, n=1)


def test_missing_prefix_invocation_fails():
    with pytest.raises(ValueError, match="explicit prefix"):
        ev.validate_invocations([], n=1)


def test_unequal_handles_preserve_distinct_trajectory_estimate():
    pop = population(); data = arrays(pop); result = ev.aggregate_arrays(data, pop)
    assert result["equal_trajectory"]["standardized_mse"][-1] == pytest.approx(.55)
    assert result["equal_handle"]["standardized_mse"][-1] == pytest.approx(.555)
    assert [r["handles"] for r in result["episodes"]] == [10, 30] + [20] * 8


@pytest.mark.parametrize("change", [
    lambda d: d["window_start"].__setitem__(0, 1),
    lambda d: d["episode_index"].__setitem__(0, 9),
    lambda d: d["standardized_mse"].__setitem__((0, 0), np.nan),
    lambda d: d["prefix_standardized_mae"].__setitem__((0, 0), -1),
    lambda d: d["feature_cosine_distance"].__setitem__((0, 0), 2.1),
])
def test_invalid_or_reordered_primitives_rejected(change):
    pop = population(); data = arrays(pop); change(data)
    with pytest.raises(ValueError): ev.aggregate_arrays(data, pop)


def test_duplicate_handle_and_unknown_trajectory_rejected():
    pop = population(); handles = copy.deepcopy(pop["handles"]); handles[0] = handles[1]
    with pytest.raises(ValueError, match="duplicate"):
        ev.population_contract("pusht", handles, pop["episode_ids"])
    handles[0] = {"episode_id": "not-registered", "start": 0}
    with pytest.raises(ValueError, match="Invalid reserved handle"):
        ev.population_contract("pusht", handles, pop["episode_ids"])


def test_review_gate_precedes_package_and_payload_access(monkeypatch, tmp_path):
    def denied(*args): raise ValueError("review not passed")
    monkeypatch.setattr(ev, "checked_registration", denied)
    monkeypatch.setattr(ev, "load_selected", lambda *args: pytest.fail("Loaded package before gate"))
    with pytest.raises(ValueError, match="review not passed"):
        ev.evaluate("anything", root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_feature_metric_is_exact_existing_helper():
    assert ev.feature_errors is ev.frozen_metrics.feature_errors
    result = ev.feature_errors(torch.tensor([[[3., 4.]]]), torch.zeros(1, 1, 2), torch.tensor([3., 2.]))
    assert result["standardized_mse"].item() == 2.5
    assert result["feature_cosine_distance"].item() == 1
