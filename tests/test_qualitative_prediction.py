"""Guard matched conditioning, true block boundaries and fixed target coordinates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location("qualitative_prediction", Path(__file__).parents[1] / "scripts/diagnose_qualitative_prediction.py")
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def trace(terminal=23):
    times = np.array([*range(0, terminal + 1, 5), *([] if terminal % 5 == 0 else [terminal])], dtype=np.int64)
    frames = np.stack([np.full((224, 224, 3), 20 + int(t), dtype=np.uint8) for t in times])
    shifted = (diagnosis.pixels_to_tensor(frames, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return {"native_times": times, "canonical_frames": frames, "shifted_frames": shifted,
            "executed_native_actions": np.arange(terminal * 2, dtype=np.float32).reshape(terminal, 2) / 100}


def test_partial_terminal_blocks_are_excluded_without_padding():
    for terminal in (21, 23):
        rows, excluded = diagnosis.windows(trace(terminal), 1)
        assert [r["anchor_native_time"] for r in rows] == [10, 15]
        assert len(excluded) == 1 and excluded[0]["anchor_native_time"] == 20
        assert rows[0]["history_native_times"] == [0, 5, 10]
        assert rows[1]["target_native_time"] == 20
    rows, excluded = diagnosis.windows(trace(50), 1)
    assert [r["anchor_native_time"] for r in rows] == [10, 15, 20] and not excluded


def test_executed_action_grouping_is_exact_and_future_disjoint():
    arrays = trace(50)
    rows, _ = diagnosis.windows(arrays, 1)
    for row in rows:
        t = row["anchor_native_time"]
        np.testing.assert_array_equal(row["past_actions"].reshape(10, 2), arrays["executed_native_actions"][t-10:t])
        np.testing.assert_array_equal(row["next_action"].reshape(5, 2), arrays["executed_native_actions"][t:t+5])
        assert row["history_frames"][-1, 0, 0, 0] == 20 + t
        assert row["next_frame"][0, 0, 0] == 25 + t


@pytest.mark.parametrize("change", ["quantized", "times_float", "times_duplicate", "wrong_support", "action_shape",
                                    "actions_nan", "action_count", "wrong_pixels", "missing_boundary", "early_partial"])
def test_invalid_replay_observations_or_action_timing_rejected(change):
    arrays = trace(50)
    if change == "quantized": arrays["canonical_frames"] = arrays["canonical_frames"].astype(np.float32) / 255
    elif change == "times_float": arrays["native_times"] = arrays["native_times"].astype(float)
    elif change == "times_duplicate": arrays["native_times"][4] = 15
    elif change == "wrong_support": arrays["native_times"][1] = 4
    elif change == "action_shape": arrays["executed_native_actions"] = arrays["executed_native_actions"].reshape(-1)
    elif change == "actions_nan": arrays["executed_native_actions"][2, 0] = np.nan
    elif change == "action_count": arrays["executed_native_actions"] = arrays["executed_native_actions"][:-1]
    elif change == "wrong_pixels": arrays["shifted_frames"][0, 0, 0, 0] ^= 1
    elif change == "missing_boundary":
        for key in ("native_times", "canonical_frames", "shifted_frames"): arrays[key] = np.delete(arrays[key], 3, axis=0)
    elif change == "early_partial": arrays["native_times"][3] = 14
    with pytest.raises(ValueError): diagnosis.windows(arrays, 1)


class FakeModel:
    def __init__(self, mode, prediction_offset=0.):
        self.config = SimpleNamespace(mode=mode, freeze_visual=True, history_length=3)
        self.latent_dim, self.action_dim = 192, 10
        self.reference_encoder, self.reference_projector = nn.Linear(2, 2), nn.Linear(2, 2)
        self.base = SimpleNamespace(encoder=deepcopy(self.reference_encoder), projector=deepcopy(self.reference_projector))
        self.base_config = {"same": True}
        self.action_mean, self.action_std = torch.zeros(10), torch.ones(10)
        self.pixel_mean, self.pixel_std = torch.zeros(3), torch.ones(3)
        self.prediction_offset = prediction_offset
        self.observed_history, self.observed_past, self.observed_future = None, None, None

    def encode_images(self, images, reference=False):
        values = images.mean(dim=(-3, -2, -1))
        if not reference: self.observed_history = images.clone()
        return values[..., None].expand(*values.shape, 192).clone()

    def infer_context(self, features, actions):
        self.observed_past = actions.clone()
        return torch.zeros(1, 32), torch.zeros(1, 32)

    def rollout_features(self, features, past, future, contexts=None):
        assert features.shape == (1, 3, 192) and past.shape == (1, 2, 10) and future.shape == (1, 1, 10)
        self.observed_future = future.clone()
        return features[:, -1:] + self.prediction_offset

    def correct_observations(self, features, context): return features


def models():
    a, b = FakeModel("factorized"), FakeModel("framewise", .1)
    b.reference_encoder = deepcopy(a.reference_encoder); b.reference_projector = deepcopy(a.reference_projector)
    b.base = SimpleNamespace(encoder=deepcopy(a.base.encoder), projector=deepcopy(a.base.projector))
    return {"factorized": a, "framewise": b}


def test_both_models_get_identical_float_inputs_and_same_canonical_target():
    window = diagnosis.windows(trace(23), 1)[0][0];pair = models()
    result = diagnosis.measure_window(pair, window, 1, "cpu")
    a, b = pair.values()
    assert torch.equal(a.observed_history, b.observed_history)
    assert torch.equal(a.observed_past, b.observed_past) and torch.equal(a.observed_future, b.observed_future)
    expected = diagnosis.pixels_to_tensor(window["history_frames"], 1)[None]
    assert torch.equal(a.observed_history, expected)
    quantized = (expected * 255).to(torch.uint8).float() / 255
    assert not torch.equal(a.observed_history, quantized)
    target = np.asarray(result["canonical_next_feature"])
    assert target.shape == (192,)
    for mode, row in result["models"].items():
        recomputed = np.mean((np.asarray(row["predicted_next_feature"]) - target) ** 2)
        assert row["next_prediction_mse"] == pytest.approx(recomputed, rel=2e-6)
    assert result["difference_mse_ours_minus_framewise"] == pytest.approx(
        result["models"]["factorized"]["next_prediction_mse"] - result["models"]["framewise"]["next_prediction_mse"])


@pytest.mark.parametrize("change", ["reference_encoder", "reference_projector", "base_encoder", "action_mean", "pixel_std", "backbone", "history", "mode", "trainable_visual"])
def test_noncommon_latent_coordinates_and_preprocessing_rejected(change):
    pair = models();diagnosis.validate_coordinates(pair);b = pair["framewise"]
    if change == "reference_encoder": b.reference_encoder.weight.data += 1
    elif change == "reference_projector": b.reference_projector.bias.data += 1
    elif change == "base_encoder": b.base.encoder.bias.data += 1
    elif change == "action_mean": b.action_mean[0] = .1
    elif change == "pixel_std": b.pixel_std[0] = 2
    elif change == "backbone": b.base_config["same"] = False
    elif change == "history": b.config.history_length = 2
    elif change == "mode": b.config.mode = "single"
    elif change == "trainable_visual": b.config.freeze_visual = False
    with pytest.raises(ValueError): diagnosis.validate_coordinates(pair)


def test_nonfinite_prediction_is_rejected():
    pair = models();pair["factorized"].prediction_offset = float("nan")
    window = diagnosis.windows(trace(), 1)[0][0]
    with pytest.raises(ValueError, match="Nonfinite"): diagnosis.measure_window(pair, window, 1, "cpu")


def original_sources():
    originals = {}
    for environment in ("pusht", "reacher"):
        for mode in diagnosis.MODES:
            rows = [{"trajectory_id": i, "observation_id": 1, "seed": i,
                     "success": int(i == (0 if mode == "factorized" else 1)),
                     "policy_eligible": 1, "success_during_context": 0} for i in range(32)]
            originals[environment, mode] = {"status": "complete", "environment": environment,
                "model_mode": mode, "training_seed": 0,
                "planning": {"status": "complete", "split": "development", "records": rows}}
    return originals


def test_population_keeps_both_traces_and_both_directions_of_discordance():
    source = original_sources()
    expected = diagnosis.expected_population(source)
    assert len(expected) == 8
    assert {v["category"] for v in expected.values()} == {"ours_only", "baseline_only"}
    for environment in ("pusht", "reacher"):
        for mode in diagnosis.MODES:
            assert expected[environment, 0, 1, mode]["category"] == "ours_only"
            assert expected[environment, 1, 1, mode]["category"] == "baseline_only"
    # Support-success tasks cannot contribute even if their controller outcomes differ.
    for mode in diagnosis.MODES:
        source["pusht", mode]["planning"]["records"][0].update(policy_eligible=0, success_during_context=1)
    assert len(diagnosis.expected_population(source)) == 6


@pytest.mark.parametrize("change", ["incomplete", "test_split", "wrong_seed", "missing_row", "duplicate",
                                    "different_keys", "success_nan", "support_mismatch", "wrong_eligibility"])
def test_original_population_is_fixed_to_complete_matched_development_records(change):
    source = original_sources(); record = source["pusht", "framewise"]; rows = record["planning"]["records"]
    if change == "incomplete": record["status"] = "running"
    elif change == "test_split": record["planning"]["split"] = "heldout"
    elif change == "wrong_seed": record["training_seed"] = 1
    elif change == "missing_row": rows.pop()
    elif change == "duplicate": rows[-1] = deepcopy(rows[0])
    elif change == "different_keys": rows[0]["trajectory_id"] = 999
    elif change == "success_nan": rows[0]["success"] = float("nan")
    elif change == "support_mismatch": rows[0].update(policy_eligible=0, success_during_context=1)
    elif change == "wrong_eligibility": rows[0]["policy_eligible"] = 0
    with pytest.raises(ValueError): diagnosis.expected_population(source)
