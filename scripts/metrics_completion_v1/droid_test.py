"""Numerical contracts and integrity gates for complementary DROID scoring."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import droid_common as c
import droid_finalize as final


def fixture_arrays():
    keys = [("a", "session1", 0), ("a", "session1", 5), ("b", "session2", 0)]
    arrays = {"episode_id": np.array(["a", "a", "b"]), "session_id": np.array(["session1", "session1", "session2"]),
              "window_start": np.array([0, 5, 0], dtype=np.int64)}
    # Unequal window counts ensure an accidental all-window mean is detected.
    for i, key in enumerate(c.METRICS):
        arrays[key] = np.stack([np.full(10, .1 + .01 * i), np.full(10, .3 + .01 * i), np.full(10, .8 + .01 * i)])
    return arrays, keys


def test_metric_formulas_independent_numpy_and_zero_vectors():
    helper = c.module("scripts/real_video_iws/evaluate.py", "droid_test_metric_helper")
    prediction = np.array([[[1., -3., 2.], [0., 0., 0.]], [[4., 2., -1.], [-1., -1., -1.]]], dtype=np.float32)
    target = np.array([[[3., 1., -2.], [0., 0., 0.]], [[-1., 0., 2.], [1., 1., 1.]]], dtype=np.float32)
    std = np.array([2., .5, 4.], dtype=np.float32)
    actual = helper.feature_errors(torch.tensor(prediction), torch.tensor(target), torch.tensor(std))
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    expected = {"standardized_mse": np.mean((difference / std) ** 2, axis=-1),
                "standardized_mae": np.mean(abs(difference) / std, axis=-1),
                "raw_dinov2_l1": np.mean(abs(difference), axis=-1)}
    p = prediction.astype(np.float64); t = target.astype(np.float64)
    expected["feature_cosine_distance"] = 1 - np.clip(np.sum(
        p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-8) *
        t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-8), axis=-1), -1, 1)
    for key in c.METRICS:
        np.testing.assert_allclose(actual[key].numpy(), expected[key], rtol=2e-7, atol=2e-7)
    assert actual["feature_cosine_distance"][0, 1] == 1  # Explicit zero-vector convention.
    for wrong_std in (torch.zeros(3), torch.tensor([1., float("nan"), 1.])):
        with pytest.raises(ValueError):
            helper.feature_errors(torch.tensor(prediction), torch.tensor(target), wrong_std)


def test_equal_episode_aggregation_and_atomic_npz_roundtrip(tmp_path):
    arrays, keys = fixture_arrays()
    episodes, summary = c.validate_arrays(arrays, keys)
    np.testing.assert_allclose(summary["standardized_mse"], .5)
    assert not np.isclose(summary["standardized_mse"][0], arrays["standardized_mse"].mean())
    assert [e["windows"] for e in episodes] == [2, 1]
    path = tmp_path / "ledger.npz"
    c.atomic_npz(arrays, path)
    with np.load(path, allow_pickle=False) as saved:
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], saved[key])
    with pytest.raises(FileExistsError):
        c.atomic_npz(arrays, path)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "reorder", "nonfinite", "negative", "cosine_range", "float_start"])
def test_population_and_numeric_fail_closed(defect):
    arrays, keys = fixture_arrays()
    if defect == "missing":
        arrays = {key: value[:-1] for key, value in arrays.items()}
    elif defect == "duplicate":
        arrays["window_start"][1] = 0
    elif defect == "reorder":
        arrays = {key: value[::-1] for key, value in arrays.items()}
    elif defect == "nonfinite":
        arrays["standardized_mae"][1, 3] = np.nan
    elif defect == "negative":
        arrays["raw_dinov2_l1"][1, 3] = -.1
    elif defect == "cosine_range":
        arrays["feature_cosine_distance"][1, 3] = 2.01
    else:
        arrays["window_start"] = arrays["window_start"].astype(float)
    with pytest.raises(ValueError):
        c.validate_arrays(arrays, keys)


def test_prior_mse_verifies_every_window_not_just_aggregate():
    arrays, keys = fixture_arrays()
    episodes, summary = c.validate_arrays(arrays, keys)
    prior = {"windows": [{"episode_id": e, "session_id": s, "window_start": w,
                           "native_mse": arrays["standardized_mse"][i].tolist()} for i, (e, s, w) in enumerate(keys)],
             "episodes": [{"episode_id": e["episode_id"], "session_id": e["session_id"], "windows": e["windows"],
                           "native_mse": e["standardized_mse"]} for e in episodes],
             "summary": {"native_mse": summary["standardized_mse"]}}
    assert c.prior_mse_check(arrays, episodes, summary, prior)["max_window_absolute_error"] == 0
    arrays["standardized_mse"][0, 0] += .01
    arrays["standardized_mse"][1, 0] -= .01  # The episode and campaign means remain identical.
    with pytest.raises(AssertionError):
        c.prior_mse_check(arrays, episodes, summary, prior)


def test_frozen_bootstrap_matches_independent_crossed_resampling():
    helper = c.module("scripts/real_video_spatial/validate_ledger.py", "droid_test_bootstrap")
    helper.METRICS = c.METRICS
    population = [("a", "s1"), ("b", "s1"), ("c", "s2")]
    rng = np.random.default_rng(42)
    matrices = [rng.uniform(.1, .9, (3, 3, 4, 10)), rng.uniform(.1, .9, (3, 3, 4, 10))]
    records = []
    for mode, matrix in zip(("first", "second"), matrices):
        for seed in range(3):
            records.append({"mode": mode, "seed": seed,
                "windows": [{"episode_id": e, "session_id": s, "window_start": 0} for e, s in population],
                "episodes": [{"episode_id": e, "session_id": s,
                              **{metric: matrix[seed, i, m].tolist() for m, metric in enumerate(c.METRICS)}}
                             for i, (e, s) in enumerate(population)]})
    actual = helper.paired_intervals(records, "first", "second", draws=101, bootstrap_seed=173)
    expected = []
    rng = np.random.default_rng(173)
    for _ in range(101):
        selected_seeds = rng.integers(0, 3, size=3)
        selected_sessions = rng.integers(0, 2, size=2)
        indices = [i for chosen in selected_sessions for i, (_, s) in enumerate(population) if s == ("s1", "s2")[chosen]]
        # Explicit sum of matched differences, including repeated sessions/seeds.
        total = sum(matrices[0][seed, index] - matrices[1][seed, index] for seed in selected_seeds for index in indices)
        expected.append(total / (len(selected_seeds) * len(indices)))
    intervals = np.quantile(expected, [.025, .975], axis=0)
    for m, metric in enumerate(c.METRICS):
        for h, effect in enumerate(actual["metrics"][metric]):
            np.testing.assert_allclose(effect["ci95"], intervals[:, m, h], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        helper.paired_intervals(records[:-1], "first", "second", draws=101, bootstrap_seed=173)


def test_finalizer_refuses_partial_model_population():
    with pytest.raises(ValueError, match="all eleven"):
        final.analyze([], [], draws=10)


def test_atomic_json_does_not_overwrite(tmp_path):
    path = tmp_path / "registration.json"
    c.atomic_json({"fixed": 1}, path)
    with pytest.raises(ValueError):
        c.atomic_json({"fixed": 2}, path)
    assert json.loads(path.read_text()) == {"fixed": 1}
