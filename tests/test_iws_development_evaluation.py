"""Independent hand calculations and population checks for IWS feature scoring."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iws_dev_eval_test", ROOT / "scripts/real_video_iws/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_metrics_have_declared_coordinate_scale_and_zero_vector_rule():
    prediction = torch.tensor([[[3., 4.], [0., 0.], [1., 0.]]])
    target = torch.tensor([[[0., 0.], [0., 0.], [-1., 0.]]])
    result = evaluation.feature_errors(prediction, target, torch.tensor([3., 2.]))
    torch.testing.assert_close(result["standardized_mse"], torch.tensor([[2.5, 0., 2./9]]))
    torch.testing.assert_close(result["standardized_mae"], torch.tensor([[1.5, 0., 1./3]]))
    torch.testing.assert_close(result["raw_dinov2_l1"], torch.tensor([[3.5, 0., 1.]]))
    torch.testing.assert_close(result["feature_cosine_distance"], torch.tensor([[1., 1., 2.]]))


def fixture():
    audit = {"records": [
        {"episode_id": "000011", "episode_index": 0, "frames": 61, "windows": 1},
        {"episode_id": "000012", "episode_index": 1, "frames": 66, "windows": 2},
        {"episode_id": "000013", "episode_index": 2, "frames": 60, "windows": 0}],
        "eligible_episodes": 2, "windows": 3}
    arrays = {"episode_index": np.array([0, 1, 1]), "window_start": np.array([0, 0, 5])}
    for key in evaluation.METRICS:
        arrays[key] = np.repeat(np.array([[9.], [1.], [3.]]), 59, axis=1)
        arrays["persistence_"+key] = arrays[key] + 10
    return arrays, audit


def test_equal_trajectory_estimate_differs_from_equal_window_estimate():
    arrays, audit = fixture()
    episodes = evaluation.aggregate_windows(arrays, audit)
    assert [r["episode_id"] for r in episodes] == ["000011", "000012"]
    assert [r["windows"] for r in episodes] == [1, 2]
    assert np.mean([r["standardized_mse_by_offset"][-1] for r in episodes]) == 5.5
    assert arrays["standardized_mse"][:, -1].mean() == pytest.approx(13/3)
    assert np.mean([r["persistence_standardized_mse_by_offset"][-1] for r in episodes]) == 15.5
    assert audit["records"][2]["windows"] == 0


@pytest.mark.parametrize("mutation", [
    lambda a: a["window_start"].__setitem__(2, 0),
    lambda a: a["episode_index"].__setitem__(2, 0),
    lambda a: a["standardized_mse"].__setitem__((0, 58), np.nan),
    lambda a: a["persistence_standardized_mse"].__setitem__((0, 58), -1),
])
def test_invalid_or_duplicate_windows_fail_closed(mutation):
    arrays, audit = fixture()
    mutation(arrays)
    with pytest.raises(ValueError):
        evaluation.aggregate_windows(arrays, audit)


def test_mismatched_horizon_and_missing_window_rejected():
    arrays, audit = fixture()
    arrays["raw_dinov2_l1"] = arrays["raw_dinov2_l1"][:, :14]
    with pytest.raises(ValueError):
        evaluation.aggregate_windows(arrays, audit)
    arrays, audit = fixture()
    arrays = {key: value[:-1] for key, value in arrays.items()}
    with pytest.raises(ValueError):
        evaluation.aggregate_windows(arrays, audit)


def test_atomic_ledger_roundtrip_preserves_all_primitives(tmp_path):
    arrays, _ = fixture()
    path = tmp_path / "windows.npz"
    evaluation.atomic_npz(path, arrays)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, loaded[key])
