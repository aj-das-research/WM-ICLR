import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("real_video_development", Path(__file__).resolve().parents[1] / "scripts/real_video_development/diagnose.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_development_refuses_test_before_loading_files():
    with pytest.raises(ValueError, match="never load test"):
        diagnostic.development_dataset("missing", "test")


def test_donor_is_different_session_and_batch_independent():
    ids = [0, 0, 1, 2, 2, 3]
    sessions = {0: "a", 1: "a", 2: "b", 3: "c"}
    indices = diagnostic.donor_indices(ids, sessions)
    assert indices.tolist() == [3, 3, 3, 5, 5, 0]
    assert all(sessions[i] != sessions[ids[j]] for i, j in zip(ids, indices))
    with pytest.raises(ValueError):
        diagnostic.donor_indices([0, 1], {0: "a", 1: "a"})


def test_hold_repeats_last_native_command_not_last_action_block():
    past = torch.arange(70.).reshape(1, 2, 35)
    future = torch.randn(1, 10, 35)
    frozen = past.clone()
    result = diagnostic.perturb_future("hold_support_command", future, past, torch.zeros(35))
    assert torch.equal(result[0, 0], torch.arange(63., 70.).repeat(5))
    assert torch.equal(result[0, 0], result[0, 9])
    assert torch.equal(past, frozen)
    assert torch.equal(diagnostic.perturb_future("reversed", future, past, torch.zeros(35)), future.flip(1))


def test_episode_weighting_and_stratum_empty():
    assert diagnostic.episode_mean([0, 0, 12], [0, 0, 1]) == 6
    assert diagnostic.episode_mean([0, 0, 12], [0, 0, 1], [False, False, False]) is None
    values = {"support_motion": np.array([1., 2., 3.]), "model_mse": np.array([[1, 2], [3, 4], [5, 6]])}
    rows = diagnostic.aggregate_rows(values, [0, 0, 1], {"support_motion": [1.5, 2.5]})
    assert rows["all"]["metrics"]["model_mse"] == [3.5, 4.5]
    assert rows["support_motion_tertile_3"]["metrics"]["model_mse"] == [5., 6.]


def test_displacement_moments_reconstruct_shrunken_prediction_error():
    summary_spec = importlib.util.spec_from_file_location("real_video_development_summary", Path(__file__).resolve().parents[1] / "scripts/real_video_development/summarize_diagnosis.py")
    summary = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary)
    rng = np.random.default_rng(42)
    residual, target = rng.normal(size=(7, 10, 11)), rng.normal(size=(7, 10, 11))
    for scale in (0., .25, .75, 1.):
        actual = np.mean((scale * residual - target) ** 2, axis=-1)
        reconstructed = summary.interpolation_error(np.mean(target ** 2, axis=-1), np.mean(residual ** 2, axis=-1), np.mean(residual * target, axis=-1), scale)
        np.testing.assert_allclose(actual, reconstructed, rtol=1e-12, atol=1e-12)


class DoubleMotionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_std", torch.ones(2))
    def predict(self, support, support_actions, future_actions):
        assert support.shape[1] == 3 and support_actions.shape[1] == 2
        steps = torch.arange(1, future_actions.shape[1] + 1).to(support)[None, :, None]
        return support[:, -1:] + 2 * steps


class CalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.episodes = [{"episode_id": "a", "split": split, "cameras": {"exterior_image_1_left": {"sha256": "fixture"}}}]
    def __len__(self):
        return 3
    def __getitem__(self, index):
        return {"features": torch.arange(13.)[:, None].repeat(1, 2), "actions": torch.zeros(12, 7), "episode_index": 0}


def test_calibration_fits_train_only_and_exact_half_scale():
    calibration_spec = importlib.util.spec_from_file_location("real_video_calibration", Path(__file__).resolve().parents[1] / "scripts/real_video_development/calibrate_residual.py")
    calibration = importlib.util.module_from_spec(calibration_spec)
    calibration_spec.loader.exec_module(calibration)
    result = calibration.fit_training_scale(DoubleMotionModel(), CalibrationDataset(), device="cpu")
    assert result["scale"] == .5
    assert result["eligible_episodes"] == 1 and result["windows"] == 3
    with pytest.raises(ValueError, match="only on training"):
        calibration.fit_training_scale(DoubleMotionModel(), CalibrationDataset("val"), device="cpu")


def test_calibration_wrapper_endpoints_and_causal_interface():
    from shiftwm.real_video_development import ResidualCalibratedWorldModel, residual_scale
    batch = CalibrationDataset()[0]
    support = batch["features"][:3][None]
    past, future = batch["actions"][:2][None], batch["actions"][2:][None]
    for scale in (0., .5, 1.):
        model = ResidualCalibratedWorldModel(DoubleMotionModel(), scale)
        result = model.predict(support, past, future)
        expected = support[:, -1:] + scale * 2 * torch.arange(1, 11.)[None, :, None]
        torch.testing.assert_close(result, expected.expand_as(result))
    assert residual_scale(0, 0) == 0
    assert residual_scale(1, -1) == 0
    assert residual_scale(1, 2) == 1
    with pytest.raises(ValueError):
        residual_scale(float("nan"), 1)
    with pytest.raises(ValueError):
        ResidualCalibratedWorldModel(DoubleMotionModel(), 1.1)


def test_calibration_json_reloads_offline_after_base_relocation(tmp_path):
    torch.set_num_threads(2)
    from shiftwm.real_video.model import RealVideoWorldModel
    from shiftwm.real_video.data import sha256
    from shiftwm.real_video_development import ResidualCalibratedWorldModel
    calibration_spec = importlib.util.spec_from_file_location("real_video_calibration_relocation", Path(__file__).resolve().parents[1] / "scripts/real_video_development/calibrate_residual.py")
    calibration = importlib.util.module_from_spec(calibration_spec)
    calibration_spec.loader.exec_module(calibration)
    torch.manual_seed(51)
    base = RealVideoWorldModel({"feature_dim": 4, "action_dim": 7, "hidden_dim": 12, "context_dim": 4, "context_hidden": 8, "depth": 1}, [0.] * 4, [1.] * 4, [0.] * 7, [1.] * 7).eval()
    torch.nn.init.normal_(base.output_projection[-1].weight, std=.03)
    config = {**base.package_config, "package_kind": calibration.training.PACKAGE_KIND,
              "metadata": {"parameter_counts": {"total": sum(p.numel() for p in base.parameters()), "trainable": sum(p.numel() for p in base.parameters() if p.requires_grad)}}}
    directory = tmp_path / "relocated"
    directory.mkdir()
    torch.save({"config": config, "epoch": 1, "step": 1, "state_dict": base.state_dict()}, directory / "model.pt")
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "package_manifest.json").write_text(json.dumps({"format_version": 1, "package_kind": calibration.training.PACKAGE_KIND, "files": {name: sha256(directory / name) for name in ("model.pt", "config.json")}}))
    record = {"package_kind": "development_train_only_residual_calibration", "format_version": 1,
              "fit": {"fit_split": "train", "displacement_energy": 4., "displacement_alignment": 2., "scale": .5},
              "base_checkpoint": "/old/nonexistent/host/path", "base_checkpoint_sha256": sha256(directory / "model.pt"),
              "wrapper_sha256": sha256(Path(__file__).resolve().parents[1] / "src/shiftwm/real_video_development.py")}
    package_path = tmp_path / "calibration.json"
    package_path.write_text(json.dumps(record))
    model = calibration.load_calibrated_package(package_path, base_checkpoint=directory)
    expected = ResidualCalibratedWorldModel(base, .5).eval()
    support, past, future = torch.randn(2, 3, 4), torch.randn(2, 2, 7), torch.randn(2, 10, 7)
    with torch.inference_mode():
        torch.testing.assert_close(model.predict(support, past, future), expected.predict(support, past, future), rtol=0, atol=0)
    record["fit"]["scale"] = .75
    package_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match training moments"):
        calibration.load_calibrated_package(package_path, base_checkpoint=directory)
