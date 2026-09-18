"""Independent causal, data-split, and cache-provenance checks for real video.

Synthetic tensors and a dummy frozen encoder test contracts, not DROID results.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from shiftwm.real_video.data import RealVideoDataset, atomic_json, sha256, training_statistics, validate_manifest
from shiftwm.real_video.model import RealVideoConfig, RealVideoWorldModel, from_config
from shiftwm.real_video.features import extract

PRIMARY = "exterior_image_1_left"
SECONDARY = "exterior_image_2_left"


@pytest.fixture(autouse=True)
def small_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model(mode="factorized", feature_dim=24):
    torch.manual_seed(818)
    config = RealVideoConfig(feature_dim=feature_dim, hidden_dim=24, depth=1,
                             context_dim=8, context_hidden=16, mode=mode)
    return RealVideoWorldModel(config, torch.linspace(-.3, .3, feature_dim),
                              torch.linspace(.8, 1.2, feature_dim), torch.linspace(-1, 1, 35),
                              torch.linspace(.5, 1.5, 35)).eval()


def batch(horizon=10, feature_dim=24):
    gen = torch.Generator().manual_seed(123)
    return {"features": torch.randn(2, 3+horizon, feature_dim, generator=gen),
            "actions": torch.randn(2, 2+horizon, 35, generator=gen)}


def train_updates(network, inputs, count=3):
    optimizer = torch.optim.Adam((p for p in network.parameters() if p.requires_grad), lr=.002)
    for _ in range(count):
        optimizer.zero_grad()
        network(inputs)["loss"].backward()
        optimizer.step()


@pytest.mark.parametrize("mode", RealVideoWorldModel.MODES)
def test_shared_initialization_is_persistence_for_horizon10(mode):
    network, inputs = model(mode), batch()
    output = network(inputs)
    assert output["predictions"].shape == (2, 10, 24)
    torch.testing.assert_close(output["predictions"], inputs["features"][:, 2:3].expand(-1, 10, -1), rtol=1e-6, atol=1e-6)


def test_all_common_parameters_have_identical_seeded_initialization_across_modes():
    models = [model(mode) for mode in RealVideoWorldModel.MODES]
    reference = models[0].state_dict()
    for network in models[1:]:
        assert network.state_dict().keys() == reference.keys()
        for key, tensor in network.state_dict().items():
            torch.testing.assert_close(tensor, reference[key], rtol=0, atol=0)


def test_actual1536_feature_coordinates_and35_action_dimensions():
    network = model(feature_dim=1536)
    result = network(batch(feature_dim=1536))
    assert result["predictions"].shape == (2, 10, 1536)
    assert torch.isfinite(result["loss"])


def test_future_target_values_cannot_change_predictions_or_support_context():
    network, inputs = model(), batch()
    train_updates(network, inputs)
    calls = []
    handle = network.dynamics_context.register_forward_pre_hook(lambda module, args: calls.append(tuple(x.detach().clone() for x in args)))
    before = network(inputs)
    changed = {"features": inputs["features"].clone(), "actions": inputs["actions"].clone()}
    changed["features"][:, 3:] = 987.
    after = network(changed)
    handle.remove()
    torch.testing.assert_close(before["predictions"], after["predictions"], rtol=0, atol=0)
    assert not torch.equal(before["loss"], after["loss"])
    for first, second in zip(calls[0], calls[1]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert calls[0][0].shape[1] == 3 and calls[0][1].shape == (2, 2, 35)
    features = inputs["features"].clone().requires_grad_()
    gradient = torch.autograd.grad(network({**inputs, "features": features})["predictions"].sum(), features)[0]
    assert torch.count_nonzero(gradient[:, 3:]) == 0
    assert torch.count_nonzero(gradient[:, :3]) > 0


def test_future_actions_are_causal_and_recursive_predictions_use_no_future_rgb():
    network, inputs = model(), batch()
    train_updates(network, inputs)
    old = network(inputs)["predictions"]
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["actions"][:, 7:] += 40
    new = network(changed)["predictions"]
    torch.testing.assert_close(old[:, :5], new[:, :5], rtol=0, atol=0)
    assert not torch.equal(old[:, 5:], new[:, 5:])
    calls = []
    handle = network.action_projection.register_forward_pre_hook(lambda module, args: calls.append(args[0].detach().clone()))
    network(inputs)
    handle.remove()
    normalized = network.normalize_actions(inputs["actions"])
    assert len(calls) == 10
    for step, actions in enumerate(calls):
        torch.testing.assert_close(actions, normalized[:, step:step+3], rtol=0, atol=0)


def test_action_free_stays_invariant_even_after_optimization():
    network, inputs = model("action_free"), batch()
    train_updates(network, inputs)
    original = network(inputs)["predictions"]
    changed = {**inputs, "actions": torch.randn_like(inputs["actions"]) * 1000}
    torch.testing.assert_close(original, network(changed)["predictions"], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in network.action_projection.parameters())


@pytest.mark.parametrize("mode", ["framewise", "constant_dynamics", "factorized"])
def test_action_gradient_is_nonzero_after_two_updates_despite_zero_gate_initialization(mode):
    network, inputs = model(mode), batch()
    train_updates(network, inputs, count=2)
    actions = inputs["actions"].clone().requires_grad_()
    prediction = network({**inputs, "actions": actions})["predictions"]
    gradient = torch.autograd.grad(prediction.square().mean(), actions)[0]
    assert torch.isfinite(gradient).all() and gradient[:, 2:].abs().max() > 1e-10
    loss = network(inputs)["loss"]
    network.zero_grad()
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in network.action_projection.parameters() if p.grad is not None) > 0


def test_constant_dynamics_context_is_episode_invariant_while_observation_context_is_not():
    network, inputs = model("constant_dynamics"), batch()
    normalized = network.normalize_features(inputs["features"][:, :3])
    actions = network.normalize_actions(inputs["actions"][:, :2])
    obs_a, dyn_a = network._context(normalized, actions)
    obs_b, dyn_b = network._context(normalized * 2 + .1, actions + 99)
    torch.testing.assert_close(dyn_a, dyn_b, rtol=0, atol=0)
    assert not torch.equal(obs_a, obs_b)


def test_factorized_context_uses_observed_actions_but_no_query():
    network, inputs = model(), batch()
    observations = network.normalize_features(inputs["features"][:, :3])
    past = network.normalize_actions(inputs["actions"][:, :2])
    obs_a, dyn_a = network._context(observations, past)
    obs_b, dyn_b = network._context(observations, past + 1)
    torch.testing.assert_close(obs_a, obs_b, rtol=0, atol=0)
    assert not torch.equal(dyn_a, dyn_b)


def test_offline_serialization_uses_vendored_predictor_without_network(tmp_path, monkeypatch):
    network, inputs = model(), batch()
    train_updates(network, inputs)
    torch.save(network.state_dict(), tmp_path / "weights.pt")
    (tmp_path / "config.json").write_text(json.dumps(network.package_config))
    import shiftwm.upstream as upstream
    monkeypatch.setattr(upstream, "ROOT", tmp_path / "no_external_checkout")
    import sys
    monkeypatch.delitem(sys.modules, "shiftwm_upstream_module", raising=False)
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *args, **kwargs: pytest.fail("Network access during package load"))
    restored = from_config(json.loads((tmp_path / "config.json").read_text()))
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True), strict=True)
    restored.eval()
    torch.testing.assert_close(restored(inputs)["predictions"], network(inputs)["predictions"], rtol=0, atol=0)


def write_cache(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    episodes, expected = [], []
    for number, (split, offset) in enumerate((("train", 0), ("train", 10), ("val", 1000), ("test", -1000))):
        frames = np.arange(15, dtype=np.int64) * 5
        features = np.arange(15 * 24, dtype=np.float32).reshape(15, 24) / 100 + offset
        native = np.arange(70 * 7, dtype=np.float32).reshape(70, 7) / 100 + offset
        actions = native.reshape(14, 35)
        path = root / f"episode{number}.npz"
        np.savez_compressed(path, features=features, actions=actions, frame_indices=frames)
        episodes.append({"episode_id": f"episode{number}", "session_id": f"site/date{number}", "split": split,
                         "steps": 14, "cameras": {PRIMARY: {"file": path.name, "sha256": sha256(path)}}})
        if split == "train":
            expected.append((features, actions))
    manifest = {"status": "complete", "feature_dim": 24, "action_dim": 35, "action_block": 5,
                "history_length": 3, "episodes": episodes}
    atomic_json(manifest, root / "manifest.json")
    return root, manifest, expected


def test_windows_join_exact_five_native7d_actions_and_never_cross_episode(tmp_path):
    root, manifest, _ = write_cache(tmp_path)
    dataset = RealVideoDataset(root, "train", horizon=10, stride=1)
    assert len(dataset) == 6
    for row in dataset:
        start, episode_index = row["window_start"], row["episode_index"]
        assert row["features"].shape == (13, 24) and row["actions"].shape == (12, 35)
        native = np.arange(70 * 7, dtype=np.float32).reshape(70, 7) / 100 + 10 * episode_index
        np.testing.assert_array_equal(row["actions"].numpy(), native[start*5:(start+12)*5].reshape(12, 35))
        np.testing.assert_array_equal(row["features"].numpy(), dataset.episodes[episode_index]["features"][start:start+13])


def test_statistics_use_only_train_primary_camera_and_sample_ddof(tmp_path):
    root, manifest, expected = write_cache(tmp_path)
    before = training_statistics(root)
    for key, index in (("feature", 0), ("action", 1)):
        joined = np.concatenate([row[index] for row in expected]).astype(np.float64)
        np.testing.assert_allclose(before[key + "_mean"], joined.mean(0), atol=1e-12)
        np.testing.assert_allclose(before[key + "_std"], joined.std(0, ddof=1), atol=1e-12)
    # Even unreadable held-out payloads cannot affect training normalization.
    for row in manifest["episodes"]:
        if row["split"] != "train":
            (root / row["cameras"][PRIMARY]["file"]).write_bytes(b"held-out bytes never consumed by fitting")
    after = training_statistics(root)
    assert before == after


@pytest.mark.parametrize("change", ["incomplete", "empty", "session_overlap", "unknown_split", "duplicate_id", "no_test"])
def test_split_or_incomplete_manifest_rejected_by_both_windows_and_statistics(tmp_path, change):
    root, manifest, _ = write_cache(tmp_path)
    if change == "incomplete": manifest["status"] = "in_progress"
    elif change == "empty": manifest["episodes"] = []
    elif change == "session_overlap": manifest["episodes"][2]["session_id"] = manifest["episodes"][0]["session_id"]
    elif change == "unknown_split": manifest["episodes"][2]["split"] = "development"
    elif change == "duplicate_id": manifest["episodes"][2]["episode_id"] = manifest["episodes"][0]["episode_id"]
    elif change == "no_test": manifest["episodes"] = manifest["episodes"][:-1]
    atomic_json(manifest, root / "manifest.json")
    for operation in (lambda: RealVideoDataset(root, "train"), lambda: training_statistics(root)):
        with pytest.raises(ValueError): operation()


def test_feature_payload_corruption_rejected(tmp_path):
    root, manifest, _ = write_cache(tmp_path)
    (root / manifest["episodes"][0]["cameras"][PRIMARY]["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        RealVideoDataset(root, "train")
    with pytest.raises(ValueError, match="changed"):
        training_statistics(root)


class DummyDino(nn.Module):
    calls = []
    loads = []
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs == {"local_files_only": True}
        cls.loads.append(str(path))
        return cls()
    def forward(self, pixel_values):
        assert not self.training and not self.weight.requires_grad and not torch.is_grad_enabled()
        self.calls.append(pixel_values.detach().clone())
        n = len(pixel_values)
        base = pixel_values.mean((1, 2, 3))[:, None, None]
        spatial = torch.arange(257, device=pixel_values.device)[None, :, None] / 100
        channels = torch.arange(384, device=pixel_values.device)[None, None, :] / 1000
        return SimpleNamespace(last_hidden_state=(base + spatial + channels).expand(n, 257, 384))


@pytest.fixture
def raw_video_fixture(tmp_path, monkeypatch):
    import transformers
    encoder_class = transformers.Dinov2Model
    monkeypatch.setattr(encoder_class, "from_pretrained", staticmethod(DummyDino.from_pretrained))
    DummyDino.calls, DummyDino.loads = [], []
    source, encoder, cache = (tmp_path / name for name in ("source", "encoder", "features"))
    source.mkdir(); encoder.mkdir()
    rows = []
    for number, split in enumerate(("train", "val", "test")):
        cameras = {}
        for camera in (PRIMARY, SECONDARY):
            images = np.stack([np.full((8, 12, 3), index + number * 30, np.uint8) for index in range(15)])
            path = source / f"{split}-{camera}.npz"
            actions = np.arange(14 * 35, dtype=np.float32).reshape(14, 35) / 100
            np.savez_compressed(path, images=images, actions=actions, frame_indices=np.arange(15, dtype=np.int64)*5)
            cameras[camera] = {"file": path.name, "sha256": sha256(path)}
        rows.append({"episode_id": split, "session_id": f"site/date{number}", "split": split, "steps": 14, "cameras": cameras})
    manifest = {"status": "complete", "episodes": rows, "action_dim": 35, "action_block": 5}
    atomic_json(manifest, source / "manifest.json")
    atomic_json({"status": "passed", "dataset_manifest_sha256": sha256(source / "manifest.json")}, source / "data_audit.json")
    (encoder / "dummy_weights.bin").write_bytes(b"frozen test encoder identity")
    atomic_json({"files": [{"file": "dummy_weights.bin", "sha256": sha256(encoder / "dummy_weights.bin")}]}, encoder / "provenance.json")
    return source, encoder, cache


def test_extractor_preserves_actions_frames_and_uses_only_registered_camera_policy(raw_video_fixture):
    source, encoder, cache = raw_video_fixture
    extract(source, cache, encoder, batch_size=4, device="cpu")
    result = json.loads((cache / "manifest.json").read_text())
    assert result["status"] == "complete" and result["feature_dim"] == 1536
    assert len(result["episodes"]) == 3
    for row in result["episodes"]:
        assert set(row["cameras"]) == ({PRIMARY, SECONDARY} if row["split"] == "test" else {PRIMARY})
        for camera, record in row["cameras"].items():
            with np.load(cache / record["file"]) as output, np.load(source / f"{row['split']}-{camera}.npz") as original:
                np.testing.assert_array_equal(output["actions"], original["actions"])
                np.testing.assert_array_equal(output["frame_indices"], original["frame_indices"])
                assert output["features"].shape == (15, 1536)
    old_calls, old_hash = len(DummyDino.calls), sha256(cache / "manifest.json")
    extract(source, cache, encoder, batch_size=4, device="cpu")
    assert len(DummyDino.calls) == old_calls and sha256(cache / "manifest.json") == old_hash


@pytest.mark.parametrize("kind", ["incomplete", "audit_failed", "audit_stale", "encoder_corrupt"])
def test_extractor_rejects_incomplete_or_unverified_sources_before_encoding(raw_video_fixture, kind):
    source, encoder, cache = raw_video_fixture
    if kind == "incomplete":
        manifest = json.loads((source / "manifest.json").read_text()); manifest["status"] = "working"
        atomic_json(manifest, source / "manifest.json")
    elif kind == "audit_failed":
        atomic_json({"status": "failed", "dataset_manifest_sha256": sha256(source / "manifest.json")}, source / "data_audit.json")
    elif kind == "audit_stale":
        atomic_json({"status": "passed", "dataset_manifest_sha256": "old identity"}, source / "data_audit.json")
    else:
        (encoder / "dummy_weights.bin").write_bytes(b"modified encoder")
    with pytest.raises(ValueError):
        extract(source, cache, encoder, batch_size=4, device="cpu")
    assert not DummyDino.loads and not DummyDino.calls


def test_changed_extraction_precision_or_batch_identity_cannot_mix_caches(raw_video_fixture):
    source, encoder, cache = raw_video_fixture
    extract(source, cache, encoder, batch_size=4, device="cpu")
    with pytest.raises(ValueError, match="identit"):
        extract(source, cache, encoder, batch_size=5, device="cpu")


def test_orphan_cache_cannot_adopt_an_unverified_new_extraction_identity(raw_video_fixture):
    source, encoder, cache = raw_video_fixture
    extract(source, cache, encoder, batch_size=4, device="cpu")
    (cache / "identity.json").unlink()
    with pytest.raises(ValueError):
        extract(source, cache, encoder, batch_size=4, device="cpu")


@pytest.mark.parametrize("mutation", ["column_frame_indices", "fractional_frame_indices", "nan_features", "nan_actions"])
def test_bad_feature_shapes_or_nonfinite_statistics_are_rejected(tmp_path, mutation):
    root, manifest, _ = write_cache(tmp_path)
    record = manifest["episodes"][0]["cameras"][PRIMARY]
    path = root / record["file"]
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if mutation == "column_frame_indices": arrays["frame_indices"] = arrays["frame_indices"][:, None]
    elif mutation == "fractional_frame_indices": arrays["frame_indices"] = arrays["frame_indices"].astype(float) + .4
    elif mutation == "nan_features": arrays["features"][0, 0] = np.nan
    else: arrays["actions"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    record["sha256"] = sha256(path)
    atomic_json(manifest, root / "manifest.json")
    with pytest.raises(ValueError):
        RealVideoDataset(root, "train")
    if mutation.startswith("nan_"):
        with pytest.raises(ValueError):
            training_statistics(root)


@pytest.mark.parametrize("mutation", ["payload", "receipt_missing", "payload_missing", "identity_receipt", "source_receipt"])
def test_completed_feature_cache_cannot_be_silently_repaired_or_relabelled(raw_video_fixture, mutation):
    source, encoder, cache = raw_video_fixture
    extract(source, cache, encoder, batch_size=4, device="cpu")
    payload = cache / "episodes" / PRIMARY / "train.npz"
    receipt = payload.with_suffix(".json")
    if mutation == "payload": payload.write_bytes(b"changed immutable cache")
    elif mutation == "receipt_missing": receipt.unlink()
    elif mutation == "payload_missing": payload.unlink()
    else:
        row = json.loads(receipt.read_text())
        row["identity_sha256" if mutation == "identity_receipt" else "source_sha256"] = "wrong hash"
        atomic_json(row, receipt)
    old_calls = len(DummyDino.calls)
    with pytest.raises(ValueError):
        extract(source, cache, encoder, batch_size=4, device="cpu")
    assert len(DummyDino.calls) == old_calls


@pytest.mark.parametrize("mutation", ["column_indices", "fractional_indices", "action_shape", "action_nonfinite"])
def test_raw_frame_and_action_contract_failure_rejected(raw_video_fixture, mutation):
    source, encoder, cache = raw_video_fixture
    manifest = json.loads((source / "manifest.json").read_text())
    record = manifest["episodes"][0]["cameras"][PRIMARY]
    path = source / record["file"]
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if mutation == "column_indices": arrays["frame_indices"] = arrays["frame_indices"][:, None]
    elif mutation == "fractional_indices": arrays["frame_indices"] = arrays["frame_indices"].astype(float) + .4
    elif mutation == "action_shape": arrays["actions"] = arrays["actions"][:, :-1]
    else: arrays["actions"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    record["sha256"] = sha256(path)
    atomic_json(manifest, source / "manifest.json")
    atomic_json({"status": "passed", "dataset_manifest_sha256": sha256(source / "manifest.json")}, source / "data_audit.json")
    with pytest.raises(ValueError):
        extract(source, cache, encoder, batch_size=4, device="cpu")
    assert not DummyDino.calls


def test_single_observation_episode_is_retained_but_has_no_prediction_windows(tmp_path):
    root, manifest, _ = write_cache(tmp_path)
    record = manifest["episodes"][0]["cameras"][PRIMARY]
    path = root / record["file"]
    np.savez_compressed(path, features=np.ones((1, 24), np.float32),
                        actions=np.empty((0, 35), np.float32), frame_indices=np.array([0], np.int64))
    record["sha256"] = sha256(path)
    manifest["episodes"][0]["steps"] = 0
    atomic_json(manifest, root / "manifest.json")
    dataset = RealVideoDataset(root, "train", horizon=10, stride=1)
    assert len(dataset.episodes) == 2 and len(dataset) == 3
    assert all(episode_index == 1 for episode_index, _ in dataset.windows)
    stats = training_statistics(root)
    assert stats["counts"] == {"feature": 16, "action": 14}
    assert np.isfinite(stats["feature_mean"]).all() and np.isfinite(stats["action_std"]).all()


def test_extractor_retains_one_frame_zero_action_episode_without_fabricating_transitions(raw_video_fixture):
    source, encoder, cache = raw_video_fixture
    manifest = json.loads((source / "manifest.json").read_text())
    original = manifest["episodes"][0]
    long_episode = deepcopy(original)
    long_episode["episode_id"], long_episode["session_id"] = "train-long", "site/date-long"
    for camera in (PRIMARY, SECONDARY):
        record = original["cameras"][camera]
        path = source / record["file"]
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        long_path = source / f"long-{camera}.npz"
        np.savez_compressed(long_path, **arrays)
        long_episode["cameras"][camera] = {"file": long_path.name, "sha256": sha256(long_path)}
        np.savez_compressed(path, images=arrays["images"][:1], actions=np.empty((0,35),np.float32),
                            frame_indices=np.array([0],np.int64))
        record["sha256"] = sha256(path)
    original["steps"] = 0
    manifest["episodes"].append(long_episode)
    atomic_json(manifest, source / "manifest.json")
    atomic_json({"status":"passed", "dataset_manifest_sha256":sha256(source / "manifest.json")}, source / "data_audit.json")
    extract(source, cache, encoder, batch_size=4, device="cpu")
    cached = json.loads((cache / "manifest.json").read_text())
    assert len(cached["episodes"]) == 4
    short = next(row for row in cached["episodes"] if row["episode_id"] == "train")
    with np.load(cache / short["cameras"][PRIMARY]["file"]) as arrays:
        assert arrays["features"].shape == (1,1536) and arrays["actions"].shape == (0,35)
        np.testing.assert_array_equal(arrays["frame_indices"], [0])
    dataset = RealVideoDataset(cache, "train", horizon=10, stride=1)
    assert len(dataset.episodes) == 2 and len(dataset) == 3
    stats = json.loads((cache / "training_statistics.json").read_text())
    assert stats["counts"] == {"action":14, "feature":16}
