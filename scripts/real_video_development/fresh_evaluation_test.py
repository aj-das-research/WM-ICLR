"""Fresh-population and causal-adapter checks on synthetic arrays only."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fresh_evaluation_common as common
import fresh_evaluation as pipeline


def manifest(rows):
    return {"status": "complete", "episodes": rows}


def row(identifier="a", session="new-session", split="test"):
    return {"episode_id": identifier, "session_id": session, "split": split}


def test_test_only_population_keeps_exact_membership():
    common.validate_population(manifest([row()]), [("a", "new-session")])


@pytest.mark.parametrize("rows", [[], [row(), row()], [row(split="train")], [row(session="wrong")], [row(identifier="extra")]])
def test_population_change_fails(rows):
    with pytest.raises(ValueError):
        common.validate_population(manifest(rows), [("a", "new-session")])


def test_scoped_adapter_restores_binding_even_on_error():
    owner = SimpleNamespace(callback="original")
    with pytest.raises(RuntimeError):
        with common.scoped_adapter(owner, "callback", "replacement"):
            assert owner.callback == "replacement"
            raise RuntimeError("deliberate failure")
    assert owner.callback == "original"


def test_statistics_are_copied_exactly_and_changed_destination_rejected(tmp_path, monkeypatch):
    reference = tmp_path / "original.json"
    reference.write_text('{"fit_split": "train", "counts": {"feature": 123}}\n')
    destination = tmp_path / "fresh"
    destination.mkdir()
    monkeypatch.setattr(pipeline, "STATS", reference)
    assert pipeline.copy_original_statistics(destination)["fit_split"] == "train"
    assert (destination / "training_statistics.json").read_bytes() == reference.read_bytes()
    (destination / "training_statistics.json").write_text('{}\n')
    with pytest.raises(ValueError):
        pipeline.copy_original_statistics(destination)


def test_existing_loader_keeps_short_episodes_and_exact_causal_windows(tmp_path, monkeypatch):
    camera = common.CAMERAS[0]
    episodes = []
    for name, frames in (("short", 1), ("eligible", 13)):
        payload = tmp_path / (name + ".npz")
        np.savez(payload, features=np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
                 actions=np.arange(max(frames - 1, 0) * 35, dtype=np.float32).reshape(frames - 1, 35),
                 frame_indices=np.arange(frames, dtype=np.int64) * 5)
        episodes.append({**row(name, "session-" + name), "steps": frames - 1,
                         "cameras": {camera: {"file": payload.name, "sha256": common.sha256(payload)}}})
    common.atomic_json({"retained": episodes}, tmp_path / "metadata_manifest.json")
    common.atomic_json({"status": "complete", "episodes": episodes, "feature_dim": 2, "action_dim": 35}, tmp_path / "manifest.json")
    monkeypatch.setattr(common, "FRESH", tmp_path)
    monkeypatch.setattr(pipeline, "CACHE", tmp_path)
    dataset = pipeline.fresh_dataset(camera, 5)
    assert len(dataset.episodes) == 2
    assert dataset.windows == [(1, 0), (1, 5)]
    assert dataset[0]["features"].shape == (8, 2)
    assert dataset[0]["actions"].shape == (7, 35)
    assert len(pipeline.fresh_dataset(camera, 10)) == 1
    # The production guard is restored; a test-only manifest cannot enter the
    # original training loader after this adapter's context exits.
    with pytest.raises(ValueError):
        pipeline.original_data.validate_manifest(dataset.manifest)

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("feature_std", torch.ones(2))
            self.calls = []
        def predict(self, support, past, future):
            self.calls.append((support.clone(), past.clone(), future.clone()))
            return support[:, -1:].expand(-1, len(future[0]), -1)

    spy = Spy()
    result = pipeline.evaluate_dataset(spy, dataset, "cpu", batch_size=128)
    assert result["episode_count"] == 1 and result["window_count"] == 2
    for support, past, future in spy.calls:
        assert support.shape == (2, 3, 2)
        assert past.shape == (2, 2, 35)
        assert future.shape == (2, 5, 35)
        torch.testing.assert_close(support[0], dataset[0]["features"][:3])


def test_bootstrap_preserves_constant_paired_difference():
    result = pipeline.crossed_session_bootstrap([[-0.2, -0.2, -0.2]] * 3,
                                                ["s1", "s1", "s2"], draws=50, seed=20260919)
    assert result["mean_difference"] == pytest.approx(-0.2)
    assert result["ci95"] == pytest.approx([-0.2, -0.2])
