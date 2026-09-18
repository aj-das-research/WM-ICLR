"""Scientific boundary tests; synthetic fixtures are not benchmark evidence."""
from copy import deepcopy

import pytest
import torch

from shiftwm.extensions.model import ExtensionWorldModel, GRUPredictor, MODES
from test_model import TinyBase, BASE_CONFIG


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(mode):
    return ExtensionWorldModel(TinyBase(), BASE_CONFIG, mode, [0.] * 4, [1.] * 4,
                               model_config={"context_dim": 4, "context_hidden": 16}).eval()


def batch():
    torch.manual_seed(21)
    return {"features": torch.randn(3, 8, 8), "reference_features": torch.randn(3, 8, 8),
            "paired_features": torch.randn(3, 8, 8), "actions": torch.randn(3, 7, 4),
            "observation_id": torch.tensor([0, 0, 1])}


@pytest.mark.parametrize("mode", MODES)
def test_recursive_targets_and_query_inaccessibility(mode):
    m, b = model(mode), batch()
    out = m(b)
    assert out["predictions"].shape == (3, 5, 8)
    torch.testing.assert_close(out["targets"], b["reference_features"][:, 3:])
    changed = deepcopy(b)
    changed["features"][:, 3:] += 400
    changed["reference_features"] += 80
    changed["paired_features"][:, 3:] -= 50
    other = m(changed)
    for name in ("predictions", "observation_context", "dynamics_context"):
        torch.testing.assert_close(out[name], other[name], rtol=0, atol=0)
    assert not torch.equal(out["prediction_loss"], other["prediction_loss"])


def test_constant_removes_only_dynamics_episode_information():
    m, b = model("constant_dynamics"), batch()
    observation, dynamics = m.infer_context(b["features"][:, :3], b["actions"][:, :2])
    observation2, dynamics2 = m.infer_context(b["features"][:, :3] * 7, b["actions"][:, :2] + 10)
    torch.testing.assert_close(dynamics, dynamics2, rtol=0, atol=0)
    assert not torch.allclose(observation, observation2)
    torch.testing.assert_close(dynamics, dynamics[:1].expand_as(dynamics), rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_query_action_causality_and_frozen_targets(mode):
    m, b = model(mode), batch()
    original = m(b)["predictions"]
    changed = deepcopy(b)
    changed["actions"][:, -1] += 3
    alternate = m(changed)["predictions"]
    torch.testing.assert_close(original[:, :-1], alternate[:, :-1], rtol=0, atol=0)
    assert not torch.allclose(original[:, -1], alternate[:, -1])
    m.train()
    m(b)["loss"].backward()
    assert all(p.grad is None for p in m.reference_encoder.parameters())
    assert all(p.grad is None for p in m.base.encoder.parameters())
    assert any(p.grad is not None for p in m.base.predictor.parameters())


def test_recurrent_predictor_is_causal_and_action_conditioned():
    predictor = GRUPredictor(8, hidden=12, layers=2).eval()
    features, actions = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    before = predictor(features, actions)
    changed = actions.clone()
    changed[:, -1] += 8
    after = predictor(features, changed)
    torch.testing.assert_close(before[:, :-1], after[:, :-1], rtol=0, atol=0)
    assert not torch.allclose(before[:, -1], after[:, -1])
