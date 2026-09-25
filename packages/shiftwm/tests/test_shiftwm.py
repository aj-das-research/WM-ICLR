"""Tests: equivalence with the research implementation, plug-in head at init, helpers."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

import shiftwm
from shiftwm import ShiftHead, ShiftWM, forecast, local_transport, skill, transport_field

# The research implementation lives in the surrounding repository (src/shiftwm/v2/models.py); it is
# loaded by file path so that it does not clash with the installed `shiftwm` package.
REF = Path(os.environ.get("SHIFTWM_REFERENCE",
                          Path(__file__).resolve().parents[3] / "src" / "shiftwm" / "v2" / "models.py"))
CKPT = Path(os.environ.get("SHIFTWM_CHECKPOINT",
                           Path(__file__).resolve().parents[3] / "results/v2s/droid/dinov2s/shiftwm/s0/best.pt"))


def _reference():
    if not REF.exists():
        pytest.skip(f"reference implementation not found at {REF}")
    spec = importlib.util.spec_from_file_location("shiftwm_reference_models", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inputs(c, b=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    n = c.grid * c.grid
    return (torch.randn(b, c.history, n, c.channels, generator=g),
            torch.randn(b, c.history - 1, c.action_dim, generator=g),
            torch.randn(b, c.horizon, c.action_dim, generator=g))


def _perturb(model, seed=2):
    """Move every parameter away from its (partly zero) initialisation so all paths are exercised."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def test_version_and_api():
    assert shiftwm.__version__ == "0.1.0"
    assert ShiftWM is shiftwm.WorldModel


@pytest.mark.parametrize("cfg", [
    dict(arm="shiftwm", channels=32, grid=8, action_dim=5, dim=64, heads=4, enc_depth=2, dec_depth=2, horizon=4),
    dict(arm="shiftwm", channels=32, grid=8, action_dim=5, dim=64, heads=4, enc_depth=1, dec_depth=1, horizon=3,
         window=0, sources=1, correction="tanh"),
    dict(arm="direct", channels=32, grid=8, action_dim=5, dim=64, heads=4, enc_depth=1, dec_depth=1, horizon=3),
    dict(arm="ar", channels=32, grid=8, action_dim=5, dim=64, heads=4, enc_depth=1, dec_depth=1, horizon=3),
])
def test_matches_reference_random_weights(cfg):
    ref_mod = _reference()
    torch.manual_seed(0)
    ref = _perturb(ref_mod.V2WorldModel(cfg)).eval()
    ours = ShiftWM.from_config(cfg).eval()
    ours.load_state_dict(ref.state_dict())
    x = _inputs(ours.config)
    with torch.no_grad():
        a, b = ref(*x), ours(*x)
    assert (a - b).abs().max().item() < 1e-5


def test_matches_reference_checkpoint():
    ref_mod = _reference()
    if not CKPT.exists():
        pytest.skip(f"checkpoint not found at {CKPT}")
    ours = ShiftWM.from_checkpoint(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref = ref_mod.V2WorldModel(ck["config"]).eval()
    ref.load_state_dict(ck["model"])
    x = _inputs(ours.config)
    with torch.no_grad():
        pa, da = ref(*x, return_details=True)
        pb, db = ours(*x, return_details=True)
    diff = max((pa - pb).abs().max().item(), *((da[k] - db[k]).abs().max().item() for k in da))
    print(f"\nmax |ours - reference| on {CKPT.name}: {diff:.2e}")
    assert diff < 1e-5


def test_forecast_details_and_field():
    torch.manual_seed(0)
    m = ShiftWM.from_config(channels=16, grid=6, action_dim=3, dim=32, heads=2, enc_depth=1, dec_depth=1, horizon=4)
    x = _inputs(m.config, b=3)
    pred, det = forecast(m, *x, return_details=True)
    assert pred.shape == (3, 4, 36, 16)
    assert det["weights"].shape == (3, 4, 36, 3 * 49) and det["gate"].shape == (3, 4, 36, 1)
    assert torch.allclose(det["weights"].sum(-1), torch.ones(3, 4, 36))
    single = forecast(m, *(t[0] for t in x))
    assert torch.allclose(single, pred[0], atol=1e-6)
    flow = transport_field(det)
    assert flow.shape == (3, 4, 36, 2) and flow.abs().max() <= 3


def test_transport_field_pure_shift():
    # weights that copy every patch from one patch to the right -> offset (+1, 0)
    g, w = 5, 3
    wts = torch.zeros(1, 1, g * g, w * w)
    wts[..., 1 * w + 2] = 1.0            # dy = 0, dx = +1
    f = transport_field({"weights": wts}, grid=g, window=w)
    assert torch.allclose(f, torch.tensor([1.0, 0.0]).expand_as(f))


def test_skill():
    hist = torch.randn(2, 3, 4, 5)
    target = torch.randn(2, 6, 4, 5)
    assert abs(skill(target, target, hist).item() - 1) < 1e-6
    assert abs(skill(hist[:, -1:].expand_as(target), target, hist).item()) < 1e-6
    assert skill(target, target, hist, per_horizon=True).shape == (6,)


def test_local_transport_matches_model_transport():
    torch.manual_seed(0)
    m = _perturb(ShiftWM.from_config(channels=16, grid=6, action_dim=3, dim=32, heads=2, enc_depth=1, dec_depth=1,
                                     horizon=2)).eval()
    hidden, memory, hist = torch.randn(2, 2, 36, 32), torch.randn(2, 3, 36, 32), torch.randn(2, 3, 36, 16)
    a, wa = m.transport(hidden, memory, hist)
    b, wb = local_transport(m.tq(hidden), m.tk(memory), hist, 6, m.config.window, m.id_bias)
    assert (a - b).abs().max() < 1e-5 and (wa - wb).abs().max() < 1e-6


def test_shifthead_init_reproduces_backbone():
    torch.manual_seed(0)
    obs = torch.randn(2, 3, 64, 24)
    pred = torch.randn(2, 4, 64, 24)
    head = ShiftHead(token_dim=24, grid=8, sources=2)
    z, det = head(pred, obs, return_details=True)
    g = torch.sigmoid(torch.tensor(-4.0))
    assert torch.allclose(det["gate"], g.expand_as(det["gate"]))                   # gate ~ 0.018 everywhere
    assert torch.allclose(z, (1 - g) * pred + g * det["moved"], atol=1e-6)
    assert (z - pred).abs().max() <= g * (det["moved"] - pred).abs().max() + 1e-6
    exact = ShiftHead(token_dim=24, grid=8, sources=2, gate_bias=float("-inf"))
    assert torch.equal(exact(pred, obs), pred)                                      # closed gate: identical


def test_shifthead_wrap_and_grad():
    torch.manual_seed(0)
    backbone = torch.nn.Linear(24, 24)

    def predictor(obs):                     # backbone returning (P, hidden)
        p = backbone(obs[:, -1:]).expand(-1, 3, -1, -1)
        return p, p
    head = ShiftHead(token_dim=24, grid=8, hidden_dim=24, sources=1, window=5)
    wrapped = head.wrap(predictor)
    obs = torch.randn(2, 2, 64, 24)
    out = wrapped(obs)
    assert out.shape == (2, 3, 64, 24)
    out.pow(2).mean().backward()
    assert head.gate.bias.grad is not None and head.gate.weight.grad.abs().sum() > 0
