"""CPU tests for the ShiftWM plug-in to the official DINO-WM predictor (src/shiftwm/v2/dinowm_plugin.py).

Run in the DINO-WM env:  environments/dinowm/.venv/bin/python -m pytest -q tests/test_dinowm_plugin.py
"""
import io
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DWM = ROOT / "external" / "dino_wm"
if not DWM.exists():
    pytest.skip("external/dino_wm not checked out", allow_module_level=True)
pytest.importorskip("einops")
sys.path[:0] = [str(DWM), str(ROOT / "src"), str(ROOT / "scripts" / "external_dinowm_plugin")]

import shims  # noqa: E402

shims.install_cpu_mask_shim(force=True)

from models.vit import ViTPredictor  # noqa: E402
from shiftwm.v2.dinowm_plugin import ShiftViTPredictor, local_transport  # noqa: E402

G, T, VIS, DIM = 5, 3, 16, 20  # tiny: 5x5 grid, 3 frames, 16 visual + 4 proprio/action channels
KW = dict(num_patches=G * G, num_frames=T, dim=DIM, depth=2, heads=2, mlp_dim=32, dim_head=8, pool="mean",
          dropout=0.0, emb_dropout=0.0)


def _pair(gate_bias=-4.0):
    torch.manual_seed(0)
    base = ViTPredictor(**KW)
    r_base = torch.rand(3)
    torch.manual_seed(0)
    plug = ShiftViTPredictor(visual_dim=VIS, key_dim=8, window=3, gate_bias=gate_bias, **KW)
    r_plug = torch.rand(3)
    return base.eval(), plug.eval(), r_base, r_plug


def test_transport_matches_shiftwm_v2():
    from shiftwm.v2.models import V2Config, V2WorldModel
    torch.manual_seed(1)
    cfg = V2Config(arm="shiftwm", channels=VIS, grid=G, dim=DIM, heads=2, enc_depth=1, dec_depth=1, window=3,
                   sources=1, key_dim=8, history=1, horizon=1, action_dim=2)
    m = V2WorldModel(cfg)
    hidden, memory, hist = torch.randn(2, 1, G * G, DIM), torch.randn(2, 1, G * G, DIM), torch.randn(2, 1, G * G, VIS)
    ref, w_ref = m.transport(hidden, memory, hist)
    out, w = local_transport(m.tq(hidden[:, 0]), m.tk(memory[:, 0]), hist[:, 0], G, 3, m.id_bias)
    assert torch.allclose(out, ref[:, 0], atol=1e-5) and torch.allclose(w, w_ref[:, 0], atol=1e-6)


def test_same_base_init_and_rng_stream():
    base, plug, r_base, r_plug = _pair()
    sb = base.state_dict()
    for k, v in sb.items():
        assert torch.equal(v, plug.state_dict()[k]), k
    assert torch.equal(r_base, r_plug)  # global RNG untouched by the head -> identical decoder init / dropout


def test_starts_at_dinowm():
    base, plug, _, _ = _pair(gate_bias=-1e4)
    x = torch.randn(2, T * G * G, DIM)
    assert torch.allclose(base(x), plug(x), atol=1e-6)
    base, plug, _, _ = _pair(gate_bias=-4.0)
    out, det = plug(x, return_details=True)
    assert torch.allclose(det["gate"], torch.sigmoid(torch.tensor(-4.0)).expand_as(det["gate"]))
    assert (out - base(x)).abs().max() < 0.2


def test_non_visual_channels_untouched_and_gate_mix():
    _, plug, _, _ = _pair(gate_bias=1e4)  # g = 1 -> visual = pure transport
    x = torch.randn(2, T * G * G, DIM)
    pred = ViTPredictor.forward(plug, x)
    out = plug(x)
    assert torch.equal(out[..., VIS:], pred[..., VIS:])
    b = x.shape[0]
    q = plug.shift_head.tq(pred.reshape(b * T, G * G, DIM))
    k = plug.shift_head.tk(x.reshape(b * T, G * G, DIM))
    moved, _ = local_transport(q, k, x.reshape(b * T, G * G, DIM)[..., :VIS], G, 3, plug.shift_head.id_bias)
    assert torch.allclose(out[..., :VIS], moved.reshape(b, T * G * G, VIS), atol=1e-5)


def test_frame_causal():
    _, plug, _, _ = _pair(gate_bias=0.0)
    with torch.no_grad():
        plug.shift_head.gate.weight.normal_()
    x = torch.randn(1, T * G * G, DIM)
    y = x.clone()
    y[:, 2 * G * G:] += torch.randn_like(y[:, 2 * G * G:])  # perturb frame 2 only
    a, b = plug(x), plug(y)
    assert torch.allclose(a[:, :2 * G * G], b[:, :2 * G * G], atol=1e-6)
    assert not torch.allclose(a[:, 2 * G * G:], b[:, 2 * G * G:])


def test_gradients_reach_head_and_pickle_roundtrip():
    _, plug, _, _ = _pair()
    plug.train()
    x = torch.randn(2, T * G * G, DIM)
    plug(x).pow(2).mean().backward()
    for n in ("tq.weight", "tk.weight", "gate.weight", "gate.bias", "id_bias"):
        p = dict(plug.shift_head.named_parameters())[n]
        assert p.grad is not None and p.grad.abs().sum() > 0, n
    buf = io.BytesIO()
    torch.save({"predictor": plug}, buf)  # upstream pickles whole modules in checkpoints
    buf.seek(0)
    re = torch.load(buf, weights_only=False)["predictor"].eval()
    plug.eval()
    assert torch.allclose(re(x), plug(x))


def test_inside_official_vworldmodel_forward_and_rollout():
    from models.proprio import ProprioceptiveEmbedding
    from models.visual_world_model import VWorldModel

    class Enc(torch.nn.Module):  # stand-in for DINOv2 (patch 14 -> 5x5 grid at 70px), same interface
        name, emb_dim, patch_size, latent_ndim = "dino_stub", VIS, 14, 2

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, VIS, 14, 14)

        def forward(self, x):
            return self.conv(x).flatten(2).transpose(1, 2)

    torch.manual_seed(0)
    pe = ProprioceptiveEmbedding(num_frames=1, in_chans=4, emb_dim=2)
    ae = ProprioceptiveEmbedding(num_frames=1, in_chans=10, emb_dim=2)
    pred = ShiftViTPredictor(visual_dim=VIS, key_dim=8, window=3, **KW)
    wm = VWorldModel(image_size=80, num_hist=T, num_pred=1, encoder=Enc(), proprio_encoder=pe, action_encoder=ae,
                     decoder=None, predictor=pred, proprio_dim=2, action_dim=2, concat_dim=1, num_action_repeat=1,
                     num_proprio_repeat=1, train_encoder=False, train_predictor=True, train_decoder=False)
    obs = {"visual": torch.randn(2, T + 1, 3, 80, 80), "proprio": torch.randn(2, T + 1, 4)}
    act = torch.randn(2, T + 1, 10)
    z_pred, _, _, loss, comps = wm(obs, act)
    assert z_pred.shape == (2, T, G * G, DIM) and torch.isfinite(loss)
    loss.backward()
    assert pred.shift_head.tq.weight.grad is not None
    with torch.no_grad():
        z_obses, z = wm.rollout({k: v[:, :T] for k, v in obs.items()}, torch.randn(2, T + 4, 10))
    assert z_obses["visual"].shape == (2, T + 5, G * G, VIS)


def test_fast_slices_identical():
    """shims.install_fast_slices returns exactly the upstream PushT slices (decode only the used frames)."""
    data = ROOT / "data" / "dinowm" / "pusht_noise"
    if not data.exists():
        pytest.skip("PushT data not downloaded")
    import numpy as np
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import load_pusht_slice_train_val
    from datasets.traj_dset import TrajSlicerDataset
    np.random.seed(0)
    ds, _ = load_pusht_slice_train_val(default_transform(224), n_rollout=3, data_path=str(data), normalize_action=True,
                                       num_hist=3, num_pred=1, frameskip=5)
    orig = TrajSlicerDataset.__getitem__
    ref = [orig(ds["train"], i) for i in (0, 7, len(ds["train"]) - 1)]
    try:
        shims.install_fast_slices()
        new = [ds["train"][i] for i in (0, 7, len(ds["train"]) - 1)]
    finally:
        TrajSlicerDataset.__getitem__ = orig
        shims._ORIG_GETITEM = None
    for (o1, a1, s1), (o2, a2, s2) in zip(ref, new):
        assert torch.equal(a1, a2) and torch.equal(s1, s2)
        for k in o1:
            assert torch.equal(o1[k], o2[k]), k
