"""CPU tests of the V-JEPA 2-AC plug-in (tiny random predictor of their class; no ViT-g, no checkpoint).

Run: source .venv/bin/activate; PYTHONPATH=src python -m pytest -q tests/test_vjepa_plugin.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import eval_vjepa2ac as ev  # noqa: E402

ev.add_repo(ROOT / "external/vjepa2")
from src.models.ac_predictor import vit_ac_predictor  # noqa: E402

import vjepa_cache as vc  # noqa: E402
import vjepa_finetune as vf  # noqa: E402
from shiftwm.v2.vjepa_plugin import PluginWM, TransportHead, make_head, window_mask  # noqa: E402

torch.set_num_threads(2)
G, D = 8, 24          # 8x8 grid (128 px / 16), token dim 24
N = G * G


def tiny_predictor(seed=0):
    torch.manual_seed(seed)
    return vit_ac_predictor(img_size=(G * 16, G * 16), patch_size=16, num_frames=64, tubelet_size=2, embed_dim=D,
                            predictor_embed_dim=32, depth=2, num_heads=2).eval()


def ln(x):
    return F.layer_norm(x, (x.shape[-1],))


def data(B=3, T=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return ln(torch.randn(B, T, N, D, generator=g)), torch.randn(B, T - 1, 7, generator=g), torch.randn(B, T, 7, generator=g)


def active_head(pred, seed=3, **kw):
    """Head with a non-trivial gate/query (so that causality tests are not vacuous)."""
    head = make_head(pred, **kw)
    torch.manual_seed(seed)
    with torch.no_grad():
        head.gate.weight.normal_(0, 0.5); head.gate.bias.fill_(0.0)
        head.src_emb.normal_(0, 0.5)
    return head


def their_forward_predictions(predictor, h, actions, states, auto_steps, tpf):
    """Verbatim logic of app/vjepa_droid/train.py::forward_predictions (extrinsics unused), flattened tokens."""
    def _step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s)
        return F.layer_norm(_z, (_z.size(-1),))
    z = h.flatten(1, 2)
    z_tf = _step_predictor(z[:, :-tpf], actions, states[:, :-1])
    _z = torch.cat([z[:, :tpf], z_tf[:, :tpf]], dim=1)
    for n in range(1, auto_steps):
        _a, _s = actions[:, : n + 1], states[:, : n + 1]
        _z_nxt = _step_predictor(_z, _a, _s)[:, -tpf:]
        _z = torch.cat([_z, _z_nxt], dim=1)
    z_ar = _z[:, tpf:]
    def loss_fn(zz, hh):
        _h = hh[:, tpf: zz.size(1) + tpf]
        return torch.mean(torch.abs(zz - _h) ** 1.0) / 1.0
    return z_tf, z_ar, loss_fn(z_tf, z) + loss_fn(z_ar, z)


# --------------------------------------------------------------------------------------------- B == released model
@pytest.mark.parametrize("auto_steps", [1, 2, 3])
def test_train_forward_matches_their_objective(auto_steps):
    pred = tiny_predictor(); wm = PluginWM(pred)
    h, a, s = data()
    z_tf, z_ar, gate = wm.train_forward(h, a, s, auto_steps)
    r_tf, r_ar, r_loss = their_forward_predictions(pred, h, a, s, auto_steps, N)
    assert gate is None and z_tf.shape == (3, 7, N, D) and z_ar.shape == (3, auto_steps, N, D)
    assert torch.allclose(z_tf.flatten(1, 2), r_tf, atol=1e-5) and torch.allclose(z_ar.flatten(1, 2), r_ar, atol=1e-5)
    assert torch.allclose(wm.loss(z_tf, z_ar, h)[0], r_loss, atol=1e-6)


@pytest.mark.parametrize("max_context", [0, 5, 8])
def test_rollout_matches_evaluator(max_context):
    pred = tiny_predictor(); wm = PluginWM(pred)
    h, _, _ = data(T=13)
    g = torch.Generator().manual_seed(5)
    s, a = torch.randn(3, 12, 7, generator=g), torch.randn(3, 12, 7, generator=g)
    ours = wm.rollout(h[:, :3], s, a, 10, max_context)
    ref = ev.rollout(pred, h[:, :3], s, a, 10, max_context)
    assert torch.equal(ours, ref)


# --------------------------------------------------------------------------------------------- C ~= B at init
def test_gate_at_init_is_near_identity():
    pred = tiny_predictor()
    B_, C_ = PluginWM(pred), PluginWM(tiny_predictor(), make_head(pred))
    head = C_.head
    assert torch.allclose(torch.sigmoid(head.gate.bias), torch.tensor(1 / (1 + np.e ** 4)).float())
    h, a, s = data()
    zb, ab, _ = B_.train_forward(h, a, s, 2)
    zc, ac, g = C_.train_forward(h, a, s, 2)
    assert abs(float(g) - 0.01799) < 1e-4
    rel = (zc - zb).norm() / zb.norm()
    assert 0 < rel < 0.03, rel                     # |Zhat - P| = g |T - P| with g = 0.018
    # gate closed exactly -> bitwise B (train objective and evaluation rollout)
    C0 = PluginWM(tiny_predictor(), make_head(pred, gate_bias=-1e4))
    z0, a0, _ = C0.train_forward(h, a, s, 2)
    assert torch.equal(z0, zb) and torch.equal(a0, ab)
    hh, _, _ = data(T=13)
    ss, aa = torch.randn(3, 12, 7), torch.randn(3, 12, 7)
    assert torch.equal(C0.rollout(hh[:, :3], ss, aa, 10), B_.rollout(hh[:, :3], ss, aa, 10))


def test_head_param_count_real_size():
    head = TransportHead(1024, 1408, 16)
    n = sum(p.numel() for p in head.parameters())
    assert n == 1024 * 64 + 1408 * 64 + 3 * 64 + 1025 + 1 == 156866, n


# --------------------------------------------------------------------------------------------- causality
def test_teacher_forced_causality_and_measured_sources():
    pred = tiny_predictor(); wm = PluginWM(pred, active_head(pred))
    h, a, s = data()
    z_tf, z_ar, _ = wm.train_forward(h, a, s, 3)
    for t in range(7):                               # block t (predicts frame t+1) must not see frames > t
        h2 = h.clone(); h2[:, t + 1:] = ln(torch.randn_like(h2[:, t + 1:]))
        z2, _, _ = wm.train_forward(h2, a, s, 3)
        assert torch.allclose(z2[:, :t + 1], z_tf[:, :t + 1], atol=1e-5)
        if t + 1 < 7:
            assert not torch.allclose(z2[:, t + 1], z_tf[:, t + 1], atol=1e-4)
    # their rollout from frame 0 only reads measured frame 0 (context = predictions afterwards)
    h3 = h.clone(); h3[:, 1:] = ln(torch.randn_like(h3[:, 1:]))
    _, z_ar3, _ = wm.train_forward(h3, a, s, 3)
    assert torch.allclose(z_ar3, z_ar, atol=1e-5)
    assert wm.tf_sources(4).tolist() == [[-1, -1, 0], [-1, 0, 1], [0, 1, 2], [1, 2, 3]]


def test_rollout_transport_reads_only_observed_history():
    pred = tiny_predictor(); wm = PluginWM(pred, active_head(pred))
    h, _, _ = data(T=13)
    s, a = torch.randn(3, 12, 7), torch.randn(3, 12, 7)
    seen = []
    orig = wm.head.forward
    def spy(hidden, frames, src_idx):
        seen.append((frames, src_idx.clone()))
        return orig(hidden, frames, src_idx)
    wm.head.forward = spy
    hist = h[:, :3]
    p1, gate = wm.rollout(hist, s, a, 10, 8, return_gate=True)
    assert len(seen) == 10 and all(f is hist and i.tolist() == [[0, 1, 2]] for f, i in seen)
    assert gate.shape == (3, 10, N) and (gate > 0).all()
    # predictions are consumed by the predictor: step k>1 depends on step-1 output (not on true future frames)
    p2 = wm.rollout(hist, s, a, 10, 8)
    assert torch.allclose(p1, p2)


# --------------------------------------------------------------------------------------------- transport
def test_transport_is_local_and_identity_limit():
    head = TransportHead(16, D, G, sources=3, window=3, key_dim=8)
    torch.manual_seed(0)
    hidden = torch.randn(2, 1, N, 16); frames = ln(torch.randn(2, 3, N, D))
    idx = torch.tensor([[0, 1, 2]])
    moved, gate, w = head(hidden, frames, idx)
    wm_ = window_mask(G, 3).repeat(1, 3)
    assert torch.all(w[:, :, ~wm_] == 0) and torch.allclose(w.sum(-1), torch.ones(2, 1, N))
    # perturbing a token outside every query's 3x3 window leaves that query unchanged
    f2 = frames.clone(); f2[:, :, 63] += 5.0                 # patch (7,7)
    m2, _, _ = head(hidden, f2, idx)
    assert torch.allclose(m2[:, 0, 0], moved[:, 0, 0]) and not torch.allclose(m2[:, 0, 63], moved[:, 0, 63])
    # huge identity bias -> copy of the same patch of the most recent observed frame
    with torch.no_grad():
        head.id_bias.fill_(200.0)
    m3, _, _ = head(hidden, frames, idx)
    assert torch.allclose(m3[:, 0], frames[:, 2], atol=1e-5)
    # unavailable slots (-1) get zero weight
    _, _, w4 = head(hidden, frames, torch.tensor([[-1, -1, 2]]))
    assert torch.all(w4[..., :2 * N] == 0)


# --------------------------------------------------------------------------------------------- cache / action reuse
def test_cache_state_action_conversion_reuses_evaluator(tmp_path):
    rng = np.random.default_rng(0)
    T = 6
    walk = np.cumsum(rng.normal(0, 0.01, (5 * (T - 1), 7)), 0) + np.array([0.4, 0, 0.3, 3.1, 0, 0, 0.2])
    np.savez(tmp_path / "ep.npz", images=np.zeros((T, 180, 320, 3), np.uint8), actions=walk.reshape(T - 1, 35),
             frame_indices=10 + 5 * np.arange(T))
    imgs, s, a = vc.load_episode(tmp_path, {"id": "x", "file": "ep.npz", "T": T})
    ref_s = ev.frame_states(walk.reshape(T - 1, 35))
    assert np.allclose(s, ref_s, atol=1e-6) and np.allclose(a[:-1], ev.poses_to_diffs(ref_s), atol=1e-6)
    assert np.all(a[-1] == 0) and s.dtype == np.float32 and a.shape == (T, 7)


# --------------------------------------------------------------------------------------------- full script path (CPU)
def fake_cache(root, eps=(("train", 14), ("train", 17), ("train", 9), ("val", 15), ("val", 13), ("test", 16), ("test", 14))):
    rows, off = [], 0
    for i, (split, T) in enumerate(eps):
        rows.append({"id": f"e{i}", "session": f"s{i % 3}", "split": split, "offset": off, "T": T, "file": ""})
        off += T
    rng = np.random.default_rng(0)
    z = torch.from_numpy(rng.normal(size=(off, N, D))).float()
    np.save(root / "tokens.npy", ln(z).numpy().astype(np.float16))
    np.save(root / "states.npy", rng.normal(size=(off, 7)).astype(np.float32))
    np.save(root / "actions.npy", rng.normal(size=(off, 7)).astype(np.float32))
    (root / "index.json").write_text(json.dumps({"episodes": rows, "complete": True}))


@pytest.mark.parametrize("arm", ["finetune", "finetune_shiftwm"])
def test_train_eval_resume_cpu(tmp_path, arm):
    fake_cache(tmp_path)
    a = vf.build_parser().parse_args(["--arm", arm, "--steps", "6", "--batch-size", "2", "--eval-every", "3",
                                      "--log-every", "2", "--eval-batch", "4", "--val-stride", "2"])
    a.device = "cpu"
    pred = tiny_predictor()
    head = make_head(pred) if arm == "finetune_shiftwm" else None
    wm = PluginWM(pred, head)
    groups = vf.param_groups(wm, a.head_lr_mult, a.weight_decay)
    assert sum(len(g["params"]) for g in groups) == len(list(wm.parameters()))
    if head is not None:
        hg = [g for g in groups if g["lr_scale"] == 10.0]
        assert any(g["weight_decay"] == 0 and any(p is head.id_bias for p in g["params"]) for g in hg)
    tr, va = vf.CachedSplit(tmp_path, "train", "cpu"), vf.CachedSplit(tmp_path, "val", "cpu")
    run, out = tmp_path / "run", tmp_path / "out"; run.mkdir(); out.mkdir()
    p0 = [p.detach().clone() for p in wm.predictor.parameters()]
    state = vf.train(a, wm, tr, va, run, out)
    assert state["step"] == 6 and [c["step"] for c in state["curve"]] == [0, 3, 6]
    assert state["best"]["step"] in (3, 6) and (run / "best.pt").exists()
    assert any(not torch.equal(p, q) for p, q in zip(p0, wm.predictor.parameters()))
    # resume: extend the budget to 9 steps; restarts from step 6 with the same optimiser state
    a.steps = 9
    wm2 = PluginWM(tiny_predictor(), make_head(tiny_predictor()) if head is not None else None)
    st2 = vf.train(a, wm2, tr, va, run, out)
    assert st2["step"] == 9 and [c["step"] for c in st2["curve"]] == [0, 3, 6, 9]
    # deadline -> resumable exit code
    a.steps, a.deadline = 12, time.time()
    with pytest.raises(SystemExit) as e:
        vf.train(a, wm2, tr, va, run, out)
    assert e.value.code == vf.EXIT_RESUME
    # evaluation + summary (+ shuffled) on test
    te = vf.CachedSplit(tmp_path, "test", "cpu")
    res = vf.evaluate(wm2, te, a, 2, True)
    assert len(res["win"]) == len(ev.window_starts(16, 3, 10, 2)) + len(ev.window_starts(14, 3, 10, 2)) == 3
    summ = vf.summarize(res, arm, "test", {})
    vf.save_eval(res, out / "test.npz", summ)
    z = np.load(out / "test.npz")
    assert z["model_mse"].shape == (2, 10) and z["model_mse_moving"].shape == (2, 10)
    assert np.all(z["model_mse_moving"] >= 0) and "action_rank_acc" in summ
    # persistence reference == eval_vjepa2ac definition
    w0 = res["win"][0]
    zt = te.Z[w0[1]:w0[1] + 13].float()
    ref = ev.metrics(zt[None, 2:3].expand(1, 10, N, D), zt[None, 3:])["mse"][0].numpy()
    assert np.allclose(res["per_win"]["persistence_mse"][0], ref, atol=1e-6)
    if head is not None:
        assert "gate_mean" in summ["per_horizon"]
