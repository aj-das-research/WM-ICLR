"""'How ShiftWM works' on a controlled toy world: every internal tensor of a trained model, step by step.

The toy world (scripts/v2/toy_world.py) uses raw 4x4 patch pixels as features, so every tensor decodes to RGB
exactly. Models are the REAL V2WorldModel (tiny config) trained by scripts/v2/toy_train.py; nothing is drawn by hand.
Outputs: figures/toy_walkthrough.pdf (mechanism, one test window) and figures/toy_horizons.pdf (forecasts + metrics).
Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_toy.py
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402
sys.path.insert(0, str(mf.ROOT / "scripts/v2"))
from toy_world import patchify, unpatchify, P  # noqa: E402
from shiftwm.v2.models import V2WorldModel  # noqa: E402

TOY = mf.ROOT / "data/toy"
RES = mf.ROOT / "results/toy"
H, K, G, W = 3, 10, 16, 7
R = W // 2
OFF = np.stack(np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="ij"), -1).reshape(-1, 2)  # (dy,dx)
STATS = json.loads((TOY / "stats.json").read_text()) if (TOY / "stats.json").exists() else None
ARMS = ("shiftwm", "direct", "ar")
# Toy frames are pixel art: nearest-neighbour upsampling shows the true 4x4 patch structure without inventing
# detail (the paper-wide lanczos default would blur patch edges that ARE the data here).
IMKW = dict(interpolation="nearest", resample=False)


def load(arm):
    ck = RES / arm / "best.pt"
    if not ck.exists():
        return None
    st = torch.load(ck, map_location="cpu")
    m = V2WorldModel(st["config"]).eval(); m.load_state_dict(st["model"])
    return m


def to_feat(frames_u8):
    f = patchify(torch.from_numpy(np.ascontiguousarray(frames_u8)).float() / 255)
    return (f - torch.tensor(STATS["feature_mean"])) / torch.tensor(STATS["feature_std"])


def to_rgb(z):
    z = np.asarray(z, np.float32) * np.array(STATS["feature_std"], np.float32) + np.array(STATS["feature_mean"], np.float32)
    return unpatchify(z).clip(0, 1)


def windows(split="test"):
    z = np.load(TOY / f"{split}.npz")
    fr, ac, lab, pos = z["frames"], z["actions"], z["labels"], z["pos"]
    E, T = fr.shape[:2]
    return fr, ac, lab, pos, [(e, s) for e in range(E) for s in range(T - H - K + 1)]


def batch(fr, ac, idx):
    e = np.array([i[0] for i in idx]); s = np.array([i[1] for i in idx])
    t = s[:, None] + np.arange(H + K)[None]
    f = to_feat(fr[e[:, None], t]); a = torch.from_numpy(ac[e[:, None], t[:, :-1]])
    return f[:, :H], a[:, :H - 1], a[:, H - 1:H - 1 + K], f[:, H:]


def implied_disp(weights):
    """weights [...,S*W*W] -> displacement (dy,dx) in px from frame t to t+k implied by the transport, per k.

    A candidate at window offset o in source frame t-s' says 'this content moved by -P*o px in (k+s') steps';
    rescaled to k steps it implies -P*o*k/(k+s'). The expected value over pi is the transport's motion estimate."""
    S = weights.shape[-1] // (W * W)
    w = weights.reshape(*weights.shape[:-1], S, W * W)
    k = np.arange(1, K + 1, dtype=np.float32).reshape(1, K, *([1] * (w.ndim - 4)))
    lag = np.arange(S - 1, -1, -1, dtype=np.float32)                                # source index -> s'
    scale = k[..., None] / (k[..., None] + lag)                                     # [1,K,...,S]
    off = -P * OFF.astype(np.float32)                                               # [WW,2]
    return np.einsum("...sj,jc->...c", w * scale[..., None], off)


def patch_labels(lab):
    """[...,64,64] labels -> majority label per patch [...,256] and its purity."""
    x = lab.reshape(*lab.shape[:-2], G, P, G, P)
    x = np.moveaxis(x, -3, -2).reshape(*lab.shape[:-2], G * G, P * P)
    counts = np.stack([(x == c).sum(-1) for c in range(5)], -1)
    return counts.argmax(-1), counts.max(-1) / (P * P)
