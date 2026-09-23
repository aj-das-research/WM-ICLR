"""Shared helpers for the ShiftWM-v2 analysis scripts (scripts/v2/{eval_transfer,train_decoder,
decode_eval,probes,flow_agreement}.py and figures/src/make_analysis_figures.py).

Conventions
- Features are standardised with the *training* cache's stats.json (train split only).
- A FeatureSplit window with local start s observes frames s..s+H-1; the last observed frame is
  t0 = s+H-1 and forecast step k (1-based) targets frame t0+k.
- RGB frames are resized to the encoder input (224x224, full-frame antialiased bilinear resize, no
  crop) exactly as in extract.encode(), so the 16x16 patch grid tiles the image in 14x14 pixel cells.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import V2WorldModel
from .train import FeatureSplit

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results/v2"
ANALYSIS = RES / "analysis"
TEST_ARMS = ("persistence", "linear", "ar_tf", "ar", "direct", "shiftwm")


# ----------------------------------------------------------------------------- caches & models
def cache_root(dataset, encoder="dinov2s"):
    return ROOT / "data/v2/features" / dataset / encoder


def read_cache(root):
    root = Path(root)
    return json.loads((root / "manifest.json").read_text()), json.loads((root / "stats.json").read_text())


def run_dirs(dataset, arm, encoder="dinov2s", seeds=None):
    """Existing run directories results/v2/<ds>/<enc>/<arm>/s<seed>.

    Learned arms: only FINISHED runs (best.pt and summary.json, which train.run writes after the final
    test evaluation), so in-progress checkpoints never leak into analysis. Set SHIFTWM_ALLOW_PARTIAL=1
    to also accept runs that only have best.pt (smoke tests)."""
    base = RES / dataset / encoder / arm
    dirs = sorted(base.glob("s*")) if seeds is None else [base / f"s{s}" for s in seeds]
    if arm in ("persistence", "linear"):
        return [d for d in dirs if d.is_dir()][:1] or ([base / "s0"] if seeds is None else [])
    partial = os.environ.get("SHIFTWM_ALLOW_PARTIAL") == "1"
    return [d for d in dirs if (d / "best.pt").exists() and (partial or (d / "summary.json").exists())]


def load_model(arm, run_dir, manifest, history=3, horizon=10, device="cpu"):
    """Trained arm from run_dir/best.pt, or a parameter-free reference arm built from the manifest."""
    if arm in ("persistence", "linear"):
        cfg = {"arm": arm, "grid": manifest["grid"], "channels": manifest["channels"],
               "action_dim": manifest["action_dim"], "history": history, "horizon": horizon}
        return V2WorldModel(cfg).to(device).eval()
    st = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    model = V2WorldModel(st["config"]).to(device).eval()
    model.load_state_dict(st["model"])
    return model


def run_config(run_dir, default_history=3, default_horizon=10):
    p = Path(run_dir) / "config.json"
    cfg = json.loads(p.read_text()) if p.exists() else {}
    return cfg.get("history", default_history), cfg.get("horizon", default_horizon), cfg


def local_starts(data):
    """Episode-local start frame of every FeatureSplit window (first window of each episode starts at 0)."""
    first = torch.full((len(data.episodes),), 1 << 62, dtype=torch.long, device=data.starts.device)
    first.scatter_reduce_(0, data.episode_of, data.starts, reduce="amin")
    return data.starts - first[data.episode_of]


@torch.no_grad()
def predict(model, hist, past, fut, details=False):
    on_cuda = hist.is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_cuda):
        if details:
            pred, det = model(hist, past, fut, return_details=True)
            return pred.float(), det
        return model(hist, past, fut).float()


# ----------------------------------------------------------------------------- frames
def _resize(images, size=224):
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


class FrameSource:
    """RGB frames aligned with a Stage-2 feature cache (DROID processed npz or Stage-1 frames)."""

    def __init__(self, cache):
        manifest, _ = read_cache(cache)
        self.kind = manifest["kind"]
        self.source = Path(manifest["source"])
        if not self.source.is_absolute():
            self.source = ROOT / self.source
        self.camera = manifest.get("camera")
        man = json.loads((self.source / "manifest.json").read_text())
        if self.kind == "droid":
            self.files = {r["episode_id"]: r["cameras"][self.camera]["file"] for r in man["episodes"]}
        else:
            self.files = {r["id"]: r["file"] for r in man["episodes"]}

    def exists(self, episode_id):
        return episode_id in self.files and (self.source / self.files[episode_id]).exists()

    def load(self, episode_id, frames=None, size=224):
        """uint8 [T or len(frames), size, size, 3] resized like the encoder input."""
        with np.load(self.source / self.files[episode_id]) as z:
            im = z["images"] if frames is None else z["images"][np.asarray(frames)]
        return _resize(im, size)

    def load_native(self, episode_id, frames):
        with np.load(self.source / self.files[episode_id]) as z:
            return z["images"][np.asarray(frames)]


# ----------------------------------------------------------------------------- decoder
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                  nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class FeatureDecoder(nn.Module):
    """Standardised patch features [B,N,C] (N=G*G) -> RGB [B,3,224,224] in [0,1].

    16x16 -> 28x28 (bilinear) -> x2 -> x2 -> x2 = 224, with residual conv blocks at every scale."""

    def __init__(self, channels=384, grid=16, size=224, widths=(512, 384, 256, 128), blocks=2):
        super().__init__()
        self.grid, self.size = grid, size
        self.base = size // 8
        self.stem = nn.Conv2d(channels, widths[0], 3, padding=1)
        self.pre = nn.Sequential(*[ResBlock(widths[0]) for _ in range(blocks)])
        ups = []
        for cin, cout in zip(widths[:-1], widths[1:]):
            ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1),
                                     *[ResBlock(cout) for _ in range(blocks)]))
        self.ups = nn.ModuleList(ups)
        self.head = nn.Sequential(nn.GroupNorm(32, widths[-1]), nn.SiLU(), nn.Conv2d(widths[-1], 3, 3, padding=1))

    def forward(self, z):
        b, n, c = z.shape
        x = z.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        x = self.stem(x)
        x = F.interpolate(x, size=(self.base, self.base), mode="bilinear", align_corners=False)
        x = self.pre(x)
        for up in self.ups:
            x = up(x)
        return torch.sigmoid(self.head(x).float())


def load_decoder(path, device="cpu"):
    st = torch.load(path, map_location=device, weights_only=False)
    dec = FeatureDecoder(**st["arch"]).to(device).eval()
    dec.load_state_dict(st["model"])
    return dec


def decoder_path(dataset, encoder="dinov2s"):
    return ANALYSIS / "decoder" / dataset / encoder / "best.pt"


@torch.no_grad()
def decode(decoder, z, batch=64):
    """z [B,N,C] standardised -> float RGB [B,3,224,224] in [0,1]."""
    out = []
    for i in range(0, len(z), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            out.append(decoder(z[i:i + batch].float()).float())
    return torch.cat(out)


# ----------------------------------------------------------------------------- image metrics
def psnr(x, y):
    """x,y [B,3,H,W] in [0,1] -> [B]."""
    mse = ((x - y) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def _gauss(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(x, y):
    """Standard SSIM (Wang et al. 2004; 11x11 Gaussian, sigma 1.5, K1=.01, K2=.03, data range 1), mean over RGB -> [B]."""
    w = _gauss(device=x.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = F.conv2d(x, w, groups=3), F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x * x, w, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=3) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean((1, 2, 3))


# ----------------------------------------------------------------------------- transport field
def window_offsets(config):
    """(oy, ox) [S*w*w] patch offsets of the transport candidates, same order as models.transport()."""
    g, win, s = config.grid, config.window, config.sources
    if not win or win >= 2 * g:
        raise ValueError("global transport: offsets depend on the query position")
    r = win // 2
    oy, ox = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    return np.tile(oy.ravel(), s).astype(np.float32), np.tile(ox.ravel(), s).astype(np.float32)


def expected_offsets(weights, config):
    """weights [...,N,S*w*w] -> expected source offset (dx, dy) [...,N] in patches, and the weight mass on
    the last observed frame [...,N]. Same computation as figures/src/make_figures.transport_arrows().
    Content motion (forward flow at the query) is -offset."""
    oy, ox = window_offsets(config)
    oy = torch.as_tensor(oy, device=weights.device); ox = torch.as_tensor(ox, device=weights.device)
    w = weights.float()
    ww = config.window * config.window
    return w @ ox, w @ oy, w[..., -ww:].sum(-1)


# ----------------------------------------------------------------------------- probes
def probe_features(z, grid=16):
    """z [B,N,C] -> [B, C + 16*C]: global mean-pool ++ 4x4 average-pooled grid."""
    b, n, c = z.shape
    x = z.float().transpose(1, 2).reshape(b, c, grid, grid)
    p4 = F.adaptive_avg_pool2d(x, 4).reshape(b, -1)
    return torch.cat((x.mean((2, 3)), p4), 1)


def pooled_flow(flow, grid=16):
    """Pixel flow [B,2,H,W] -> patch-unit flow [B,2,grid,grid] (average over each patch cell)."""
    h = flow.shape[-1]
    return F.adaptive_avg_pool2d(flow, grid) * (grid / h)


def bootstrap_ci(values, n=2000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def split_for(root, split, history, horizon, device, stats, stride, max_episodes=None, tasks=None):
    return FeatureSplit(root, split, history, horizon, device, stats, stride, tasks, max_episodes)


def fmt_or_pend(v, digits=2):
    return r"\pend" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def isqrt(n):
    return int(math.isqrt(n))
