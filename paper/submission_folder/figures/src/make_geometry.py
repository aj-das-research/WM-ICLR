"""Feature-space geometry of forecasting (real held-out DROID data, trained checkpoints).

(a) One moving patch at k=10: the S*w*w observed candidate features (sized/coloured by learned transport
    weight), the observed feature at the same location, the true future, and the ShiftWM / Direct / AR
    forecasts, in a 2-D PCA of these vectors.
Also records (JSON only, not drawn) each forecast's distance to the nearest observed candidate feature on moving
patches; all forecasts lie closer to observed features than the true future does (the future contains new content).
Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_geometry.py
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402
from shiftwm.v2.models import V2WorldModel  # noqa: E402
from shiftwm.v2.train import FeatureSplit  # noqa: E402

K = 10


def load(arm):
    for base in ("results/v2s", "results/v2"):
        ck = sorted((mf.ROOT / base / "droid/dinov2s" / arm).glob("s*/best.pt"))
        if ck:
            st = torch.load(ck[0], map_location="cpu")
            m = V2WorldModel(st["config"]).eval(); m.load_state_dict(st["model"]); return m
    return None


def candidates(hist, cfg):
    """[B,N,S*w*w,C] observed candidate features in each patch's window (zeros where padded) + validity."""
    b, h, n, c = hist.shape
    g, w, s = cfg.grid, cfg.window, cfg.sources
    r = w // 2
    t = hist[:, -s:].permute(0, 1, 3, 2).reshape(b * s, c, g, g)
    u = F.unfold(t, w, padding=r).reshape(b, s, c, w * w, n).permute(0, 4, 1, 3, 2).reshape(b, n, s * w * w, c)
    valid = F.unfold(torch.ones(1, 1, g, g), w, padding=r)[0].T.bool().repeat(1, s)
    return u, valid


def main():
    torch.set_num_threads(4)
    models = {a: load(a) for a in ("shiftwm", "direct", "ar")}
    fig = plt.figure(figsize=(3.1, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.0001], wspace=0.0, left=0.07, right=0.99, top=0.99, bottom=0.1)
    if any(m is None for m in models.values()):
        for i in range(2):
            mf.pending(fig.add_subplot(gs[0, i]), "geometry")
        fig.savefig(mf.FIG / "geometry.pdf"); return
    root = mf.ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    data = FeatureSplit(root, "test", 3, K, "cpu", stats, stride=8, max_episodes=24)
    idx = torch.arange(0, len(data), max(1, len(data) // 16))[:16]
    hist, past, fut, target = data.batch(idx)
    with torch.no_grad():
        p_sw, det = models["shiftwm"](hist, past, fut, return_details=True)
        p_dir = models["direct"](hist, past, fut); p_ar = models["ar"](hist, past, fut)
    cfg = models["shiftwm"].config
    cand, valid = candidates(hist, cfg)                                    # [B,N,M,C]
    y = target[:, K - 1]                                                   # [B,N,C]
    change = ((y - hist[:, -1]) ** 2).mean(-1)
    moving = change >= change.quantile(0.75, dim=-1, keepdim=True)
    # (b) distance to the nearest observed candidate, normalised by the typical spacing of observed features
    def nn_dist(v):
        d = ((cand - v[:, :, None]) ** 2).sum(-1).sqrt().masked_fill(~valid[None], float("inf"))
        return d.min(-1).values
    spacing = torch.cdist(hist[:, -1], hist[:, -1]).median()
    dists = {"true future": nn_dist(y), "ShiftWM": nn_dist(p_sw[:, K - 1]), "Direct": nn_dist(p_dir[:, K - 1]),
             "AR": nn_dist(p_ar[:, K - 1])}
    names = list(dists)
    vals = [(dists[k][moving] / spacing).numpy() for k in names]
    # (a) one moving patch in 2-D
    b0 = int(torch.argmax(moving.float().sum(1) * 0 + change.max(1).values))
    i0 = int(torch.argmax(change[b0] * det["gate"][b0, K - 1, :, 0]))
    w = det["weights"][b0, K - 1, i0].numpy(); cv = cand[b0, i0].numpy(); ok = valid[i0].numpy()
    pts = {"obs": hist[b0, -1, i0].numpy(), "true": y[b0, i0].numpy(), "ShiftWM": p_sw[b0, K - 1, i0].numpy(),
           "Direct": p_dir[b0, K - 1, i0].numpy(), "AR": p_ar[b0, K - 1, i0].numpy()}
    X = np.concatenate([cv[ok], np.stack(list(pts.values()))]); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False); P = Vt[:2].T
    ax = fig.add_subplot(gs[0, 0])
    q = (cv[ok] - mu) @ P; ww = w[ok] / w[ok].max()
    ax.scatter(q[:, 0], q[:, 1], s=4 + 60 * ww, c=ww, cmap="Greens", vmin=-0.2, vmax=1, edgecolors="none", zorder=1,
               label="observed candidates (size: $\\pi$)")
    style = {"obs": ("#9CC3E4", "s", "observed, same patch"), "true": (mf.INK, "*", "true future"),
             "ShiftWM": (mf.METHODS["shiftwm"][1], "o", "ShiftWM"), "Direct": (mf.METHODS["direct"][1], "D", "Direct"),
             "AR": (mf.METHODS["ar"][1], "^", "AR")}
    for k, v in pts.items():
        c, mk, lab = style[k]; xy = (v - mu) @ P
        ax.scatter(xy[0], xy[1], c=c, marker=mk, s=70 if k == "true" else 34, edgecolors="white", linewidths=0.6,
                   zorder=4, label=lab)
    o = (pts["obs"] - mu) @ P; s_ = (pts["ShiftWM"] - mu) @ P
    ax.annotate("", xy=s_, xytext=o, arrowprops=dict(arrowstyle="-|>", color=mf.METHODS["shiftwm"][1], lw=1.0), zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel("feature PC 1", fontsize=6.8, labelpad=1); ax.set_ylabel("feature PC 2", fontsize=6.8, labelpad=1)
    ax.legend(fontsize=6.2, loc="best", handletextpad=0.3, borderaxespad=0.3, framealpha=0.85, markerscale=0.8)
    fig.savefig(mf.FIG / "geometry.pdf"); fig.savefig(mf.FIG / "geometry_preview.png", dpi=200)
    med = {k: float(np.median(v)) for k, v in zip(names, vals)}
    (mf.RES / "analysis/geometry").mkdir(parents=True, exist_ok=True)
    (mf.RES / "analysis/geometry/nn_distance_medians.json").write_text(json.dumps(med, indent=1))
    print("median normalised distance to nearest observed feature:", {k: round(v, 3) for k, v in med.items()})


if __name__ == "__main__":
    main()
