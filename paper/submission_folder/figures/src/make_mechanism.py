"""'How ShiftWM works' figure: intermediate tensors of trained models on a held-out DROID window.

(a) PCA->RGB views of the 16x16xC feature grids: observed Z0, true Z_k, forecasts of ShiftWM / Direct / AR,
    and ShiftWM's additive decomposition (1-g)Z0 + g T + r.
(c) Forecast sharpness: spatial variance of predicted grids relative to the truth, per horizon
    (results/v2/analysis/sharpness, computed by scripts/v2/sharpness.py on all DROID test windows).
Everything is computed from logged checkpoints and real features; nothing is drawn by hand.
Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_mechanism.py
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402
from shiftwm.v2.models import V2WorldModel  # noqa: E402

K = 10


def load_model(arm):
    ck = sorted((mf.RES / "droid/dinov2s" / arm).glob("s*/best.pt"))
    if not ck:
        return None
    st = torch.load(ck[0], map_location="cpu")
    m = V2WorldModel(st["config"]).eval(); m.load_state_dict(st["model"])
    return m


def window(ep):
    root = mf.ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    with np.load(root / f"episodes/{ep}.npz") as z:
        f, a = z["features"].astype(np.float32), z["actions"]
    fm, fs = np.array(stats["feature_mean"]), np.array(stats["feature_std"])
    am, ast = np.array(stats["action_mean"]), np.array(stats["action_std"])
    fz = (f.reshape(len(f), -1, f.shape[-1]) - fm) / fs
    an = (a - am) / ast
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    return t(fz[None, :3]), t(an[None, :2]), t(an[None, 2:2 + K]), fz[3:3 + K]


def pca_rgb(fit, *grids):
    """Fit 3-component PCA on `fit` [M,C]; map each grid [N,C] to a 16x16x3 RGB image with shared scaling."""
    mu = fit.mean(0); U, S, Vt = np.linalg.svd(fit - mu, full_matrices=False)
    P = Vt[:3].T
    proj = [(g - mu) @ P for g in grids]
    lo, hi = np.percentile(np.concatenate(proj), [1, 99], axis=0)
    out = []
    for p in proj:
        x = np.clip((p - lo) / (hi - lo + 1e-8), 0, 1)
        out.append(x.reshape(16, 16, 3))
    return out


def main():
    ep = mf.pick_teaser_episode()
    hist, past, fut, truth = window(ep)
    models = {a: load_model(a) for a in ("shiftwm", "direct", "ar")}
    fig = plt.figure(figsize=(5.5, 3.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.05], wspace=0.22, hspace=0.7,
                          left=0.01, right=0.985, top=0.9, bottom=0.12)
    if any(m is None for m in models.values()):
        mf.pending(fig.add_subplot(gs[:, :]), "mechanism"); fig.savefig(mf.FIG / "mechanism.pdf"); plt.close(fig); return
    with torch.no_grad():
        p_sw, det = models["shiftwm"](hist, past, fut, return_details=True)
        p_dir = models["direct"](hist, past, fut)
        p_ar = models["ar"](hist, past, fut)
    z0 = hist[0, -1].numpy(); k = K - 1
    g = det["gate"][0, k].numpy(); corr = det["correction"][0, k].numpy(); pred = p_sw[0, k].numpy()
    term_id = (1 - g) * z0
    term_tr = pred - term_id - corr                               # = g * T_k, exact identity of Eq. (2)
    fit = np.concatenate([truth.reshape(-1, truth.shape[-1]), z0])
    imgs = pca_rgb(fit, z0, truth[k], pred, p_dir[0, k].numpy(), p_ar[0, k].numpy())
    # (a) forecasts in feature space
    sub = gs[0, :].subgridspec(1, 5, wspace=0.08)
    names = ["observed $Z_0$", "true $Z_{10}$", "ShiftWM", "Direct", "AR"]
    errs = [None, None] + [float(((x - truth[k]) ** 2).mean()) for x in (pred, p_dir[0, k].numpy(), p_ar[0, k].numpy())]
    for i, (im, n, e) in enumerate(zip(imgs, names, errs)):
        ax = fig.add_subplot(sub[0, i]); ax.imshow(im, interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(n, fontsize=7, fontweight="bold" if n == "ShiftWM" else "normal", pad=2)
        ax.set_xlabel(f"MSE {e:.3f}" if e is not None else ("input" if i == 0 else "target"), fontsize=6.5, labelpad=1.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#009E73" if n == "ShiftWM" else "#C9CED6"); sp.set_linewidth(1.6 if n == "ShiftWM" else 0.6)
    # (b) decomposition
    sub2 = gs[1, 0].subgridspec(1, 7, width_ratios=[1, 0.26, 1, 0.26, 1, 0.26, 1], wspace=0.0)
    dimgs = pca_rgb(np.concatenate([fit, term_id, term_tr]), term_id, term_tr, pred)
    cn = np.linalg.norm(corr, axis=-1).reshape(16, 16)
    panels = [(dimgs[0], "stay", "$(1-g)Z_0$"), ("+", None, None), (dimgs[1], "move", "$g\\,T$"), ("+", None, None),
              (cn, "correct", "$\\|r\\|$"), ("=", None, None), (dimgs[2], "forecast", "$\\hat Z_{10}$")]
    for i, (im, n, eq) in enumerate(panels):
        ax = fig.add_subplot(sub2[0, i]); ax.set_xticks([]); ax.set_yticks([])
        if isinstance(im, str):
            ax.set_axis_off(); ax.text(0.5, 0.5, im, ha="center", va="center", fontsize=12, color=mf.INK, fontweight="bold")
            continue
        ax.imshow(im, cmap="magma" if im.ndim == 2 else None, interpolation="nearest", aspect="auto")
        ax.set_title(n, fontsize=7, pad=2, fontweight="bold"); ax.set_xlabel(eq, fontsize=6.8, labelpad=1.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#C9CED6")
    # (c) sharpness vs horizon
    ax = fig.add_subplot(gs[1, 1])
    sp_path = mf.RES / "analysis/sharpness/droid_dinov2s.json"
    if sp_path.exists():
        sh = json.loads(sp_path.read_text()); kk = np.arange(1, K + 1)
        for arm in ("persistence", "ar", "direct", "shiftwm"):
            if arm not in sh:
                continue
            lab, col, ls, mk = mf.METHODS[arm]
            ax.plot(kk, sh[arm]["mean"], color=col, ls=ls, marker=mk, markevery=3, lw=2 if arm == "shiftwm" else 1.3)
            ax.fill_between(kk, sh[arm]["lo"], sh[arm]["hi"], color=col, alpha=0.15, lw=0)
            ax.annotate(lab.split(" (")[0], (kk[-1], sh[arm]["mean"][-1]), xytext=(3, 0), textcoords="offset points",
                        fontsize=6.3, va="center", color=mf.INK)
        ax.set_xlim(0.5, 13.2); ax.set_xticks([1, 4, 7, 10])
        ax.set_xlabel("forecast step $k$", fontsize=7, labelpad=1); ax.set_ylabel("kept spatial contrast", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6.5)
    else:
        mf.pending(ax, "sharpness")
    for t, x, y in (("(a) Forecasts in feature space (PCA$\\to$RGB, $k{=}10$, held-out DROID)", 0.01, 0.955),
                    ("(b) How ShiftWM composes a forecast", 0.01, 0.5), ("(c) Forecasts stay sharp", 0.63, 0.5)):
        fig.text(x, y, t, fontsize=7.8, fontweight="bold", color=mf.INK)
    fig.savefig(mf.FIG / "mechanism.pdf"); fig.savefig(mf.FIG / "mechanism_preview.png", dpi=170)
    plt.close(fig)
    print("mechanism figure from episode", ep)


if __name__ == "__main__":
    main()
