"""'How ShiftWM works' on a controlled toy world: every internal tensor of a trained model, step by step.

The toy world (scripts/v2/toy_world.py) uses raw 4x4 patch pixels as features, so every tensor decodes to RGB
exactly. Models are the REAL V2WorldModel (tiny config) trained by scripts/v2/toy_train.py; nothing is drawn by hand.
Outputs: figures/toy_walkthrough.pdf (mechanism, one test window) and figures/toy_horizons.pdf (forecasts + metrics).
Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_toy.py
"""
import json
import os
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
RES = Path(os.environ.get("TOY_RES", mf.ROOT / "results/toy"))
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


# ---------------------------------------------------------------------------------------------- analysis
def true_disp(pos, t, k):
    """Ground-truth displacement (dy,dx) px of square and disc between frames t and t+k. pos cols: sq x,y; disc x,y."""
    d = pos[t + k] - pos[t]
    return np.array([d[1], d[0]]), np.array([d[3], d[2]])


@torch.no_grad()
def run(model, fr, ac, idx, details=False):
    """Streams (pred, details, target) per mini-batch (materialising all transport weights would need ~2 GB)."""
    for i in range(0, len(idx), 8):
        hist, past, fut, tgt = batch(fr, ac, idx[i:i + 8])
        r = model(hist, past, fut, return_details=True) if details else (model(hist, past, fut), None)
        yield r[0].numpy(), None if r[1] is None else {k: v.numpy() for k, v in r[1].items()}, tgt.numpy()


def analyse():
    """Per-horizon test MSE of every arm (pixel units) + transport-offset error vs ground-truth motion (cached)."""
    cache = RES / "analysis.json"
    cks = [RES / a / "best.pt" for a in ARMS if (RES / a / "best.pt").exists()]
    if cache.exists() and all(c.stat().st_mtime < cache.stat().st_mtime for c in cks):
        return json.loads(cache.read_text())
    fr, ac, lab, pos, wins = windows("test")
    std = np.array(STATS["feature_std"], np.float32)
    res = {"n_windows": len(wins), "mse": {}}
    pers = []
    for arm in ARMS:
        m = load(arm)
        if m is None:
            continue
        se = np.zeros(K); epe = {"square": np.zeros(K), "disc": np.zeros(K), "square_zero": np.zeros(K),
                                "disc_zero": np.zeros(K), "bg": np.zeros(K)}
        cnt = {c: np.zeros(K) for c in ("square", "disc", "bg")}
        gate = {c: np.zeros(K) for c in ("square", "disc", "bg", "cross")}; gcnt = {c: np.zeros(K) for c in gate}
        corr = {c: np.zeros(K) for c in gate}
        j = 0
        for pred, det, tgt in run(m, fr, ac, wins, details=(arm == "shiftwm")):
            se += (((pred - tgt) * std) ** 2).mean((2, 3)).sum(0)
            if arm == "shiftwm":
                disp = implied_disp(det["weights"])                                      # [B,K,N,2]
                for b in range(len(pred)):
                    e, s = wins[j + b]; t = s + H - 1
                    for k in range(1, K + 1):
                        pl, pur = patch_labels(lab[e, t + k]); pl0, _ = patch_labels(lab[e, t])
                        dsq, ddc = true_disp(pos[e], t, k)
                        for c, lbl, gt in (("square", 1, dsq), ("disc", 2, ddc), ("bg", 0, np.zeros(2))):
                            sel = (pl == lbl) & (pur == 1) & ((pl0 == lbl) | (lbl != 0))
                            if sel.any():
                                epe[c][k - 1] += np.linalg.norm(disp[b, k - 1, sel] - gt, axis=-1).sum()
                                if c != "bg":
                                    epe[c + "_zero"][k - 1] += sel.sum() * np.linalg.norm(gt)
                                cnt[c][k - 1] += sel.sum()
                        cn = np.linalg.norm(det["correction"][b, k - 1] * std, axis=-1)
                        for c, sel in (("square", (pl == 1) & (pur == 1)), ("disc", (pl == 2) & (pur == 1)),
                                       ("bg", (pl == 0) & (pur == 1)), ("cross", (pl == 4) & (pl0 != 4))):
                            if sel.any():
                                gate[c][k - 1] += det["gate"][b, k - 1, sel, 0].sum(); corr[c][k - 1] += cn[sel].sum()
                                gcnt[c][k - 1] += sel.sum()
            j += len(pred)
        res["mse"][arm] = (se / len(wins)).tolist()
        if arm == "shiftwm":
            res["epe"] = {c: (v / np.maximum(cnt[c.split("_")[0]], 1)).tolist() for c, v in epe.items()}
            res["gate"] = {c: (v / np.maximum(gcnt[c], 1)).tolist() for c, v in gate.items()}
            res["corr"] = {c: (v / np.maximum(gcnt[c], 1)).tolist() for c, v in corr.items()}
        print(arm, np.round(res["mse"][arm], 4), flush=True)
    se = np.zeros(K)
    for i in range(0, len(wins), 64):
        hist, _, _, tgt = batch(fr, ac, wins[i:i + 64])
        se += ((((hist[:, -1:] - tgt).numpy()) * std) ** 2).mean((2, 3)).sum(0)
    res["mse"]["persistence"] = (se / len(wins)).tolist()
    for a in ARMS:
        ev = RES / a / "eval.json"
        if ev.exists():
            e = json.loads(ev.read_text()); res.setdefault("train", {})[a] = {x: e.get(x) for x in
                                                                               ("steps", "batch", "train_sec", "device")}
    cache.write_text(json.dumps(res, indent=1))
    return res


def pick_window(fr, lab, pos, wins):
    """Deterministic example: large square motion, disc motion, and a big cross reveal within the horizon."""
    best, arg = -1, None
    for e, s in wins:
        t = s + H - 1
        dsq, _ = true_disp(pos[e], t, K)
        sq = pos[e, t, :2]
        if min(sq.min(), 64 - 12 - sq.max()) < 8:
            continue
        reveal = (lab[e, t + K] == 4).sum() - (lab[e, t] == 4).sum()
        # keep the square away from the disc and the occluder so every object is readable
        dsd = np.linalg.norm(pos[e, t, :2] + 6 - pos[e, t, 2:4]); dso = np.linalg.norm(pos[e, t, :2] - pos[e, t, 4:6])
        if dsd < 20 or dso < 22 or np.linalg.norm(pos[e, t, 2:4] - pos[e, t, 4:6] - 8) < 18:
            continue
        sc = np.linalg.norm(dsq) + 0.05 * reveal
        if sc > best:
            best, arg = sc, (e, s)
    return arg


# ---------------------------------------------------------------------------------------------- drawing
UP = 8   # 64x64 -> 512x512; the toy is pixel art, so nearest-neighbour upsampling is exact (no invented detail)
HL, QC = "#FFD400", "#00E5FF"   # highlight (revealed cross) / query patch colours


def show(ax, img, dim=1.0):
    img = np.repeat(np.repeat(np.asarray(img), UP, 0), UP, 1)
    ax.imshow(1 - dim * (1 - img) if dim != 1 else img, extent=(0, 64, 64, 0), **IMKW)
    ax.set_xlim(0, 64); ax.set_ylim(64, 0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(mf.MUTED); s.set_linewidth(0.5)


def patch_grid(ax, alpha=0.35):
    for v in range(4, 64, 4):
        ax.axhline(v, color="white", lw=0.25, alpha=alpha); ax.axvline(v, color="white", lw=0.25, alpha=alpha)


def box(ax, y, x, h, w, color, lw=1.0, ls="-"):
    """Full 4-sided rectangle in pixel coordinates (top-left y,x)."""
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=color, lw=lw, ls=ls, zorder=5))


def grid_map(ax, m, cmap, vmin, vmax):
    ax.imshow(m.reshape(G, G), extent=(0, 64, 64, 0), cmap=cmap, vmin=vmin, vmax=vmax, **IMKW)
    ax.set_xlim(0, 64); ax.set_ylim(64, 0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(mf.MUTED); s.set_linewidth(0.5)


def outlines(ax, labels, classes=(1, 2), color="white", lw=0.5):
    for c in classes:
        m = (labels == c).astype(float)
        if m.any():
            ax.contour(np.arange(64) + 0.5, np.arange(64) + 0.5, m, levels=[0.5], colors=color, linewidths=lw)


def cbar(fig, ax, im_or_cmap, vmin, vmax, label):
    cax = ax.inset_axes([0.0, -0.13, 1.0, 0.055])
    sm = plt.cm.ScalarMappable(cmap=im_or_cmap, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.set_ticks([vmin, vmax]); cb.ax.set_xticklabels([f"{vmin:g}", f"{vmax:g}"])
    cb.ax.text(0.5, -1.9, label, transform=cb.ax.transAxes, ha="center", va="top", fontsize=5.5, color=mf.INK)


def title(ax, s):
    ax.set_title(s, fontsize=6.8, pad=2.5, loc="left", fontweight="normal")


def example(models, ks=5):
    fr, ac, lab, pos, wins = windows("test")
    e, s = pick_window(fr, lab, pos, wins)
    t = s + H - 1
    out = {"e": e, "s": s, "t": t, "fr": fr[e], "lab": lab[e], "pos": pos[e]}
    for arm, m in models.items():
        pred, det, tgt = next(run(m, fr, ac, [(e, s)], details=(arm == "shiftwm")))
        out[arm] = pred[0]; out["tgt"] = tgt[0]
        if det is not None:
            out["det"] = {k: v[0] for k, v in det.items()}
    return out


def query_patch(ex, k):
    """Centre-most patch fully covered by the square at t+k (grid row, col)."""
    pl, pur = patch_labels(ex["lab"][ex["t"] + k])
    cand = np.where((pl == 1) & (pur == 1))[0]
    sq = ex["pos"][ex["t"] + k, :2] + 6                                            # square centre (x,y)
    cy, cx = cand // G * P + 2, cand % G * P + 2
    q = cand[np.argmin((cy - sq[1]) ** 2 + (cx - sq[0]) ** 2)]
    return q // G, q % G


def mse_px(a, b):
    std = np.array(STATS["feature_std"], np.float32)
    return float((((a - b) * std) ** 2).mean())


def walkthrough(ex, ks=5):
    t, det = ex["t"], ex["det"]
    fig = plt.figure(figsize=(5.5, 2.62))
    gs = fig.add_gridspec(2, 6, left=0.012, right=0.988, top=0.9, bottom=0.12, wspace=0.12, hspace=0.62)
    qr, qc = query_patch(ex, ks)
    qy, qx = qr * P, qc * P
    Ssrc = 3
    w = det["weights"][ks - 1, qr * G + qc].reshape(Ssrc, W, W)
    vmax_w = float(w.max())
    # --- row 1: observed frames with patch grid, query patch and its 7x7 window; transport weights per source
    for i in range(Ssrc):
        ax = fig.add_subplot(gs[0, i]); lag = Ssrc - 1 - i
        show(ax, ex["fr"][t - lag] / 255); patch_grid(ax)
        box(ax, qy - R * P, qx - R * P, W * P, W * P, QC, lw=0.9, ls=(0, (2, 1)))
        box(ax, qy, qx, P, P, QC, lw=1.0)
        # true source of the query's content in this frame (square moves rigidly)
        d = ex["pos"][t + ks, :2] - ex["pos"][t - lag, :2]
        ax.plot(qx + 2 - d[0], qy + 2 - d[1], marker="o", ms=3.2, mfc="none", mec="white", mew=0.8, zorder=6)
        title(ax, ("(a) " if i == 0 else "") + (f"frame $t{-lag}$" if lag else "frame $t$ (last)"))
        axw = fig.add_subplot(gs[0, 3 + i])
        axw.imshow(w[i], cmap="Greens", vmin=0, vmax=vmax_w, extent=(-R - .5, R + .5, R + .5, -R - .5), **IMKW)
        axw.set_xticks([]); axw.set_yticks([]); axw.grid(False)
        for sp in axw.spines.values():
            sp.set_visible(True); sp.set_color(QC); sp.set_linewidth(0.9); sp.set_linestyle((0, (2, 1)))
        m = w[i].sum()
        if m > 1e-3:
            ey, ex_ = (w[i] * OFF[:, 0].reshape(W, W)).sum() / m, (w[i] * OFF[:, 1].reshape(W, W)).sum() / m
            if np.hypot(ey, ex_) > 0.15:
                axw.annotate("", xy=(ex_, ey), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=mf.INK, lw=0.9,
                                                                               mutation_scale=6), zorder=6)
        axw.plot(-d[0] / P, -d[1] / P, marker="o", ms=4, mfc="none", mec=mf.INK, mew=0.7, zorder=7)
        axw.plot(0, 0, marker="+", ms=4, color=mf.MUTED, mew=0.6)
        title(axw, ("(b) " if i == 0 else "") + (rf"$\pi$ on $t{-lag}$" if lag else r"$\pi$ on $t$") + f" ({m:.2f})")
    # --- row 2: transport field, gate, correction, forecast vs truth vs Direct at horizon ks
    disp = implied_disp(det["weights"][None])[0, ks - 1]                             # [N,2] px
    g = det["gate"][ks - 1, :, 0]
    ax = fig.add_subplot(gs[1, 0]); show(ax, ex["fr"][t] / 255, dim=0.45)
    yy, xx = np.divmod(np.arange(G * G), G)
    sel = (g > 0.25) & (np.linalg.norm(disp, axis=1) > 0.75)
    ax.quiver(xx[sel] * P + 2, yy[sel] * P + 2, disp[sel, 1], disp[sel, 0], angles="xy", scale_units="xy", scale=1,
              color="#C0392B", width=0.012, headwidth=3.5, headlength=3.5, headaxislength=3.2, zorder=5)
    title(ax, f"(c) transport field, $k{{=}}{ks}$")
    ax = fig.add_subplot(gs[1, 1]); grid_map(ax, g, "viridis", 0, 1)
    outlines(ax, ex["lab"][t + ks]); cbar(fig, ax, "viridis", 0, 1, "gate $g$"); title(ax, "(d) gate $g$")
    std = np.array(STATS["feature_std"], np.float32)
    cn = np.linalg.norm(det["correction"][ks - 1] * std, axis=-1) / np.sqrt(P * P * 3)   # RMS per pixel value
    vmax_c = float(np.ceil(cn.max() * 20) / 20)
    ax = fig.add_subplot(gs[1, 2]); grid_map(ax, cn, "magma", 0, vmax_c)
    new = (ex["lab"][t + ks] == 4) & (ex["lab"][t] != 4)
    if new.any():
        ys, xs = np.where(new)
        box(ax, ys.min() // P * P, xs.min() // P * P, (ys.max() // P + 1 - ys.min() // P) * P,
            (xs.max() // P + 1 - xs.min() // P) * P, HL, lw=1.0)
    cbar(fig, ax, "magma", 0, vmax_c, "RMS correction"); title(ax, "(e) correction $\\|c\\|$")
    for j, (key, name) in enumerate((("shiftwm", "ShiftWM"), ("tgt", "truth"), ("direct", "Direct"))):
        if key not in ex:
            continue
        ax = fig.add_subplot(gs[1, 3 + j]); show(ax, to_rgb(ex[key][ks - 1]))
        if key != "tgt":
            ax.text(0.97, 0.03, f"MSE {1e3 * mse_px(ex[key][ks - 1], ex['tgt'][ks - 1]):.1f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.5, color="white",
                    bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))
        box(ax, qy, qx, P, P, QC, lw=0.8)
        title(ax, ("(f) " if j == 0 else "") + f"{name}, $t{{+}}{ks}$")
    fig.savefig(mf.FIG / "toy_walkthrough.pdf"); fig.savefig(mf.FIG / "toy_walkthrough_preview.png", dpi=250)
    plt.close(fig)


def horizons(ex, res, ks=(1, 5, 10)):
    rows = [("tgt", "Truth"), ("shiftwm", "ShiftWM"), ("direct", "Direct"), ("ar", "AR")]
    rows = [r for r in rows if r[0] in ex]
    fig = plt.figure(figsize=(5.5, 2.5))
    n = len(rows)
    gl = fig.add_gridspec(n, len(ks), left=0.045, right=0.47, top=0.91, bottom=0.03, wspace=0.05, hspace=0.08)
    new = lambda k: (ex["lab"][ex["t"] + k] == 4) & (ex["lab"][ex["t"]] != 4)
    for i, (key, name) in enumerate(rows):
        for j, k in enumerate(ks):
            ax = fig.add_subplot(gl[i, j]); show(ax, to_rgb(ex[key][k - 1]))
            if i == 0:
                ax.set_title(f"$t{{+}}{k}$", fontsize=7, pad=2)
            if j == 0:
                ax.set_ylabel(name, fontsize=7, labelpad=2)
            if key != "tgt":
                ax.text(0.97, 0.03, f"{1e3 * mse_px(ex[key][k - 1], ex['tgt'][k - 1]):.1f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=5.5, color="white",
                        bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))
            m = new(k)
            if m.any() and key == "tgt":
                ys, xs = np.where(m)
                box(ax, ys.min() - 1, xs.min() - 1, ys.max() - ys.min() + 3, xs.max() - xs.min() + 3, HL, lw=0.8)
    kk = np.arange(1, K + 1)
    ax = fig.add_axes([0.575, 0.6, 0.4, 0.31])
    for arm in ("persistence", "ar", "direct", "shiftwm"):
        if arm in res["mse"]:
            lab_, col, ls, mk = mf.METHODS[arm]
            ax.plot(kk, 1e3 * np.array(res["mse"][arm]), color=col, ls=ls, marker=mk, ms=2.6, lw=1.2,
                    label=lab_.replace(" (ours)", ""))
    ax.set_ylabel(r"test MSE ($\times10^{-3}$)", fontsize=6.5); ax.set_xticks([1, 5, 10])
    ax.tick_params(labelsize=6); ax.legend(fontsize=5.8, ncol=2, loc="upper left", handlelength=1.6)
    ax.set_title("(b) forecast error", fontsize=7, loc="left", fontweight="normal")
    ax = fig.add_axes([0.575, 0.12, 0.4, 0.31])
    if "epe" in res:
        for c, col in (("square", "#C0392B"), ("disc", "#2E6FD8")):
            ax.plot(kk, res["epe"][c], color=col, marker="o", ms=2.6, lw=1.2, label=f"{c}: transport")
            ax.plot(kk, res["epe"][c + "_zero"], color=col, ls=(0, (3, 2)), lw=1.0, label=f"{c}: no motion")
    ax.set_xlabel("horizon $k$", fontsize=6.5); ax.set_ylabel("endpoint error (px)", fontsize=6.5)
    ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.6, ncol=2, loc="upper left", handlelength=1.8)
    ax.set_title("(c) implied offset vs true motion", fontsize=7, loc="left", fontweight="normal")
    fig.text(0.045, 0.975, "(a) forecasts (MSE $\\times10^{-3}$ in corner)", fontsize=7, va="top", color=mf.INK)
    fig.savefig(mf.FIG / "toy_horizons.pdf"); fig.savefig(mf.FIG / "toy_horizons_preview.png", dpi=250)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    models = {a: m for a in ARMS if (m := load(a)) is not None}
    res = analyse()
    print(json.dumps({k: v for k, v in res.items() if k != "mse"}, indent=0)[:1500])
    ex = example(models)
    print("example window", ex["e"], ex["s"])
    walkthrough(ex)
    horizons(ex, res)


if __name__ == "__main__":
    main()
