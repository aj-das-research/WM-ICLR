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
H, K, G, W = 3, 5, 16, 11
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


CROSS_AREA = 57   # pixels of the fully revealed cross (two 3x11 bars sharing a 3x3 centre)


def pick_window(fr, lab, pos, wins):
    """Fixed selection rule (stated in the caption): among test windows where, in every frame t-2..t+K, the square,
    disc and occluder/cross are fully visible and never touch, the square moves in every observed step, and the cross
    is partly revealed at t (30-80% of its pixels) and fully revealed at t+K, take the one with the largest square
    displacement t-2 -> t+K (ties: first window in test order)."""
    best, arg = -1.0, None
    for e, s in wins:
        t = s + H - 1
        ok = True
        for u in range(t - 2, t + K + 1):
            sq, dc, oc = pos[e, u, :2] + 6, pos[e, u, 2:4], pos[e, u, 4:6] + 8
            cr = pos[e, 0, 4:6] + 8                                                   # cross centre = initial occluder centre
            if ((lab[e, u] == 1).sum() < 144 or (lab[e, u] == 2).sum() < 100 or np.abs(sq - oc).max() < 16
                    or np.abs(sq - cr).max() < 14 or np.linalg.norm(sq - dc) < 20 or np.abs(dc - oc).max() < 16
                    or np.abs(dc - cr).max() < 14):
                ok = False; break
        if not ok:
            continue
        rev = (lab[e, t] == 4).sum() / CROSS_AREA
        if not 0.3 <= rev <= 0.8 or (lab[e, t + K] == 4).sum() < CROSS_AREA:
            continue
        steps = np.abs(np.diff(pos[e, t - 2:t + 1, :2], axis=0)).sum(1)
        if (steps == 0).any():
            continue
        sc = float(np.linalg.norm(pos[e, t + K, :2] - pos[e, t - 2, :2]))
        if sc > best:
            best, arg = sc, (e, s)
    return arg


# ---------------------------------------------------------------------------------------------- drawing
# Toy frames are pixel art: nearest-neighbour upsampling (imshow interpolation="nearest") reproduces the exact 4x4
# patch structure without inventing detail, so it is the faithful way to enlarge them.
QC, HL, ARW = "#00B8D9", "#FFB000", "#C0392B"   # query/window, revealed-cross highlight, motion arrows
CAP = 6.4                                        # panel-title font size


def frame_ax(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#B8BDC4"); s.set_linewidth(0.5)


def show(ax, img, dim=1.0, extent=(0, 64, 64, 0)):
    img = np.asarray(img)
    ax.imshow(1 - dim * (1 - img) if dim != 1 else img, extent=extent, **IMKW)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); frame_ax(ax)


def patch_grid(ax, lo=0, hi=64, alpha=0.5):
    for v in range(lo + 4, hi, 4):
        ax.axhline(v, color="white", lw=0.2, alpha=alpha); ax.axvline(v, color="white", lw=0.2, alpha=alpha)


def box(ax, y, x, h, w, color, lw=1.0, ls="-", z=5):
    """Full 4-sided rectangle in pixel coordinates (top-left y,x)."""
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=color, lw=lw, ls=ls, zorder=z, joinstyle="miter"))


def grid_map(ax, m, cmap, vmin, vmax):
    ax.imshow(m.reshape(G, G), extent=(0, 64, 64, 0), cmap=cmap, vmin=vmin, vmax=vmax, **IMKW)
    ax.set_xlim(0, 64); ax.set_ylim(64, 0); frame_ax(ax)


def outlines(ax, labels, classes=(1, 2), color="white", lw=0.5):
    for c in classes:
        m = (labels == c).astype(float)
        if m.any():
            ax.contour(np.arange(64) + 0.5, np.arange(64) + 0.5, m, levels=[0.5], colors=color, linewidths=lw)


def cbar(fig, ax, cmap, vmin, vmax, label):
    cax = ax.inset_axes([0.08, -0.12, 0.84, 0.05])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5.2, length=1.2, pad=0.8)
    cb.set_ticks([vmin, vmax]); cb.ax.set_xticklabels([f"{vmin:g}", f"{vmax:g}"])
    cb.ax.set_xlabel(label, fontsize=5.4, labelpad=-5.5, color=mf.INK)


def title(ax, s, loc="center"):
    ax.set_title(s, fontsize=CAP, pad=2.2, loc=loc, fontweight="normal", color=mf.INK)


def tag(ax, s):
    ax.text(0.97, 0.03, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color="white",
            bbox=dict(boxstyle="square,pad=0.18", fc="#1f2630", ec="none", alpha=0.7), zorder=8)


def corner(ax, s):
    ax.text(0.04, 0.96, s, transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=mf.INK, zorder=8,
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.85))


def span_title(fig, a0, a1, s):
    """One title centred over axes a0..a1, at the height of a normal panel title."""
    p0, p1 = a0.get_position(), a1.get_position()
    pad = 2.2 / 72 / fig.get_figheight()                                            # same 2.2 pt pad as title()
    fig.text((p0.x0 + p1.x1) / 2, p0.y1 + pad, s, ha="center", va="baseline", fontsize=CAP, color=mf.INK)


def cross_box(ax, lab_now, pad=1):
    ys, xs = np.where(lab_now == 4)
    if len(ys):
        box(ax, ys.min() - pad, xs.min() - pad, ys.max() - ys.min() + 1 + 2 * pad, xs.max() - xs.min() + 1 + 2 * pad,
            HL, lw=0.9)


def example(models):
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
    """Patch nearest the square's centre at t+k that the square fully covers (grid row, col)."""
    pl, pur = patch_labels(ex["lab"][ex["t"] + k])
    cand = np.where((pl == 1) & (pur == 1))[0]
    sq = ex["pos"][ex["t"] + k, :2] + 6
    cy, cx = cand // G * P + 2, cand % G * P + 2
    q = cand[np.argmin((cy - sq[1]) ** 2 + (cx - sq[0]) ** 2)]
    return q // G, q % G


def mse_px(a, b):
    std = np.array(STATS["feature_std"], np.float32)
    return float((((a - b) * std) ** 2).mean())


def walkthrough(ex, ks=3):
    """Two rows of six equal panels. Row 1: observe -> query -> match. Row 2: move -> gate -> correct -> compose."""
    t, det = ex["t"], ex["det"]
    fig = plt.figure(figsize=(5.5, 2.55))
    L, Rm, top, bot, hs, ws = 0.01, 0.99, 0.925, 0.1, 0.3, 0.1
    gs = fig.add_gridspec(2, 6, left=L, right=Rm, top=top, bottom=bot, wspace=ws, hspace=hs)
    qr, qc = query_patch(ex, ks)
    qy, qx = qr * P, qc * P
    w = det["weights"][ks - 1, qr * G + qc].reshape(-1, W, W)                        # [S,W,W] (source 0 = t-2)
    S_ = w.shape[0]
    src = int(np.argmax(w.sum((1, 2))))                                              # frame holding most of pi
    lag = S_ - 1 - src
    # (i) observe: three frames, patch grid, square outline at t for reference
    for i in range(3):
        ax = fig.add_subplot(gs[0, i]); show(ax, ex["fr"][t - 2 + i] / 255); patch_grid(ax)
        corner(ax, ["$t{-}2$", "$t{-}1$", "$t$"][i])
    span_title(fig, fig.axes[0], fig.axes[2], "(i) observe: three frames, 16$\\times$16 patches")
    # (ii) query + window on the source frame
    ax = fig.add_subplot(gs[0, 3]); show(ax, ex["fr"][t - lag] / 255); patch_grid(ax)
    box(ax, qy - R * P, qx - R * P, W * P, W * P, QC, lw=0.9, ls=(0, (2.5, 1.2)))
    box(ax, qy, qx, P, P, QC, lw=1.1)
    title(ax, "(ii) query + window"); corner(ax, f"$t{{-}}{lag}$" if lag else "$t$")
    # (iii) transport weights pi on the window crop of that frame; arrow = expected source offset
    ax = fig.add_subplot(gs[0, 4])
    ext = (qx - R * P, qx + (R + 1) * P, qy + (R + 1) * P, qy - R * P)
    show(ax, ex["fr"][t - lag] / 255, dim=0.35, extent=(0, 64, 64, 0))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_facecolor("#EEF0F2")
    wm = np.ma.masked_less(w[src], 0.01 * w[src].max())
    ax.imshow(wm, extent=ext, cmap="Greens", vmin=0, vmax=float(w[src].max()), alpha=0.9, zorder=3, **IMKW)
    box(ax, qy, qx, P, P, QC, lw=1.1)
    m = w[src].sum()
    oy, ox = (w[src] * OFF[:, 0].reshape(W, W)).sum() / m, (w[src] * OFF[:, 1].reshape(W, W)).sum() / m
    ax.annotate("", xy=(qx + 2 + P * ox, qy + 2 + P * oy), xytext=(qx + 2, qy + 2), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=mf.INK, lw=0.9, mutation_scale=6, shrinkA=0, shrinkB=0))
    d = ex["pos"][t + ks, :2] - ex["pos"][t - lag, :2]                                # true source (square is rigid)
    ax.plot(qx + 2 - d[0], qy + 2 - d[1], marker="o", ms=4.2, mfc="none", mec=HL, mew=1.0, zorder=7)
    title(ax, "(iii) match: weights $\\pi$"); corner(ax, "zoom")
    for sp in ax.spines.values():                                                     # dashed frame = the window in (ii)
        sp.set_color(QC); sp.set_linewidth(0.9); sp.set_linestyle((0, (2.5, 1.2)))
    # (iv) motion field implied by pi
    disp = implied_disp(det["weights"][None])[0, ks - 1]
    g = det["gate"][ks - 1, :, 0]
    ax = fig.add_subplot(gs[0, 5]); show(ax, ex["fr"][t] / 255, dim=0.4)
    yy, xx = np.divmod(np.arange(G * G), G)
    sel = np.linalg.norm(disp, axis=1) > 2.0
    ax.quiver(xx[sel] * P + 2, yy[sel] * P + 2, disp[sel, 1], disp[sel, 0], angles="xy", scale_units="xy", scale=1,
              color=ARW, width=0.011, headwidth=3.2, headlength=3.2, headaxislength=2.9, zorder=5)
    title(ax, "(iv) implied motion")
    # (v) gate, (vi) correction
    ax = fig.add_subplot(gs[1, 0]); grid_map(ax, g, "viridis", 0, 1); outlines(ax, ex["lab"][t + ks])
    cbar(fig, ax, "viridis", 0, 1, "keep $\\leftrightarrow$ move"); title(ax, "(v) gate $g$")
    std = np.array(STATS["feature_std"], np.float32)
    cn = np.sqrt(((det["correction"][ks - 1] * std) ** 2).mean(-1))                # RMS per pixel value
    vc = float(max(0.05, np.ceil(cn.max() * 20) / 20))
    ax = fig.add_subplot(gs[1, 1]); grid_map(ax, cn, "magma", 0, vc)
    new = (ex["lab"][t + ks] == 4) & (ex["lab"][t] != 4)
    if new.any():
        cross_box(ax, np.where(new, 4, 0))
    cbar(fig, ax, "magma", 0, vc, "RMS"); title(ax, "(vi) correction $\\|c\\|$")
    # (vii) compose: ShiftWM forecast vs truth vs Direct
    for j, (key, name) in enumerate((("tgt", "truth"), ("shiftwm", "ShiftWM"), ("direct", "Direct"))):
        ax = fig.add_subplot(gs[1, 2 + j])
        if key not in ex:
            show(ax, np.ones((64, 64, 3))); ax.text(0.5, 0.5, "Direct\npending", ha="center", va="center", fontsize=6,
                                       color=mf.MUTED, transform=ax.transAxes); continue
        show(ax, to_rgb(ex[key][ks - 1]))
        if key == "tgt":
            cross_box(ax, ex["lab"][t + ks])
        else:
            tag(ax, f"MSE {1e3 * mse_px(ex[key][ks - 1], ex['tgt'][ks - 1]):.1f}")
        corner(ax, name)
    span_title(fig, fig.axes[-3], fig.axes[-1], f"(vii) forecast $t{{+}}{ks}$: $(1{{-}}g)\\,z_t + g\\,\\tilde z + c$ vs. truth")
    # (legend) shared colour key
    ax = fig.add_subplot(gs[1, 5]); ax.set_axis_off()
    items = [(QC, "-", "query / window"), (HL, "o", "true source"), (HL, "-", "revealed cross"),
             (mf.INK, ">", "expected offset"), (ARW, ">", "motion arrows"), ("#9aa0a6", "c", "object outline")]
    for i, (c, kind, txt) in enumerate(items):
        y = 0.95 - i * 0.18
        if kind == "c":
            ax.add_patch(Rectangle((0.02, y - 0.05), 0.12, 0.1, fill=True, fc="#2d6b5f", ec="white", lw=0.8,
                                   transform=ax.transAxes))
        elif kind == "-":
            ax.add_patch(Rectangle((0.02, y - 0.05), 0.12, 0.1, fill=False, ec=c, lw=1.0, transform=ax.transAxes))
        elif kind == "o":
            ax.plot(0.08, y, marker="o", ms=4, mfc="none", mec=c, mew=1.0, transform=ax.transAxes)
        else:
            ax.annotate("", xy=(0.15, y), xytext=(0.01, y), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, mutation_scale=6))
        ax.text(0.2, y, txt, transform=ax.transAxes, va="center", fontsize=5.4, color=mf.INK)
    fig.savefig(mf.FIG / "toy_walkthrough.pdf"); fig.savefig(mf.FIG / "toy_walkthrough_preview.png", dpi=300)
    plt.close(fig)


def horizons(ex, res, ks=(1, 3, 5)):
    rows = [(r, n) for r, n in (("tgt", "truth"), ("shiftwm", "ShiftWM"), ("direct", "Direct"), ("ar", "AR"))
            if r in ex]
    n, t = len(rows), ex["t"]
    top_in, bot_in = 0.32, 0.34                        # header (panel + column titles) / room for (c)'s x labels
    fh = 0.62 * n + top_in + bot_in
    fig = plt.figure(figsize=(5.5, fh))
    gl = fig.add_gridspec(n, len(ks) + 1, left=0.05, right=0.5, top=1 - top_in / fh, bottom=bot_in / fh,
                          wspace=0.06, hspace=0.08)
    for i, (key, name) in enumerate(rows):
        ax = fig.add_subplot(gl[i, 0])
        show(ax, ex["fr"][t] / 255)
        if i == 0:
            title(ax, "input $t$")
        if i:
            ax.set_visible(False)
        for j, k in enumerate(ks):
            ax = fig.add_subplot(gl[i, j + 1]); show(ax, to_rgb(ex[key][k - 1]))
            if i == 0:
                title(ax, f"$t{{+}}{k}$"); cross_box(ax, ex["lab"][t + k])
            else:
                tag(ax, f"{1e3 * mse_px(ex[key][k - 1], ex['tgt'][k - 1]):.1f}")
    for i, (key, name) in enumerate(rows):                                            # row labels at the left edge
        p = fig.axes[1 + i * (len(ks) + 1)].get_position()
        fig.text(0.045, (p.y0 + p.y1) / 2, name, rotation=90, ha="right", va="center", fontsize=CAP, color=mf.INK)
    # (b) and (c) span exactly the image grid: top of (b) = top of the first image row, bottom of (c) = bottom
    # of the last image row, so all three panels share their top and bottom lines.
    cols = len(ks) + 1
    for a in fig.axes:
        a.apply_aspect()
    y_top = fig.axes[1].get_position().y1
    y_bot = fig.axes[(n - 1) * cols + 1].get_position().y0
    gap = 0.34 / fh                                     # room for (b)'s tick labels and (c)'s title
    h = (y_top - y_bot - gap) / 2
    kk = np.arange(1, K + 1)
    ax = fig.add_axes([0.6, y_top - h, 0.38, h])
    for arm in ("persistence", "ar", "direct", "shiftwm"):
        if arm in res["mse"]:
            lab_, col, ls, mk = mf.METHODS[arm]
            ax.plot(kk, 1e3 * np.array(res["mse"][arm]), color=col, ls=ls, marker=mk, ms=2.6, lw=1.2,
                    label=lab_.replace(" (ours)", ""))
    ax.set_ylabel(r"MSE ($\times10^{-3}$)", fontsize=6); ax.set_xticks(kk); ax.tick_params(labelsize=5.8)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))   # headroom so the legend clears the curves
    ax.legend(fontsize=5.4, ncol=4, loc="upper left", handlelength=1.6, columnspacing=0.8)
    title(ax, "(b) test error vs horizon", loc="left")
    ax = fig.add_axes([0.6, y_bot, 0.38, h])
    if "epe" in res:
        for c, col in (("square", "#C0392B"), ("disc", "#2E6FD8")):
            ax.plot(kk, res["epe"][c], color=col, marker="o", ms=2.6, lw=1.2, label=f"{c}")
            ax.plot(kk, res["epe"][c + "_zero"], color=col, ls=(0, (3, 2)), lw=0.9)
    ax.plot([], [], color=mf.MUTED, ls=(0, (3, 2)), lw=0.9, label="no-motion ref.")
    ax.set_xlabel("horizon $k$", fontsize=6); ax.set_ylabel("offset error (px)", fontsize=6)
    ax.set_xticks(kk); ax.tick_params(labelsize=5.8)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))   # headroom so the legend clears the curves
    ax.legend(fontsize=5.4, ncol=3, loc="upper left", handlelength=1.6, columnspacing=0.8)
    title(ax, "(c) transport offset vs true motion", loc="left")
    fig.text(0.05, 1 - 0.04 / fh, "(a) forecasts (corner: MSE$\\times10^{3}$)", fontsize=CAP, va="top", color=mf.INK)
    fig.savefig(mf.FIG / "toy_horizons.pdf"); fig.savefig(mf.FIG / "toy_horizons_preview.png", dpi=300)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    models = {a: m for a in ARMS if (m := load(a)) is not None}
    res = analyse()
    print(json.dumps(res.get("epe", {}))[:600])
    ex = example(models)
    print("example window", ex["e"], ex["s"], "arms", list(models))
    walkthrough(ex)
    horizons(ex, res)


if __name__ == "__main__":
    main()
