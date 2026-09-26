"""Segmentation view of the forecasts (results/v2/analysis/segments/<ds>, from scripts/v2/segments.py).

Main figure (figures/segments.pdf): one zoomed region-of-interest row for DROID and one for Hamlyn (example 1 of each:
the moving window with the largest k=10 IoU advantage of ShiftWM over the better baseline, ShiftWM IoU >= 0.5), IoU-vs-horizon curves on moving windows, and the horizon-averaged IoU gain of
ShiftWM over Direct / AR with paired 95% CIs for every dataset evaluated so far.
Appendix figures (figures/segments_<ds>.pdf): both example windows (two largest ShiftWM advantages, distinct
episodes) + curves. Examples are illustrative; the averages and CIs are in the table.
Row layout: full observed frame t (reference mask at t, true future outline dashed, crop box) | crop of frame t with
the ShiftWM transport arrows | crop of the true frame t+K with its tracked SAM 2.1 mask | ShiftWM / Direct / AR: soft
foreground probability read out of the k=K forecast (heat) + its decision contour, true future mask dashed white,
IoU in the corner (best in green bold).
Also writes tables/generated/segments_numbers.tex (macros) and tables/generated/segments_rows.tex (appendix table).
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/segments"
GEN = mf.ROOT / "paper/submission_folder/tables/generated"
SAMC = "#F2A900"          # reference-mask colour
G = 16
ORDER = ["droid", "openh_hamlyn", "bridge", "fractal", "iws_pusht", "iws_box", "iws_rope", "plan_pusht", "plan_reacher",
         "plan_tworoom"]
NAME = {"droid": "DROID", "openh_hamlyn": "Hamlyn", "bridge": "Bridge", "fractal": "RT-1", "iws_pusht": "IWS Push-T",
        "iws_box": "IWS Box", "iws_rope": "IWS Rope", "plan_pusht": "Push-T (sim)", "plan_reacher": "Reacher",
        "plan_tworoom": "Two-Room"}
MAC = {"droid": "Droid", "openh_hamlyn": "Hamlyn", "bridge": "Bridge", "fractal": "Fractal", "iws_pusht": "IwsPusht",
       "iws_box": "IwsBox", "iws_rope": "IwsRope", "plan_pusht": "PlanPusht", "plan_reacher": "PlanReacher",
       "plan_tworoom": "PlanTworoom"}
LEARN = ("shiftwm", "direct", "ar")
BRIGHT_GREEN = "#7CF2C4"


def summaries():
    out = {}
    for ds in ORDER:
        p = D / ds / "summary.json"
        out[ds] = json.loads(p.read_text()) if p.exists() else {"status": "pending"}
    return out


def smooth(score, h, w):
    """16x16 patch score -> [h,w] bicubic spline through the patch centres (the encoder squashes the frame)."""
    yc = (np.arange(G) + 0.5) * h / G - 0.5; xc = (np.arange(G) + 0.5) * w / G - 0.5
    return RectBivariateSpline(yc, xc, score.reshape(G, G), kx=3, ky=3)(np.arange(h), np.arange(w))


def overlay(ax, alpha_map, color):
    rgba = np.zeros(alpha_map.shape + (4,)); rgba[..., :3] = to_rgb(color); rgba[..., 3] = np.clip(alpha_map, 0, 1)
    ax.imshow(rgba, aspect="equal", interpolation="antialiased")


def contour(ax, field, color, lw, ls="-", level=0.5):
    if field.min() < level < field.max():
        ax.contour(field, levels=[level], colors=[color], linewidths=lw, linestyles=[ls])


def corner(ax, text, color="white", bold=False, fs=6.2):
    ax.text(0.96, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", bbox=dict(fc="#1F2A37", ec="none", alpha=0.75, pad=0.8))


def crop_box(m0, mk, h, w, margin=0.18):
    ys, xs = np.nonzero(m0 | mk)
    if len(ys) == 0:
        return 0, 0, min(h, w)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(y1 - y0, x1 - x0) * (1 + 2 * margin)
    side = int(np.clip(side, min(h, w) * 0.35, min(h, w)))
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    top = int(np.clip(cy - side / 2, 0, h - side)); left = int(np.clip(cx - side / 2, 0, w - side))
    return top, left, side


def arrows(ax, off, gate, sel, h, w, box, n=6):
    """Source (query + expected source offset) -> query, for the strongest-moving patches inside the crop."""
    top, left, side = box
    sx, sy = w / G, h / G
    mag = np.linalg.norm(off, axis=-1) * gate
    chosen = []
    for i in np.argsort(-mag):
        yi, xi = divmod(i, G)
        qx, qy = (xi + 0.5) * sx, (yi + 0.5) * sy
        if not sel[i] or mag[i] < 0.4 or not (left <= qx <= left + side and top <= qy <= top + side):
            continue
        if all(max(abs(yi - divmod(j, G)[0]), abs(xi - divmod(j, G)[1])) >= 2 for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    for i in chosen:
        yi, xi = divmod(i, G)
        qx, qy = (xi + 0.5) * sx, (yi + 0.5) * sy
        a = ax.annotate("", xy=(qx, qy), xytext=(qx + off[i, 0] * sx, qy + off[i, 1] * sy),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.18", color=mf.METHODS["shiftwm"][1],
                                        lw=1.2, shrinkA=0, shrinkB=0))
        a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])


def cell_centres(h, w):
    gy, gx = np.divmod(np.arange(G * G), G)
    return (gx + 0.5) * w / G, (gy + 0.5) * h / G


def coverage(mask):
    """Pixel mask [h,w] -> 16x16 area coverage (same area-average as scripts/v2/segments.py)."""
    h, w = mask.shape
    yi = (np.arange(G * 45) * h // (G * 45)); xi = (np.arange(G * 40) * w // (G * 40))
    up = mask[yi][:, xi].astype(float)
    return up.reshape(G, 45, G, 40).mean((1, 3)).ravel()


def cell_outline(ax, lab, h, w, color, lw):
    """Exact outline of the labelled 16x16 cells (no smoothing): boundary edges between labelled/unlabelled cells."""
    from matplotlib.collections import LineCollection
    L = lab.reshape(G, G); sx, sy = w / G, h / G; segs = []
    P = np.pad(L, 1)
    for i in range(G):
        for j in range(G):
            if not L[i, j]:
                continue
            x0, x1, y0, y1 = j * sx - 0.5, (j + 1) * sx - 0.5, i * sy - 0.5, (i + 1) * sy - 0.5
            if not P[i, j + 1]: segs.append([(x0, y0), (x1, y0)])        # noqa: E701 top
            if not P[i + 2, j + 1]: segs.append([(x0, y1), (x1, y1)])    # noqa: E701 bottom
            if not P[i + 1, j]: segs.append([(x0, y0), (x0, y1)])        # noqa: E701 left
            if not P[i + 1, j + 2]: segs.append([(x1, y0), (x1, y1)])    # noqa: E701 right
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, capstyle="projecting",
                                     path_effects=[pe.Stroke(linewidth=lw + 1.0, foreground="white"), pe.Normal()]))


def roi_row(fig, gs, Z, b, titles=True, label=None):
    names = list(Z["names"]); H = int(Z["history"]); K = Z["score"].shape[2]; tau = float(Z["tau"])
    frames, masks = Z["frames"][b], Z["masks"][b]
    obs, fut = frames[H - 1], frames[H - 1 + K]
    h, w = obs.shape[:2]
    m0, mk = masks[0].astype(bool), masks[K].astype(bool)
    learned = [n for n in LEARN if n in names]
    cx, cy = cell_centres(h, w)
    ck = coverage(mk); true_c = np.array([(ck * cx).sum() / ck.sum(), (ck * cy).sum() / ck.sum()])
    labs = {n: Z["score"][b][names.index(n)][K - 1] > 0 for n in learned}
    pred_c = {n: np.array([cx[labs[n]].mean(), cy[labs[n]].mean()]) if labs[n].any() else None for n in learned}
    errs = {n: float(Z["place"][b][names.index(n)][K - 1]) for n in learned}
    ext = np.zeros((h, w), bool)
    for pc in list(pred_c.values()) + [true_c]:
        if pc is not None:
            ext[int(np.clip(pc[1], 0, h - 1)), int(np.clip(pc[0], 0, w - 1))] = True
    top, left, side = crop_box(m0 | ext, mk, h, w)
    top_e = min(round(errs[n]) for n in learned)
    best = {n for n in learned if round(errs[n]) == top_e}              # ties at the shown precision share the mark
    axes = []
    # full frame thumbnail
    ax = fig.add_subplot(gs[0]); ax.imshow(obs, aspect="equal")
    overlay(ax, m0 * 0.35, SAMC); contour(ax, m0.astype(float), SAMC, 0.6)
    contour(ax, mk.astype(float), "white", 0.6, ls=(0, (2, 1.4)))
    dim = np.zeros((h, w, 4)); dim[..., 3] = 0.45
    dim[max(top, 0):top + side, max(left, 0):left + side, 3] = 0
    ax.imshow(dim, extent=(-0.5, w - 0.5, h - 0.5, -0.5), aspect="equal", zorder=3)
    ax.add_patch(Rectangle((left - 0.5, top - 0.5), side, side, fill=False, ec="#FFD23F", lw=1.3, zorder=4))
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5); axes.append(ax)
    if label:
        ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="white",
                fontweight="bold", bbox=dict(fc="#1F2A37", ec="none", alpha=0.75, pad=0.8))

    def crop(a):
        a.set_xlim(left - 0.5, left + side - 0.5); a.set_ylim(top + side - 0.5, top - 0.5)
    # observed crop + transport
    ax = fig.add_subplot(gs[1]); ax.imshow(obs, aspect="equal")
    overlay(ax, m0 * 0.3, SAMC); contour(ax, m0.astype(float), SAMC, 0.7)
    sel = Z["score"][b][names.index("shiftwm")][K - 1] > 0
    arrows(ax, Z["offsets"][b][K - 1], Z["gate"][b][K - 1], sel, h, w, (top, left, side))
    crop(ax); axes.append(ax)
    # true future crop
    ax = fig.add_subplot(gs[2]); ax.imshow(fut, aspect="equal")
    overlay(ax, mk * 0.38, SAMC); contour(ax, mk.astype(float), SAMC, 0.8)
    corner(ax, f"moved {100 * float(Z['motion'][b]):.0f}%")
    crop(ax); axes.append(ax)
    for c, n in enumerate(learned):
        ax = fig.add_subplot(gs[3 + c]); ax.imshow(fut, aspect="equal")
        col = mf.METHODS[n][1]
        rgba = np.zeros((G, G, 4)); rgba[..., :3] = to_rgb(col); rgba[..., 3] = labs[n].reshape(G, G) * 0.2
        ax.imshow(rgba, extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="nearest", aspect="equal")
        cell_outline(ax, labs[n], h, w, col, 1.0)
        contour(ax, mk.astype(float), "white", 0.8, ls=(0, (2.2, 1.5)))
        ax.plot(*true_c, marker="x", ms=5.5, mew=1.6, color="white", zorder=6,
                path_effects=[pe.Stroke(linewidth=2.8, foreground="#1F2A37"), pe.Normal()])
        if pred_c[n] is not None:
            pc = pred_c[n]
            if np.linalg.norm(pc - true_c) > 2:
                a = ax.annotate("", xy=tuple(true_c), xytext=tuple(pc), zorder=5,
                                arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.16", color="white", lw=1.0,
                                                shrinkA=2.5, shrinkB=3.5))
                a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.2, foreground="#1F2A37"), pe.Normal()])
            ax.plot(*pc, marker="o", ms=4.6, mfc=col, mec="white", mew=0.9, zorder=7)
        corner(ax, f"{errs[n]:.0f} px", color=BRIGHT_GREEN if n in best else "white", bold=n in best)
        crop(ax); axes.append(ax)
    for c, a in enumerate(axes):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if titles:
            t = ["observed $t$", "$t$ + transport", f"true $t{{+}}{K}$ (SAM 2.1)", "ShiftWM (ours)", "Direct", "AR"][c]
            a.set_title(t, fontsize=6.9, pad=2.2, color=mf.METHODS["shiftwm"][1] if c == 3 else mf.INK,
                        fontweight="bold")
    return axes


def place_curve(ax, P, sub, title, ylabel=True):
    iou_curve(ax, P, sub, title, ylabel, ylab="placement error (px)")


def iou_curve(ax, R, sub, title, ylabel=True, ylab="IoU"):
    K = len(R[sub]["shiftwm"]["mean"]); k = np.arange(1, K + 1)
    for n in ("persistence", "ar", "direct", "shiftwm"):
        if n not in R[sub]:
            continue
        lab, col, ls, mk = mf.METHODS[n]
        r = R[sub][n]
        ax.plot(k, r["mean"], color=col, ls=ls, marker=mk, markevery=3, ms=2.6, lw=1.7 if n == "shiftwm" else 1.1,
                label="ShiftWM" if n == "shiftwm" else lab.split(" (")[0], zorder=3 if n == "shiftwm" else 2)
        ax.fill_between(k, r["lo"], r["hi"], color=col, alpha=0.13, lw=0)
    ax.set_xticks([1, 5, K]); ax.set_xlim(0.7, K + 0.3)
    ax.set_xlabel("forecast step $k$", fontsize=6.6, labelpad=1); ax.tick_params(labelsize=6.1, pad=1.5)
    if ylabel:
        ax.set_ylabel(ylab, fontsize=6.6, labelpad=1.5)
    ax.set_title(title, fontsize=6.9, pad=2.5, loc="left")


def gain_panel(ax, S, sub="moving"):
    """Horizon-averaged IoU gain of ShiftWM over Direct and AR, paired 95% CI over episodes, per dataset."""
    done = [ds for ds in ORDER if S[ds].get("status") == "done" and "placement" in S[ds]]
    y = np.arange(len(done))[::-1]
    for off, n in ((0.14, "direct"), (-0.14, "ar")):
        col, mk = mf.METHODS[n][1], mf.METHODS[n][3]
        for yy, ds in zip(y, done):
            r = S[ds]["placement"]["px"][sub]
            if n not in r:
                continue
            d0 = r[n]["avg"]["diff_sw_minus"]                          # ShiftWM - baseline error; plot the reduction
            d = {"mean": -d0["mean"], "lo": -d0["hi"], "hi": -d0["lo"]}
            sig = d["lo"] > 0 or d["hi"] < 0
            ax.errorbar(d["mean"], yy + off, xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                        fmt=mk, ms=3.2, color=col, mfc=col if sig else "white", lw=1.0, capsize=1.5, capthick=0.8)
    ax.axvline(0, color=mf.MUTED, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([NAME[d] for d in done], fontsize=6.3)
    ax.tick_params(labelsize=6.1, pad=1.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("error removed by ShiftWM (px)", fontsize=6.6, labelpad=1)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=mf.METHODS[n][1], marker=mf.METHODS[n][3], ms=3.2, lw=0) for n in ("direct", "ar")]
    ax.legend(hs, ["vs Direct", "vs AR"], fontsize=5.6, loc="best", handletextpad=0.2, borderaxespad=0.2,
              labelspacing=0.2, title="filled: CI excludes 0", title_fontsize=5.4)


def load_examples(ds):
    return np.load(D / ds / "examples.npz", allow_pickle=True)


def dataset_figure(ds, S):
    if (D / ds / "decoded_examples.npz").exists() and S.get("decoded_segment"):
        return dataset_figure_decoded(ds, S)
    Z = load_examples(ds); R = S["results"][S["primary_labeller"]]
    asp = Z["frames"][0].shape[2] / Z["frames"][0].shape[1]
    wr = [asp, 1, 1, 1, 1, 1]
    fig = plt.figure(figsize=(5.5, 2.95))
    for r in range(2):
        g = fig.add_gridspec(1, 6, left=0.005, right=0.995, top=0.93 - r * 0.285, bottom=0.93 - r * 0.285 - 0.25,
                             wspace=0.035, width_ratios=wr)
        roi_row(fig, [g[0, i] for i in range(6)], Z, r, titles=r == 0, label=f"example {r + 1}")
    bot = fig.add_gridspec(1, 3, left=0.075, right=0.995, top=0.29, bottom=0.085, wspace=0.3, width_ratios=[1, 1, 0.75])
    P = S["placement"]["px"]
    a1 = fig.add_subplot(bot[0]); place_curve(a1, P, "all", f"all windows ({P['all']['windows']})")
    a2 = fig.add_subplot(bot[1], sharey=a1); place_curve(a2, P, "moving", f"moving windows ({P['moving']['windows']})", False)
    al = fig.add_subplot(bot[2]); al.set_axis_off()
    hs, ls = a1.get_legend_handles_labels()
    idx = [ls.index(x) for x in ("ShiftWM", "Direct", "AR", "Persistence") if x in ls]
    al.legend([hs[i] for i in idx], [ls[i] for i in idx], loc="center left", fontsize=6.4, borderaxespad=0)
    fig.savefig(mf.FIG / f"segments_{ds}.pdf"); fig.savefig(mf.FIG / f"segments_{ds}_preview.png", dpi=200)
    plt.close(fig)


WORD = {"droid": "arm", "openh_hamlyn": "instrument"}     # what the reference mask is, per dataset
EX_MIN_IOU = 0.3


def example_ok(Z, b):
    """Display rule for a saved decoded example (on top of the selection rule in segments.py): the reference mask at t
    must touch the image border (a robot arm or instrument shaft always enters the frame; a mask that does not is a
    detector error on tissue/objects that QC does not catch), and ShiftWM's decoded segment must overlap the true
    target mask (IoU >= EX_MIN_IOU)."""
    m = Z["mask_t"][b].astype(bool)
    border = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
    return bool(border and Z["iou"][b][list(Z["names"]).index("shiftwm")] >= EX_MIN_IOU)


def load_decoded(ds):
    """decoded_examples.npz restricted to the examples that pass example_ok (order kept)."""
    Z = np.load(D / ds / "decoded_examples.npz", allow_pickle=True)
    keep = [b for b in range(len(Z["episode"])) if example_ok(Z, b)]
    out = {k: (Z[k][keep] if Z[k].ndim and len(Z[k]) == len(Z["episode"]) and k != "names" else Z[k]) for k in Z.files}
    out["dropped"] = [str(Z["episode"][b]) for b in range(len(Z["episode"])) if b not in keep]
    return out


def _centroid(m):
    ys, xs = np.nonzero(m)
    return None if len(xs) == 0 else np.array([xs.mean(), ys.mean()])


def decoded_row(fig, gs, Z, b, titles, label, word="tool"):
    """observed t (tool filled; inset: full frame + crop box) | true t+K (target filled) | ShiftWM / Direct / AR: each
    method's OWN decoded k=K forecast of the SAME window, with its own SAM 2.1 segment filled in the method colour,
    the true-target outline dashed white, an arrow from the predicted to the target centroid, and the placement error.
    All five columns share one square crop around the tool (aspect preserved)."""
    names = list(Z["names"]); K = int(Z["k"])
    obs, fut, m0, mk = Z["obs"][b], Z["fut"][b], Z["mask_t"][b].astype(bool), Z["mask_true"][b].astype(bool)
    learned = [n for n in LEARN if n in names]
    segs = {n: Z["seg"][b][names.index(n)].astype(bool) for n in learned}
    # badge = distance between the centroids of the drawn masks (predicted vs. true), so number and drawing agree
    _tc = _centroid(mk)
    errs = {}
    for n_ in learned:
        _pc = _centroid(segs[n_])
        errs[n_] = float("nan") if _pc is None or _tc is None else float(np.linalg.norm(_pc - _tc))
    h, w = obs.shape[:2]
    union = m0 | mk
    for n in learned:
        union |= segs[n]
    top, left, side = crop_box(union, mk, h, w, margin=0.12)
    tc = _centroid(mk)
    finite = [round(errs[n]) for n in learned if np.isfinite(errs[n])]
    best = {n for n in learned if np.isfinite(errs[n]) and finite and round(errs[n]) == min(finite)}

    def finish(ax, title, tcol=mf.INK):
        ax.set_xlim(left - 0.5, left + side - 0.5); ax.set_ylim(top + side - 0.5, top - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if titles:
            ax.set_title(title, fontsize=6.8, pad=2.2, fontweight="bold", color=tcol)
    # observed t + inset thumbnail
    ax = fig.add_subplot(gs[0]); ax.imshow(obs, aspect="equal")
    overlay(ax, m0 * 0.45, SAMC); contour(ax, m0.astype(float), SAMC, 1.0)
    finish(ax, "observed $t$")
    iw, ih = 0.5, 0.5 * h / w
    q = union[top:top + side, left:left + side]; hh, ww = q.shape
    corners = {(0.0, 0.0): q[hh // 2:, :ww // 2], (1 - iw, 0.0): q[hh // 2:, ww // 2:],
               (0.0, 1 - ih): q[:hh // 2, :ww // 2], (1 - iw, 1 - ih): q[:hh // 2, ww // 2:]}
    pos = min(corners, key=lambda k: corners[k].mean())                  # the corner with the least tool in it
    ins = ax.inset_axes([pos[0], pos[1], iw, ih])
    ins.imshow(obs, aspect="equal"); ins.set_xticks([]); ins.set_yticks([])
    ins.add_patch(Rectangle((left - 0.5, top - 0.5), side, side, fill=False, ec="white", lw=0.7))
    for sp in ins.spines.values():
        sp.set_edgecolor("white"); sp.set_linewidth(0.6)
    if label:
        lx = 0.03 if pos[0] > 0 or pos[1] == 0.0 else 0.97
        ax.text(lx, 0.95 if pos[1] == 0.0 else 0.05, label, transform=ax.transAxes, ha="left" if lx < 0.5 else "right",
                va="top" if pos[1] == 0.0 else "bottom", fontsize=6.2, color="white",
                fontweight="bold", bbox=dict(fc="#1F2A37", ec="none", alpha=0.75, pad=0.8))
    # target
    ax = fig.add_subplot(gs[1]); ax.imshow(fut, aspect="equal")
    overlay(ax, mk * 0.45, SAMC); contour(ax, mk.astype(float), "white", 1.0)
    finish(ax, f"target: true $t{{+}}{K}$")
    for c, n in enumerate(learned):
        col = mf.METHODS[n][1]
        ax = fig.add_subplot(gs[2 + c]); ax.imshow(Z["decoded"][b][names.index(n)], aspect="equal")
        overlay(ax, segs[n] * 0.45, col); contour(ax, segs[n].astype(float), col, 1.1)
        contour(ax, mk.astype(float), "white", 0.9, ls=(0, (2.2, 1.5)))
        pc = _centroid(segs[n])
        if pc is not None and tc is not None and np.linalg.norm(pc - tc) > 4:
            a = ax.annotate("", xy=tuple(tc), xytext=tuple(pc), zorder=5,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.17", color="white", lw=1.0,
                                            shrinkA=1.5, shrinkB=1.5))
            a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.3, foreground="#1F2A37"), pe.Normal()])
        txt = f"{errs[n]:.0f} px" if np.isfinite(errs[n]) else f"no {word}"
        corner(ax, txt, color=BRIGHT_GREEN if n in best else "white", bold=n in best)
        finish(ax, ("ShiftWM (ours)" if n == "shiftwm" else mf.METHODS[n][0].split(" (")[0]),
               mf.METHODS["shiftwm"][1] if n == "shiftwm" else mf.INK)


LEGEND_LINE = "filled = predicted WHAT (segmented in each model's decoded $k{=}10$ forecast);  dashed = where it should be"


def decoded_panels(ax_m, ax_g, S, sub="moving"):
    done = [ds for ds in ORDER if S[ds].get("decoded_segment")]
    y = np.arange(len(done))[::-1]
    for off, n in ((0.2, "shiftwm"), (0.0, "direct"), (-0.2, "ar")):
        col, mk = mf.METHODS[n][1], mf.METHODS[n][3]
        for yy, ds in zip(y, done):
            r = S[ds]["decoded_segment"]["iou"][sub][n]
            ax_m.errorbar(r["mean"], yy + off, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt=mk, ms=3.2,
                          color=col, lw=1.0, capsize=1.5, capthick=0.8, label=n if yy == y[0] else None)
    ax_m.set_yticks(y); ax_m.set_yticklabels([NAME[d] for d in done], fontsize=6.3); ax_m.grid(axis="y", visible=False)
    ax_m.set_ylim(y.min() - 0.5, y.max() + 1.1); ax_m.tick_params(labelsize=6.1, pad=1.5)
    ax_m.set_xlabel("IoU of the decoded segment", fontsize=6.6, labelpad=1)
    for off, n in ((0.12, "direct"), (-0.12, "ar")):
        col, mk = mf.METHODS[n][1], mf.METHODS[n][3]
        for yy, ds in zip(y, done):
            d = S[ds]["decoded_segment"]["iou"][sub][n]["diff_sw_minus"]
            sig = d["lo"] > 0 or d["hi"] < 0
            ax_g.errorbar(100 * d["mean"], yy + off, xerr=[[100 * (d["mean"] - d["lo"])], [100 * (d["hi"] - d["mean"])]],
                          fmt=mk, ms=3.2, color=col, mfc=col if sig else "white", lw=1.0, capsize=1.5, capthick=0.8)
    ax_g.axvline(0, color=mf.MUTED, lw=0.8)
    ax_g.set_yticks(y); ax_g.set_yticklabels([]); ax_g.grid(axis="y", visible=False)
    ax_g.set_ylim(y.min() - 0.5, y.max() + 1.1); ax_g.tick_params(labelsize=6.1, pad=1.5)
    ax_g.set_xlabel("ShiftWM gain (IoU points); filled: CI excludes 0", fontsize=6.6, labelpad=1)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=mf.METHODS[n][1], marker=mf.METHODS[n][3], ms=3.2, lw=0) for n in ("shiftwm", "direct", "ar")]
    ax_m.legend(hs, ["ShiftWM", "Direct", "AR"], fontsize=5.6, loc="upper center", ncol=3, handletextpad=0.2,
                borderaxespad=0.15, columnspacing=0.8)
    ax_g.legend(hs[1:], ["vs Direct", "vs AR"], fontsize=5.6, loc="upper center", ncol=2, handletextpad=0.2,
                borderaxespad=0.15, columnspacing=0.8)


def _rows_layout(fig_w=5.5, ncol=5, wspace_in=0.04):
    col_w = (fig_w * 0.99 - (ncol - 1) * wspace_in) / ncol
    return col_w


def dataset_figure_decoded(ds, S):
    Z = load_decoded(ds)
    n = len(Z["episode"]); cw = _rows_layout()
    H_in = 0.2 + n * (cw + 0.06) + 0.18 + 1.05
    fig = plt.figure(figsize=(5.5, H_in))
    top0 = 1 - 0.2 / H_in; row_h = cw / H_in
    for r in range(n):
        t = top0 - r * (row_h + 0.06 / H_in)
        g = fig.add_gridspec(1, 5, left=0.005, right=0.995, top=t, bottom=t - row_h, wspace=0.04 / cw)
        decoded_row(fig, [g[0, i] for i in range(5)], Z, r, titles=r == 0, label=f"example {r + 1}", word=WORD[ds])
    y_leg = top0 - n * (row_h + 0.06 / H_in) + 0.02 / H_in
    fig.text(0.5, y_leg, LEGEND_LINE.replace("WHAT", WORD[ds]), ha="center", va="top", fontsize=6.4, color=mf.INK)
    P = S["placement"]["px"]
    bot = fig.add_gridspec(1, 3, left=0.075, right=0.995, top=0.8 / H_in, bottom=0.26 / H_in, wspace=0.3,
                           width_ratios=[1, 1, 0.75])
    a1 = fig.add_subplot(bot[0]); place_curve(a1, P, "all", f"patch-level placement, all ({P['all']['windows']})")
    a2 = fig.add_subplot(bot[1], sharey=a1)
    place_curve(a2, P, "moving", f"moving windows ({P['moving']['windows']})", False)
    al = fig.add_subplot(bot[2]); al.set_axis_off()
    hs, ls = a1.get_legend_handles_labels()
    idx = [ls.index(x) for x in ("ShiftWM", "Direct", "AR", "Persistence") if x in ls]
    al.legend([hs[i] for i in idx], [ls[i] for i in idx], loc="center left", fontsize=6.4, borderaxespad=0)
    fig.savefig(mf.FIG / f"segments_{ds}.pdf"); fig.savefig(mf.FIG / f"segments_{ds}_preview.png", dpi=200)
    plt.close(fig)


def main_figure(S):
    rows = [ds for ds in ("droid", "openh_hamlyn") if (D / ds / "decoded_examples.npz").exists()
            and S[ds].get("decoded_segment")]
    if not rows:
        return main_figure_placement(S)
    Zs = {ds: load_decoded(ds) for ds in rows}
    rows = [ds for ds in rows if len(Zs[ds]["episode"])]
    cw = _rows_layout(); n = len(rows)
    H_in = 0.2 + n * (cw + 0.06) + 0.18 + 1.05
    fig = plt.figure(figsize=(5.5, H_in))
    top0 = 1 - 0.2 / H_in; row_h = cw / H_in
    for r, ds in enumerate(rows):
        t = top0 - r * (row_h + 0.06 / H_in)
        g = fig.add_gridspec(1, 5, left=0.005, right=0.995, top=t, bottom=t - row_h, wspace=0.04 / cw)
        decoded_row(fig, [g[0, i] for i in range(5)], Zs[ds], 0, titles=r == 0, label=NAME[ds], word=WORD[ds])
    y_leg = top0 - n * (row_h + 0.06 / H_in) + 0.02 / H_in
    fig.text(0.5, y_leg, LEGEND_LINE.replace("WHAT", " / ".join(WORD[ds] for ds in rows)), ha="center", va="top", fontsize=6.4,
             color=mf.INK)
    bot = fig.add_gridspec(1, 2, left=0.1, right=0.99, top=0.8 / H_in, bottom=0.26 / H_in, wspace=0.12)
    am = fig.add_subplot(bot[0]); ag = fig.add_subplot(bot[1])
    decoded_panels(am, ag, S)
    am.set_title("(b) moving windows: IoU of decoded segment (95% CI)", fontsize=6.9, pad=2.5, loc="left")
    ag.set_title("(c) paired gain of ShiftWM", fontsize=6.9, pad=2.5, loc="left")
    fig.savefig(mf.FIG / "segments.pdf"); fig.savefig(mf.FIG / "segments_preview.png", dpi=220)
    plt.close(fig)


def main_figure_placement(S):
    rows = [ds for ds in ("droid", "openh_hamlyn") if S[ds].get("status") == "done"]
    if not rows:
        fig, ax = plt.subplots(figsize=(5.5, 2.6)); mf.pending(ax, "segmentation view"); fig.savefig(mf.FIG / "segments.pdf"); return
    Zs = {ds: load_examples(ds) for ds in rows}
    fig = plt.figure(figsize=(5.5, 1.2 + 0.86 * len(rows)))
    H_in = fig.get_figheight()
    row_h = 0.78 / H_in; top0 = 1 - 0.17 / H_in
    for r, ds in enumerate(rows):
        Z = Zs[ds]; asp = Z["frames"][0].shape[2] / Z["frames"][0].shape[1]
        t = top0 - r * (row_h + 0.05 / H_in)
        g = fig.add_gridspec(1, 6, left=0.005, right=0.995, top=t, bottom=t - row_h, wspace=0.035,
                             width_ratios=[asp, 1, 1, 1, 1, 1])
        roi_row(fig, [g[0, i] for i in range(6)], Z, 0, titles=r == 0, label=NAME[ds])
    bot_top = 0.92 / H_in
    bot = fig.add_gridspec(1, 3, left=0.07, right=0.995, top=bot_top, bottom=0.3 / H_in, wspace=0.42,
                           width_ratios=[1, 1, 1.15])
    for c, ds in enumerate(("droid", "openh_hamlyn")):
        if S[ds].get("status") != "done":
            continue
        a = fig.add_subplot(bot[c])
        place_curve(a, S[ds]["placement"]["px"], "moving", f"({'bc'[c]}) {NAME[ds]}: moving windows", ylabel=c == 0)
        if c == 0:
            lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.4 * (hi_ - lo_))
            a.legend(fontsize=5.3, loc="upper left", ncol=2, handlelength=1.4, borderaxespad=0.15, labelspacing=0.1,
                     handletextpad=0.3, columnspacing=0.6)
    ag = fig.add_subplot(bot[2]); gain_panel(ag, S)
    ag.set_title("(d) mean over $k$, vs Direct / AR", fontsize=6.9, pad=2.5, loc="left")
    fig.savefig(mf.FIG / "segments.pdf"); fig.savefig(mf.FIG / "segments_preview.png", dpi=220)
    plt.close(fig)


def fmt_ci(d):
    return f"{100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


def write_tex(S):
    GEN.mkdir(parents=True, exist_ok=True)
    L = ["% generated by paper/submission_folder/figures/src/make_segments.py from results/v2/analysis/segments"]
    rows = []
    tag = {"shiftwm": "SW", "direct": "Di", "ar": "AR", "persistence": "Pers", "oracle": "Oracle"}
    for ds in ORDER:
        s = S[ds]; m = "seg" + MAC[ds]
        if s.get("status") != "done":
            continue
        R = s["results"][s["primary_labeller"]]; K = s["horizon"]
        for sub, pre in (("all", ""), ("moving", "Mov")):
            for n, t in tag.items():
                if n not in R[sub]:
                    continue
                r = R[sub][n]
                L.append(f"\\newcommand{{\\{m}{pre}{t}Avg}}{{{r['avg']['mean']:.2f}}}")
                L.append(f"\\newcommand{{\\{m}{pre}{t}Five}}{{{r['mean'][4]:.2f}}}")
                L.append(f"\\newcommand{{\\{m}{pre}{t}Last}}{{{r['mean'][K - 1]:.2f}}}")
                if n != "shiftwm":
                    d = r["avg"]["diff_sw_minus"]
                    L.append(f"\\newcommand{{\\{m}{pre}Gain{t}}}{{{100 * d['mean']:+.1f}}}")
                    L.append(f"\\newcommand{{\\{m}{pre}Gain{t}CI}}{{[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]}}")
                    for kk, nm in ((4, "Five"), (K - 1, "Last")):
                        L.append(f"\\newcommand{{\\{m}{pre}Gain{t}{nm}}}{{{100 * r['diff_sw_minus']['mean'][kk]:+.1f}}}")
                        L.append(f"\\newcommand{{\\{m}{pre}Gain{t}{nm}CI}}{{[{100 * r['diff_sw_minus']['lo'][kk]:+.1f}, "
                                 f"{100 * r['diff_sw_minus']['hi'][kk]:+.1f}]}}")
        L.append(f"\\newcommand{{\\{m}Windows}}{{{R['all']['windows']:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}MovWindows}}{{{R['moving']['windows']:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}Episodes}}{{{R['all']['episodes']}}}")
        L.append(f"\\newcommand{{\\{m}Considered}}{{{s['windows_considered']:,}}}".replace(",", "{,}"))
        rej = s["windows_rejected"]
        for k_, v in rej.items():
            L.append(f"\\newcommand{{\\{m}Rej{''.join(p.capitalize() for p in k_.split('_'))}}}{{{v:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}Rejected}}{{{sum(rej.values()):,}}}".replace(",", "{,}"))
        for l, v in s["oracle_iou_k_last_by_labeller"].items():
            L.append(f"\\newcommand{{\\{m}Oracle{l.capitalize()}}}{{{v:.2f}}}")
        first = True
        for n in ("persistence", "ar", "direct", "shiftwm"):
            if n not in R["all"]:
                continue
            cells = []
            for sub, key in (("all", "avg"), ("all", 4), ("all", K - 1), ("moving", "avg"), ("moving", K - 1)):
                r = R[sub][n]
                v = r["avg"]["mean"] if key == "avg" else r["mean"][key]
                txt = f"{v:.3f}"
                if n == "shiftwm":
                    base = [b for b in ("direct", "ar") if b in R[sub]]
                    los = [R[sub][b]["avg"]["diff_sw_minus"]["lo"] if key == "avg" else R[sub][b]["diff_sw_minus"]["lo"][key]
                           for b in base]
                    if base and min(los) > 0:
                        txt = f"\\good{{{txt}}}"
                cells.append(txt)
            lab = {"persistence": "Persistence", "ar": "AR", "direct": "Direct", "shiftwm": "\\ours{}"}[n]
            name = f"\\multirow{{4}}{{*}}{{{NAME[ds]}}}" if first else ""
            if n == "shiftwm":
                gd = R["moving"]["direct"]["avg"]["diff_sw_minus"]
                ga = R["moving"]["ar"]["avg"]["diff_sw_minus"] if "ar" in R["moving"] else None
                g_ = lambda d: (f"\\good{{{100 * d['mean']:+.1f}}}" if d["lo"] > 0 else f"{100 * d['mean']:+.1f}") + \
                    f" {{\\tiny[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]}}"
                cells += [g_(gd), g_(ga) if ga else "--"]
            else:
                cells += ["", ""]
            rows.append(f"{name} & {lab} & " + " & ".join(cells) + " \\\\")
            first = False
        rows.append("\\midrule")
    # placement (centroid distance, display px): macros + table rows
    prow = []
    for ds in ORDER:
        s_ = S[ds]; m = "seg" + MAC[ds] + "Place"
        if s_.get("status") != "done" or "placement" not in s_:
            continue
        P = s_["placement"]["px"]; PP = s_["placement"]["patches"]; K = s_["horizon"]
        hd, wd = s_["placement"]["display_hw"]
        L.append(f"\\newcommand{{\\{m}Frame}}{{${wd}{{\\times}}{hd}$}}")
        for sub, pre in (("moving", ""), ("all", "All")):
            for n, t in tag.items():
                if n not in P[sub]:
                    continue
                r = P[sub][n]
                L.append(f"\\newcommand{{\\{m}{pre}{t}}}{{{r['avg']['mean']:.1f}}}")
                L.append(f"\\newcommand{{\\{m}{pre}{t}Last}}{{{r['mean'][K - 1]:.1f}}}")
                L.append(f"\\newcommand{{\\{m}{pre}{t}Patch}}{{{PP[sub][n]['avg']['mean']:.2f}}}")
                if n != "shiftwm":
                    d = r["avg"]["diff_sw_minus"]            # reduction = baseline - ShiftWM
                    L.append(f"\\newcommand{{\\{m}{pre}Red{t}}}{{{-d['mean']:+.1f}}}")
                    L.append(f"\\newcommand{{\\{m}{pre}Red{t}CI}}{{[{-d['hi']:+.1f}, {-d['lo']:+.1f}]}}")
                    dl = r["diff_sw_minus"]
                    L.append(f"\\newcommand{{\\{m}{pre}Red{t}Last}}{{{-dl['mean'][K - 1]:+.1f}}}")
                    L.append(f"\\newcommand{{\\{m}{pre}Red{t}LastCI}}{{[{-dl['hi'][K - 1]:+.1f}, {-dl['lo'][K - 1]:+.1f}]}}")
        first = True
        for n in ("persistence", "ar", "direct", "shiftwm"):
            if n not in P["all"]:
                continue
            cells = []
            for sub, key in (("all", "avg"), ("all", K - 1), ("moving", "avg"), ("moving", 4), ("moving", K - 1)):
                r = P[sub][n]
                v = r["avg"]["mean"] if key == "avg" else r["mean"][key]
                txt = f"{v:.1f}"
                if n == "shiftwm":
                    his = [P[sub][b_]["avg"]["diff_sw_minus"]["hi"] if key == "avg" else P[sub][b_]["diff_sw_minus"]["hi"][key]
                           for b_ in ("direct", "ar") if b_ in P[sub]]
                    if his and max(his) < 0:
                        txt = f"\\good{{{txt}}}"
                cells.append(txt)
            lab = {"persistence": "Persistence", "ar": "AR", "direct": "Direct", "shiftwm": "\\ours{}"}[n]
            if n == "shiftwm":
                f_ = lambda d: (f"\\good{{{-d['mean']:+.1f}}}" if d["hi"] < 0 else f"{-d['mean']:+.1f}") + \
                    f" {{\\tiny[{-d['hi']:+.1f}, {-d['lo']:+.1f}]}}"
                cells += [f_(P["moving"]["direct"]["avg"]["diff_sw_minus"]), f_(P["moving"]["ar"]["avg"]["diff_sw_minus"])]
            else:
                cells += ["", ""]
            prow.append((f"\\multirow{{4}}{{*}}{{{NAME[ds]}}}" if first else "") + f" & {lab} & " + " & ".join(cells) + " \\\\")
            first = False
        prow.append("\\midrule")
    while prow and prow[-1] == "\\midrule":
        prow.pop()
    while rows and rows[-1] == "\\midrule":
        rows.pop()
    (GEN / "segments_place_rows.tex").write_text("\n".join(prow) + "\n")
    # decoded-segment metric (segment the tool/arm in each method's decoded k=K forecast)
    dtag = {"shiftwm": "SW", "direct": "Di", "ar": "AR", "decoded_truth": "Truth"}
    for ds in ORDER:
        Dd = S[ds].get("decoded_segment")
        if not Dd:
            continue
        m = "seg" + MAC[ds] + "Dec"
        for sub, pre in (("moving", ""), ("all", "All")):
            for n, t in dtag.items():
                r = Dd["iou"][sub][n]
                L.append(f"\\newcommand{{\\{m}{pre}{t}}}{{{r['mean']:.2f}}}")
                L.append(f"\\newcommand{{\\{m}{pre}{t}CI}}{{[{r['lo']:.2f}, {r['hi']:.2f}]}}")
                if n in ("direct", "ar"):
                    d = r["diff_sw_minus"]
                    L.append(f"\\newcommand{{\\{m}{pre}Gain{t}}}{{{100 * d['mean']:+.1f}}}")
                    L.append(f"\\newcommand{{\\{m}{pre}Gain{t}CI}}{{[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]}}")
                    pl = Dd["place_px"][sub][n]["diff_sw_minus"]
                    L.append(f"\\newcommand{{\\{m}{pre}PlaceRed{t}}}{{{-pl['mean']:+.1f}}}")
                    L.append(f"\\newcommand{{\\{m}{pre}PlaceRed{t}CI}}{{[{-pl['hi']:+.1f}, {-pl['lo']:+.1f}]}}")
                L.append(f"\\newcommand{{\\{m}{pre}Place{t}}}{{{Dd['place_px'][sub][n]['mean']:.1f}}}")
            L.append(f"\\newcommand{{\\{m}{pre}Windows}}{{{Dd['iou'][sub]['windows']:,}}}".replace(",", "{,}"))
        for n, t in dtag.items():
            L.append(f"\\newcommand{{\\{m}Det{t}}}{{{100 * Dd['detection_rate'][n]:.0f}}}")
    # robustness check (scripts/v2/segments_border_check.py): windows whose reference mask at t touches the frame border
    for ds in ORDER:
        fb = D / ds / "border_restricted.json"
        if not fb.exists():
            continue
        B = json.loads(fb.read_text()); m = "seg" + MAC[ds] + "Border"; r = B["border"]["moving"]
        L.append(f"\\newcommand{{\\{m}Windows}}{{{B['windows_border']:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}Total}}{{{B['windows_total']:,}}}".replace(",", "{,}"))
        nx = B["windows_total"] - B["windows_border"]
        L.append(f"\\newcommand{{\\{m}Excluded}}{{{nx} of {B['windows_total']:,} ({100 * nx / B['windows_total']:.0f}\\%)}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}Moving}}{{{B['moving_border']:,}}}".replace(",", "{,}"))
        for n, t in (("shiftwm", "SW"), ("direct", "Di"), ("ar", "AR")):
            L.append(f"\\newcommand{{\\{m}Place{t}}}{{{r['place_px'][n]['mean']:.1f}}}")
            L.append(f"\\newcommand{{\\{m}IoU{t}}}{{{r['iou'][n]['mean']:.3f}}}")
            if n != "shiftwm":
                d = r["iou"][n]["diff_sw_minus"]; pl = r["place_px"][n]["diff_sw_minus"]
                L.append(f"\\newcommand{{\\{m}Gain{t}}}{{{100 * d['mean']:+.1f}}}")
                L.append(f"\\newcommand{{\\{m}Gain{t}CI}}{{[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]}}")
                L.append(f"\\newcommand{{\\{m}Red{t}}}{{{-pl['mean']:+.1f}}}")
                L.append(f"\\newcommand{{\\{m}Red{t}CI}}{{[{-pl['hi']:+.1f}, {-pl['lo']:+.1f}]}}")
    (GEN / "segments_numbers.tex").write_text("\n".join(L) + "\n")
    if rows and rows[-1].endswith("\\midrule"):
        rows[-1] = rows[-1][: -len(" \\midrule")]
    (GEN / "segments_rows.tex").write_text("\n".join(rows) + "\n")
    figs = []
    for ds in ORDER:
        if S[ds].get("status") != "done":
            continue
        if S[ds].get("decoded_segment") and (D / ds / "decoded_examples.npz").exists():
            Zd = np.load(D / ds / "decoded_examples.npz", allow_pickle=True)
            met = bool(Zd["thresholds_met"])
            rule = ("moving windows (distinct episodes) where the \\ours{} segment matches the target (IoU $\\ge 0.6$ or "
                    "placement in its best quartile) while Direct and AR both fail (IoU $\\le 0.3$ or placement error "
                    "$\\ge 2\\times$ that of \\ours{}), ranked by the IoU margin over the better baseline"
                    if met else "the moving windows with the largest IoU margin of \\ours{} over the better baseline "
                    "(no window met the success/failure thresholds)")
            figs.append("\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=\\linewidth]{segments_%s.pdf}\n"
                        "  \\caption{\\textbf{Decoded forecasts, %s} (target: %s). Each column: that method's own $k{=}10$ forecast of the same window, decoded and segmented; dashed: true mask. Selection rule: \\cref{app:segments}.%s}\n  \\label{fig:segments-%s}\n\\end{figure}"
                        % (ds, NAME[ds], S[ds]["target"], "" if rule else "", ds.replace("_", "-")))
            continue
        R = S[ds]["results"][S[ds]["primary_labeller"]]
        figs.append("\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=\\linewidth]{segments_%s.pdf}\n"
                    "  \\caption{\\textbf{Arm placement, %s} (target: %s; %d windows, %d episodes). Same true frame $t{+}10$ in every column; examples selected as in \\cref{app:segments}. Bottom: placement error vs.\\ horizon.}\n"
                    "  \\label{fig:segments-%s}\n\\end{figure}" % (ds, NAME[ds], S[ds]["target"], R["all"]["windows"],
                                                                   R["all"]["episodes"], ds.replace("_", "-")))
    (GEN / "segments_figs.tex").write_text("\n".join(figs) + "\n")
    # caption of fig:segments and the decoded-segment paragraph follow whichever figure style was drawn
    dec_ok = [ds for ds in ("droid", "openh_hamlyn") if S[ds].get("decoded_segment")
              and (D / ds / "decoded_examples.npz").exists()]
    if dec_ok:
        dropped = {ds: load_decoded(ds)["dropped"] for ds in dec_ok}
        nd = {ds: len(v) for ds, v in dropped.items() if v}
        drop_note = (" This removes " + " and ".join(f"{v} of the {NAME[ds]} windows" for ds, v in nd.items())
                     + " (reference mask on tissue); the saved outputs do not allow re-selecting replacements, so "
                     "fewer examples are shown there.") if nd else ""
        cap = ("\\textbf{Where does each model put the arm or instrument?} (a) Each model's own decoded $k{=}10$ "
               "forecast of the same window, arm (DROID) or instrument (Hamlyn) segmented by SAM~2.1 (filled) vs.\\ the true target (dashed); zoomed, inset: full frame. "
               "(b,c) Decoded-segment IoU on moving windows and paired gain of \\ours{}.")
        par = ["\\paragraph{Decoded forecasts.} To show the placement in pixels we decode every $k{=}10$ forecast with the "
               "trained feature decoder and segment the arm or instrument in the \\emph{decoded} image with one rule for all models "
               "(top Grounding~DINO box for the dataset prompt $\\rightarrow$ SAM~2.1; no segment if the detection score "
               "is below 0.3), then compare it with the SAM~2.1 mask of the true frame (patch-level IoU)."]
        for ds in dec_ok:
            Dd = S[ds]["decoded_segment"]; r = Dd["iou"]["moving"]
            f = lambda d: f"{100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
            par.append(f"On {Dd['iou']['moving']['windows']} moving {NAME[ds]} windows the decoded-segment IoU is "
                       f"{r['shiftwm']['mean']:.2f} for \\ours{{}}, {r['direct']['mean']:.2f} for Direct and "
                       f"{r['ar']['mean']:.2f} for AR (decoded true features: {r['decoded_truth']['mean']:.2f}); paired "
                       f"gain of \\ours{{}} {f(r['direct']['diff_sw_minus'])} IoU points vs.\\ Direct and "
                       f"{f(r['ar']['diff_sw_minus'])} vs.\\ AR (detection rate {100 * Dd['detection_rate']['shiftwm']:.0f}\\%/"
                       f"{100 * Dd['detection_rate']['direct']:.0f}\\%/{100 * Dd['detection_rate']['ar']:.0f}\\%).")
        par.append("The examples in \\cref{fig:segments} are selected by a fixed rule that favours \\ours{}: moving windows "
                   "where its decoded segment matches the target (IoU $\\ge 0.6$ or placement error in its best quartile) "
                   "while Direct and AR both miss it (IoU $\\le 0.3$ or at least twice the placement error), ranked by the "
                   "IoU margin over the better baseline, one per episode, top two (a third if its margin is within 10\\% of "
                   "the second). A selected window is shown only if its reference mask at $t$ touches the image border (an "
                   "arm or instrument shaft always enters the frame; a mask that does not is a detector error on tissue "
                   "or objects that QC does not catch) and the decoded segment of \\ours{} overlaps the true mask (IoU "
                   "$\\ge 0.3$)." + drop_note + " The examples illustrate the effect; the averages above measure it. "
                   "Every model is shown on the same window and the same frame.")
        (GEN / "segments_decoded_text.tex").write_text(" ".join(par) + "\n")
    else:
        cap = ("\\textbf{Where does each model put the arm?} (a) Predicted arm region (outline), its centroid (dot) and the "
               "true centroid ($\\times$) on the same true frame. (b,c) Placement error vs.\\ horizon. (d) Error removed "
               "by \\ours{}.")
        (GEN / "segments_decoded_text.tex").write_text("\n")
    (GEN / "segments_caption.tex").write_text(cap + "\n")
    with open(GEN / "segments_numbers.tex", "a") as fh:
        fh.write("\\newcommand{\\segCaption}{" + cap + "}\n")


def main():
    S = summaries()
    write_tex(S)
    for ds in ORDER:
        if S[ds].get("status") == "done":
            dataset_figure(ds, S[ds])
    main_figure(S)
    print("wrote segments:", [ds for ds in ORDER if S[ds].get("status") == "done"])


def examples_figure(datasets=("droid", "openh_hamlyn")):
    """Appendix qualitative figure: only the decoded example rows (robot arm on DROID, instruments on Hamlyn), no
    curves; the aggregate placement / IoU results are in fig:segments."""
    rows = []
    for ds in datasets:
        Z = load_decoded(ds)
        names = list(Z["names"])
        for r in range(len(Z["episode"])):
            tc = _centroid(Z["mask_true"][r].astype(bool))
            d = {}
            for a in ("shiftwm", "direct", "ar"):
                pc = _centroid(Z["seg"][r][names.index(a)].astype(bool))
                d[a] = np.inf if pc is None or tc is None else float(np.linalg.norm(pc - tc))
            if d["shiftwm"] < min(d["direct"], d["ar"]):      # keep windows where ShiftWM places the target best
                rows.append((ds, Z, r))
    n = len(rows); cw = _rows_layout()
    H_in = 0.2 + n * (cw + 0.06) + 0.2
    fig = plt.figure(figsize=(5.5, H_in))
    top0 = 1 - 0.2 / H_in; row_h = cw / H_in
    lab = {"droid": "DROID", "openh_hamlyn": "Hamlyn"}
    cnt = {}
    for i, (ds, Z, r) in enumerate(rows):
        cnt[ds] = cnt.get(ds, 0) + 1
        t = top0 - i * (row_h + 0.06 / H_in)
        g = fig.add_gridspec(1, 5, left=0.005, right=0.995, top=t, bottom=t - row_h, wspace=0.04 / cw)
        decoded_row(fig, [g[0, j] for j in range(5)], Z, r, titles=i == 0, label=f"{lab[ds]} {cnt[ds]}", word=WORD[ds])
    y_leg = top0 - n * (row_h + 0.06 / H_in) + 0.02 / H_in
    fig.text(0.5, y_leg, "filled = predicted arm or instrument (segmented in each model's decoded $k=10$ forecast);  "
             "dashed = where it should be", ha="center", va="top", fontsize=6.4, color=mf.INK)
    fig.savefig(mf.FIG / "segments_examples.pdf"); fig.savefig(mf.FIG / "segments_examples_preview.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
