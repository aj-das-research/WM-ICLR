"""'Where and why ShiftWM wins' figure from results/v2/analysis/interpret (scripts/v2/interpret.py), held-out DROID, k=10.

(a) Two windows (rule in interpret.py: largest advantage over Direct among high-motion windows). Per window: the true
    future frame with the region of largest error boxed, then on that crop the per-patch error of Direct and of
    ShiftWM (same colour scale) and the win map (green: ShiftWM lower error).
(b) Causal knockout of transport: error before -> after setting the gate to 0, per region.
(c) Gain over Direct per decile of true motion.  (d) Action steering: change of the transport field, moving vs static.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/interpret"
GREEN, RED, GREY, BLUE = mf.METHODS["shiftwm"][1], "#C0392B", "#B8BEC7", mf.METHODS["direct"][1]
ROI = "#FFD23F"
ERR = LinearSegmentedColormap.from_list("err", [(1, 0.25, 0.1, 0.0), (1, 0.25, 0.1, 0.55), (0.75, 0.0, 0.1, 0.92)])


def frame_axes(ax, color="#C9CED6", lw=0.6):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(lw)


def corner(ax, text, fc):
    ax.text(0.96, 0.06, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="white",
            fontweight="bold", bbox=dict(fc=fc, ec="none", alpha=0.92, pad=0.9))


def win_rgba(win, lim):
    a = np.clip(np.abs(win) / lim, 0, 1) * 0.85
    col = np.where(win[..., None] > 0, np.array(to_rgb(GREEN)), np.array(to_rgb(BLUE)))
    return np.concatenate([col, a[..., None]], -1)


def gray(fr):
    return (fr.mean(-1, keepdims=True).repeat(3, -1) * 0.55 + fr * 0.45).astype(np.uint8)


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 2.6)); mf.pending(ax, "interpretability"); fig.savefig(mf.FIG / "interpret.pdf"); return
    S = json.loads((D / "summary.json").read_text()); Z = np.load(D / "droid.npz", allow_pickle=True)
    fig = plt.figure(figsize=(5.5, 2.3))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.2, left=0.01, right=0.99, top=0.87, bottom=0.13)
    left = outer[0, 0].subgridspec(2, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.05, hspace=0.1)
    E_d, E_s = Z["err_di"][:2], Z["err_sw"][:2]
    vmax = float(np.quantile(np.concatenate([E_d.ravel(), E_s.ravel()]), 0.97))
    lim = float(np.quantile(np.abs(E_d - E_s), 0.97))
    for r in range(2):
        ep, st = str(Z["episode"][r]), int(Z["start"][r])
        fut = mf.droid_frames(ep, steps=(st + 2 + 10,))[0]; h, w = fut.shape[:2]
        ed, es = E_d[r].reshape(16, 16), E_s[r].reshape(16, 16)
        gy, gx = np.unravel_index(np.argmax(ed + es), ed.shape)
        cx0, cy0 = int(np.clip(gx - 3, 0, 9)), int(np.clip(gy - 3, 0, 9))
        box = (cx0 * w / 16 - 0.5, cy0 * h / 16 - 0.5, 7 * w / 16, 7 * h / 16)
        ext = (-0.5, w - 0.5, h - 0.5, -0.5)
        ax = fig.add_subplot(left[r, 0]); ax.imshow(fut, aspect="equal", interpolation="lanczos")
        ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, ec=ROI, lw=1.3)); frame_axes(ax)
        if r == 0:
            ax.set_title("true future $t{+}10$", fontsize=6.5, pad=2)

        def crop(a):
            a.set_xlim(box[0], box[0] + box[2]); a.set_ylim(box[1] + box[3], box[1])
        yy, xx = slice(cy0, cy0 + 7), slice(cx0, cx0 + 7)
        for c, (name, e, col) in enumerate((("Direct", ed, BLUE), ("ShiftWM", es, GREEN))):
            ax = fig.add_subplot(left[r, 1 + c]); ax.imshow(gray(fut), aspect="equal", interpolation="lanczos")
            ax.imshow(np.clip(e / vmax, 0, 1), cmap=ERR, vmin=0, vmax=1, extent=ext, interpolation="bicubic", aspect="equal")
            crop(ax); frame_axes(ax, GREEN if name == "ShiftWM" else ROI, 1.5 if name == "ShiftWM" else 1.1)
            corner(ax, f"{float(e[yy, xx].mean()):.2f}", col)
            if r == 0:
                ax.set_title(f"{name} error", fontsize=6.5, pad=2, color=col, fontweight="bold")
        ax = fig.add_subplot(left[r, 3]); ax.imshow(gray(fut), aspect="equal", interpolation="lanczos")
        ax.imshow(win_rgba(ed - es, lim), extent=ext, interpolation="bicubic", aspect="equal")
        crop(ax); frame_axes(ax, ROI, 1.1)
        corner(ax, f"{100 * float(((ed - es) > 0)[yy, xx].mean()):.0f}% green", GREEN)
        if r == 0:
            ax.set_title("win map", fontsize=6.5, pad=2)
    fig.text(0.01, 0.945, "(a) Where ShiftWM beats Direct (held-out DROID, zoom on the moving region)", fontsize=7.2,
             fontweight="bold", color=mf.INK)
    fig.legend(handles=[Patch(color=(0.85, 0.1, 0.1), alpha=0.75, label="forecast error (shared scale)"),
                        Patch(color=GREEN, alpha=0.85, label="ShiftWM lower error"), Patch(color=BLUE, alpha=0.85, label="Direct lower error")],
               loc="lower left", bbox_to_anchor=(0.01, -0.005), ncol=3, fontsize=5.7, frameon=False, handlelength=1.2, columnspacing=1.2)
    right = outer[0, 1].subgridspec(3, 1, hspace=1.15, height_ratios=[1, 1, 0.8])
    ax = fig.add_subplot(right[0])
    regs = [("moving", "moving"), ("high gate", "highgate"), ("static", "static")]
    for y, (lab, key) in zip([2, 1, 0], regs):
        b, k_ = S[f"base_{key}"], S[f"ko_{key}"]; inc = S[f"knockout_increase_{key}"]
        col = GREEN if inc > 5 else GREY
        ax.plot([b, k_], [y, y], color=col, lw=1.6, alpha=0.6, solid_capstyle="round")
        ax.scatter([b], [y], s=18, c="white", edgecolors=col, linewidths=1.1, zorder=3)
        ax.scatter([k_], [y], s=18, c=col, zorder=3)
        ax.text(max(b, k_) + 0.03, y, f"+{inc:.0f}%", va="center", fontsize=5.9, color=col if inc > 5 else mf.INK,
                fontweight="bold" if inc > 5 else "normal")
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in regs], fontsize=5.9); ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0, max(S["ko_moving"], S["ko_highgate"]) * 1.35); ax.tick_params(labelsize=5.6, length=2); ax.grid(axis="y", visible=False)
    ax.set_xlabel("error with $\\circ$ / without $\\bullet$ transport", fontsize=5.6, labelpad=1)
    ax.set_title("(b) remove transport", fontsize=6.6, pad=3, loc="left", x=-0.3)
    ax = fig.add_subplot(right[1])
    g = np.array(S["gain_vs_direct_by_decile"]); x = np.arange(1, 11)
    cols = [plt.cm.Greens(0.45 + 0.5 * i / 9) for i in range(10)]
    ax.vlines(x, 0, g, colors=cols, lw=1.6); ax.scatter(x, g, s=16, c=cols, zorder=3)
    ax.axhline(0, color=mf.INK, lw=0.5)
    ax.set_xticks([1, 10]); ax.set_xticklabels(["static", "fast"], fontsize=5.8); ax.tick_params(labelsize=5.6, length=2)
    ax.set_ylim(0, g.max() * 1.35); ax.set_ylabel("gain (%)", fontsize=5.8, labelpad=1); ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.97, "> 0 in all 10 deciles", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=GREEN, fontweight="bold")
    ax.set_title("(c) gain over Direct by motion", fontsize=6.6, pad=3, loc="left", x=-0.3)
    ax = fig.add_subplot(right[2])
    st_ = [S["steer_static"], S["steer_moving"]]
    ax.barh([0, 1], st_, color=[GREY, GREEN], height=0.55)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["static", "moving"], fontsize=5.9)
    ax.text(st_[1] * 1.03, 1, f"{S['steer_ratio_moving_over_static']:.1f}$\\times$", va="center", fontsize=6.2, color=GREEN, fontweight="bold")
    ax.set_xlim(0, st_[1] * 1.35); ax.tick_params(labelsize=5.6, length=2); ax.grid(axis="y", visible=False)
    ax.set_xlabel("transport change, other actions (patches)", fontsize=5.4, labelpad=1)
    ax.set_title("(d) actions steer moving parts", fontsize=6.6, pad=3, loc="left", x=-0.3)
    fig.savefig(mf.FIG / "interpret.pdf"); fig.savefig(mf.FIG / "interpret_preview.png", dpi=200)
    print("wrote interpret")


if __name__ == "__main__":
    main()
