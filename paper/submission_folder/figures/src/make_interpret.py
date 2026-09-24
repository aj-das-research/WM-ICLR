"""'Where and why ShiftWM wins' figure from results/v2/analysis/interpret (scripts/v2/interpret.py).

(a) Two held-out DROID windows (largest and 75th-percentile motion): future frame with the per-patch win map
    (Direct error minus ShiftWM error at k=10; green = ShiftWM better), a zoom on the region of largest motion,
    and the learned gate.  (b) Causal knockout of transport.  (c) Gain over Direct per motion decile.
    (d) Action steering: change of the transport field under counterfactual actions, moving vs. static patches.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/interpret"
GREEN, RED, GREY = mf.METHODS["shiftwm"][1], "#C0392B", "#B8BEC7"


def win_rgba(win, lim):
    """Green where ShiftWM is better, red where worse; transparent where the two methods are equal."""
    from matplotlib.colors import to_rgb
    a = np.clip(np.abs(win) / lim, 0, 1) * 0.8
    col = np.where(win[..., None] > 0, np.array(to_rgb(GREEN)), np.array(to_rgb(RED)))
    return np.concatenate([col, a[..., None]], -1)


def up(m, h, w):
    """Smooth upsampling of a 16x16 map to the frame size (bicubic via imshow extent handled by caller)."""
    return m.reshape(16, 16)


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 2.6)); mf.pending(ax, "interpretability"); fig.savefig(mf.FIG / "interpret.pdf"); return
    S = json.loads((D / "summary.json").read_text()); Z = np.load(D / "droid.npz", allow_pickle=True)
    fig = plt.figure(figsize=(5.5, 2.55))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.3, 0.9, 0.9, 0.06, 1.45], height_ratios=[1, 1], wspace=0.06, hspace=0.32,
                          left=0.01, right=0.99, top=0.9, bottom=0.12)
    win_all = Z["err_di"] - Z["err_sw"]
    lim = float(np.quantile(np.abs(win_all[:2]), 0.98))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("win", [RED, "#FFFFFF", GREEN])
    for r in range(2):
        ep, st = str(Z["episode"][r]), int(Z["start"][r])
        fut = mf.droid_frames(ep, steps=(st + 2 + 10,))[0]
        h, w = fut.shape[:2]
        win = win_all[r].reshape(16, 16); gate = Z["gate"][r].reshape(16, 16)
        chg_map = Z["err_di"][r].reshape(16, 16)
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(fut, aspect="equal")
        ax.imshow(win_rgba(win, lim), extent=(-0.5, w - 0.5, h - 0.5, -0.5), aspect="equal", interpolation="bicubic")
        gy, gx = np.unravel_index(np.argmax(chg_map), chg_map.shape)
        cx0 = int(np.clip(gx - 3, 0, 9)); cy0 = int(np.clip(gy - 3, 0, 9))
        ax.add_patch(Rectangle((cx0 * w / 16, cy0 * h / 16), 7 * w / 16, 7 * h / 16, fill=False, ec="#FFD23F", lw=1.2))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("future frame + win map", fontsize=6.5, pad=2)
        # zoom
        az = fig.add_subplot(gs[r, 1]); az.imshow(fut, aspect="equal")
        az.imshow(win_rgba(win, lim), extent=(-0.5, w - 0.5, h - 0.5, -0.5), aspect="equal", interpolation="bicubic")
        az.set_xlim(cx0 * w / 16, (cx0 + 7) * w / 16); az.set_ylim((cy0 + 7) * h / 16, cy0 * h / 16)
        az.set_xticks([]); az.set_yticks([])
        for sp in az.spines.values():
            sp.set_edgecolor("#FFD23F"); sp.set_linewidth(1.2)
        if r == 0:
            az.set_title("zoom: moving region", fontsize=6.5, pad=2)
        ag = fig.add_subplot(gs[r, 2]); ag.imshow(fut, aspect="equal")
        ag.imshow(gate, cmap="viridis", alpha=0.6, extent=(-0.5, w - 0.5, h - 0.5, -0.5), aspect="equal", interpolation="bicubic", vmin=0, vmax=1)
        ag.set_xticks([]); ag.set_yticks([])
        if r == 0:
            ag.set_title("learned gate", fontsize=6.5, pad=2)
        share = 100 * float((win > 0).mean())
        ax.text(0.02, 0.04, f"ShiftWM better on {share:.0f}% of patches", transform=ax.transAxes, fontsize=5.6, color="white",
                fontweight="bold", bbox=dict(fc="#1F2A37", ec="none", alpha=0.6, pad=1))
    # (b) knockout, (c) motion deciles, (d) steering -- stacked in the right column
    sub = gs[:, 4].subgridspec(3, 1, hspace=0.95)
    ax = fig.add_subplot(sub[0])
    ko = [S["knockout_increase_moving"], S["knockout_increase_highgate"], S["knockout_increase_static"]]
    ax.barh([2, 1, 0], ko, color=[GREEN, GREEN, GREY], height=0.62)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["moving", "high gate", "static"], fontsize=6.2)
    for y, v in zip([2, 1, 0], ko):
        ax.text(v + 1, y, f"+{v:.0f}%", va="center", fontsize=6.2, color=GREEN if v > 5 else mf.INK, fontweight="bold" if v > 5 else "normal")
    ax.set_xlim(0, max(ko) * 1.35); ax.tick_params(labelsize=6); ax.grid(axis="y", visible=False)
    ax.set_title("(b) error increase without transport", fontsize=6.6, pad=2, loc="left")
    ax = fig.add_subplot(sub[1])
    g = np.array(S["gain_vs_direct_by_decile"])
    ax.bar(np.arange(1, 11), g, color=[GREEN if v > 0 else RED for v in g], width=0.75)
    ax.set_xticks([1, 5, 10]); ax.set_xticklabels(["static", "", "fast"], fontsize=6); ax.tick_params(labelsize=6)
    ax.set_ylabel("%", fontsize=6, labelpad=1); ax.grid(axis="x", visible=False)
    ax.set_title("(c) gain over Direct, by motion decile", fontsize=6.6, pad=2, loc="left")
    ax = fig.add_subplot(sub[2])
    st_ = [S["steer_moving"], S["steer_static"]]
    ax.barh([1, 0], st_, color=[GREEN, GREY], height=0.62)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["moving", "static"], fontsize=6.2)
    ax.text(st_[0] * 1.02, 1, f"{S['steer_ratio_moving_over_static']:.1f}$\\times$", va="center", fontsize=6.4, color=GREEN, fontweight="bold")
    ax.set_xlim(0, st_[0] * 1.35); ax.tick_params(labelsize=6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("change of transport (patches)", fontsize=6, labelpad=1)
    ax.set_title("(d) actions steer moving parts", fontsize=6.6, pad=2, loc="left")
    fig.text(0.01, 0.955, "(a) Where ShiftWM beats Direct on held-out DROID (green: lower error)", fontsize=7.4, fontweight="bold", color=mf.INK)
    fig.savefig(mf.FIG / "interpret.pdf"); fig.savefig(mf.FIG / "interpret_preview.png", dpi=200)
    print("wrote interpret")


if __name__ == "__main__":
    main()
