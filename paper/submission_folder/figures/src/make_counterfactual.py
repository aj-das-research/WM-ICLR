"""Counterfactual-actions figure from results/v2/analysis/counterfactual (scripts/v2/counterfactual.py), held-out DROID.

(a) One fixed history (the teaser window) forecast under four real command sequences: the true one and three
    re-anchored alternatives from other test episodes (left / right / near-static, fixed rule). Per plan: the commanded
    end-effector path (top view, robot frame), ShiftWM's transport field at k=10 over the observed frame, and its k=10
    forecast (PCA->RGB shared with the observed and true grids, as Fig. 11).
(b) Over all test windows: how much the forecast changes when the plan is swapped, on moving and static patches.
(c) Error increase on moving patches when the plan is swapped (the true plan should give the better forecast).
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402
from make_mechanism import pca_rgb  # noqa: E402

D = mf.RES / "analysis/counterfactual"
GREEN, BLUE, ORANGE, GREY = mf.METHODS["shiftwm"][1], mf.METHODS["direct"][1], mf.METHODS["ar"][1], "#8C95A1"
K = 10
LABEL = lambda n: "true plan" if n == "true" else f"alt.: {n}"
KMAP = plt.get_cmap("viridis")


def frame_axes(ax, edge="#C9CED6", lw=0.6):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(lw)


NW, NH = 320, 180                                   # native DROID frame size: frames and grids are shown at 16:9


def path_panel(ax, disp, lim):
    """Top view of the commanded end-effector path (robot frame: +x forward = up, +y left = left), in cm."""
    d = np.concatenate([np.zeros((1, 3)), disp]) * 100
    ax.plot(-d[:, 1], d[:, 0], color=mf.INK, lw=0.7, zorder=2)
    ax.scatter(-d[1:, 1], d[1:, 0], c=np.arange(1, K + 1), cmap=KMAP, vmin=1, vmax=K, s=6, zorder=3, edgecolors="none")
    ax.scatter([0], [0], marker="o", s=9, facecolors="white", edgecolors=mf.INK, linewidths=0.6, zorder=4)
    ax.set_box_aspect(NH / NW); ax.set_xlim(-lim * NW / NH, lim * NW / NH); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#E6E8EB", lw=0.5, zorder=0); ax.axvline(0, color="#E6E8EB", lw=0.5, zorder=0)
    frame_axes(ax)
    ax.text(0.03, 0.06, f"$\\Delta z$ {d[-1, 2]:+.0f} cm", transform=ax.transAxes, fontsize=4.8, color=mf.INK)


def show(ax, im, interp="lanczos"):
    ax.imshow(im, extent=(0, NW, NH, 0), aspect="auto", interpolation=interp)
    ax.set_box_aspect(NH / NW); frame_axes(ax)


def arrows(ax, img, dx, dy, gate, g=16, thr=0.5):
    """Gate-weighted content motion -g*o (patches) at k=10 where it exceeds `thr` patches, over the observed frame."""
    show(ax, img * 0.8 + 0.2 * 0.35)
    xs, ys = (np.arange(g) + 0.5) * NW / g, (np.arange(g) + 0.5) * NH / g
    X, Y = np.meshgrid(xs, ys)
    mu, mv = (-dx * gate).reshape(g, g), (-dy * gate).reshape(g, g)
    m = np.hypot(mu, mv) > thr
    ax.quiver(X[m], Y[m], mu[m] * NW / g, mv[m] * NH / g, color="#FFE066", angles="xy", scale_units="xy", scale=1,
              width=0.012, headwidth=3.2, headlength=3.2, headaxislength=2.8, edgecolor=mf.INK, linewidth=0.25)
    ax.set_xlim(0, NW); ax.set_ylim(NH, 0)
    return int(m.sum())


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 2.4)); mf.pending(ax, "counterfactual"); fig.savefig(mf.FIG / "counterfactual.pdf"); return
    S = json.loads((D / "summary.json").read_text()); X = np.load(D / "example.npz", allow_pickle=True)
    names = [str(n) for n in X["names"]]
    fig = plt.figure(figsize=(5.5, 2.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 4.1], left=0.045, right=0.725, top=0.865, bottom=0.075, wspace=0.13)
    g0 = outer[0, 0].subgridspec(3, 1, hspace=0.12); g1 = outer[0, 1].subgridspec(3, 4, wspace=0.05, hspace=0.12)
    z0, true = X["z0"], X["true"][K - 1]
    fit = np.concatenate([X["true"].reshape(-1, z0.shape[-1]), z0])
    imgs = pca_rgb(fit, z0, true, *[X["shiftwm"][p, K - 1] for p in range(len(names))])
    obs = X["frame_obs"].astype(float) / 255
    # column 0: observed frame, true future frame, true Z10
    for r, (im, t) in enumerate(((obs, "observed"), (X["frame_true"], "true $t_0{+}10$"), (imgs[1], "true $Z_{10}$"))):
        ax = fig.add_subplot(g0[r, 0]); show(ax, im, "bicubic" if r == 2 else "lanczos")
        ax.set_ylabel(t, fontsize=5.6, labelpad=1.5)
    lim = 100 * max(np.abs(X["disp"][:, :, :2]).max(), 0.02) * 1.15
    for p, n in enumerate(names):
        col = p + 1
        ax = fig.add_subplot(g1[0, p]); path_panel(ax, X["disp"][p], lim)
        ax.set_title(LABEL(n), fontsize=6.2, pad=2, fontweight="bold" if n == "true" else "normal",
                     color=GREEN if n == "true" else mf.INK)
        ax = fig.add_subplot(g1[1, p]); arrows(ax, obs, X["dx"][p, K - 1], X["dy"][p, K - 1], X["gate"][p, K - 1])
        ax = fig.add_subplot(g1[2, p]); show(ax, imgs[2 + p], "bicubic")
        lab = "$\\hat Z_{10}$, true plan" if n == "true" else f"change (moving) {X['change_shiftwm_moving'][p]:.2f}"
        ax.set_xlabel(lab, fontsize=5.0, labelpad=1)
        if p == 0:
            for r, t in enumerate(("path (top)", "motion $-g\\,o$", "forecast")):
                fig.axes[-3 + r].set_ylabel(t, fontsize=5.6, labelpad=1.5)
    fig.text(0.01, 0.95, "(a) One history, four real command sequences (ShiftWM, $k{=}10$)", fontsize=6.9, fontweight="bold",
             color=mf.INK)
    # (b) change of the forecast when the plan is swapped, all windows
    k = np.arange(1, K + 1)
    ax = fig.add_axes([0.795, 0.575, 0.195, 0.29])
    for a, col, mk in (("ar", ORANGE, "^"), ("direct", BLUE, "D"), ("shiftwm", GREEN, "o")):
        for reg, ls in (("moving", "-"), ("static", (0, (2.5, 1.5)))):
            v = S[f"change_{a}_{reg}"]["by_k"]
            ax.plot(k, v, color=col, ls=ls, lw=1.5 if a == "shiftwm" else 1.0, marker=mk if reg == "moving" else None,
                    ms=2.2, markevery=3)
    ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=5.4, length=2)
    ax.set_ylabel("$\\|f(a')-f(a)\\|^2$", fontsize=5.8, labelpad=1)
    ax.set_title("(b) forecast change", fontsize=6.4, pad=2)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=GREY, lw=1), Line2D([], [], color=GREY, lw=1, ls=(0, (2.5, 1.5)))], ["moving", "static"],
              fontsize=5.0, loc="upper left", frameon=False, handlelength=1.6, handletextpad=0.3, borderaxespad=0.1)
    # (c) error increase on moving patches
    ax = fig.add_axes([0.795, 0.13, 0.195, 0.29])
    for a, col, mk in (("ar", ORANGE, "^"), ("direct", BLUE, "D"), ("shiftwm", GREEN, "o")):
        v = np.array(S[f"err_alt_{a}_moving"]["by_k"]) - np.array(S[f"err_true_{a}_moving"]["by_k"])
        ax.plot(k, v, color=col, lw=1.5 if a == "shiftwm" else 1.0, marker=mk, ms=2.2, markevery=3)
    ax.axhline(0, color=mf.INK, lw=0.5)
    h = [Line2D([], [], color=c, marker=m, ms=2.5, lw=1) for c, m in ((GREEN, "o"), (BLUE, "D"), (ORANGE, "^"))]
    ax.legend(h, ["ShiftWM", "Direct", "AR"], fontsize=5.0, loc="upper left", frameon=False, handlelength=1.4,
              handletextpad=0.3, borderaxespad=0.1)
    ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=5.4, length=2)
    ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1); ax.set_ylabel("err$(a')-$err$(a)$", fontsize=5.8, labelpad=1)
    ax.set_title("(c) error increase, moving", fontsize=6.4, pad=2)
    fig.savefig(mf.FIG / "counterfactual.pdf"); fig.savefig(mf.FIG / "counterfactual_preview.png", dpi=200)
    g = lambda key: S[key]["k10"]                                            # noqa: E731
    lines = [f"\\def\\cfWindows{{{S['windows']:,}}}".replace(",", "{,}"), f"\\def\\cfEpisodes{{{S['episodes']}}}"]
    for a, t in (("shiftwm", "S"), ("direct", "D"), ("ar", "A")):
        lines += [f"\\def\\cfChgMov{t}{{{g(f'change_{a}_moving'):.3f}}}", f"\\def\\cfChgSta{t}{{{g(f'change_{a}_static'):.3f}}}",
                  f"\\def\\cfInc{t}{{{S[f'err_increase_{a}_moving_k10'][0]:.3f}}}",
                  f"\\def\\cfSel{t}{{{S[f'selectivity_{a}_k10'][0]:.1f}}}",
                  f"\\def\\cfErrAlt{t}{{{g(f'err_alt_{a}_moving'):.2f}}}", f"\\def\\cfErrTrue{t}{{{g(f'err_true_{a}_moving'):.2f}}}"]
    (mf.ROOT / "paper/submission_folder/tables/generated/counterfactual_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote counterfactual")


if __name__ == "__main__":
    main()
