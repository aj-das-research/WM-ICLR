"""Transport-trail figure from results/v2/analysis/trails (scripts/v2/trails.py), held-out DROID, ShiftWM seed 0.

(a) Three windows at the 25th / 50th / 75th percentile of the trail error (fixed rule, typical cases): for moving
    patches, ShiftWM's expected source location i + o_k (solid, circles) and the RAFT backward-flow trail i + b_k
    (dashed, squares) for k = 1..10, coloured by k, over the observed frame (zoomed on the moving region).
(b) Trail endpoint error vs k on all moving patches: ShiftWM, gate-weighted ShiftWM, zero motion, and the reach limit
    of the 7x7 transport window (95% episode-bootstrap CIs).
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/trails"
GREEN, GREY = mf.METHODS["shiftwm"][1], "#8C95A1"
K, G, PX = 10, 16, 14
KMAP = plt.get_cmap("viridis")
N_SHOW = 5


def frame_axes(ax, edge="#C9CED6", lw=0.6):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(lw)


def trail(ax, pts, style):
    """pts [K+1,2] pixel positions (query first); segments coloured by k; markers at every k or only at k=K."""
    seg = np.stack([pts[:-1], pts[1:]], 1)
    lc = LineCollection(seg, cmap=KMAP, norm=plt.Normalize(1, K), lw=style["lw"], linestyles=style["ls"], zorder=style["z"],
                        capstyle="round")
    lc.set_array(np.arange(1, K + 1)); ax.add_collection(lc)
    sel = np.arange(1, K + 1) if style["every"] else np.array([K])
    ax.scatter(pts[sel, 0], pts[sel, 1], c=sel, cmap=KMAP, vmin=1, vmax=K, s=style["s"], marker=style["m"],
               edgecolors=style["ec"], linewidths=0.4, zorder=style["z"] + 1)


NW, NH, BOX = 320, 180, 0.8                              # native frame size (trails are drawn at the true aspect)


def spread(cand, score, n, gap=2):
    """Greedy: highest-score patches whose query positions are >= `gap` patches apart (Chebyshev)."""
    out = []
    for i in cand[np.argsort(-score, kind="stable")]:
        if all(max(abs(i // G - j // G), abs(i % G - j % G)) >= gap for j in out):
            out.append(i)
        if len(out) == n:
            break
    return out


def example(ax, X, b):
    img = X["frame_obs"][b]
    f = X["flow"][b]                                    # [K,2,N] patches
    o = np.stack((X["dx"][b], X["dy"][b]), 1)           # [K,2,N]
    mov = np.where(np.hypot(*f[-1]) > 0.5)[0]
    show = spread(mov, np.hypot(*f[-1][:, mov]), N_SHOW)
    ys, xs = np.divmod(np.arange(G * G), G)
    sc = np.array([NW / G, NH / G])                     # patch -> native pixels
    c = np.stack((xs + 0.5, ys + 0.5), 1) * sc
    tr = {i: (np.concatenate([c[i][None], c[i] + o[:, :, i] * sc]), np.concatenate([c[i][None], c[i] + f[:, :, i] * sc]))
          for i in show}
    P = np.concatenate([np.concatenate(v) for v in tr.values()])
    lo, hi = P.min(0) - 12, P.max(0) + 12
    w = max(hi[0] - lo[0], (hi[1] - lo[1]) / BOX, 100); h = w * BOX
    cx, cy = np.clip((lo + hi) / 2, [w / 2, h / 2], [NW - w / 2, NH - h / 2]) if w <= NW and h <= NH else ((lo + hi) / 2)
    ax.imshow(img, extent=(0, NW, NH, 0), aspect="auto", interpolation="lanczos", alpha=0.85)
    for i, (ps, pf) in tr.items():
        trail(ax, pf, dict(lw=0.9, ls=(0, (2.2, 1.2)), z=3, s=14, m="s", ec="white", every=False))
        trail(ax, ps, dict(lw=1.3, ls="-", z=5, s=9, m="o", ec=mf.INK, every=True))
        ax.scatter(*c[i], s=14, marker="+", c="white", linewidths=0.9, zorder=7)
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy + h / 2, cy - h / 2)
    ax.set_box_aspect(BOX)
    frame_axes(ax)


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 2.0)); mf.pending(ax, "trails"); fig.savefig(mf.FIG / "trails.pdf"); return
    S = json.loads((D / "summary.json").read_text()); X = np.load(D / "examples.npz", allow_pickle=True)
    fig = plt.figure(figsize=(5.5, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.06, left=0.01, right=0.99, top=0.85, bottom=0.23)
    for b in range(3):
        ax = fig.add_subplot(gs[0, b]); example(ax, X, b)
        ax.set_title(f"{int(X['percentiles'][b])}th-percentile window", fontsize=6.3, pad=2)
        ax.set_xlabel(f"EPE$_{{10}}$ {X['win_epe'][b]:.2f} patches ({int(X['win_moving'][b])} moving)", fontsize=5.3, labelpad=1.5)
    fig.text(0.01, 0.95, "(a) Where moving patches are fetched from, $k{=}1,\\dots,10$ (ShiftWM solid; RAFT flow dashed, square at $k{=}10$)",
             fontsize=6.9, fontweight="bold", color=mf.INK)
    # (b) aggregate
    ax = fig.add_subplot(gs[0, 3])
    pos = ax.get_position(); ax.set_position([pos.x0 + 0.055, pos.y0, pos.width - 0.055, pos.height])
    k = np.arange(1, K + 1)
    for m, lab, col, ls, lw in (("epe_zero", "zero motion", GREY, (0, (3, 2)), 1.0),
                                ("epe_gated", "ShiftWM $g\\,o$", "#7FCDB5", "-", 1.0),
                                ("epe", "ShiftWM $o$", GREEN, "-", 1.6),
                                ("epe_reach", "reach limit", mf.INK, ":", 1.0)):
        mu = np.array(S[m]["mean"]); ci = np.array(S[m]["ci"])
        ax.plot(k, mu, color=col, ls=ls, lw=lw, marker="o" if m == "epe" else None, ms=2.2, label=lab)
        ax.fill_between(k, ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
    ax.set_xlim(0.6, 10.4); ax.set_ylim(-0.05, 1.95); ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=5.6, length=2)
    ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1); ax.set_ylabel("trail endpoint error (patches)", fontsize=5.8, labelpad=1)
    ax.set_title("(b) all moving patches", fontsize=6.6, pad=2)
    ax.legend(fontsize=5.1, loc="upper left", frameon=False, handlelength=1.4, handletextpad=0.4, borderaxespad=0.2)
    sm = fig.add_axes([0.29, 0.085, 0.16, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(1, K), KMAP), cax=sm, orientation="horizontal", ticks=[1, 5, 10])
    cb.ax.tick_params(labelsize=5.2, length=1.5); cb.outline.set_linewidth(0.4)
    fig.text(0.28, 0.096, "horizon $k$", fontsize=5.6, ha="right", va="center", color=mf.INK)
    fig.savefig(mf.FIG / "trails.pdf"); fig.savefig(mf.FIG / "trails_preview.png", dpi=200)
    e, z, gt = S["epe"]["mean"], S["epe_zero"]["mean"], S["epe_gated"]["mean"]
    lines = [f"\\def\\trWindows{{{S['windows']:,}}}".replace(",", "{,}"), f"\\def\\trEpisodes{{{S['episodes']}}}",
             f"\\def\\trPatches{{{S['moving_patches']:,}}}".replace(",", "{,}"),
             f"\\def\\trEpeOne{{{e[0]:.2f}}}", f"\\def\\trEpeTen{{{e[-1]:.2f}}}", f"\\def\\trZeroOne{{{z[0]:.2f}}}",
             f"\\def\\trZeroTen{{{z[-1]:.2f}}}", f"\\def\\trGatedTen{{{gt[-1]:.2f}}}",
             f"\\def\\trReachTen{{{S['epe_reach']['mean'][-1]:.2f}}}", f"\\def\\trOutTen{{{100 * S['share_out'][-1]:.0f}}}",
             f"\\def\\trCosTen{{{S['cos']['mean'][-1]:.2f}}}"]
    (mf.ROOT / "paper/submission_folder/tables/generated/trails_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote trails")


if __name__ == "__main__":
    main()
