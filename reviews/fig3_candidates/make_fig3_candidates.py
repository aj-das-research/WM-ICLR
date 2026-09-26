"""Figure 3 redesign candidates (previews only; the paper's featurespace.pdf is untouched).

Target slot: left half of a two-column top float, 0.5\\linewidth ~= 2.75 in wide, figure ~1.5 in tall.
All geometry is authored in inches (axes = whole figure, 1 data unit = 1 in), so font sizes are
print sizes. Schematic, not measured data: positions and weights are illustrative (as in the original).

Run:  .venv/bin/python reviews/fig3_candidates/make_fig3_candidates.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, FancyBboxPatch, RegularPolygon
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))
from layout_quality import audit_figure, issue_message  # noqa: E402

OUT = Path(__file__).resolve().parent
W, H = 2.75, 1.50

# Paper palette (make_figures.py / featurespace.tex)
GREEN, BLUE, ORANGE, AMBER = "#009E73", "#0072B2", "#D55E00", "#E69F00"
INK, SUB, OBS = "#1F2A37", "#5B6573", "#52627A"
DGREEN = "#00704F"          # text-safe dark green (>=4.5:1 on white)
DAMBER = "#9A6700"          # text-safe dark amber
DORANGE = "#B34700"
BAND = "#E9ECF1"

FS = 6.8     # content labels
FS_S = 6.2   # supporting labels (>= 6 pt floor)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": FS, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "text.color": INK,
})


def canvas():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def badge(ax, x, y, n, color, r=0.052):
    ax.add_patch(Circle((x, y), r, color=color, zorder=9, lw=0))
    ax.text(x, y - 0.004, str(n), color="white", fontsize=FS_S, fontweight="bold",
            ha="center", va="center", zorder=10)


def arrow(ax, p, q, color, lw=0.9, ls="-", rad=0.0, head=4.5, z=5, shrinkA=0, shrinkB=0):
    a = FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle=f"-|>,head_length={head*0.1:.2f},head_width={head*0.06:.2f}",
                        color=color, lw=lw, ls=ls, zorder=z, shrinkA=shrinkA, shrinkB=shrinkB,
                        mutation_scale=10)
    ax.add_patch(a)
    return a


def star(ax, x, y, s=0.055, z=8):
    ax.add_patch(RegularPolygon((x, y), 5, radius=s, color=INK, zorder=z, lw=0))
    # 5-point star via path
    ang = np.pi / 2 + np.arange(10) * np.pi / 5
    rr = np.where(np.arange(10) % 2 == 0, s, s * 0.42)
    pts = np.c_[x + rr * np.cos(ang), y + rr * np.sin(ang)]
    ax.patches[-1].remove()
    ax.add_patch(Polygon(pts, closed=True, color=INK, zorder=z, lw=0))


def diamond(ax, x, y, s=0.045, color=GREEN, z=8):
    ax.add_patch(Polygon([(x, y + s), (x + s, y), (x, y - s), (x - s, y)], closed=True,
                         facecolor=color, edgecolor="white", lw=0.5, zorder=z))


def smooth(pts, n=200):
    """Catmull-Rom spline through pts."""
    P = np.asarray(pts, float)
    P = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n // (len(P) - 3), endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return np.array(out)


def finish(fig, name):
    issues = audit_figure(fig, min_font_pt=6.0, display_width_inches=W)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}_print.png", dpi=150)   # ~print-size proof
    plt.close(fig)
    return [issue_message(i) for i in issues]


# ---------------------------------------------------------------------------------------------
# (A) Same concept, re-laid-out wide and short: curved band of real patch features across the width.
# ---------------------------------------------------------------------------------------------
def candidate_a():
    fig, ax = canvas()
    ctrl = [(0.08, 0.50), (0.60, 0.66), (1.20, 0.80), (1.80, 0.86), (2.35, 0.82), (2.72, 0.72)]
    c = smooth(ctrl)
    ax.plot(c[:, 0], c[:, 1], color=BAND, lw=19, solid_capstyle="butt", zorder=0)
    ax.plot(c[:, 0], c[:, 1], color="#B9C1CC", lw=0.4, ls=(0, (1, 1.5)), zorder=1)
    ax.text(0.03, 0.17, "real patch features", fontsize=FS_S, style="italic", color=OBS,
            ha="left", va="top", rotation=13, rotation_mode="anchor")
    ax.text(2.72, 1.47, r"feature space $\mathbb{R}^C$", fontsize=FS_S, style="italic", color=SUB,
            ha="right", va="top")

    def on(x, dy=0.0):
        return (x, float(np.interp(x, c[:, 0], c[:, 1])) + dy)

    z0 = on(0.52, -0.075)
    cands = [on(0.24, 0.06), on(0.86, 0.08), on(1.18, 0.00), on(1.50, 0.07), on(1.88, 0.01), on(2.22, 0.06)]
    pis = np.array([0.03, 0.03, 0.04, 0.12, 0.55, 0.23])      # rho(i) = index 4
    wts = np.r_[0.02, pis]; wts = wts / wts.sum()
    allp = np.array([z0] + cands)
    T = (wts[:, None] * allp).sum(0)
    g = 0.86
    zg = (1 - g) * np.array(z0) + g * T
    zt = np.array([T[0] + 0.12, 1.24])
    zh = zt + np.array([-0.07, -0.05])

    # transport lines (width = pi)
    ax.add_patch(Polygon(allp[[0, 1]].tolist() + [cands[1], cands[2], cands[3], cands[4], cands[5]] + [z0],
                         closed=True, color=GREEN, alpha=0.07, lw=0, zorder=1))
    for p, w in zip(allp, wts):
        ax.plot([p[0], T[0]], [p[1], T[1]], color=GREEN, lw=0.25 + 4.2 * w, alpha=0.85,
                solid_capstyle="round", zorder=3)
    # Direct: synthesise the whole displacement
    arrow(ax, z0, (zt[0] - 0.05, zt[1] + 0.01), BLUE, lw=1.0, rad=-0.33, head=4.2, z=4)
    # AR: recursion drifts off the band
    ar = np.array([z0, (0.74, 0.44), (0.92, 0.40), (1.08, 0.28), (1.30, 0.26), (1.45, 0.15), (1.66, 0.13)])
    ax.plot(ar[:-1, 0], ar[:-1, 1], color=ORANGE, lw=0.8, ls=(0, (2.4, 1.3)), zorder=4)
    arrow(ax, ar[-2], ar[-1], ORANGE, lw=0.8, head=3.6, z=4)
    ax.scatter(ar[1:-1, 0], ar[1:-1, 1], s=4, color=ORANGE, zorder=5, lw=0)
    # gate and correction
    ax.plot([z0[0], zg[0]], [z0[1], zg[1]], color=AMBER, lw=1.0, ls=(0, (1, 1)), zorder=5)
    ax.add_patch(Circle(zg, 0.022, color=AMBER, zorder=7, ec="white", lw=0.3))
    arrow(ax, zg, zh, "#444", lw=0.8, head=3.6, z=7, shrinkB=2.5)
    # points
    for p, w in zip(cands, pis):
        ax.add_patch(Circle(p, 0.018 + 0.05 * w, color=OBS, ec="white", lw=0.4, zorder=6))
    ax.add_patch(Circle(z0, 0.033, facecolor="white", edgecolor=SUB, lw=1.0, zorder=7))
    diamond(ax, *T, s=0.04)
    ax.add_patch(Circle(zh, 0.028, color=GREEN, ec="white", lw=0.5, zorder=8))
    star(ax, *zt, s=0.05)

    # labels
    ax.text(z0[0] - 0.02, z0[1] - 0.045, r"$\mathbf{z}_{0,i}$ (keep)", fontsize=FS, color=SUB, ha="right", va="top")
    rho = cands[4]
    ax.text(rho[0] + 0.05, rho[1] + 0.035, r"$\mathbf{z}_{0,\rho(i)}$", fontsize=FS, color=OBS, ha="left", va="bottom")
    ax.text(T[0] + 0.035, T[1] - 0.05, r"$\mathbf{T}_{k,i}$", fontsize=FS, color=DGREEN, ha="left", va="top")
    ax.text(zh[0] - 0.045, zh[1] - 0.005, r"$\hat{\mathbf{z}}_{k,i}$", fontsize=FS, color=DGREEN, ha="right", va="center")
    ax.text(zt[0] + 0.07, zt[1], r"true $\mathbf{z}_{k,i}$", fontsize=FS, color=INK, ha="left", va="center")
    mid = (np.array(z0) + zg) / 2
    ax.text(mid[0] + 0.02, mid[1] - 0.035, r"gate $g$", fontsize=FS_S, color=DAMBER, ha="center", va="top",
            rotation=np.degrees(np.arctan2(zg[1] - z0[1], zg[0] - z0[0])), rotation_mode="anchor")
    rm = (zg + zh) / 2
    ax.text(rm[0] + 0.03, rm[1] - 0.02, r"$\mathbf{r}$", fontsize=FS, color="#333", ha="left", va="center")

    badge(ax, 0.10, 1.38, 2, BLUE)
    ax.text(0.18, 1.38, "Direct: synthesise", fontsize=FS, color=BLUE, ha="left", va="center")
    badge(ax, 1.76, 0.40, 1, GREEN)
    ax.text(1.84, 0.40, r"$\mathbf{ShiftWM}$: mix", fontsize=FS, color=DGREEN, ha="left", va="center")
    badge(ax, 1.76, 0.13, 3, ORANGE)
    ax.text(1.84, 0.13, "AR: errors compound", fontsize=FS, color=DORANGE, ha="left", va="center")
    return finish(fig, "A_wide_featurespace")


# ---------------------------------------------------------------------------------------------
# (B) Observed -> forecast strip: three aligned lanes with a shared output column.
# ---------------------------------------------------------------------------------------------
def candidate_b():
    fig, ax = canvas()
    xo, xout = 0.30, 2.50               # observed column centre, output column centre
    # column headers
    ax.text(xo + 0.13, 1.46, r"observed ($t\leq 0$)", fontsize=FS_S, color=SUB, ha="center", va="top", style="italic")
    ax.text(xout + 0.02, 1.46, "step $k$", fontsize=FS_S, color=SUB, ha="center", va="top", style="italic")
    ax.plot([2.28, 2.28], [0.05, 1.36], color="#D5DAE1", lw=0.5, ls=(0, (2, 2)), zorder=0)

    def chip(x, y, s=0.07, fc=OBS, ec="white", lw=0.4, z=6):
        ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    def target(y):
        star(ax, xout + 0.10, y + 0.02, s=0.045)

    # ---- lane 1: ShiftWM ------------------------------------------------------------------
    y1 = 1.02
    ax.add_patch(FancyBboxPatch((0.03, 0.72), 2.21, 0.62, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#EEF8F4", ec="none", zorder=0))
    # window W(i): 3x3 grid of candidate features, z0,i centre, rho(i) upper right
    gx, gy, d = xo + 0.08, y1, 0.085
    pis = {(-1, 1): .03, (0, 1): .06, (1, 1): .55, (-1, 0): .02, (0, 0): .04, (1, 0): .22,
           (-1, -1): .02, (0, -1): .03, (1, -1): .03}
    T = np.array([1.05, y1 + 0.02])
    for (i, j), p in pis.items():
        x, y = gx + i * d, gy + j * d
        ax.plot([x, T[0]], [y, T[1]], color=GREEN, lw=0.25 + 5.0 * p, alpha=0.85, solid_capstyle="round", zorder=3)
    for (i, j), p in pis.items():
        x, y = gx + i * d, gy + j * d
        if (i, j) == (0, 0):
            chip(x, y, fc="white", ec=SUB, lw=0.9, z=7)
        else:
            chip(x, y, fc=OBS if (i, j) != (1, 1) else "#2E3A4D", z=6)
    ax.text(gx - 1.5 * d - 0.03, gy + d, r"$\mathcal{W}(i)$", fontsize=FS_S, color=SUB, ha="right", va="center")
    ax.text(gx + d + 0.02, gy + 1.5 * d - 0.005, r"$\mathbf{z}_{0,\rho(i)}$", fontsize=FS_S, color=OBS, ha="center", va="bottom")
    diamond(ax, *T, s=0.045)
    ax.text(T[0], T[1] + 0.06, r"$\mathbf{T}_{k,i}$", fontsize=FS, color=DGREEN, ha="center", va="bottom")
    ax.text((gx + T[0]) / 2 + 0.07, T[1] + 0.075, r"move ($\pi$)", fontsize=FS_S, color=DGREEN, ha="center", va="bottom")
    # keep path: z0,i -> gate mixer (below)
    gate = np.array([1.52, y1 - 0.02])
    keep_y = 0.80
    kx = gx - 1.5 * d - 0.035
    ax.plot([gx - 0.035, kx, kx, gate[0]], [gy, gy, keep_y, keep_y], color=SUB, lw=0.8, zorder=2)
    ax.text(0.95, keep_y + 0.015, r"keep $\mathbf{z}_{0,i}$", fontsize=FS_S, color=SUB, ha="center", va="bottom")
    arrow(ax, (gate[0], keep_y), (gate[0], gate[1] - 0.075), SUB, lw=0.8, head=3.4)
    arrow(ax, (T[0] + 0.05, T[1]), (gate[0] - 0.075, gate[1] + 0.01), GREEN, lw=1.0, head=3.6)
    # gate node: slider 1-g / g
    ax.add_patch(Circle(gate, 0.07, fc="white", ec=AMBER, lw=1.0, zorder=6))
    ax.text(*gate, "$g$", fontsize=FS, color=DAMBER, ha="center", va="center", zorder=7)
    ax.text(gate[0] + 0.03, keep_y + 0.07, r"$1{-}g$", fontsize=FS_S, color=SUB, ha="left", va="center")
    # correction r
    plus = np.array([1.92, gate[1]])
    arrow(ax, (gate[0] + 0.07, gate[1]), (plus[0] - 0.05, plus[1]), INK, lw=0.8, head=3.4)
    ax.add_patch(Circle(plus, 0.05, fc="white", ec=INK, lw=0.7, zorder=6))
    ax.text(*plus, "+", fontsize=FS + 0.5, color=INK, ha="center", va="center", zorder=7)
    arrow(ax, (plus[0], plus[1] + 0.22), (plus[0], plus[1] + 0.05), "#444", lw=0.8, head=3.4)
    ax.text(plus[0] + 0.03, plus[1] + 0.17, r"$\mathbf{r}$ correct", fontsize=FS_S, color="#333", ha="left", va="center")
    arrow(ax, (plus[0] + 0.05, plus[1]), (xout - 0.05, plus[1]), GREEN, lw=1.0, head=3.6)
    ax.add_patch(Circle((xout, plus[1]), 0.03, color=GREEN, ec="white", lw=0.5, zorder=8))
    ax.text(xout, plus[1] - 0.05, r"$\hat{\mathbf{z}}_{k,i}$", fontsize=FS, color=DGREEN, ha="center", va="top")
    target(plus[1])
    badge(ax, 0.09, 1.29, 1, GREEN)
    ax.text(0.16, 1.29, r"$\mathbf{ShiftWM}$", fontsize=FS, color=DGREEN, ha="left", va="center")

    # ---- lane 2: Direct -------------------------------------------------------------------
    y2 = 0.50
    badge(ax, 0.09, y2 + 0.06, 2, BLUE)
    ax.text(0.16, y2 + 0.06, "Direct", fontsize=FS, color=BLUE, ha="left", va="center")
    chip(xo + 0.08, y2 - 0.04, fc="white", ec=SUB, lw=0.9)
    ax.text(xo + 0.08, y2 - 0.09, r"$\mathbf{z}_{0,i}$", fontsize=FS_S, color=SUB, ha="center", va="top")
    arrow(ax, (xo + 0.13, y2 - 0.04), (xout - 0.05, y2 - 0.04), BLUE, lw=1.0, head=3.6)
    ax.text(1.35, y2 - 0.02, r"synthesise $\mathbf{z}_{0,\rho(i)}{-}\mathbf{z}_{0,i}$ from $\mathbf{h}$",
            fontsize=FS_S, color=BLUE, ha="center", va="bottom")
    ax.add_patch(Circle((xout, y2 - 0.04), 0.03, color=BLUE, ec="white", lw=0.5, zorder=8))
    target(y2 - 0.04)

    # ---- lane 3: AR -----------------------------------------------------------------------
    y3 = 0.17
    badge(ax, 0.09, y3 + 0.06, 3, ORANGE)
    ax.text(0.16, y3 + 0.06, "AR", fontsize=FS, color=DORANGE, ha="left", va="center")
    chip(xo + 0.08, y3 - 0.04, fc="white", ec=SUB, lw=0.9)
    xs = [xo + 0.08, 0.95, 1.45, 1.95, xout]
    ys = [y3 - 0.04, y3 - 0.03, y3 - 0.06, y3 - 0.02, y3 - 0.10]
    for a in range(len(xs) - 1):
        arrow(ax, (xs[a] + 0.05, ys[a]), (xs[a + 1] - 0.04, ys[a + 1]), ORANGE, lw=0.8,
              rad=-0.25, head=3.4)
    for x, y, rr in zip(xs[1:], ys[1:], [0.028, 0.04, 0.055, 0.07]):
        ax.add_patch(Circle((x, y), rr, fc=ORANGE, alpha=0.13, lw=0, zorder=4))
        ax.add_patch(Circle((x, y), 0.022, color=ORANGE, ec="white", lw=0.4, zorder=7))
    ax.text(1.20, y3 + 0.09, "feed forecasts back: errors compound", fontsize=FS_S, color=DORANGE,
            ha="center", va="bottom")
    target(y3 - 0.02 + 0.04)
    return finish(fig, "B_observed_to_forecast")


# ---------------------------------------------------------------------------------------------
# (C) Unrolled band: real patch features as one axis; pi as a stem histogram; geometry of Eq. (2).
# ---------------------------------------------------------------------------------------------
def candidate_c():
    fig, ax = canvas()
    yl = 0.62                                   # the unrolled band of real patch features
    x0, x1 = 0.08, 2.70
    ax.add_patch(FancyBboxPatch((x0, yl - 0.045), x1 - x0, 0.09, boxstyle="round,pad=0,rounding_size=0.045",
                                fc=BAND, ec="none", zorder=0))
    ax.text(x1, yl + 0.06, "real patch\nfeatures", fontsize=FS_S, style="italic", color=OBS, ha="right", va="bottom",
            linespacing=0.95)
    ax.text(x1, 1.47, r"feature space $\mathbb{R}^C$", fontsize=FS_S, style="italic", color=SUB, ha="right", va="top")

    xs = np.array([0.30, 0.52, 0.80, 1.08, 1.36, 1.64, 1.92, 2.20])
    iz0, irho = 1, 6
    pis = np.array([0.02, 0.03, 0.03, 0.04, 0.06, 0.18, 0.46, 0.18])
    pis = pis / pis.sum()
    T = float((pis * xs).sum())
    g = 0.80
    xg = (1 - g) * xs[iz0] + g * T
    zh = np.array([xg + 0.10, yl + 0.52])
    zt = np.array([xg + 0.20, yl + 0.56])

    # pi stems (hanging below band, height ~ pi) -- the transport weights
    hmax = 0.27
    for x, p in zip(xs, pis):
        if p > 0:
            ax.plot([x, x], [yl, yl - 0.05 - hmax * p / pis.max()], color=GREEN, lw=2.2, solid_capstyle="butt",
                    alpha=0.85, zorder=2)
    # converge to T (weights) - light lines from each candidate to T
    for x, p in zip(xs, pis):
        ax.plot([x, T], [yl, yl], color=GREEN, lw=0.2 + 3.5 * p, alpha=0.0)
    ax.text(xs[irho] + 0.08, yl - 0.05 - hmax + 0.02, r"$\pi_{k,i}$", fontsize=FS, color=DGREEN, ha="left", va="center")
    # candidates
    for k, x in enumerate(xs):
        if k == iz0:
            continue
        r = 0.022 + 0.035 * pis[k] / pis.max()
        ax.add_patch(Circle((x, yl), r, color=OBS, ec="white", lw=0.4, zorder=6))
    ax.add_patch(Circle((xs[iz0], yl), 0.034, fc="white", ec=SUB, lw=1.0, zorder=7))
    diamond(ax, T, yl, s=0.045)

    # gate: move from z0 toward T by g (amber bracket above band)
    yb = yl + 0.10
    ax.plot([xs[iz0], xs[iz0], xg, xg], [yl + 0.05, yb, yb, yl + 0.05], color=AMBER, lw=0.9, zorder=3)
    ax.text((xs[iz0] + xg) / 2 + 0.08, yb + 0.012, r"gate $g$", fontsize=FS_S,
            color=DAMBER, ha="center", va="bottom")
    ax.add_patch(Circle((xg, yl), 0.02, color=AMBER, ec="white", lw=0.3, zorder=8))
    # correction r
    arrow(ax, (xg, yl + 0.03), zh, "#444", lw=0.8, head=3.6, z=7, shrinkB=2.8)
    ax.text((xg + zh[0]) / 2 + 0.04, (yl + zh[1]) / 2 - 0.02, r"$\mathbf{r}$", fontsize=FS, color="#333", ha="left")
    ax.add_patch(Circle(zh, 0.028, color=GREEN, ec="white", lw=0.5, zorder=8))
    star(ax, *zt, s=0.05)
    ax.text(zh[0] - 0.045, zh[1], r"$\hat{\mathbf{z}}_{k,i}$", fontsize=FS, color=DGREEN, ha="right", va="center")
    ax.text(zt[0] + 0.07, zt[1] + 0.005, r"true $\mathbf{z}_{k,i}$", fontsize=FS, color=INK, ha="left", va="center")

    # Direct: synthesise the displacement from scratch
    arrow(ax, (xs[iz0], yl + 0.04), (zt[0] - 0.05, zt[1] + 0.02), BLUE, lw=1.0, rad=-0.30, head=4.0, z=4)

    # AR: recursion drifts off the band (below-left region, clear of the stems)
    ar = np.array([(xs[iz0], yl - 0.035), (0.62, 0.40), (0.78, 0.36), (0.88, 0.24), (1.04, 0.21), (1.12, 0.10)])
    ax.plot(ar[:-1, 0], ar[:-1, 1], color=ORANGE, lw=0.8, ls=(0, (2.4, 1.3)), zorder=4)
    arrow(ax, ar[-2], ar[-1], ORANGE, lw=0.8, head=3.4)
    ax.scatter(ar[1:-1, 0], ar[1:-1, 1], s=4, color=ORANGE, zorder=5, lw=0)

    # labels
    ax.text(xs[iz0] - 0.05, yl + 0.045, r"$\mathbf{z}_{0,i}$", fontsize=FS, color=SUB, ha="right", va="bottom")
    ax.text(xs[irho], yl + 0.045, r"$\mathbf{z}_{0,\rho(i)}$", fontsize=FS, color=OBS, ha="left", va="bottom")
    ax.text(T, yl - 0.06, r"$\mathbf{T}_{k,i}$", fontsize=FS, color=DGREEN, ha="center", va="top")

    badge(ax, 0.10, 1.38, 2, BLUE)
    ax.text(0.18, 1.38, "Direct: synthesise", fontsize=FS, color=BLUE, ha="left", va="center")
    badge(ax, 2.06, 0.98, 1, GREEN)
    ax.text(2.14, 0.98, r"$\mathbf{ShiftWM}$", fontsize=FS, color=DGREEN, ha="left", va="center")
    badge(ax, 1.30, 0.10, 3, ORANGE)
    ax.text(1.38, 0.10, "AR: errors compound", fontsize=FS, color=DORANGE, ha="left", va="center")
    return finish(fig, "C_unrolled_band")


if __name__ == "__main__":
    report = {"A": candidate_a(), "B": candidate_b(), "C": candidate_c()}
    for k, v in report.items():
        print(k, "issues:", v if v else "none")
    (OUT / "qa_audit.json").write_text(json.dumps(report, indent=2))
