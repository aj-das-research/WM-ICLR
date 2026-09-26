"""Candidate redesigns of the ShiftWM method figure (Figure 2, label fig:method).

All raster insets are the real-data assets produced by
paper/submission_folder/figures/src/arch_assets.py (held-out DROID episode droid-c70170b3ca00f756c4bb35c8,
checkpoint results/v2s/droid/dinov2s/shiftwm/s0/best.pt, step k=10, patch 51), copied read-only into ./assets.
Transport weights, window masses, gate value and the 12-D per-patch vector colours are parsed from
assets/head_data.tex. Everything else is vector geometry.

Usage (repo root):  .venv/bin/python reviews/arch_candidates/make_arch_candidates.py [A|B|C ...]
Outputs: reviews/arch_candidates/cand_<X>.{pdf,svg,png} and a layout audit printed to stdout.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle
from matplotlib.transforms import Affine2D

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))
try:
    from layout_quality import audit_figure, issue_message  # skill QA
except Exception:  # pragma: no cover
    audit_figure = None

# ------------------------------------------------------------------ style (paper palette, Okabe-Ito derived)
INK = "#1F2A37"
SUB = "#5B6573"
OBS = "#52627A"        # observed features (slate), as in the current Figure 2
ACT = "#7B4FA0"        # actions
OURS = "#009E73"       # ShiftWM / transport
OURS_D = "#00664A"
GATE = "#E69F00"       # gate
GATE_D = "#8A5A00"
DIRECT = "#0072B2"     # Direct baseline
AR = "#D55E00"         # AR baseline
PANEL = "#F3F5F8"
PANEL_G = "#EAF6F1"
BOXEDGE = "#8993A1"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6,
})
FS = 6.8     # body label size (pt) at 5.5 in print width
FS_S = 6.2   # smallest label
FS_T = 7.6   # panel titles


def parse_head_data():
    txt = (ASSETS / "head_data.tex").read_text()
    d = {}
    for name, val in re.findall(r"\\def\\(\w+)\{([^}]*)\}", txt):
        d[name] = val
    win = np.array([[float(x) for x in d[f"win{c}"].split(",")] for c in "ABC"]).reshape(3, 7, 7)
    vecs = {k: ["#" + h for h in d[f"vec{k}"].split(",")] for k in ("Zzero", "Ttr", "Rcor", "Zhat", "Ztrue")}
    mass = [int(d[f"mass{c}"]) for c in "ABC"]
    return win, vecs, mass, float(d["gateval"])


WIN, VECS, MASS, GATEVAL = parse_head_data()


def img(name):
    return plt.imread(ASSETS / name)


# ------------------------------------------------------------------ primitives (coordinates in inches)
def new_fig(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def panel(ax, x0, y0, x1, y1, color, r=0.06, z=0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=color, ec="none", zorder=z))


def box(ax, x0, y0, x1, y1, ec=BOXEDGE, fc="white", lw=0.8, r=0.035, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def T(ax, x, y, s, fs=FS, c=INK, ha="center", va="center", w="normal", z=6, **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, fontweight=w, zorder=z, **kw)


def arrow(ax, pts, c=INK, lw=0.8, head=True, z=3, ls="-", hw=2.6, hl=3.4):
    pts = np.asarray(pts, float)
    if len(pts) > 2:
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=c, lw=lw, zorder=z, solid_capstyle="butt", ls=ls,
                solid_joinstyle="miter")
    a, b = pts[-2], pts[-1]
    style = f"-|>,head_width={hw/72*2.2:.3f},head_length={hl/72*2.2:.3f}" if head else "-"
    ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle=style, mutation_scale=10, lw=lw, color=c,
                                 zorder=z, shrinkA=0, shrinkB=0, linestyle=ls, joinstyle="miter",
                                 capstyle="butt"))


def image(ax, name, x0, y0, w, h, edge=INK, lw=0.4, z=3, alpha=1.0):
    ax.imshow(img(name), extent=(x0, x0 + w, y0, y0 + h), zorder=z, interpolation="lanczos", alpha=alpha)
    ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=edge, lw=lw, zorder=z + 0.1))


def grid_lines(ax, x0, y0, w, h, n, c="white", lw=0.25, alpha=0.6, z=3.2):
    for t in np.linspace(0, 1, n + 1)[1:-1]:
        ax.plot([x0 + t * w] * 2, [y0, y0 + h], color=c, lw=lw, alpha=alpha, zorder=z)
        ax.plot([x0, x0 + w], [y0 + t * h] * 2, color=c, lw=lw, alpha=alpha, zorder=z)


def vec_chip(ax, x0, y0, cols, cw=0.085, ch=0.040, edge=INK, lw=0.5, z=4):
    """Vertical feature-vector chip (top entry first)."""
    n = len(cols)
    for j, c in enumerate(cols):
        ax.add_patch(Rectangle((x0, y0 + (n - 1 - j) * ch), cw, ch, fc=c, ec="white", lw=0.25, zorder=z))
    ax.add_patch(Rectangle((x0, y0), cw, n * ch, fc="none", ec=edge, lw=lw, zorder=z + 0.1))
    return x0 + cw / 2, y0, y0 + n * ch


def window(ax, x0, y0, W, cell, ident=False, z=4, edge=OURS_D):
    """7x7 transport window with real (max-normalised) weights; row 0 on top."""
    for r in range(7):
        for c in range(7):
            v = np.sqrt(W[r, c])
            col = np.array(matplotlib.colors.to_rgb(OURS)) * v + (1 - v) * np.ones(3)
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (6 - r) * cell), cell, cell, fc=col, ec="white",
                                   lw=0.25, zorder=z))
    ax.add_patch(Rectangle((x0, y0), 7 * cell, 7 * cell, fc="none", ec=edge, lw=0.6, zorder=z + 0.1))
    if ident:
        ax.add_patch(Rectangle((x0 + 3 * cell, y0 + 3 * cell), cell, cell, fc="none", ec=GATE_D, lw=1.0,
                               zorder=z + 0.2))


def stack(ax, names, x0, y0, s, dx, dy, edge=INK, grid=16, z=3):
    """Offset stack of square images, back to front; returns front lower-left."""
    n = len(names)
    for k, nm in enumerate(names):
        xx, yy = x0 + (n - 1 - k) * dx, y0 + (n - 1 - k) * dy
        image(ax, nm, xx, yy, s, s, edge=edge, lw=0.4, z=z + k * 0.3)
        if grid:
            grid_lines(ax, xx, yy, s, s, grid, lw=0.15, alpha=0.35, z=z + k * 0.3 + 0.1)
    return x0, y0


def finish(fig, name, width_in):
    out = HERE / f"cand_{name}"
    issues = []
    if audit_figure is not None:
        try:
            issues = audit_figure(fig, min_font_pt=6.0, display_width_inches=width_in)
        except Exception as e:  # report but do not fail
            print("audit error", e)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=400)
    print(f"[{name}] saved {out}.pdf/.svg/.png ; layout issues: {len(issues)}")
    for it in issues[:40]:
        try:
            print("   ", issue_message(it))
        except Exception:
            print("   ", it)
    plt.close(fig)


# ================================================================== Candidate A: "h decides where, Z supplies what"
def zstack(ax, x0, y0, s, d, n=3, front="head_obs.png"):
    """Observed feature grids: back grids as slate tiles (content hidden), front = real PCA view of Z_0."""
    for k in range(n - 1):
        xx, yy = x0 + (n - 1 - k) * d, y0 + (n - 1 - k) * d
        ax.add_patch(Rectangle((xx, yy), s, s, fc="#C9D0DA", ec=OBS, lw=0.4, zorder=3 + k * 0.3))
        grid_lines(ax, xx, yy, s, s, 16, lw=0.15, alpha=0.5, z=3.05 + k * 0.3)
    image(ax, front, x0, y0, s, s, edge=OBS, lw=0.4, z=4)
    grid_lines(ax, x0, y0, s, s, 16, lw=0.15, alpha=0.35, z=4.1)


def frames_encoder(ax, yc, x0=0.07, fw=0.38, tw=0.315):
    """Observed frames -> frozen DINOv2 -> Z grids. Returns (x of Z stack centre, top of stack, right edge)."""
    for k, nm in enumerate(["observed_0.png", "observed_1.png", "observed_2.png"]):
        image(ax, nm, x0 + k * 0.045, yc - 0.03 - k * 0.05 + 0.05, fw, fw * 0.5625, lw=0.35, z=3 + k)
    fx1 = x0 + fw + 0.09
    arrow(ax, [(fx1 + 0.005, yc), (fx1 + 0.065, yc)])
    tx0, tx1 = fx1 + 0.065, fx1 + 0.065 + tw
    ax.add_patch(Polygon([(tx0, yc - 0.21), (tx0, yc + 0.21), (tx1, yc + 0.12), (tx1, yc - 0.12)], closed=True,
                         fc="#E4E7EC", ec=BOXEDGE, lw=0.7, zorder=3))
    T(ax, (tx0 + tx1) / 2, yc + 0.045, "DINOv2", fs=FS_S)
    T(ax, (tx0 + tx1) / 2, yc - 0.06, "frozen", fs=FS_S, c=SUB, style="italic")
    arrow(ax, [(tx1, yc), (tx1 + 0.07, yc)])
    zx = tx1 + 0.08
    zstack(ax, zx, yc - 0.19, 0.30, 0.045)
    return x0, fx1, zx, tx0, tx1


def design_A():
    W, H = 5.5, 2.05
    fig, ax = new_fig(W, H)
    XS = 3.12                                              # split between predictor and head panels
    panel(ax, 0.02, 0.02, XS - 0.04, H - 0.02, PANEL)
    panel(ax, XS + 0.02, 0.02, W - 0.02, H - 0.02, PANEL_G)
    T(ax, 0.10, H - 0.11, "Predictor", fs=FS_T, w="bold", ha="left")
    T(ax, 0.68, H - 0.11, "(shared with the baselines; or V-JEPA 2-AC, DINO-WM)", fs=FS_S, c=SUB, ha="left")
    T(ax, XS + 0.10, H - 0.11, "ShiftWM head: gated transport", fs=FS_T, w="bold", c=OURS_D, ha="left")

    yc = 1.12
    x0, fx1, zx, tx0, tx1 = frames_encoder(ax, yc)
    T(ax, x0 + 0.24, yc - 0.30, "observed frames", fs=FS_S, c=SUB)
    T(ax, zx + 0.17, yc - 0.30, r"$\mathbf{Z}_{-H+1:0}$", fs=FS_S, c=OBS)
    zr = zx + 0.30 + 0.09                                  # right edge of stack

    # ---- memory encoder
    ex0, ex1, ey0, ey1 = 1.46, 2.16, yc - 0.25, yc + 0.25
    box(ax, ex0, ey0, ex1, ey1)
    cx = (ex0 + ex1) / 2
    T(ax, cx, yc + 0.155, "Memory encoder", fs=FS_S, w="bold")
    T(ax, cx, yc + 0.035, r"$L_e$ layers", fs=FS_S)
    T(ax, cx, yc - 0.075, "+ space, frame,", fs=FS_S, c=SUB)
    T(ax, cx, yc - 0.175, "action emb.", fs=FS_S, c=SUB)
    arrow(ax, [(zr, yc), (ex0, yc)], c=OBS)

    # ---- decoder, drawn as K parallel copies
    dx0, dx1, dy0, dy1 = 2.32, 2.94, yc - 0.25, yc + 0.25
    for k in (2, 1):
        box(ax, dx0 + k * 0.035, dy0 + k * 0.035, dx1 + k * 0.035, dy1 + k * 0.035, fc="#FAFBFC", lw=0.6)
    box(ax, dx0, dy0, dx1, dy1)
    cx = (dx0 + dx1) / 2
    T(ax, cx, yc + 0.155, r"Decoder $\times L_d$", fs=FS_S, w="bold")
    T(ax, cx, yc + 0.045, "self-attn: patches", fs=FS_S)
    T(ax, cx, yc - 0.06, r"cross-attn: $\mathbf{M}$", fs=FS_S)
    T(ax, cx, yc - 0.165, r"AdaLN$(\mathbf{c}_k)$", fs=FS_S, c=ACT)
    T(ax, cx + 0.035, dy1 + 0.135, r"all $K$ steps in parallel", fs=FS_S, c=SUB)
    arrow(ax, [(ex1, yc), (dx0, yc)])
    T(ax, 2.31, yc - 0.075, r"$\mathbf{M}$", fs=FS)

    # ---- actions
    ya = 0.36
    ax.imshow(img("actions_spark.png"), extent=(0.12, 0.72, ya - 0.08, ya + 0.08), zorder=3)
    T(ax, 0.42, ya - 0.16, r"actions $\mathbf{a}_{-H+1:K-1}$", fs=FS_S, c=ACT)
    arrow(ax, [(0.76, ya), (1.62, ya), (1.62, ey0)], c=ACT, lw=0.7)
    T(ax, 1.55, 0.60, r"$\mathbf{a}_{<0}$", fs=FS_S, c=ACT, ha="right")
    gx0, gx1 = 1.93, 2.37
    arrow(ax, [(1.62, ya), (gx0, ya)], c=ACT, lw=0.7)
    T(ax, 1.775, ya + 0.075, r"$\mathbf{a}_{0:K-1}$", fs=FS_S, c=ACT)
    box(ax, gx0, ya - 0.11, gx1, ya + 0.11, ec=ACT, lw=0.7)
    T(ax, (gx0 + gx1) / 2, ya + 0.035, "causal", fs=FS_S)
    T(ax, (gx0 + gx1) / 2, ya - 0.06, "GRU", fs=FS_S)
    pc = 2.55
    ax.add_patch(Circle((pc, ya), 0.045, fc="white", ec=ACT, lw=0.7, zorder=4))
    T(ax, pc, ya + 0.002, "+", fs=FS, c=ACT)
    arrow(ax, [(gx1, ya), (pc - 0.045, ya)], c=ACT, lw=0.7)
    arrow(ax, [(pc, 0.12), (pc, ya - 0.045)], c=ACT, lw=0.7)
    T(ax, pc + 0.05, 0.13, r"$\mathbf{e}_k$", fs=FS_S, c=ACT, ha="left")
    arrow(ax, [(pc, ya + 0.045), (pc, dy0)], c=ACT, lw=0.7)
    T(ax, pc + 0.04, 0.63, r"$\mathbf{c}_k$", fs=FS, c=ACT, ha="left")

    # ================= head panel
    cell = 0.054
    wx = [3.40, 3.87, 4.34]; wy = yc - 3.5 * cell + 0.02
    for s in range(3):
        window(ax, wx[s], wy, WIN[s], cell, ident=(s == 2))
        T(ax, wx[s] + 3.5 * cell, wy + 7 * cell + 0.07,
          [r"$\mathbf{Z}_{-2}$", r"$\mathbf{Z}_{-1}$", r"$\mathbf{Z}_{0}$"][s] + f" {MASS[s]}%", fs=FS_S, c=OBS)
    wtop = wy + 7 * cell
    bx0, bx1, by = wx[0], wx[2] + 7 * cell, wtop + 0.15
    ax.plot([bx0, bx0, bx1, bx1], [by - 0.025, by, by, by - 0.025], color=OBS, lw=0.7, zorder=3)
    ys = 1.72
    zc = zx + 0.17
    arrow(ax, [(zc, yc + 0.20), (zc, ys), ((bx0 + bx1) / 2, ys), ((bx0 + bx1) / 2, by + 0.005)],
          c=OBS, lw=1.5, hw=3.2, hl=3.8)
    T(ax, 1.95, ys + 0.075, r"observed features $\mathbf{Z}$ also go straight to the head, which moves them",
      fs=FS_S, c=OBS, style="italic")
    # h: q into the windows; bus to gate and correction
    hx = 3.02
    arrow(ax, [(dx1, yc), (hx, yc)], head=False)
    ax.add_patch(Circle((hx, yc), 0.014, fc=INK, ec="none", zorder=5))
    arrow(ax, [(hx, yc), (hx, yc - 0.05), (wx[0] - 0.015, yc - 0.05)], lw=0.7)
    T(ax, 3.21, yc - 0.005, r"$\mathbf{q}$", fs=FS_S)
    T(ax, hx + 0.03, yc - 0.33, r"$\mathbf{h}_{k,i}$", fs=FS, ha="left")
    ky = yc + 0.12
    arrow(ax, [(2.25, yc), (2.25, 1.615), (3.26, 1.615), (3.26, ky), (wx[0] - 0.015, ky)], c=SUB, lw=0.6)
    ax.add_patch(Circle((2.25, yc), 0.012, fc=INK, ec="none", zorder=5))
    T(ax, 3.32, ky + 0.07, r"$\mathbf{k}$", fs=FS_S, c=SUB)
    ex = wx[2] + 7 * cell + 0.07
    T(ax, ex, yc + 0.20, r"$\pi_{k,i}=\mathrm{softmax}_{\mathcal{W}(i)}$", fs=FS_S, ha="left", c=OURS_D)
    T(ax, ex, yc + 0.095, r"$(\mathbf{q}^{\!\top}\mathbf{k}_j+\beta\,\mathbf{1}_{\rm id})$", fs=FS_S, ha="left", c=OURS_D)
    T(ax, ex, yc - 0.02, r"$\mathcal{W}(i)$: $w{\times}w$ in", fs=FS_S, ha="left", c=SUB)
    T(ax, ex, yc - 0.115, r"last $S$ frames", fs=FS_S, ha="left", c=SUB)
    ax.add_patch(Rectangle((ex, yc - 0.245), 0.05, 0.05, fc="none", ec=GATE_D, lw=1.0, zorder=5))
    T(ax, ex + 0.08, yc - 0.22, r"patch $i$", fs=FS_S, ha="left", c=GATE_D)

    # ---- output equation on real per-patch vectors
    ch, cw = 0.034, 0.08
    cy = 0.27
    top = cy + 12 * ch
    mid = cy + 6 * ch
    xs = dict(z0=3.60, T=wx[1] + 3.5 * cell - cw / 2, r=4.34, zh=4.60)
    T(ax, 3.42, mid, r"$(1{-}g)$", fs=FS, c=GATE_D)
    vec_chip(ax, xs["z0"], cy, VECS["Zzero"], cw, ch, edge=GATE_D, lw=0.9)
    T(ax, 3.82, mid, r"$+\,g$", fs=FS, c=GATE_D)
    vec_chip(ax, xs["T"], cy, VECS["Ttr"], cw, ch, edge=OURS_D, lw=0.9)
    T(ax, 4.25, mid, "$+$", fs=FS + 1)
    vec_chip(ax, xs["r"], cy, VECS["Rcor"], cw, ch, edge=INK, lw=0.5)
    T(ax, 4.51, mid, "$=$", fs=FS + 1)
    vec_chip(ax, xs["zh"], cy, VECS["Zhat"], cw, ch, edge=OURS, lw=1.2)
    for key, lab, col in (("z0", r"$\mathbf{Z}_{0,i}$", OBS), ("T", r"$\mathbf{T}_{k,i}$", OURS_D),
                          ("r", r"$\mathbf{r}_{k,i}$", INK), ("zh", r"$\hat{\mathbf{Z}}_{k,i}$", OURS_D)):
        T(ax, xs[key] + cw / 2, cy - 0.07, lab, fs=FS_S, c=col)
    T(ax, xs["z0"] + cw / 2, top + 0.055, "keep", fs=FS_S, c=SUB)
    T(ax, xs["r"] + cw / 2, top + 0.055, "new", fs=FS_S, c=SUB)
    tx = xs["T"] + cw / 2
    arrow(ax, [(tx, wy - 0.01), (tx, top + 0.01)], c=OURS_D, lw=0.9)
    T(ax, tx + 0.04, wy - 0.085, r"$\sum_j \pi(j)\,\mathbf{Z}_j$", fs=FS_S, c=OURS_D, ha="left")
    arrow(ax, [(hx, yc), (hx, 0.10), (3.30, 0.10)], lw=0.6, head=False)
    T(ax, 3.32, 0.09, r"$g=\sigma(\mathbf{w}_g^{\!\top}\mathbf{h}{+}b_g)$,", fs=FS_S, c=GATE_D, ha="left")
    T(ax, 4.10, 0.09, r"$\mathbf{r}=\mathbf{s}\odot\mathbf{W}_o\mathbf{h}$", fs=FS_S, c=INK, ha="left")
    sx = 4.86
    stack(ax, ["head_hat_k10.png", "head_hat_k5.png", "head_hat_k1.png"], sx, 0.33, 0.33, 0.05, 0.05, edge=OURS_D)
    arrow(ax, [(xs["zh"] + cw + 0.02, mid), (sx - 0.02, mid)], c=OURS_D, lw=0.9)
    T(ax, sx + 0.22, 0.235, r"$\hat{\mathbf{Z}}_{1:K}$", fs=FS_S, c=OURS_D)
    T(ax, W - 0.07, 0.09, r"$g{=}0$: Direct head", fs=FS_S, c=DIRECT, ha="right")
    finish(fig, "A", W)


# ================================================================== Candidate C: "rays through the observed volume"
def card(ax, name, x0, y0, s, edge=OBS, lw=0.5, z=3, grid=True):
    image(ax, name, x0, y0, s, s, edge=edge, lw=lw, z=z)
    if grid:
        grid_lines(ax, x0, y0, s, s, 16, lw=0.12, alpha=0.35, z=z + 0.05)


def patch_xy(x0, y0, s, r, c):
    return x0 + (c + 0.5) / 16 * s, y0 + (1 - (r + 0.5) / 16) * s


# ================================================================== Candidate A, v2 (polished after selection)
def design_A_v2():
    W, H = 5.5, 2.25
    fig, ax = new_fig(W, H)
    XL, XR = 3.10, 3.16                                    # left panel right edge / right panel left edge
    panel(ax, 0.02, 0.02, XL, H - 0.02, PANEL)
    panel(ax, XR, 0.02, W - 0.02, H - 0.02, PANEL_G)
    ty = H - 0.10
    T(ax, 0.10, ty, "Predictor", fs=FS_T, w="bold", ha="left")
    T(ax, 0.68, ty, "(shared with the baselines; or V-JEPA 2-AC, DINO-WM)", fs=FS_S, c=SUB, ha="left")
    T(ax, XR + 0.10, ty, "ShiftWM head: gated transport", fs=FS_T, w="bold", c=OURS_D, ha="left")

    yc = 1.12                                              # main row centre
    x0, fx1, zx, tx0, tx1 = frames_encoder(ax, yc, fw=0.34, tw=0.36)
    T(ax, x0 + 0.24, yc - 0.30, "observed frames", fs=FS_S, c=SUB)
    T(ax, zx + 0.17, yc - 0.30, r"$\mathbf{Z}_{-H+1:0}$", fs=FS_S, c=OBS)
    zr = zx + 0.30 + 0.09

    # ---- memory encoder
    ex0, ex1, ey0, ey1 = 1.46, 2.16, yc - 0.25, yc + 0.25
    box(ax, ex0, ey0, ex1, ey1)
    cx = (ex0 + ex1) / 2
    T(ax, cx, yc + 0.155, "Memory encoder", fs=FS_S, w="bold")
    T(ax, cx, yc + 0.04, r"$L_e$ layers", fs=FS_S)
    T(ax, cx, yc - 0.075, "+ space, frame,", fs=FS_S, c=SUB)
    T(ax, cx, yc - 0.175, "action emb.", fs=FS_S, c=SUB)
    arrow(ax, [(zr, yc), (ex0, yc)], c=OBS)

    # ---- decoder (K parallel copies, offset straight up-right)
    dx0, dx1, dy0, dy1 = 2.36, 2.94, yc - 0.25, yc + 0.25
    for k in (2, 1):
        box(ax, dx0 + k * 0.025, dy0 + k * 0.035, dx1 + k * 0.025, dy1 + k * 0.035, fc="#FAFBFC", lw=0.6)
    box(ax, dx0, dy0, dx1, dy1)
    cx = (dx0 + dx1) / 2
    T(ax, cx, yc + 0.155, r"Decoder $\times L_d$", fs=FS_S, w="bold")
    T(ax, cx, yc + 0.045, "patch self-attn", fs=FS_S)
    T(ax, cx, yc - 0.06, r"cross-attn: $\mathbf{M}$", fs=FS_S)
    T(ax, cx, yc - 0.165, r"AdaLN$(\mathbf{c}_k)$", fs=FS_S, c=ACT)
    T(ax, cx + 0.025, dy1 + 0.155, r"all $K$ steps in parallel", fs=FS_S, c=SUB)
    mj = 2.215                                              # M junction
    arrow(ax, [(ex1, yc), (dx0, yc)])
    ax.add_patch(Circle((mj, yc), 0.013, fc=INK, ec="none", zorder=5))
    T(ax, 2.29, yc + 0.075, r"$\mathbf{M}$", fs=FS)

    # ---- actions
    ya = 0.42
    ax.imshow(img("actions_spark.png"), extent=(0.14, 0.72, ya - 0.08, ya + 0.08), zorder=3)
    T(ax, 0.43, ya - 0.17, r"actions $\mathbf{a}_{-H+1:K-1}$", fs=FS_S, c=ACT)
    ax_ = 1.62
    arrow(ax, [(0.76, ya), (ax_, ya), (ax_, ey0)], c=ACT, lw=0.7)
    T(ax, ax_ - 0.05, 0.66, r"$\mathbf{a}_{<0}$", fs=FS_S, c=ACT, ha="right")
    gx0, gx1 = 1.95, 2.37
    arrow(ax, [(ax_, ya), (gx0, ya)], c=ACT, lw=0.7)
    T(ax, (ax_ + gx0) / 2, ya + 0.085, r"$\mathbf{a}_{0:K-1}$", fs=FS_S, c=ACT)
    box(ax, gx0, ya - 0.11, gx1, ya + 0.11, ec=ACT, lw=0.7)
    T(ax, (gx0 + gx1) / 2, ya + 0.04, "causal", fs=FS_S)
    T(ax, (gx0 + gx1) / 2, ya - 0.055, "GRU", fs=FS_S)
    pc = 2.56
    ax.add_patch(Circle((pc, ya), 0.045, fc="white", ec=ACT, lw=0.7, zorder=4))
    T(ax, pc, ya + 0.002, "+", fs=FS, c=ACT)
    arrow(ax, [(gx1, ya), (pc - 0.045, ya)], c=ACT, lw=0.7)
    arrow(ax, [(pc, 0.17), (pc, ya - 0.045)], c=ACT, lw=0.7)
    T(ax, pc + 0.05, 0.20, r"$\mathbf{e}_k$", fs=FS_S, c=ACT, ha="left")
    arrow(ax, [(pc, ya + 0.045), (pc, dy0)], c=ACT, lw=0.7)
    T(ax, pc + 0.05, 0.68, r"$\mathbf{c}_k$", fs=FS, c=ACT, ha="left")

    # ================= head panel: transport windows
    cell = 0.054
    ws = 7 * cell
    wx = [3.36, 3.80, 4.24]; wy = 0.97
    for s in range(3):
        window(ax, wx[s], wy, WIN[s], cell, ident=(s == 2))
        T(ax, wx[s] + ws / 2, wy + ws + 0.075,
          [r"$\mathbf{Z}_{-2}$", r"$\mathbf{Z}_{-1}$", r"$\mathbf{Z}_{0}$"][s] + f" {MASS[s]}%", fs=FS_S, c=OBS)
    wtop = wy + ws
    bx0, bx1, by = wx[0], wx[2] + ws, wtop + 0.165
    ax.plot([bx0, bx0, bx1, bx1], [by - 0.025, by, by, by - 0.025], color=OBS, lw=0.7, zorder=3)
    # lanes (top to bottom): title, italic stream label, stream, k-wire, "parallel" label
    ys, yk = 1.88, 1.72
    zc = zx + 0.17
    arrow(ax, [(zc, yc + 0.20), (zc, ys), ((bx0 + bx1) / 2, ys), ((bx0 + bx1) / 2, by + 0.005)],
          c=OBS, lw=1.5, hw=3.2, hl=3.8)
    T(ax, 1.97, ys + 0.10, r"observed features $\mathbf{Z}$ also go straight to the head, which moves them",
      fs=FS_S, c=OBS, style="italic")
    # h -> q ; M -> k
    hx = 3.03
    arrow(ax, [(dx1, yc), (hx, yc)], head=False)
    ax.add_patch(Circle((hx, yc), 0.014, fc=INK, ec="none", zorder=5))
    qy = yc - 0.06
    arrow(ax, [(hx, yc), (hx, qy), (wx[0] - 0.015, qy)], lw=0.7)
    T(ax, 3.20, qy + 0.07, r"$\mathbf{q}$", fs=FS_S)
    ky = wy + ws - 0.07
    kx = 3.27
    arrow(ax, [(mj, yc), (mj, yk), (kx, yk), (kx, ky), (wx[0] - 0.015, ky)], c=SUB, lw=0.6)
    T(ax, kx - 0.04, ky + 0.05, r"$\mathbf{k}$", fs=FS_S, c=SUB)
    T(ax, hx + 0.05, 0.86, r"$\mathbf{h}_{k,i}$", fs=FS, ha="left")
    # Eq. 1 legend
    ex = wx[2] + ws + 0.06
    T(ax, ex, wy + 0.33, r"$\pi_{k,i}=\mathrm{softmax}_{\mathcal{W}(i)}$", fs=FS_S, ha="left", c=OURS_D)
    T(ax, ex, wy + 0.225, r"$(\mathbf{q}^{\!\top}\mathbf{k}_j+\beta\,\mathbf{1}_{\rm id})$", fs=FS_S, ha="left", c=OURS_D)
    T(ax, ex, wy + 0.11, r"$\mathcal{W}(i)$: $w{\times}w$, last $S$", fs=FS_S, ha="left", c=SUB)
    ax.add_patch(Rectangle((ex, wy - 0.005), 0.05, 0.05, fc="none", ec=GATE_D, lw=1.0, zorder=5))
    T(ax, ex + 0.08, wy + 0.02, r"patch $i$", fs=FS_S, ha="left", c=GATE_D)

    # ---- output equation on real per-patch vectors, even pitch
    ch, cw = 0.030, 0.08
    cy = 0.42
    top = cy + 12 * ch
    mid = cy + 6 * ch
    tx = wx[1] + ws / 2                                    # T chip centred under the Z_-1 window column
    pitch = 0.40
    cen = dict(z0=tx - pitch, T=tx, r=tx + pitch, zh=tx + 2 * pitch - 0.08)
    for key, cols, ec, lw in (("z0", VECS["Zzero"], GATE_D, 0.9), ("T", VECS["Ttr"], OURS_D, 0.9),
                              ("r", VECS["Rcor"], INK, 0.5), ("zh", VECS["Zhat"], OURS, 1.2)):
        vec_chip(ax, cen[key] - cw / 2, cy, cols, cw, ch, edge=ec, lw=lw)
    T(ax, cen["z0"] - cw / 2 - 0.05, mid, r"$(1{-}g)$", fs=FS, c=GATE_D, ha="right")
    T(ax, (cen["z0"] + cen["T"]) / 2, mid, r"$+\;g$", fs=FS, c=GATE_D)
    T(ax, (cen["T"] + cen["r"]) / 2, mid, "$+$", fs=FS + 1)
    T(ax, (cen["r"] + cen["zh"]) / 2, mid, "$=$", fs=FS + 1)
    for key, lab, col, word in (("z0", r"$\mathbf{Z}_{0,i}$", OBS, "keep"), ("T", r"$\mathbf{T}_{k,i}$", OURS_D, "move"),
                                ("r", r"$\mathbf{r}_{k,i}$", INK, "new"), ("zh", r"$\hat{\mathbf{Z}}_{k,i}$", OURS_D, "")):
        T(ax, cen[key], cy - 0.065, lab, fs=FS_S, c=col)
        if word:
            T(ax, cen[key], cy - 0.155, word, fs=FS_S, c=SUB, style="italic")
    arrow(ax, [(tx, wy - 0.01), (tx, top + 0.015)], c=OURS_D, lw=0.9)
    T(ax, tx + 0.05, (wy + top) / 2 - 0.005, r"$\Sigma_j\,\pi(j)\,\mathbf{Z}_j$", fs=FS_S, c=OURS_D, ha="left")
    # h bus -> gate & correction formulas (inside the green panel, clear of edges)
    fy = 0.12
    arrow(ax, [(hx, qy), (hx, fy), (XR + 0.08, fy)], lw=0.6)
    T(ax, XR + 0.11, fy, r"$g=\sigma(\mathbf{w}_g^{\!\top}\mathbf{h}{+}b_g)$", fs=FS_S, c=GATE_D, ha="left")
    T(ax, 4.10, fy, r"$\mathbf{r}=\mathbf{s}\odot\mathbf{W}_o\mathbf{h}$", fs=FS_S, c=INK, ha="left")
    # forecast stack
    sx, ss = 4.95, 0.30
    stack(ax, ["head_hat_k10.png", "head_hat_k5.png", "head_hat_k1.png"], sx, cy + 0.02, ss, 0.045, 0.045,
          edge=OURS_D)
    arrow(ax, [(cen["zh"] + cw / 2 + 0.03, mid), (sx - 0.03, mid)], c=OURS_D, lw=0.9)
    T(ax, sx + ss / 2, cy - 0.065, r"$\hat{\mathbf{Z}}_{1:K}$", fs=FS_S, c=OURS_D)
    T(ax, W - 0.12, fy, r"$g{=}0$: Direct head", fs=FS_S, c=DIRECT, ha="right")
    finish(fig, "A_v2", W)


def design_C():
    import matplotlib.patheffects as pe
    W, H = 5.5, 2.05
    fig, ax = new_fig(W, H)
    s = 0.36
    panel(ax, 0.02, 0.02, W - 0.02, H - 0.02, PANEL)
    panel(ax, 1.66, 0.70, 4.02, H - 0.05, PANEL_G, r=0.05, z=0.5)
    T(ax, 0.10, H - 0.11, "Observed grids", fs=FS_T, w="bold", ha="left")
    T(ax, 1.76, H - 0.13, "ShiftWM head: move, keep, correct", fs=FS_T, w="bold", c=OURS_D, ha="left")
    T(ax, 4.12, H - 0.11, "Forecast", fs=FS_T, w="bold", ha="left", c=INK)

    # --- frames -> frozen DINOv2 -> column of observed grids (time runs downward)
    cx0 = 1.04
    rows = [1.48, 1.10, 0.72]
    names = ["obs_m2.png", "obs_m1.png", "obs_0.png"]
    frames = ["observed_0.png", "observed_1.png", "observed_2.png"]
    labs = [r"$\mathbf{Z}_{-2}$", r"$\mathbf{Z}_{-1}$", r"$\mathbf{Z}_{0}$"]
    ax.add_patch(Polygon([(0.53, 0.70), (0.53, 1.86), (0.76, 1.76), (0.76, 0.80)], closed=True,
                         fc="#E4E7EC", ec=BOXEDGE, lw=0.7, zorder=3))
    T(ax, 0.645, 1.28, "frozen DINOv2", fs=FS_S, rotation=90)
    tnode = (2.38, 1.62)
    cell = s / 16
    raw = WIN / WIN.sum()
    thr = np.sort(raw.ravel())[-15]
    for f in range(3):
        y0 = rows[f]; yc = y0 + s / 2
        image(ax, frames[f], 0.09, yc - 0.095, 0.34, 0.19, lw=0.35)
        arrow(ax, [(0.44, yc), (0.53, yc)], lw=0.6)
        arrow(ax, [(0.76, yc), (cx0 - 0.01, yc)], lw=0.6, c=OBS)
        T(ax, 0.90, yc + 0.07, labs[f], fs=FS_S, c=OBS)
        T(ax, 0.90, yc - 0.075, f"{MASS[f]}%", fs=FS_S, c=OURS_D)
        card(ax, names[f], cx0, y0, s, z=3)
        for r in range(7):
            for c in range(7):
                v = np.sqrt(WIN[f, r, c])
                ax.add_patch(Rectangle((cx0 + c * cell, y0 + s - (r + 1) * cell), cell, cell,
                                       fc=OURS, alpha=0.85 * v, ec="none", zorder=3.2))
        ax.add_patch(Rectangle((cx0, y0 + s - 7 * cell), 7 * cell, 7 * cell, fc="none", ec="white", lw=0.9, zorder=3.3))
        ax.add_patch(Rectangle((cx0, y0 + s - 7 * cell), 7 * cell, 7 * cell, fc="none", ec=OURS_D, lw=0.5, zorder=3.35))
        for r in range(7):
            for c in range(7):
                if raw[f, r, c] >= thr:
                    px, py = patch_xy(cx0, y0, s, r, c)
                    ax.plot([px, tnode[0]], [py, tnode[1]], color=OURS_D, lw=0.25 + 9 * raw[f, r, c],
                            alpha=0.7, zorder=4, solid_capstyle="round")
                    ax.add_patch(Circle((px, py), 0.010, fc=OURS_D, ec="none", zorder=4.1))
    T(ax, 0.28, 0.60, "input frames", fs=FS_S, c=SUB)
    ix, iy = patch_xy(cx0, rows[2], s, 3, 3)
    ax.add_patch(Rectangle((ix - cell / 2, iy - cell / 2), cell, cell, fc="none", ec=GATE_D, lw=1.0, zorder=9))

    # --- head
    ax.add_patch(Circle(tnode, 0.10, fc="white", ec=OURS_D, lw=1.0, zorder=12))
    T(ax, tnode[0] - 0.005, tnode[1] + 0.01, r"$\mathbf{T}_{k,i}$", fs=FS_S, c=OURS_D, z=13)
    T(ax, 1.76, 1.80, r"$\pi_{k,i}$: softmax over the $w{\times}w$ window $\mathcal{W}(i)$, last $S$ grids",
      fs=FS_S, c=OURS_D, ha="left")
    cx, cy = 3.25, iy
    ax.add_patch(Circle((cx, cy), 0.07, fc="white", ec=INK, lw=0.9, zorder=12))
    T(ax, cx, cy + 0.004, "+", fs=FS + 1.5, z=13)
    arrow(ax, [(tnode[0] + 0.10, tnode[1]), (cx, tnode[1]), (cx, cy + 0.07)], c=OURS_D, lw=0.9)
    T(ax, 2.50, 1.535, r"move: $g\sum_j\pi(j)\,\mathbf{Z}_j$", fs=FS_S, c=OURS_D, ha="left")
    kp = np.array([(ix + cell / 2, iy), (cx - 0.07, cy)])
    ax.plot(kp[:, 0], kp[:, 1], color=PANEL_G, lw=4.5, zorder=9.8)
    arrow(ax, kp, c=GATE_D, lw=0.9, z=10)
    T(ax, 2.00, cy + 0.075, r"keep: $(1{-}g)\,\mathbf{Z}_{0,i}$", fs=FS_S, c=GATE_D)
    # h -> query, gate, correction
    yl = 0.43
    hx = 2.80
    arrow(ax, [(2.64, yl + 0.03), (hx, yl + 0.03)], head=False)
    ax.add_patch(Circle((hx, yl + 0.03), 0.013, fc=INK, ec="none", zorder=5))
    arrow(ax, [(hx, yl + 0.03), (hx, 0.80), (cx, 0.80), (cx, cy - 0.07)], lw=0.8)
    arrow(ax, [(hx, 0.80), (hx, 1.40), (tnode[0], 1.40), (tnode[0], tnode[1] - 0.10)], lw=0.7)
    T(ax, 2.55, 1.34, "query", fs=FS_S, c=SUB, ha="left")
    T(ax, cx + 0.04, 0.855, r"correct: $\mathbf{r}_{k,i}$", fs=FS_S, ha="left")
    T(ax, hx + 0.035, 0.62, r"$\mathbf{h}_{k,i}$", fs=FS, ha="left")
    T(ax, hx + 0.035, 0.50, r"($g,\mathbf{r}$ from $\mathbf{h}$)", fs=FS_S, c=SUB, ha="left")
    T(ax, cx + 0.05, 1.30, r"$g{=}\sigma(\mathbf{w}_g^{\!\top}\mathbf{h}{+}b_g)$", fs=FS_S, c=GATE_D, ha="left")

    # --- forecast deck and true future
    ox, oy = 4.18, 0.95
    for j, nm in enumerate(["head_hat_k1.png", "head_hat_k5.png"]):
        card(ax, nm, ox + 0.14 - j * 0.07, oy + 0.18 - j * 0.09, 0.55, edge=OURS_D, z=3 + j)
    card(ax, "head_hat_k10.png", ox, oy, 0.55, edge=OURS_D, lw=0.9, z=6)
    pc = 0.55 / 16
    ax.add_patch(Rectangle((ox + 3 * pc, oy + 0.55 - 4 * pc), pc, pc, fc="none", ec="white", lw=1.4, zorder=8))
    ax.add_patch(Rectangle((ox + 3 * pc, oy + 0.55 - 4 * pc), pc, pc, fc="none", ec=OURS_D, lw=0.8, zorder=8.1))
    tx, ty = ox + 3.5 * pc, oy + 0.55 - 3.5 * pc
    arrow(ax, [(cx + 0.07, cy), (3.95, cy), (3.95, ty), (tx - pc / 2 - 0.01, ty)], c=OURS_D, lw=1.0, z=9)
    T(ax, 3.62, cy + 0.075, r"$\hat{\mathbf{Z}}_{k,i}$", fs=FS_S, c=OURS_D)
    T(ax, ox + 0.275, oy - 0.07, r"$\hat{\mathbf{Z}}_{k}$, $k{=}10$", fs=FS_S, c=OURS_D)
    T(ax, ox + 0.40, oy + 0.86, r"all $K$ steps at once", fs=FS_S, c=SUB)
    T(ax, 4.94, oy + 0.24, r"$\approx$", fs=FS + 2)
    card(ax, "head_true.png", 5.01, oy, 0.42, edge=INK, z=4)
    T(ax, 5.01 + 0.21, oy - 0.07, r"true $\mathbf{Z}_{10}$", fs=FS_S)
    T(ax, 4.78, 0.74, r"error: copy $\mathbf{Z}_0$ 0.44 $\to$ ShiftWM 0.28", fs=FS_S, c=INK)

    # --- predictor lane
    T(ax, 0.10, 0.44, "Predictor", fs=FS, w="bold", ha="left")
    T(ax, 0.10, 0.335, "shared; or V-JEPA", fs=FS_S, c=SUB, ha="left")
    T(ax, 0.10, 0.245, "2-AC, DINO-WM", fs=FS_S, c=SUB, ha="left")
    ex0, ex1 = 0.92, 1.62
    box(ax, ex0, yl - 0.15, ex1, yl + 0.15)
    T(ax, (ex0 + ex1) / 2, yl + 0.06, "Memory encoder", fs=FS_S, w="bold")
    T(ax, (ex0 + ex1) / 2, yl - 0.065, r"$L_e$ layers + emb.", fs=FS_S)
    arrow(ax, [(cx0 + s / 2, rows[2] - 0.01), (cx0 + s / 2, yl + 0.15)], c=OBS, lw=0.8)
    dx0, dx1 = 1.76, 2.58
    for k in (2, 1):
        box(ax, dx0 + k * 0.03, yl - 0.15 + k * 0.03, dx1 + k * 0.03, yl + 0.15 + k * 0.03, fc="#FAFBFC", lw=0.6)
    box(ax, dx0, yl - 0.15, dx1, yl + 0.15)
    T(ax, (dx0 + dx1) / 2, yl + 0.06, r"Decoder $\times L_d$", fs=FS_S, w="bold")
    T(ax, (dx0 + dx1) / 2, yl - 0.065, r"attn to $\mathbf{M}$, AdaLN$(\mathbf{c}_k)$", fs=FS_S)
    arrow(ax, [(ex1, yl), (dx0, yl)])
    T(ax, (ex1 + dx0) / 2, yl + 0.07, r"$\mathbf{M}$", fs=FS_S)
    ax.imshow(img("actions_spark.png"), extent=(0.95, 1.35, 0.06, 0.15), zorder=3)
    T(ax, 0.90, 0.105, r"actions $\mathbf{a}$", fs=FS_S, c=ACT, ha="right")
    arrow(ax, [(1.28, 0.17), (1.28, yl - 0.15)], c=ACT, lw=0.6)
    box(ax, 1.62, 0.05, 2.20, 0.16, ec=ACT, lw=0.6, r=0.02)
    T(ax, 1.91, 0.105, r"GRU $+\,\mathbf{e}_k$", fs=FS_S, c=ACT)
    arrow(ax, [(1.38, 0.105), (1.62, 0.105)], c=ACT, lw=0.6)
    arrow(ax, [(2.20, 0.105), (2.34, 0.105), (2.34, yl - 0.15)], c=ACT, lw=0.6)
    T(ax, 2.38, 0.19, r"$\mathbf{c}_k$", fs=FS_S, c=ACT, ha="left")
    T(ax, 4.78, 0.42, r"$g{=}0$ reduces to the", fs=FS_S, c=DIRECT)
    T(ax, 4.78, 0.32, r"Direct head $\mathbf{Z}_0{+}\mathbf{r}$", fs=FS_S, c=DIRECT)
    T(ax, 4.78, 0.15, "no forecast is fed back", fs=FS_S, c=SUB, style="italic")
    finish(fig, "C", W)


if __name__ == "__main__":
    which = sys.argv[1:] or ["A"]
    for w in which:
        globals()[f"design_{w}"]()
