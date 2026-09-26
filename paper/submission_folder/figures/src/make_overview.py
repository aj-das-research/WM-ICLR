"""Evaluation overview (App. A.2, fig:overview), radial: what is evaluated, on what, with which predictors. Scope only.

centre   the task (real frames -> real ShiftWM forecast grids) and the scale headline
ring 1   evaluation axes (coloured arcs, curved names) and their metrics (curved, outside the arc)
ring 2   model-coverage dots per benchmark (filled marker = predictor evaluated on that benchmark)
ring 3   benchmarks as real circular thumbnails, with a two-line label (name; embodiment, action, test size, horizon)
All facts come from the paper (tab:datasets, Sec. 4, App. B.3, App. D.4-D.5, tab:main, tab:plugin, fig:planning).
Thumbnails use fixed rules (see make_glossary.thumbs; DROID camera 2 = same instant from the second exterior camera).
Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_overview.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge
from matplotlib.textpath import TextPath
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402
import make_glossary as G  # noqa: E402

W, H = 5.5, 3.7
FS = mf.FS_NOTE
M = mf.METHODS
SL = mf.BACKBONE
# predictors: (name, colour, marker)
MODELS = [("ShiftWM", M["shiftwm"][1], "o"), ("Direct", M["direct"][1], "D"), ("AR", M["ar"][1], "^"),
          ("AR-TF", M["ar_tf"][1], "s"), ("persistence", M["persistence"][1], "o"), ("linear", "#B8BEC7", "o"),
          ("V-JEPA 2-AC", SL, "o"), ("V-JEPA 2-AC + head", M["shiftwm"][1], "p"), ("DINO-WM", SL, "s"),
          ("DINO-WM + head", M["shiftwm"][1], "P")]
# axes: key, name, colour, metrics, angular share
AXES = [("F", "held-out forecasting", mf.INK, "MSE · skill · action ranking", 140),
        ("Z", "zero-shot", "#3C5A99", "camera 2 · MSE", 62),
        ("P", "plug-in head", SL, "latent MSE · SSIM · LPIPS", 88),
        ("A", "analyses", M["shiftwm"][1], "oracle · gate off · IoU", 70)]
# benchmarks per axis: (thumb, name, facts, models evaluated)
M6 = [0, 1, 2, 3, 4, 5]
BENCH = {
    "F": [("lt", "Language-Table", "xArm · 2-D · 264 test · K=10", M6),
          ("knot", "Open-H Hamlyn", "dVRK · 80-D · 153 test · K=10", M6),
          ("droid", "DROID", "Franka · 35-D · 132 test · K=10", M6),
          ("iws", "IWS PushT / Box / Rope", "bimanual · 20/70/40-D · 3\u00d7200 test · K=12", [0, 1, 2, 3, 4])],
    "Z": [("droid2", "DROID camera 2", "unseen view · 132 test", M6)],
    "P": [("droid10", "V-JEPA 2-AC on DROID", "1.3B · same budget · K=10", [6, 7]),
          ("pushtd", "DINO-WM PushT", "official code and split", [8, 9])],
    "A": [("droid", "DROID test", "oracle move · segments", [0, 1, 2])],
}
ORDER = ["F", "Z", "P", "A"]


def frames():
    T = G.thumbs()
    ep = mf.pick_teaser_episode()
    T["droid2"] = G._square(mf.droid_frames(ep, camera="exterior_image_2_left", steps=(2,))[0])
    T["droid10"] = G._square(mf.droid_frames(ep, steps=(12,))[0])
    T["pushtd"] = np.load(mf.RES / "analysis/qual_best/dinowm_pusht.npz")["frame_obs"][0]
    man = json.loads((mf.ROOT / "data/v2/frames/iws_pusht/manifest.json").read_text())   # first IWS PushT test handle
    r = next(r for r in man["episodes"] if r["split"] == "test")
    with np.load(mf.ROOT / "data/v2/frames/iws_pusht" / r["file"]) as z:
        T["iws"] = z["images"][0]
    return T


def test_units():
    """Held-out real test units from tab:datasets (tables/datasets.tex): episodes for DROID/Hamlyn/Language-Table,
    recording handles for IWS (test count is per task; multiplied by the number of listed tasks)."""
    import re
    eps, handles = 0, 0
    for line in (mf.ROOT / "paper/submission_folder/tables/datasets.tex").read_text().splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 4 or not re.search(r"/\s*[\d{},]+\s*/", cells[2]):
            continue
        test = int(re.sub(r"[^0-9]", "", cells[2].split("/")[-1]))
        if cells[0].startswith("IWS"):
            handles += test * len(cells[0].replace("IWS", "").split("/"))
        else:
            eps += test
    return eps, handles


def curved(ax, text, cx, cy, r, a_mid, size=FS, color="white", weight="normal"):
    """Text set along a circle of radius r (inches), centred at angle a_mid (deg); upright in both halves."""
    fp = FontProperties(family=mf.plt.rcParams["font.family"], size=size, weight=weight)
    widths = [TextPath((0, 0), ch, prop=fp).get_extents().width / 72 if ch != " " else size * 0.28 / 72
              for ch in text]
    widths = [max(w_, size * 0.2 / 72) + size * 0.03 / 72 for w_ in widths]
    top = 0 < (a_mid % 360) < 180
    total = sum(widths) / r * 180 / np.pi
    a = a_mid + total / 2 if top else a_mid - total / 2
    for ch, w_ in zip(text, widths):
        da = w_ / r * 180 / np.pi
        ac = a - da / 2 if top else a + da / 2
        x, y = cx + r * np.cos(np.deg2rad(ac)), cy + r * np.sin(np.deg2rad(ac))
        rot = ac - 90 if top else ac + 90
        ax.text(x, y, ch, fontsize=size, color=color, fontweight=weight, ha="center", va="center", rotation=rot,
                rotation_mode="anchor")
        a = a - da if top else a + da


def main():
    T = frames()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    cx, cy = W / 2, H / 2
    r_hub, r1a, r1b, r_met, r_dot, r_th, s_th = 0.55, 0.59, 0.75, 0.83, 0.95, 1.17, 0.15
    pol = lambda r, a: (cx + r * np.cos(np.deg2rad(a)), cy + r * np.sin(np.deg2rad(a)))
    # ---------------- hub: the task and the scale headline
    ax.add_patch(Circle((cx, cy), r_hub, fc="#F4F7F6", ec="#C9D3CF", lw=0.6, zorder=1))
    f = 0.19
    for i in range(3):
        x0, y0 = cx - 0.36 + 0.035 * i, cy + 0.1 + 0.035 * i
        ax.imshow(T["droid"], extent=(x0, x0 + f, y0, y0 + f), zorder=3 + i)
        ax.add_patch(Rectangle((x0, y0), f, f, fill=False, ec="white", lw=0.5, zorder=3 + i))
    ax.add_patch(FancyArrowPatch((cx - 0.09, cy + 0.22), (cx + 0.05, cy + 0.22), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.7, color=mf.INK, zorder=7))
    for i, k in enumerate(("k1", "k5", "k10")):
        x0, y0 = cx + 0.08 + 0.035 * i, cy + 0.1 + 0.035 * i
        ax.imshow(imread(G.ASSETS / f"head_hat_{k}.png"), extent=(x0, x0 + f, y0, y0 + f), zorder=3 + i)
        ax.add_patch(Rectangle((x0, y0), f, f, fill=False, ec="white", lw=0.5, zorder=3 + i))
    n_bench = sum(len(v) for v in BENCH.values()) - 2          # V-JEPA and analyses rows reuse the DROID benchmark
    eps, handles = test_units()
    ax.text(cx, cy + 0.04, f"{n_bench} benchmarks · {len(MODELS)} predictors\n{len(AXES)} axes · paired tests (Holm)\n"
            f"real test: {eps} episodes\n+ {handles} IWS handles", fontsize=FS,
            color=mf.INK, ha="center", va="top", linespacing=1.1)
    print("headline:", n_bench, "benchmarks,", len(MODELS), "predictors,", len(AXES), "axes,", eps, "episodes +", handles,
          "IWS handles")
    # ---------------- rings
    a = 90 + AXES[0][4] / 2                                    # forecasting centred on the top
    spans = {}
    for key, name, col, met, share in AXES:
        a0, a1 = a - share, a
        spans[key] = (a0, a1)
        ax.add_patch(Wedge((cx, cy), r1b, a0 + 0.8, a1 - 0.8, width=r1b - r1a, fc=col, ec="none", zorder=2))
        curved(ax, name, cx, cy, (r1a + r1b) / 2, (a0 + a1) / 2, color="white", weight="bold")
        curved(ax, met, cx, cy, r_met, (a0 + a1) / 2, color=col)
        a = a0
    # ---------------- benchmarks: thumbnails, labels, model dots
    for key, *_ in AXES:
        a0, a1 = spans[key]
        items = BENCH[key]
        col = next(c for k, _, c, _, _ in AXES if k == key)
        for n, (tid, name, facts, mods) in enumerate(items):
            am = a1 - (a1 - a0) * (n + 0.5) / len(items)
            x, y = pol(r_th, am)
            im = ax.imshow(T[tid], extent=(x - s_th, x + s_th, y - s_th, y + s_th), zorder=4, interpolation="lanczos")
            im.set_clip_path(Circle((x, y), s_th, transform=ax.transData))
            ax.add_patch(Circle((x, y), s_th, fill=False, ec=col, lw=0.9, zorder=5))
            # dots: evaluated predictors, tangential row between the metrics arc and the thumbnail
            k_ = len(mods); step = 0.062 / r_dot * 180 / np.pi
            for i, j in enumerate(mods):
                ad = am + (i - (k_ - 1) / 2) * step * (-1 if 0 < am % 360 < 180 else 1)
                xd, yd = pol(r_dot, ad)
                _, c, mk = MODELS[j]
                ax.scatter([xd], [yd], s=9 if mk != "*" else 16, marker=mk, color=c, edgecolors="none" if mk != "x" else None,
                           linewidths=0.8 if mk == "x" else 0, zorder=6)
            # label outside the thumbnail
            ca, sa = np.cos(np.deg2rad(am)), np.sin(np.deg2rad(am))
            lx, ly = pol(r_th + s_th + 0.05, am)
            ha = "left" if ca > 0.05 else ("right" if ca < -0.05 else "center")
            va = "center" if abs(sa) < 0.75 else ("bottom" if sa > 0 else "top")
            dy = {"center": 0.0, "bottom": 0.105, "top": 0.0}[va]
            ax.text(lx, ly + dy + (0.005 if va == "center" else 0), name, fontsize=FS, color=col, fontweight="bold",
                    ha=ha, va="bottom" if va != "top" else "top")
            ax.text(lx, ly + dy - (0.005 if va == "center" else 0) - (0.105 if va == "top" else 0), facts, fontsize=FS,
                    color=mf.INK, ha=ha, va="top")
    # bold benchmark names: overlay the first line in bold (same position) -- simpler: draw names separately
    # ---------------- model legend (bottom-left corner), compact two columns
    lx0, ly0 = 0.03, 1.25
    ax.text(lx0, ly0 + 0.02, "predictors", fontsize=FS, color=mf.MUTED, fontweight="bold", va="bottom")
    for j, (name, c, mk) in enumerate(MODELS):
        x, y = lx0 + 0.05, ly0 - 0.07 - j * 0.088
        ax.scatter([x], [y], s=9 if mk != "*" else 16, marker=mk, color=c, linewidths=0.8 if mk == "x" else 0)
        ax.text(x + 0.06, y, name, fontsize=FS, color=mf.INK, va="center")
    mf.qa(fig, "overview", W)
    fig.savefig(mf.FIG / "overview.pdf"); fig.savefig(mf.FIG / "overview_preview.png", dpi=200)
    plt.close(fig)
    print("wrote overview")


if __name__ == "__main__":
    main()
