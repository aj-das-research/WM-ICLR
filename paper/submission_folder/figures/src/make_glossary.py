"""Visual glossary (App. A.2, fig:glossary): a hub (the forecasting task) with five sectors of terms.

Definitions reuse the wording of the former glossary table (paper/removed_floats/tab_glossary.tex) and of Sec. 4.
Thumbnails are real frames, chosen by a fixed rule (no manual picking):
  DROID            frame t of the teaser window (make_figures.pick_teaser_episode), centre square crop
  Language-Table   first test episode, first frame
  Open-H Hamlyn    first test episode of knot_tying / peg_transfer / suturing_1, first frame (native aspect, centre crop)
  PushT/TwoRoom/Reacher  first episode of the LeWM planning frame cache, first frame (224x224 environment render)
  Wall             first saved observation of the DINO-WM Wall qualitative cache (results/v2/analysis/qual_best)
Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_glossary.py
"""
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch, Wedge
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

sys.path.insert(0, str(mf.ROOT / "src"))
W, H = 5.5, 4.45
FS, FT = mf.FS_NOTE, 6.8
M = mf.METHODS
SECT = {"robot": "#3C5A99", "surg": "#9C4F6B", "sim": "#8A6D1E", "wm": mf.BACKBONE, "ours": M["shiftwm"][1]}


# ------------------------------------------------------------------------------------------------ real thumbnails
def _square(im):
    h, w = im.shape[:2]; s = min(h, w)
    return im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def thumbs():
    from shiftwm.v2 import analysis as A
    out = {}
    ep = mf.pick_teaser_episode()
    out["droid"] = _square(mf.droid_frames(ep, steps=(2,))[0])
    def first(ds, task=None, native=False):
        root = mf.ROOT / "data/v2/frames" / ds
        man = json.loads((root / "manifest.json").read_text())
        rows = [r for r in man["episodes"] if (task is None or r.get("task") == task)]
        test = [r for r in rows if r.get("split") == "test"] or rows
        r = test[0]
        if native:
            fs = A.FrameSource(A.cache_root(ds))
            return _square(fs.load_native(r["id"], [0])[0])
        with np.load(root / r["file"]) as z:
            return z["images"][0]
    out["lt"] = first("language_table")
    for k, t in (("knot", "knot_tying"), ("peg", "peg_transfer"), ("sut", "suturing_1")):
        try:
            out[k] = first("openh_hamlyn", t, native=True)
        except Exception as e:  # noqa: BLE001
            print("native Hamlyn frame unavailable, using cache frame:", e)
            out[k] = first("openh_hamlyn", t)
    for k in ("pusht", "tworoom", "reacher"):
        out[k] = first(f"plan_{k}")
    out["wall"] = np.load(mf.RES / "analysis/qual_best/dinowm_wall.npz")["frame_obs"][0]
    return out


# ------------------------------------------------------------------------------------------------ drawing helpers
def card(ax, x0, y0, x1, y1, title, col):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="white", ec=col, lw=0.8, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, y1 - 0.17), x1 - x0, 0.17, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=col, ec=col, lw=0.8, zorder=2))
    ax.add_patch(Rectangle((x0, y1 - 0.17), x1 - x0, 0.08, fc=col, ec="none", zorder=2))
    ax.text(x0 + 0.07, y1 - 0.085, title, fontsize=FT, color="white", fontweight="bold", va="center", zorder=3)


def item(ax, x, y, w, term, text, glyph=None, img=None, col=mf.INK, chars=None):
    """One entry, top-left at (x, y). Returns the height used."""
    tx, tw = x, w
    if img is not None:
        s = 0.36
        ax.imshow(img, extent=(x, x + s, y - s, y), interpolation="lanczos", zorder=3)
        ax.add_patch(Rectangle((x, y - s), s, s, fill=False, ec="#9AA3AE", lw=0.4, zorder=4))
        tx, tw = x + s + 0.06, w - s - 0.06
    elif glyph is not None:
        glyph(ax, x + 0.055, y - 0.075)
        tx, tw = x + 0.15, w - 0.15
    n = chars or int(tw * 27)                             # ~27 characters per inch at 6.2 pt STIX (measured)
    lines = textwrap.wrap(text, n)
    ax.text(tx, y, term, fontsize=FS, color=col, fontweight="bold", va="top", zorder=3)
    ax.text(tx, y - 0.093, "\n".join(lines), fontsize=FS, color=mf.INK, va="top", linespacing=1.0, zorder=3)
    hh = 0.093 + 0.09 * len(lines) + 0.03
    return max(hh, 0.40 if img is not None else hh)


def dot(col, shape="s"):
    def g(ax, x, y):
        if shape == "s":
            ax.add_patch(Rectangle((x - 0.035, y - 0.035), 0.07, 0.07, fc=col, ec="none", zorder=3))
        else:
            ax.scatter([x], [y], s=12, marker=shape, color=col, zorder=3, linewidths=0)
    return g


def gridglyph(kind):
    """3x3 patch-grid icon for a ShiftWM / evaluation term."""
    G, O = M["shiftwm"][1], "#D5DAE1"
    def g(ax, x, y):
        c = 0.034
        x0, y0 = x - 1.5 * c, y - 1.5 * c
        fill = {"feat": [[1] * 3] * 3, "moving": [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
                "persist": [[0] * 3] * 3}.get(kind, [[0] * 3] * 3)
        for i in range(3):
            for j in range(3):
                col = {"feat": ["#52627A", "#7A8BA3", "#A9B6C8"][(i + j) % 3], "moving": mf.LOSS if fill[i][j] else O,
                       "persist": M["persistence"][1]}.get(kind, O)
                if kind in ("window", "oracle") and (i, j) == (1, 1):
                    col = G
                if kind == "oracle" and (i, j) == (0, 2):
                    col = "#00543D"
                if kind == "gate":
                    col = mf.GATE if j == 2 else O
                if kind == "corr":
                    col = "#B5367A" if (i, j) == (1, 1) else O
                ax.add_patch(Rectangle((x0 + j * c, y0 + (2 - i) * c), c * 0.9, c * 0.9, fc=col, ec="none", zorder=3))
        if kind == "transport":
            ax.add_patch(FancyArrowPatch((x0, y0 + 0.2 * c), (x0 + 3 * c, y0 + 2.8 * c), arrowstyle="-|>",
                                         mutation_scale=4, lw=0.7, color=G, zorder=4))
        if kind == "window":
            ax.add_patch(Rectangle((x0 - 0.008, y0 - 0.008), 3 * c + 0.012, 3 * c + 0.012, fill=False, ec=G, lw=0.6, zorder=4))
        if kind in ("rollout", "tf"):
            ax.add_patch(FancyArrowPatch((x0 + 3 * c, y0 + 1.5 * c), (x0, y0 + 1.5 * c),
                                         connectionstyle="arc3,rad=0.9" if kind == "rollout" else "arc3,rad=0",
                                         arrowstyle="-|>", mutation_scale=4, lw=0.7,
                                         color=M["ar"][1] if kind == "rollout" else M["ar_tf"][1], zorder=4))
    return g


# ------------------------------------------------------------------------------------------------ figure
def main():
    T = thumbs()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    cx, cy, R = W / 2, 3.62, 0.7
    # ring of five coloured arcs around the hub, one per sector, with spokes to the cards
    ring = [("robot", 110, 170), ("sim", 190, 250), ("ours", 255, 285), ("wm", 290, 350), ("surg", 10, 70)]
    for k, a0, a1 in ring:
        ax.add_patch(Wedge((cx, cy), R + 0.1, a0, a1, width=0.07, fc=SECT[k], ec="none", alpha=0.85, zorder=1))
    spokes = {"robot": ((cx - 0.68, cy + 0.37), (1.76, cy + 0.37)), "surg": ((cx + 0.68, cy + 0.37), (W - 1.76, cy + 0.37)),
              "sim": ((cx - 0.72, cy - 0.35), (1.76, 2.25)), "wm": ((cx + 0.72, cy - 0.35), (W - 1.76, 2.25)),
              "ours": ((cx, cy - 0.8), (cx, 2.75))}
    for k, (p, q) in spokes.items():
        ax.plot(*zip(p, q), color=SECT[k], lw=0.8, zorder=1, alpha=0.8)
    # hub: the task
    ax.add_patch(Circle((cx, cy), R, fc="#F2F8F5", ec=M["shiftwm"][1], lw=1.0, zorder=2))
    ax.text(cx, cy + 0.47, "the task", fontsize=FT, fontweight="bold", color=mf.INK, ha="center", va="center", zorder=3)
    s = 0.26
    for i, dx in enumerate((-0.46, -0.38, -0.30)):                          # 3 observed frames (real, teaser window)
        ax.imshow(T["droid"], extent=(cx + dx, cx + dx + s, cy - 0.02 + 0.04 * i, cy + s - 0.02 + 0.04 * i),
                  zorder=3 + i)
        ax.add_patch(Rectangle((cx + dx, cy - 0.02 + 0.04 * i), s, s, fill=False, ec="white", lw=0.4, zorder=3 + i))
    ax.add_patch(FancyArrowPatch((cx + 0.02, cy + 0.15), (cx + 0.2, cy + 0.15), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.7, color=mf.INK, zorder=6))
    for i in range(4):                                                       # future feature grids (schematic)
        for j in range(4):
            ax.add_patch(Rectangle((cx + 0.24 + j * 0.065, cy + 0.02 + i * 0.065), 0.058, 0.058,
                                   fc=["#52627A", "#7A8BA3", "#A9B6C8"][(i * 3 + j) % 3], ec="none", zorder=5))
    ax.text(cx, cy - 0.09, "3 observed frames\n+ future actions $\\rightarrow$\nnext 10 frozen DINOv2\npatch-feature grids",
            fontsize=FS, color=mf.INK, ha="center", va="top", linespacing=1.0, zorder=6)
    # ---- sector cards
    cw = 1.74
    def column(x0, y_top, y_bot, key, title, entries):
        card(ax, x0, y_bot, x0 + cw, y_top, title, SECT[key])
        y = y_top - 0.23
        for e in entries:
            y -= item(ax, x0 + 0.07, y, cw - 0.13, *e) + 0.015
        if y < y_bot:
            print("overflow", title, round(y_bot - y, 3))
    column(0.0, H - 0.01, 2.42, "robot", "Robot data", [
        ("DROID", "real Franka arm episodes; split by recording session", None, T["droid"], SECT["robot"]),
        ("Language-Table", "xArm pushing blocks; long unscripted play data", None, T["lt"], SECT["robot"]),
        ("Episode / session", "one demonstration; episodes recorded consecutively at one site",
         dot(SECT["robot"]), None, SECT["robot"]),
        ("End-effector", "the tool at the arm's tip, a two-finger gripper", dot(SECT["robot"]), None, SECT["robot"]),
        ("Action block", "all commands of one model step, concatenated", dot(SECT["robot"]), None, SECT["robot"])])
    column(W - cw, H - 0.01, 2.42, "surg", "Surgical data", [
        ("Open-H Hamlyn", "dVRK recordings of seven practice tasks", None, T["knot"], SECT["surg"]),
        ("Practice task", "a standard surgeon exercise, e.g. peg transfer", None, T["peg"], SECT["surg"]),
        ("Phantom", "an artificial practice model of tissue", None, T["sut"], SECT["surg"]),
        ("dVRK", "da Vinci Research Kit, surgical robot", dot(SECT["surg"]), None, SECT["surg"]),
        ("Scene camera", "external camera, not an endoscope", dot(SECT["surg"]), None, SECT["surg"])])
    column(0.0, 2.32, 0.0, "sim", "Simulated suites (plug-in, planning)", [
        ("PushT", "push a T-shaped block to a target pose", None, T["pusht"], SECT["sim"]),
        ("Wall", "2-D agent moves through a door in a wall", None, T["wall"], SECT["sim"]),
        ("TwoRoom", "2-D navigation between two rooms (LeWM)", None, T["tworoom"], SECT["sim"]),
        ("Reacher", "a two-link arm reaches a target (LeWM)", None, T["reacher"], SECT["sim"]),
        ("MPC / CEM", "plan, execute the start, re-plan; sample and refit", dot(SECT["sim"]), None, SECT["sim"])])
    column(W - cw, 2.32, 0.0, "wm", "World models and baselines", [
        ("V-JEPA 2-AC", "published action-conditioned WM, post-trained on DROID", dot(mf.BACKBONE, "o"), None, mf.BACKBONE),
        ("DINO-WM", "published WM on DINOv2 patch features", dot(mf.BACKBONE, "o"), None, mf.BACKBONE),
        ("LeWM", "released end-to-end WM; planning reference", dot(mf.BACKBONE, "*"), None, mf.BACKBONE),
        ("Direct", "regresses each future grid: $\\mathbf{Z}_0$ + learned residual", dot(M["direct"][1], "D"), None,
         M["direct"][1]),
        ("AR", "residual to the latest grid, rolled out recursively", dot(M["ar"][1], "^"), None, M["ar"][1]),
        ("AR-TF", "DINO-WM-style: teacher-forced next step, rolled out", dot(M["ar_tf"][1], "s"), None, M["ar_tf"][1])])
    # ShiftWM + evaluation terms: two narrow columns under the hub
    x0, x1, yt, yb = 1.82, W - 1.82, 2.75, 0.0
    card(ax, x0, yb, x1, yt, "ShiftWM and evaluation terms", SECT["ours"])
    half = (x1 - x0 - 0.1) / 2
    left = [("Patch features", "frozen DINOv2 16\u00d716 grid, 384-D each", gridglyph("feat")),
            ("Transport", "each patch takes content from observed ones", gridglyph("transport")),
            ("Transport window", "w\u00d7w patches in each of the last S frames", gridglyph("window")),
            ("Gate", "per-patch weight: keep vs. move", gridglyph("gate")),
            ("Correction", "added residual for new content", gridglyph("corr")),
            ("Oracle move", "observed feature closest to the truth", gridglyph("oracle"))]
    right = [("Persistence", "copy the last observed grid", gridglyph("persist")),
             ("Moving patches", "25% with the largest true change", gridglyph("moving")),
             ("Skill", "share of persistence error removed", dot(M["shiftwm"][1])),
             ("Rollout", "own predictions fed back as inputs", gridglyph("rollout")),
             ("Teacher forcing", "train on true past, not own output", gridglyph("tf")),
             ("Session split", "no recording session in two splits", dot(SECT["robot"]))]
    for col_x, entries in ((x0 + 0.04, left), (x0 + 0.06 + half, right)):
        y = yt - 0.23
        for t_, d_, g_ in entries:
            y -= item(ax, col_x, y, half, t_, d_, glyph=g_, col=mf.INK, chars=int((half - 0.15) * 25)) + 0.012
        if y < yb:
            print("overflow ours", round(yb - y, 3))
    mf.qa(fig, "glossary", W)
    fig.savefig(mf.FIG / "glossary.pdf"); fig.savefig(mf.FIG / "glossary_preview.png", dpi=200)
    plt.close(fig)
    print("wrote glossary")


if __name__ == "__main__":
    main()
