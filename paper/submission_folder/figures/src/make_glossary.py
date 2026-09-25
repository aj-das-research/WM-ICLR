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


# ------------------------------------------------------------------------------------------------ radial layout
from matplotlib.image import imread  # noqa: E402

ASSETS = Path(__file__).parent / "arch_assets"
OURS2 = "#00785A"                                   # evaluation terms: darker shade of the ShiftWM green
# sectors: key, ring label, colour, angle span (deg, counter-clockwise from +x), side of the text block
SECTORS = [("ours", "ShiftWM", SECT["ours"], 66, 114), ("robot", "robot data", SECT["robot"], 119, 167),
           ("sim", "simulated suites", SECT["sim"], 172, 246), ("eval", "evaluation", OURS2, 251, 289),
           ("wm", "world models", SECT["wm"], 294, 353), ("surg", "surgical data", SECT["surg"], -2, 64)]
TERMS = {
    "robot": [("Episode / session", "one demonstration; episodes recorded consecutively at one site"),
              ("End-effector", "the tool at the arm's tip, a two-finger gripper"),
              ("Action block", "all commands of one model step, concatenated"),
              ("Language-Table", "xArm pushing blocks; long unscripted play data"),
              ("DROID", "real Franka arm episodes; split by recording session")],
    "surg": [("dVRK", "da Vinci Research Kit, a surgical robot"),
             ("Scene camera", "an external camera, not an endoscope"),
             ("Open-H Hamlyn", "dVRK recordings of seven practice tasks"),
             ("Practice task", "a standard surgeon exercise, e.g. peg transfer"),
             ("Phantom", "an artificial practice model of tissue")],
    "sim": [("PushT", "push a T-shaped block to a target pose"),
            ("Wall", "2-D agent moves through a door in a wall"),
            ("TwoRoom", "2-D navigation between two rooms (LeWM)"),
            ("Reacher", "a two-link arm reaches a target (LeWM)"),
            ("MPC / CEM", "plan, act, re-plan; sample and refit")],
    "wm": [("V-JEPA 2-AC", "published action-conditioned WM, post-trained on DROID"),
           ("DINO-WM", "published WM on DINOv2 patch features"),
           ("LeWM", "released end-to-end WM; planning reference"),
           ("Direct", "regresses each future grid as $\\mathbf{Z}_0$ + residual"),
           ("AR", "residual to the latest grid, rolled out recursively"),
           ("AR-TF", "DINO-WM-style: teacher-forced next step, rolled out")],
    "ours": [("Patch features", "frozen DINOv2 16×16 grid"),
             ("Transport window", "w×w patches in the last S frames"),
             ("Transport", "content moved from observed patches"),
             ("Correction", "added residual for new content"),
             ("Gate", "per-patch weight: keep vs. move"),
             ("Oracle move", "observed feature closest to the truth")],
    "eval": [("Persistence", "copy the last observed grid"),
             ("Rollout", "own predictions fed back as inputs"),
             ("Moving patches", "25% with the largest true change"),
             ("Teacher forcing", "train on true past, not own output"),
             ("Skill", "share of persistence error removed"),
             ("Session split", "no recording session in two splits")],
}
THUMB = {"robot": ["lt", "droid"], "surg": ["sut", "peg", "knot"], "sim": ["pusht", "wall", "tworoom", "reacher"]}
TERM_COL = {"wm": {"Direct": M["direct"][1], "AR": M["ar"][1], "AR-TF": M["ar_tf"][1]}}


ENTRY_TXT = {}


def entry(ax, x, y, w, term, text, col, ha="left"):
    """Term (bold, sector colour) over its definition; returns the height used."""
    lines = textwrap.wrap(text, max(12, int(w * 27)))
    ENTRY_TXT[term] = ax.text(x, y, term, fontsize=FS, color=col, fontweight="bold", va="top", ha=ha)
    ax.text(x, y - 0.088, "\n".join(lines), fontsize=FS, color=mf.INK, va="top", ha=ha, linespacing=1.0)
    return 0.088 + 0.088 * len(lines) + 0.028


def main():
    T = thumbs()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    cx, cy = W / 2, H / 2
    r_hub, r_in, r_out, r_th, s_th = 0.6, 0.64, 0.8, 0.975, 0.135
    pol = lambda r, a: (cx + r * np.cos(np.deg2rad(a)), cy + r * np.sin(np.deg2rad(a)))
    # ---------------- centre disc: the task, drawn with real objects
    ax.add_patch(Circle((cx, cy), r_hub, fc="#F4F7F6", ec="#C9D3CF", lw=0.6, zorder=1))
    ax.text(cx, cy + 0.44, "the task", fontsize=FT, fontweight="bold", color=mf.INK, ha="center", va="center")
    f = 0.24
    for i in range(3):                                            # 3 observed frames (teaser window, frame t)
        x0, y0 = cx - 0.45 + 0.045 * i, cy + 0.02 + 0.045 * i
        ax.imshow(T["droid"], extent=(x0, x0 + f, y0, y0 + f), zorder=3 + i)
        ax.add_patch(Rectangle((x0, y0), f, f, fill=False, ec="white", lw=0.5, zorder=3 + i))
    ax.add_patch(FancyArrowPatch((cx - 0.1, cy + 0.17), (cx + 0.07, cy + 0.17), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.7, color=mf.INK, zorder=7))
    for i, k in enumerate(("k1", "k5", "k10")):                   # real ShiftWM forecasts at k = 1, 5, 10 (PCA colours)
        x0, y0 = cx + 0.1 + 0.05 * i, cy + 0.02 + 0.045 * i
        ax.imshow(imread(ASSETS / f"head_hat_{k}.png"), extent=(x0, x0 + f, y0, y0 + f), zorder=3 + i,
                  interpolation="nearest")
        ax.add_patch(Rectangle((x0, y0), f, f, fill=False, ec="white", lw=0.5, zorder=3 + i))
    ax.text(cx, cy - 0.06, "3 frames + actions\n$\\rightarrow$ 10 future\npatch-feature grids",
            fontsize=FS, color=mf.INK, ha="center", va="top", linespacing=1.05)
    # ---------------- middle ring: sector arcs with tangential labels
    for key, lab, col, a0, a1 in SECTORS:
        ax.add_patch(Wedge((cx, cy), r_out, a0, a1, width=r_out - r_in, fc=col, ec="white", lw=0.8, zorder=2))
        am = (a0 + a1) / 2
        x, y = pol((r_in + r_out) / 2, am)
        rot = am - 90 if 0 < am % 360 < 180 else am + 90
        ax.text(x, y, lab, fontsize=FS, color="white", fontweight="bold", ha="center", va="center", rotation=rot,
                rotation_mode="anchor", zorder=3)
    # ---------------- outer ring: real thumbnails on the circle, within their sector
    anchors = {}
    for key, _, col, a0, a1 in SECTORS:
        ids = THUMB.get(key, [])
        for n, tid in enumerate(ids):
            a = a0 + (a1 - a0) * (n + 0.5) / len(ids)
            x, y = pol(r_th, a)
            im = ax.imshow(T[tid], extent=(x - s_th, x + s_th, y - s_th, y + s_th), zorder=4, interpolation="lanczos")
            clip = Circle((x, y), s_th, transform=ax.transData)
            im.set_clip_path(clip)
            ax.add_patch(Circle((x, y), s_th, fill=False, ec=col, lw=0.9, zorder=5))
            anchors[tid] = (x, y, a)
    # ---------------- text blocks: left / right columns and top / bottom bands, no boxes
    colw = 1.5
    def block(key, x, y_top, w, ncol=1, ha="left", at=None):
        _, lab, col, a0, a1 = next(sct for sct in SECTORS if sct[0] == key)
        ax.text(x if ha == "left" else x + w, y_top, lab.upper() if key not in ("ours",) else "SHIFTWM TERMS",
                fontsize=FS, color=col, fontweight="bold", va="top", ha=ha)
        ax.plot([x, x + w], [y_top - 0.105] * 2, color=col, lw=0.6)
        ys, pos = [y_top - 0.14] * ncol, {}
        cwid = (w - 0.08 * (ncol - 1)) / ncol
        for n, (t_, d_) in enumerate(TERMS[key]):
            c = n % ncol
            xx = x + c * (cwid + 0.08)
            if at and t_ in at:                       # align thumbnail terms with their thumbnail on the ring
                ys[c] = min(ys[c], at[t_])
            pos[t_] = (xx, ys[c])
            ys[c] -= entry(ax, xx if ha == "left" else xx + cwid, ys[c], cwid, t_, d_,
                           TERM_COL.get(key, {}).get(t_, col), ha=ha)
        return min(ys), pos
    top_cols = {}
    link = {"droid": ("robot", "DROID"), "lt": ("robot", "Language-Table"), "knot": ("surg", "Open-H Hamlyn"),
            "peg": ("surg", "Practice task"), "sut": ("surg", "Phantom"), "pusht": ("sim", "PushT"),
            "wall": ("sim", "Wall"), "tworoom": ("sim", "TwoRoom"), "reacher": ("sim", "Reacher")}
    at = {term: anchors[tid][1] + 0.045 for tid, (_, term) in link.items()}
    lo, top_cols["robot"] = block("robot", 0.03, H - 0.02, colw, at=at)
    print("robot bottom", round(lo, 3))
    lo, top_cols["sim"] = block("sim", 0.03, min(lo - 0.06, 2.42), colw, at=at)
    print("sim bottom", round(lo, 3))
    lo, top_cols["surg"] = block("surg", W - colw - 0.03, H - 0.02, colw, ha="right", at=at)
    print("surg bottom", round(lo, 3))
    lo, top_cols["wm"] = block("wm", W - colw - 0.03, min(lo - 0.06, 2.08), colw, ha="right")
    print("wm bottom", round(lo, 3))
    bw = 2.2
    lo, _ = block("ours", cx - bw / 2, H - 0.02, bw, ncol=2)
    print("ours bottom", round(lo, 3), "circle top", round(cy + r_th + s_th, 3))
    lo, _ = block("eval", cx - bw / 2, cy - r_th - s_th - 0.08, bw, ncol=2)
    print("eval bottom", round(lo, 3))
    # ---------------- leaders: thumbnail -> its term (sector colour, thin)
    for tid, (key, term) in link.items():
        x, y, a = anchors[tid]
        right = key == "surg"
        bb = ENTRY_TXT[term].get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        ex = bb.x0 - 0.04 if right else bb.x1 + 0.04
        ey = (bb.y0 + bb.y1) / 2
        u = np.array([ex - x, ey - y]); u = u / np.linalg.norm(u)      # leave the thumbnail towards its term
        sx, sy = x + s_th * u[0], y + s_th * u[1]
        col = next(sct[2] for sct in SECTORS if sct[0] == key)
        ax.plot([sx, ex], [sy, ey], color=col, lw=0.5, alpha=0.8, zorder=1)
        ax.add_patch(Circle((ex, ey), 0.012, fc=col, ec="none"))
    mf.qa(fig, "glossary", W)
    fig.savefig(mf.FIG / "glossary.pdf"); fig.savefig(mf.FIG / "glossary_preview.png", dpi=200)
    plt.close(fig)
    print("wrote glossary")


if __name__ == "__main__":
    main()
