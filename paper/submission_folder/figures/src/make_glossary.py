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


EXTRA = [("Window, horizon", "$H{=}3$ observed and $K{=}10$ future steps; $k$ steps ahead"),
         ("Open / closed loop", "forecasting without new observations / re-planning after each chunk"),
         ("Success", "the final state is within the task's tolerance of the goal"),
         ("Proprioception", "the robot's measured joint and gripper state"),
         ("Ex vivo", "tissue removed from the body")]
IN_OVERVIEW = {"DROID", "Language-Table", "Open-H Hamlyn", "PushT", "Wall", "TwoRoom", "Reacher", "V-JEPA 2-AC",
               "DINO-WM", "LeWM"}                      # shown with their facts in the overview figure (fig:overview)
LIST_GROUPS = [("robot", "Robot data"), ("surg", "Surgical data"), ("sim", "Planning"), ("wm", "Baselines"),
               ("ours", "ShiftWM"), ("eval", "Evaluation"), ("other", "Other")]


def write_list():
    """Glossary description list typeset under the figure (tables/generated/glossary_list.tex): every term + definition,
    grouped by the figure's sectors, sector colours on the group names, two balanced columns."""
    def tex(t):
        return (t.replace("%", "\\%").replace("\u00d7", "$\\times$").replace("16$\\times$16", "$16{\\times}16$")
                .replace("w$\\times$w", "$w{\\times}w$").replace(" S frames", " $S$ frames"))
    groups = []
    for key, name in LIST_GROUPS:
        items = [t for t in TERMS.get(key, EXTRA if key == "other" else []) if t[0] not in IN_OVERVIEW]
        if key in ("robot", "surg"):                  # listing order: images first, as in the figure
            items = sorted(items, key=lambda t: t[0] not in ("DROID", "Language-Table", "Open-H Hamlyn",
                                                               "Practice task", "Phantom"))
        col = {"other": "#243447", "eval": OURS2}.get(key, SECT.get(key, "#243447")).lstrip("#")
        rows = [f"\\multicolumn{{2}}{{@{{}}l}}{{\\textcolor[HTML]{{{col}}}{{\\textbf{{{name}}}}}}}\\\\[1pt]"]
        rows += ["\\textbf{" + tex(t) + "} & " + tex(d) + " \\\\" for t, d in items]
        groups.append((len(items) + 1, rows))
    tot = sum(n for n, _ in groups); left, acc = [], 0
    for n, r in groups:
        if acc + n / 2 <= tot / 2 or not left:
            left.append(r); acc += n
    right = [r for _, r in groups[len(left):]]
    col = lambda gs: ("\\begin{tabular}[t]{@{}>{\\raggedright\\arraybackslash}p{0.31\\linewidth}"
                      ">{\\raggedright\\arraybackslash}p{0.67\\linewidth}@{}}\n"
                      + "\n\\addlinespace[2pt]\n".join("\n".join(g) for g in gs) + "\n\\end{tabular}")
    out = ("% Generated by figures/src/make_glossary.py (same term list as glossary.pdf)\n"
           "{\\scriptsize\\setlength{\\tabcolsep}{2pt}\\renewcommand{\\arraystretch}{1.0}\n"
           "\\begin{minipage}[t]{0.49\\linewidth}\n" + col(left) + "\n\\end{minipage}\\hfill\n"
           "\\begin{minipage}[t]{0.49\\linewidth}\n" + col(right) + "\n\\end{minipage}}\n")
    (mf.ROOT / "paper/submission_folder/tables/generated/glossary_list.tex").write_text(out)


def main():
    T = thumbs()
    W_, H_ = 5.5, 1.9
    fig = plt.figure(figsize=(W_, H_))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_); ax.set_ylim(0, H_); ax.set_aspect("equal"); ax.axis("off")
    cx, cy = W_ / 2, H_ / 2
    r_hub, r_in, r_out, r_th, s_th = 0.56, 0.6, 0.72, 0.97, 0.17
    pol = lambda r, a: (cx + r * np.cos(np.deg2rad(a)), cy + r * np.sin(np.deg2rad(a)))
    # ---------------- centre disc: the task, with real objects
    ax.add_patch(Circle((cx, cy), r_hub, fc="#F4F7F6", ec="#C9D3CF", lw=0.6, zorder=1))
    ax.text(cx, cy + 0.4, "the task", fontsize=FT, fontweight="bold", color=mf.INK, ha="center", va="center")
    f = 0.22
    for i in range(3):
        x0, y0 = cx - 0.42 + 0.04 * i, cy + 0.03 + 0.04 * i
        ax.imshow(T["droid"], extent=(x0, x0 + f, y0, y0 + f), zorder=3 + i)
        ax.add_patch(Rectangle((x0, y0), f, f, fill=False, ec="white", lw=0.5, zorder=3 + i))
    ax.add_patch(FancyArrowPatch((cx - 0.1, cy + 0.17), (cx + 0.06, cy + 0.17), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.7, color=mf.INK, zorder=7))
    for i, k in enumerate(("k1", "k5", "k10")):
        x0, y0 = cx + 0.1 + 0.045 * i, cy + 0.03 + 0.04 * i
        ax.imshow(imread(ASSETS / f"head_hat_{k}.png"), extent=(x0, x0 + f, y0, y0 + f), zorder=3 + i,
                  interpolation="nearest")
        ax.add_patch(Rectangle((x0, y0), f, f, fill=False, ec="white", lw=0.5, zorder=3 + i))
    ax.text(cx, cy - 0.06, "3 frames + actions\n$\\rightarrow$ 10 future\npatch-feature grids", fontsize=FS,
            color=mf.INK, ha="center", va="top", linespacing=1.05)
    # ---------------- ring: six sectors; image sectors on the sides, term sectors top / bottom / lower right
    s_th = 0.12
    sectors = [("ours", "ShiftWM", SECT["ours"], 62, 118), ("robot", "robot data", SECT["robot"], 120, 180),
               ("sim", "simulated", SECT["sim"], 182, 240), ("eval", "evaluation", OURS2, 242, 298),
               ("wm", "world models", SECT["wm"], 300, 358), ("surg", "surgical data", SECT["surg"], 0, 60)]
    for key, lab, col, a0, a1 in sectors:
        ax.add_patch(Wedge((cx, cy), r_out, a0 + 1, a1 - 1, width=r_out - r_in, fc=col, ec="none", zorder=2))
        am = (a0 + a1) / 2
        x, y = pol((r_in + r_out) / 2, am)
        rot = am - 90 if 0 < am % 360 < 180 else am + 90
        ax.text(x, y, lab, fontsize=FS, color="white", fontweight="bold", ha="center", va="center", rotation=rot,
                rotation_mode="anchor", zorder=3)
    # ---------------- real thumbnails on the circle (image sectors), name outside each
    names = {"droid": "DROID", "lt": "Language-Table", "knot": "Open-H Hamlyn", "peg": "practice task",
             "sut": "phantom", "pusht": "PushT", "wall": "Wall", "tworoom": "TwoRoom", "reacher": "Reacher"}
    angles = {"lt": 145, "droid": 165, "pusht": 181, "wall": 197, "tworoom": 213, "reacher": 229,
              "knot": 37, "peg": 20, "sut": 3}
    colof = {"lt": "robot", "droid": "robot", "pusht": "sim", "wall": "sim", "tworoom": "sim", "reacher": "sim",
             "knot": "surg", "peg": "surg", "sut": "surg"}
    r_th = r_out + 0.06 + s_th
    for tid, a in angles.items():
        col = SECT[colof[tid]]
        x, y = pol(r_th, a)
        im = ax.imshow(T[tid], extent=(x - s_th, x + s_th, y - s_th, y + s_th), zorder=4, interpolation="lanczos")
        im.set_clip_path(Circle((x, y), s_th, transform=ax.transData))
        ax.add_patch(Circle((x, y), s_th, fill=False, ec=col, lw=0.8, zorder=5))
        lx = x + (s_th + 0.05) * (1 if np.cos(np.deg2rad(a)) > 0 else -1)
        ax.text(lx, y, names[tid], fontsize=FS, color=col, fontweight="bold", va="center",
                ha="left" if np.cos(np.deg2rad(a)) > 0 else "right")
    # ---------------- term sectors: names only (definitions are listed under the figure)
    ax.text(*pol(r_out + 0.07, 90), "transport · window · gate · correction", fontsize=FS,
            color=mf.INK, ha="center", va="bottom")
    ax.text(*pol(r_out + 0.11, 270), "persistence · skill · rollout · moving patches", fontsize=FS,
            color=mf.INK, ha="center", va="top")
    ax.text(*pol(r_out + 0.2, 318), "V-JEPA 2-AC · DINO-WM · LeWM\nDirect · AR · AR-TF", fontsize=FS,
            color=mf.INK, ha="left", va="center", linespacing=1.15)
    mf.qa(fig, "glossary", W_)
    fig.savefig(mf.FIG / "glossary.pdf"); fig.savefig(mf.FIG / "glossary_preview.png", dpi=200)
    plt.close(fig)
    write_list()
    print("wrote glossary")


if __name__ == "__main__":
    main()
