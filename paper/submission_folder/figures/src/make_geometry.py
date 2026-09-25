"""Feature-space geometry figure from results/v2/analysis/geometry (scripts/v2/geometry.py), held-out DROID, k=10.

(a) Three moving patches chosen by a fixed rule (see geometry.py), each in a 2-D PCA of its own observed transport
    candidates: candidates sized by learned transport weight, the observed feature at the same location (stay), the
    true future and the ShiftWM / Direct / AR forecasts.
(b) Moving-patch error at k=10: persistence, AR, Direct, ShiftWM and the oracle transport (best single observed feature
    in the window, chosen with the true future; a model-free reference).
(c) Per-patch error difference Direct - ShiftWM on all moving patches (right of 0: ShiftWM better).
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

D = mf.RES / "analysis/geometry"
GREEN, BLUE, ORANGE = mf.METHODS["shiftwm"][1], mf.METHODS["direct"][1], mf.METHODS["ar"][1]
GREY = "#8C95A1"


def example(ax, X, b, legend):
    ok = X["valid"][b]; cv = X["cand"][b][ok]; w = X["weights"][b][ok]
    pts = {k: X[k][b] for k in ("obs", "true", "shiftwm", "direct", "ar")}
    Y = np.concatenate([cv, np.stack(list(pts.values()))]); mu = Y.mean(0)
    _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False); P = Vt[:2].T
    q = (cv - mu) @ P; ww = w / w.max()
    o = np.argsort(ww)
    ax.scatter(q[o, 0], q[o, 1], s=3 + 55 * ww[o], c=ww[o], cmap="Greens", vmin=-0.25, vmax=1, edgecolors="none", zorder=1)
    sty = {"obs": (mf.METHODS["persistence"][1], "s", 24, "stay $\\mathbf{z}_{0,i}$"), "true": (mf.INK, "*", 90, "true future"),
           "shiftwm": (GREEN, "o", 38, "ShiftWM"), "direct": (BLUE, "D", 26, "Direct"), "ar": (ORANGE, "^", 30, "AR")}
    xy = {k: (v - mu) @ P for k, v in pts.items()}
    ax.annotate("", xy=xy["shiftwm"], xytext=xy["obs"], zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.9, shrinkA=3, shrinkB=3))
    for k, (c, m, s, lab) in sty.items():
        ax.scatter(*xy[k], c=c, marker=m, s=s, edgecolors="white", linewidths=0.6, zorder=4 if k != "true" else 5, label=lab)
    e = {k: float(((pts[k] - pts["true"]) ** 2).mean()) for k in ("shiftwm", "direct", "ar")}
    txt = " ".join(f"{n} {e[k]:.2f}" for k, n in (("shiftwm", "S"), ("direct", "D"), ("ar", "A")))
    ax.text(0.03, 0.03, "err " + txt, transform=ax.transAxes, fontsize=mf.FS_NOTE, color=mf.INK,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#C9CED6"); sp.set_linewidth(0.6)
    ax.set_aspect("equal", adjustable="datalim")
    if legend:
        ax.legend(fontsize=mf.FS_NOTE, loc="upper left", bbox_to_anchor=(0.0, -0.03), ncol=5, frameon=False,
                  handletextpad=0.2, columnspacing=0.9, markerscale=0.9)


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 2.2)); mf.pending(ax, "geometry"); fig.savefig(mf.FIG / "geometry.pdf"); return
    S = json.loads((D / "summary.json").read_text()); X = np.load(D / "examples.npz")
    mv = S["regions"]["moving"]
    fig = plt.figure(figsize=(5.5, 2.15))
    Wf, Hf = fig.get_figwidth(), fig.get_figheight()
    y0, y1 = 0.47, 1.78                                           # axes band (inches)
    box_ = lambda x0, x1: fig.add_axes([x0 / Wf, y0 / Hf, (x1 - x0) / Wf, (y1 - y0) / Hf])
    for b in range(3):
        ax = box_(0.05 + b * 0.93, 0.05 + b * 0.93 + 0.87); example(ax, X, b, legend=(b == 0))
        ax.set_title(f"moving patch {b + 1}", fontsize=mf.FS_LABEL, pad=2)
    fig.text(0.01, 0.975, "(a) One patch in feature space (PCA of its observed candidates; size = transport weight)",
             fontsize=mf.FS_TITLE, fontweight="bold", color=mf.INK, va="top")
    # (b) moving-patch error (horizontal bars)
    ax = box_(3.45, 4.3)
    names = [("oracle", "oracle move", "#D5DAE1"), ("shiftwm", "ShiftWM", GREEN), ("direct", "Direct", BLUE), ("ar", "AR", ORANGE),
             ("persistence", "stay", GREY)]
    v = [mv[k]["k10"] for k, _, _ in names]
    bars = ax.barh(range(5), v, color=[c for *_, c in names], height=0.66)
    bars[0].set_hatch("////"); bars[0].set_edgecolor(GREY); bars[0].set_linewidth(0.4)
    for y, (k, _, _) in enumerate(names):
        good = k == "shiftwm"
        ax.text(v[y] + 0.02, y, f"{v[y]:.2f}", va="center", fontsize=mf.FS_NOTE, color=GREEN if good else mf.INK,
                fontweight="bold" if good else "normal")
    ax.set_yticks(range(5)); ax.set_yticklabels([n for _, n, _ in names], fontsize=mf.FS_TICK)
    ax.tick_params(axis="x", labelsize=mf.FS_TICK); ax.set_xlim(0, max(v) * 1.28); ax.grid(axis="y", visible=False)
    ax.set_title("(b) Error, moving patches", fontsize=7, pad=3, loc="right")
    sh = mv["oracle_gain_share"]
    ax.set_xlabel(f"oracle gain recovered:\nS {100 * sh['shiftwm']:.0f}%, D {100 * sh['direct']:.0f}%, A {100 * sh['ar']:.0f}%",
                  fontsize=mf.FS_NOTE, labelpad=2)
    # (c) per-patch difference
    ax = box_(4.5, 5.44)
    bins = X["bins"]; c = (bins[:-1] + bins[1:]) / 2; h = X["diff_direct"] / X["diff_direct"].sum()
    ax.bar(c, h, width=bins[1] - bins[0], color=np.where(c > 0, GREEN, "#E3A79F"), edgecolor="none")
    ax.axvline(0, color=mf.INK, lw=0.6)
    wr = S["moving_patch_winrate"]
    ax.text(0.97, 0.95, f"ShiftWM better on\n{100 * wr['vs_direct']:.0f}% of moving patches", transform=ax.transAxes,
            ha="right", va="top", fontsize=mf.FS_NOTE, color=GREEN, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    ax.set_ylim(0, h.max() * 1.45); ax.set_xlim(-1.2, 1.2); ax.set_yticks([]); ax.tick_params(labelsize=mf.FS_TICK); ax.grid(False)
    ax.set_xlabel("Direct err $-$ ShiftWM err", fontsize=mf.FS_NOTE, labelpad=1)
    ax.set_title("(c) Per patch, vs. Direct", fontsize=7, pad=3, loc="right")
    mf.qa(fig, "geometry", 5.5)
    fig.savefig(mf.FIG / "geometry.pdf"); fig.savefig(mf.FIG / "geometry_preview.png", dpi=200)
    al = S["regions"]["all"]
    mac = {"geoPersist": mv["persistence"]["k10"], "geoOracle": mv["oracle"]["k10"], "geoShift": mv["shiftwm"]["k10"],
           "geoDirect": mv["direct"]["k10"], "geoAR": mv["ar"]["k10"]}
    lines = [f"\\def\\{k}{{{v:.2f}}}" for k, v in mac.items()]
    pct = {"geoOracleRed": 100 * (1 - mv["oracle"]["k10"] / mv["persistence"]["k10"]),
           "geoShareShift": 100 * mv["oracle_gain_share"]["shiftwm"], "geoShareDirect": 100 * mv["oracle_gain_share"]["direct"],
           "geoShareAR": 100 * mv["oracle_gain_share"]["ar"], "geoWinDirect": 100 * wr["vs_direct"], "geoWinAR": 100 * wr["vs_ar"]}
    lines += [f"\\def\\{k}{{{v:.0f}}}" for k, v in pct.items()]
    lines += [f"\\def\\geoWindows{{{S['windows']:,}}}".replace(",", "{,}"), f"\\def\\geoEpisodes{{{S['episodes']}}}"]
    (mf.ROOT / "paper/submission_folder/tables/generated/geometry_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote geometry")


if __name__ == "__main__":
    main()
