"""'Anatomy of the gains' figure from results/v2/analysis/anatomy (scripts/v2/anatomy.py), held-out DROID.

(a,b) Error reduction of ShiftWM vs Direct / vs AR per horizon x decile of true per-patch change.
(c)   Composition of the forecast: mean gate and share of the departure from Z0 carried by moved features, per horizon.
(d)   Moving-patch error per horizon: stay, AR, Direct, ShiftWM and the model-free oracle move (95% episode CIs).
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/anatomy"
GREEN, BLUE, ORANGE, GREY = mf.METHODS["shiftwm"][1], mf.METHODS["direct"][1], mf.METHODS["ar"][1], "#8C95A1"
CMAP = LinearSegmentedColormap.from_list("gain", ["#F4FBF8", "#9ED9C3", GREEN, "#00543D"])


def heat(ax, m, title, vmax, cbar=False, fig=None):
    m = np.asarray(m)
    im = ax.imshow(m, origin="lower", cmap=CMAP, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest",
                   extent=(0.5, 10.5, 0.5, 10.5))
    ax.set_xticks([1, 5, 10]); ax.set_yticks([1, 10]); ax.set_yticklabels(["static", "fast"], fontsize=5.6)
    ax.tick_params(labelsize=5.6, length=2); ax.grid(False)
    ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1)
    lo, hi = m.min(), m.max()
    ax.set_title(title + (f": +{lo:.0f} to +{hi:.0f}% error" if lo > 0 else ""), fontsize=6.3, pad=2,
                 color=GREEN if lo > 0 else mf.INK)
    return im


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 1.9)); mf.pending(ax, "anatomy"); fig.savefig(mf.FIG / "anatomy.pdf"); return
    S = json.loads((D / "summary.json").read_text())
    fig = plt.figure(figsize=(5.5, 1.85))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.05, 1.05], wspace=0.42, left=0.06, right=0.995, top=0.86, bottom=0.2)
    shades = {1: "#9ED9C3", 5: "#2FB58A", 10: "#00543D"}
    for j, (base, title) in enumerate((("direct", "(a) gain over Direct"), ("ar", "(b) gain over AR"))):
        m = np.asarray(S["gain_map"][base])                                  # [decile, k]
        ax = fig.add_subplot(gs[0, j]); x = np.arange(1, 11)
        ax.axhline(0, color=mf.INK, lw=0.5)
        for kk, col in shades.items():
            ax.plot(x, m[:, kk - 1], color=col, lw=1.5 if kk == 10 else 1.1, marker="o", ms=2.2, label=f"$k{{=}}{kk}$")
        ax.fill_between(x, m.min(1), m.max(1), color=GREEN, alpha=0.08, lw=0)
        ax.set_xticks([1, 10]); ax.set_xticklabels(["static", "fast"], fontsize=5.6)
        ax.set_xlabel("true motion (decile)", fontsize=5.8, labelpad=0)
        ax.tick_params(labelsize=5.6, length=2); ax.set_ylim(bottom=min(0, m.min()) - 0.04 * m.max(), top=m.max() * 1.3)
        ax.set_title(title, fontsize=6.5, pad=2)
        if j == 0:
            ax.set_ylabel("error reduction (%)", fontsize=5.8, labelpad=1)
            ax.legend(fontsize=5.2, frameon=False, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
        ax.text(0.5 if j == 0 else 0.97, 0.05 if j == 0 else 0.95, f"> 0 in all {m.size} bins", transform=ax.transAxes, ha="center" if j == 0 else "right", va="bottom" if j == 0 else "top", fontsize=5.4,
                color=GREEN, fontweight="bold") if m.min() > 0 else None
    # (c) composition
    ax = fig.add_subplot(gs[0, 2]); k = np.arange(1, 11); c = S["composition"]
    ax.plot(k, c["moving"]["gate"], color=GREEN, lw=1.3, marker="o", ms=2.4, label="gate, moving")
    ax.plot(k, c["static"]["gate"], color=GREEN, lw=1.0, ls=(0, (3, 1.5)), label="gate, static")
    ax.plot(k, c["moving"]["move_share"], color=mf.INK, lw=1.2, marker="s", ms=2.2, label="moved share, moving")
    ax.set_ylim(0, 1.05); ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=5.6, length=2)
    ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1)
    ax.set_title("(c) forecast composition", fontsize=6.5, pad=2)
    ax.legend(fontsize=5.0, loc="lower right", frameon=False, handlelength=1.6, labelspacing=0.25)
    # (d) moving-patch error vs horizon
    ax = fig.add_subplot(gs[0, 3]); mc = S["moving_curves"]
    for a, lab, col, ls in (("persistence", "stay", GREY, (0, (3, 1.5))), ("ar", "AR", ORANGE, "-"), ("direct", "Direct", BLUE, "-"),
                            ("shiftwm", "ShiftWM", GREEN, "-"), ("oracle", "oracle move", "#B6BDC7", (0, (1, 1.2)))):
        ax.plot(k, mc[a]["mean"], color=col, ls=ls, lw=1.5 if a == "shiftwm" else 1.0, label=lab, zorder=3 if a == "shiftwm" else 2)
        ax.fill_between(k, mc[a]["lo"], mc[a]["hi"], color=col, alpha=0.18, lw=0)
    ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=5.6, length=2)
    ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1); ax.set_ylabel("error, moving patches", fontsize=5.8, labelpad=1)
    ax.set_title("(d) moving-part error", fontsize=6.5, pad=2); ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=5.0, loc="upper left", frameon=False, handlelength=1.6, labelspacing=0.25)
    fig.savefig(mf.FIG / "anatomy.pdf"); fig.savefig(mf.FIG / "anatomy_preview.png", dpi=200)
    gd, ga = np.asarray(S["gain_map"]["direct"]), np.asarray(S["gain_map"]["ar"])
    mv = c["moving"]
    lines = [f"\\def\\anaDirMin{{{gd.min():.0f}}}", f"\\def\\anaDirMax{{{gd.max():.0f}}}",
             f"\\def\\anaARMin{{{ga.min():.0f}}}", f"\\def\\anaARMax{{{ga.max():.0f}}}",
             f"\\def\\anaGateOne{{{mv['gate'][0]:.2f}}}", f"\\def\\anaGateTen{{{mv['gate'][-1]:.2f}}}",
             f"\\def\\anaMoveShare{{{100 * np.mean(mv['move_share']):.0f}}}",
             f"\\def\\anaCellsDir{{{int((gd > 0).sum())}}}", f"\\def\\anaCellsAR{{{int((ga > 0).sum())}}}"]
    (mf.ROOT / "paper/submission_folder/tables/generated/anatomy_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote anatomy")


if __name__ == "__main__":
    main()
