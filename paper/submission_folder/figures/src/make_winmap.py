"""Per-episode win map from results/v2/analysis/winmap (scripts/v2/winmap.py), held-out DROID, seed 0.

(a) Mean true feature change of every test episode (rows, sorted; the marginal of (b, c)).
(b,c) Relative error difference (ShiftWM - baseline) / baseline per episode x horizon; green = ShiftWM better,
      red = worse. Triangles mark episodes whose horizon-averaged difference is > 0 (ShiftWM loses).
(d) Horizon-averaged difference vs episode motion, for both baselines.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/winmap"
GREEN, BLUE, ORANGE, GREY = mf.METHODS["shiftwm"][1], mf.METHODS["direct"][1], mf.METHODS["ar"][1], "#8C95A1"
LOSS = "#B8433A"
CMAP = LinearSegmentedColormap.from_list("win", ["#00543D", GREEN, "#9ED9C3", "#F7F7F7", "#EDB3AB", "#D2685C", LOSS])
V = 0.3


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 2.2)); mf.pending(ax, "winmap"); fig.savefig(mf.FIG / "winmap.pdf"); return
    S = json.loads((D / "summary.json").read_text()); Z = np.load(D / "winmap.npz", allow_pickle=True)
    n = len(Z["motion"]); y = np.arange(n)
    fig = plt.figure(figsize=(5.5, 2.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.42, 1, 1, 0.055, 1.25], wspace=0.17, left=0.045, right=0.995, top=0.87,
                          bottom=0.17)
    # (a) motion marginal
    ax = fig.add_subplot(gs[0, 0])
    ax.barh(y, Z["motion"], height=1.0, color=GREY, lw=0)
    ax.set_ylim(-0.5, n - 0.5); ax.set_xlim(Z["motion"].max() * 1.05, 0)
    ax.set_yticks([0, n - 1]); ax.set_yticklabels(["1", str(n)], fontsize=5.6)
    ax.set_ylabel(f"test episode (sorted by motion)", fontsize=5.8, labelpad=0)
    ax.tick_params(labelsize=5.4, length=2); ax.grid(False)
    ax.set_xticks([0, 0.5]); ax.set_xlabel("true change", fontsize=5.8, labelpad=1)
    ax.set_title("(a) motion", fontsize=6.5, pad=2)
    norm = TwoSlopeNorm(0, -V, V)
    for j, (base, lab) in enumerate((("direct", "Direct"), ("ar", "AR"))):
        ax = fig.add_subplot(gs[0, 1 + j])
        r = Z[f"rel_{base}"]
        im = ax.imshow(r, origin="lower", aspect="auto", cmap=CMAP, norm=norm, interpolation="nearest",
                       extent=(0.5, 10.5, -0.5, n - 0.5))
        lose = np.where(Z[f"lose_{base}"])[0]
        ax.scatter(np.full(len(lose), 11.0), lose, marker="<", s=14, c=LOSS, edgecolors="white", linewidths=0.4,
                   clip_on=False, zorder=5)
        ax.set_xlim(0.5, 10.5); ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks([1, 5, 10]); ax.set_yticks([]); ax.tick_params(labelsize=5.6, length=2); ax.grid(False)
        ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1)
        s = S[base]
        ax.set_title(f"({'bc'[j]}) vs. {lab}: {100 * s['win_frac_avg']:.0f}% of episodes won", fontsize=6.3, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#C9CED6"); sp.set_linewidth(0.6)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax, extend="both", ticks=[-0.3, -0.15, 0, 0.15, 0.3])
    cb.ax.set_yticklabels(["$-$30%", "$-$15%", "0", "+15%", "+30%"], fontsize=5.2); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5)
    cb.set_label("(ShiftWM $-$ base) / base", fontsize=5.6, labelpad=1)
    # (d) horizon-averaged difference vs motion
    ax = fig.add_subplot(gs[0, 4])
    pos = ax.get_position(); ax.set_position([pos.x0 + 0.1, pos.y0, pos.width - 0.1, pos.height])
    ax.axhspan(0, 30, color=LOSS, alpha=0.07, lw=0)
    ax.axhline(0, color=mf.INK, lw=0.6)
    for base, lab, col, mk in (("direct", "vs. Direct", BLUE, "D"), ("ar", "vs. AR", ORANGE, "^")):
        r = 100 * Z[f"rel_{base}"].mean(1)
        ax.scatter(Z["motion"], r, s=6, c=col, marker=mk, alpha=0.75, edgecolors="none", label=lab)
        lo = Z[f"lose_{base}"]
        ax.scatter(Z["motion"][lo], r[lo], s=16, facecolors="none", edgecolors=LOSS, linewidths=0.7, marker="o")
    ax.set_xscale("log"); ax.set_ylim(-75, 22)
    ax.text(0.97, 0.955, "ShiftWM\nworse", transform=ax.transAxes, ha="right", va="top", linespacing=0.95, fontsize=5.3, color=LOSS)
    ax.tick_params(labelsize=5.6, length=2)
    ax.set_xlabel("episode motion (true change)", fontsize=5.8, labelpad=1)
    ax.set_ylabel("rel. difference, mean $k$ (%)", fontsize=5.8, labelpad=1)
    sd, sa = S["direct"]["spearman_motion_vs_rel"][0], S["ar"]["spearman_motion_vs_rel"][0]
    ax.set_title("(d) gain vs. motion", fontsize=6.3, pad=2)
    ax.legend(fontsize=5.3, loc="lower left", frameon=False, handletextpad=0.1, borderaxespad=0.2, markerscale=1.6)
    ax.text(0.97, 0.05, f"Spearman $\\rho$\nD {sd:.2f}, A {sa:.2f}", transform=ax.transAxes, fontsize=5.2, color=mf.INK, ha="right")
    fig.savefig(mf.FIG / "winmap.pdf"); fig.savefig(mf.FIG / "winmap_preview.png", dpi=200)
    sD, sA = S["direct"], S["ar"]
    lines = [f"\\def\\wmEpisodes{{{S['episodes']}}}", f"\\def\\wmWinDirect{{{100 * sD['win_frac_avg']:.0f}}}",
             f"\\def\\wmWinAR{{{100 * sA['win_frac_avg']:.0f}}}", f"\\def\\wmLoseDirect{{{sD['n_lose_avg']}}}",
             f"\\def\\wmLoseAR{{{sA['n_lose_avg']}}}", f"\\def\\wmCellsDirect{{{100 * sD['cells_lose']:.1f}}}",
             f"\\def\\wmCellsAR{{{100 * sA['cells_lose']:.1f}}}", f"\\def\\wmRhoDirect{{{sd:.2f}}}",
             f"\\def\\wmRhoAR{{{sa:.2f}}}", f"\\def\\wmLoseAllSeedsDirect{{{sD['n_lose_all_seeds']}}}",
             f"\\def\\wmLoseAllSeedsAR{{{sA['n_lose_all_seeds']}}}",
             f"\\def\\wmMedQOneDirect{{{-100 * sD['median_rel_by_motion_quartile'][0]:.0f}}}",
             f"\\def\\wmMedQFourDirect{{{-100 * sD['median_rel_by_motion_quartile'][-1]:.0f}}}",
             f"\\def\\wmMedQOneAR{{{-100 * sA['median_rel_by_motion_quartile'][0]:.0f}}}",
             f"\\def\\wmMedQFourAR{{{-100 * sA['median_rel_by_motion_quartile'][-1]:.0f}}}"]
    (mf.ROOT / "paper/submission_folder/tables/generated/winmap_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote winmap")


def compact():
    """Main-text panel: horizon-averaged relative difference per test episode vs. episode motion (panel (d) alone)."""
    if not (D / "summary.json").exists():
        return
    S = json.loads((D / "summary.json").read_text()); Z = np.load(D / "winmap.npz", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(2.05, 1.72))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.86, bottom=0.2)
    ax.axhspan(0, 30, color=LOSS, alpha=0.07, lw=0)
    ax.axhline(0, color=mf.INK, lw=0.6)
    for base, lab, col, mk in (("direct", "vs. Direct", BLUE, "D"), ("ar", "vs. AR", ORANGE, "^")):
        r = 100 * Z[f"rel_{base}"].mean(1)
        won = 100 * S[base]["win_frac_avg"]
        ax.scatter(Z["motion"], r, s=6, c=col, marker=mk, alpha=0.75, edgecolors="none", label=f"{lab} ({won:.0f}% won)")
        lo = Z[f"lose_{base}"]
        ax.scatter(Z["motion"][lo], r[lo], s=16, facecolors="none", edgecolors=LOSS, linewidths=0.7, marker="o")
    ax.set_xscale("log"); ax.set_ylim(-75, 22)
    ax.text(0.97, 0.955, "ShiftWM worse", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=LOSS)
    ax.tick_params(labelsize=5.8, length=2)
    ax.set_xlabel("episode motion (true change)", fontsize=6.2, labelpad=1)
    ax.set_ylabel("rel. error difference (%)", fontsize=6.2, labelpad=1)
    ax.set_title(f"{S['episodes']} held-out DROID episodes", fontsize=6.6, pad=2)
    ax.legend(fontsize=5.6, loc="lower right", frameon=False, handletextpad=0.1, borderaxespad=0.2, markerscale=1.6)
    fig.savefig(mf.FIG / "winmap_compact.pdf"); fig.savefig(mf.FIG / "winmap_compact_preview.png", dpi=250)
    plt.close(fig)
    print("wrote winmap_compact")


if __name__ == "__main__":
    main()
    compact()
