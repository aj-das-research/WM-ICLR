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
    from scipy.ndimage import zoom
    fig = plt.figure(figsize=(5.5, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.02, 1.02], wspace=0.36, left=0.055, right=0.995, top=0.86, bottom=0.2)
    # (a,b) gain landscapes: smooth filled contours over horizon x motion decile
    for j, (base, title) in enumerate((("direct", "(a) gain over Direct (%)"), ("ar", "(b) gain over AR (%)"))):
        m = np.asarray(S["gain_map"][base])                                  # [decile, k]
        f = zoom(m, 6, order=3, mode="nearest")
        x = np.linspace(1, 10, f.shape[1]); y = np.linspace(1, 10, f.shape[0])
        lv = np.linspace(0, np.ceil(f.max() / 5) * 5, 11) if base == "ar" else np.linspace(0, np.ceil(f.max()), 8)
        ax = fig.add_subplot(gs[0, j])
        ax.contourf(x, y, f, levels=lv, cmap=CMAP, extend="both")
        cl = ax.contour(x, y, f, levels=lv[1::2], colors="white", linewidths=0.5, alpha=0.9)
        ax.clabel(cl, fmt="%d", fontsize=4.8, inline=True, inline_spacing=2)
        pk = np.unravel_index(np.argmax(m), m.shape)
        ax.scatter(pk[1] + 1, pk[0] + 1, marker="*", s=40, c="white", edgecolors=mf.INK, linewidths=0.5, zorder=4)
        ax.set_xticks([1, 5, 10]); ax.set_yticks([1, 10]); ax.set_yticklabels(["static", "fast"] if j == 0 else ["", ""], fontsize=5.6)
        ax.tick_params(labelsize=5.6, length=2); ax.grid(False)
        ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1)
        if j == 0:
            ax.set_ylabel("true motion (decile)", fontsize=5.8, labelpad=0)
        ax.set_title(title, fontsize=6.5, pad=2)
        if m.min() > 0:
            ax.text(0.97, 0.96, f"+{m.min():.0f}% to +{m.max():.0f}% ($\\star$ max)", transform=ax.transAxes, fontsize=5.0, color=mf.INK, ha="right", va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
    # (c) forecast composition on moving patches: stacked keep / move shares
    k = np.arange(1, 11); c = S["composition"]
    ax = fig.add_subplot(gs[0, 2]); g = np.array(c["moving"]["gate"])
    ax.stackplot(k, g, 1 - g, colors=[GREEN, "#D5DAE1"], alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.plot(k, c["static"]["gate"], color=mf.INK, lw=0.9, ls=(0, (3, 1.5)))
    ax.text(6.5, 0.35, "move  $g$", fontsize=6.2, color="white", fontweight="bold", ha="center")
    ax.text(6.5, 0.92, "keep  $1{-}g$", fontsize=6.2, color=mf.INK, ha="center")
    ax.text(9.8, c["static"]["gate"][-1] - 0.06, "gate on static patches", fontsize=5.0, color="white", ha="right", va="top")
    ax.set_xlim(1, 10); ax.set_ylim(0, 1); ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=5.6, length=2); ax.grid(False)
    ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1); ax.set_ylabel("share of forecast", fontsize=5.8, labelpad=1)
    ax.set_title("(c) moving parts are moved", fontsize=6.5, pad=2)
    # (d) share of the oracle-move gain recovered on moving patches
    ax = fig.add_subplot(gs[0, 3]); mc = S["moving_curves"]
    st, orc = np.array(mc["persistence"]["mean"]), np.array(mc["oracle"]["mean"])
    rec = {a: 100 * (st - np.array(mc[a]["mean"])) / (st - orc) for a in ("shiftwm", "direct", "ar")}
    best = np.maximum(rec["direct"], rec["ar"])
    ax.fill_between(k, best, rec["shiftwm"], color=GREEN, alpha=0.22, lw=0)
    for a, lab, col in (("ar", "AR", ORANGE), ("direct", "Direct", BLUE), ("shiftwm", "ShiftWM", GREEN)):
        ax.plot(k, rec[a], color=col, lw=1.6 if a == "shiftwm" else 1.0, marker="o", ms=2.2, label=lab)
        dy = {"shiftwm": 0, "direct": 2.2, "ar": -2.2}[a]
        ax.text(10.25, rec[a][-1] + dy, f"{rec[a][-1]:.0f}%", fontsize=5.4, va="center", color=col,
                fontweight="bold" if a == "shiftwm" else "normal")
    ax.set_xlim(0.7, 11.6); ax.set_ylim(20, 90); ax.set_xticks([1, 5, 10]); ax.tick_params(labelsize=5.6, length=2)
    ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1); ax.set_ylabel("recovered (%)", fontsize=5.8, labelpad=1)
    ax.set_title("(d) oracle gain recovered", fontsize=6.5, pad=2)
    ax.legend(fontsize=5.2, frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0.3)
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
