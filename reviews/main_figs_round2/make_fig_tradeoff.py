"""Main-text half-width plot (round 2): forecast skill vs. action-ranking accuracy, DROID and Open-H (Hamlyn).

Numbers: paper/submission_folder/figures/src/make_tradeoff.py collect() (same runs as Table 1 via make_figures.root_for:
results/v2s/<ds>/dinov2s/<arm>/s*/eval_test.npz), unchanged:
  x = skill = % of persistence error removed (mean over horizons), seed mean;
  y = action-ranking accuracy = 100 - ranking error (the share of test windows where the true actions give a lower
      error than another episode's actions; same definition as Table 1 "rank."), seed mean;
  bars = 95% cluster-bootstrap CIs (recording sessions for DROID, episodes for Hamlyn).
DROID: 3 seeds per arm; Hamlyn: 1 seed. Ablations and single-seed dots of the appendix version are not drawn.
Language-Table (appendix Fig.) is omitted here: there Direct ranks actions marginally better (1.8% vs 1.9% error,
overlapping CIs); see README.

Usage (repo root): PYTHONPATH=src .venv/bin/python reviews/main_figs_round2/make_fig_tradeoff.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker  # noqa: F401

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper/submission_folder/figures/src"))
import make_tradeoff as TR  # noqa: E402
mf = TR.mf

FS, FT = mf.FS_NOTE, mf.FS_TICK
PANELS = (("droid", "DROID (3 seeds)"), ("openh_hamlyn", "Open-H surgery (1 seed)"))
# label offsets (points) per dataset/arm -- layout only
OFF = {"droid": {"shiftwm": (-5, 3, "right"), "direct": (-5, -3, "right"), "ar": (-4, -5, "right"),
                 "ar_tf": (4, -5, "left")},
       "openh_hamlyn": {"shiftwm": (-5, 4, "right"), "direct": (-5, 1, "right"), "ar": (4, -8, "left"),
                        "ar_tf": (4, -5, "left")}}


def main():
    data = TR.collect()
    W, H = 2.7, 1.72
    fig = plt.figure(figsize=(W, H))
    L, R, B, T, G = 0.36, 0.06, 0.30, 0.20, 0.30
    pw = (W - L - R - G) / 2
    led = {}
    for j, (ds, title) in enumerate(PANELS):
        ax = fig.add_axes([(L + j * (pw + G)) / W, B / H, pw / W, (H - B - T) / H])
        led[ds] = {}
        for arm in ("ar_tf", "ar", "direct", "shiftwm"):
            p = data[ds]["arms"][arm]
            _, col, _, mk = mf.METHODS[arm]
            x, y = p["x"], 100 - p["y"]
            ylo, yhi = 100 - p["yci"][1], 100 - p["yci"][0]
            ax.errorbar(x, y, xerr=[[x - p["xci"][0]], [p["xci"][1] - x]], yerr=[[y - ylo], [yhi - y]], fmt="none",
                        ecolor=col, elinewidth=0.7, capsize=0, alpha=0.8, zorder=3)
            ax.scatter(x, y, s=30 if arm == "shiftwm" else 20, marker=mk, color=col, edgecolor="white", lw=0.5,
                       zorder=5)
            dx, dy, ha = OFF[ds][arm]
            ax.annotate(TR.SHORT[arm], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=FS, color=col, fontweight="bold" if arm == "shiftwm" else "normal", zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
            led[ds][arm] = dict(skill=x, skill_ci=list(p["xci"]), rank_acc=y, rank_acc_ci=[ylo, yhi], seeds=p["n"])
        ax.set_title(title, fontsize=mf.FS_LABEL, pad=3, color=mf.INK, fontweight="bold")
        ax.tick_params(labelsize=FT, length=2, pad=1.5)
        ax.grid(True, lw=0.5, color="#EEF0F3")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20 if ds == "droid" else 5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        xs = [v for a in data[ds]["arms"].values() for v in a["xci"]]
        ys = [100 - v for a in data[ds]["arms"].values() for v in a["yci"]]
        xr, yr = max(xs) - min(xs), max(ys) - min(ys)
        ax.set_xlim(min(xs) - 0.1 * xr, max(xs) + 0.1 * xr); ax.set_ylim(min(ys) - 0.08 * yr, max(ys) + 0.1 * yr)
        if j == 0:
            ax.annotate("", xy=(0.24, 0.97), xytext=(0.06, 0.79), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.18", color=mf.MUTED, lw=0.8))
            ax.text(0.19, 0.80, "better", transform=ax.transAxes, fontsize=FS, color=mf.MUTED, ha="left", va="center",
                    rotation=0)
            ax.set_ylabel(r"action-ranking acc. (%) $\uparrow$", fontsize=FS, labelpad=1.5)
    fig.text((L + (W - L - R)) / 2 / W, 0.03, r"skill: % of persistence error removed $\rightarrow$", ha="center",
             va="bottom", fontsize=FS, color=mf.INK)
    mf.qa(fig, "fig_tradeoff", display_width=W)
    fig.savefig(HERE / "fig_tradeoff.pdf"); fig.savefig(HERE / "fig_tradeoff.png", dpi=400)
    (HERE / "ledger_tradeoff.json").write_text(json.dumps(led, indent=1, default=float))
    print(json.dumps(led, indent=1, default=float))


if __name__ == "__main__":
    main()
