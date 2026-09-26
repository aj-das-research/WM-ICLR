"""Appendix figure replacing the IWS per-task table: each matched baseline's test error relative to ShiftWM (%, mean over
horizons k=1..12) on the 200 official RLA-WM validation handles of each IWS task, seed 0, with 95% bootstrap intervals
over handles (paired: the same resampled handles for both methods). Data: scripts/v2/make_tables.load()."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import make_figures as mf

sys.path.insert(0, str(mf.ROOT / "scripts/v2"))
import make_tables as T                                                          # noqa: E402

TASKS = [("iws_pusht", "PushT"), ("iws_box", "Box"), ("iws_rope", "Rope")]
ARMS = ["ar_tf", "ar", "direct"]


def rel(b, s, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    boots = 100 * (b[idx].mean(1) / s[idx].mean(1) - 1)
    return 100 * (b.mean() / s.mean() - 1), np.percentile(boots, [2.5, 97.5])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.25), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.8, bottom=0.3, wspace=0.12)
    for ax, (task, name) in zip(axes, TASKS):
        s = T.load(task, "shiftwm")["mse"].mean(1)
        for y, arm in enumerate(ARMS):
            b = T.load(task, arm)["mse"].mean(1)
            v, (lo, hi) = rel(b, s)
            lab, col, _, mk = mf.METHODS[arm]
            ax.barh(y, v, color=col, alpha=0.85, height=0.62, zorder=3)
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=mf.INK, elinewidth=0.7, capsize=1.6, zorder=4)
            ax.text(max(v, hi) + 0.6, y, f"+{v:.1f}%", va="center", ha="left", fontsize=mf.FS_NOTE, color=mf.INK)
        ax.axvline(0, color=mf.METHODS["shiftwm"][1], lw=1.4, zorder=5)
        ax.set_title(name, fontsize=mf.FS_LABEL, fontweight="bold", color=mf.INK, pad=2)
        ax.set_xlim(0, 31); ax.grid(axis="x", color=mf.GRID, lw=0.5); ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=mf.FS_TICK, length=1.5, pad=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(ARMS)))
    axes[0].set_yticklabels([mf.METHODS[a][0].split(" (")[0] for a in ARMS])
    axes[0].invert_yaxis()
    fig.text(0.56, 0.04, "error above ShiftWM (%, mean over horizons; green line: ShiftWM)", ha="center",
             fontsize=mf.FS_NOTE, color=mf.INK)
    mf.qa(fig, "iws_tasks", 5.5)
    fig.savefig(mf.FIG / "iws_tasks.pdf"); fig.savefig(mf.FIG / "iws_tasks_preview.png", dpi=220)
    print("wrote iws_tasks")


if __name__ == "__main__":
    main()
