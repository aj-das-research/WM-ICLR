"""Candidate redesigns of Figure 5 (fig:winmap-main): per-episode comparison on held-out DROID, seed 0.

Data (read-only):
  results/v2/analysis/winmap/{winmap.npz,summary.json}   (scripts/v2/winmap.py) -- the paper's metric:
      rel[e,k] = (err_ShiftWM - err_base) / err_base, averaged over k; "win" = mean_k rel < 0  -> 127/130 vs both.
  results/v2s/droid/dinov2s/<arm>/s0/eval_test.npz       -- per-episode, per-horizon feature MSE (absolute errors).
      Absolute gain = mean_k err_base - mean_k err_ShiftWM (horizon-mean error). Under this aggregation ShiftWM is
      lower on 125/130 episodes vs Direct (two extra near-ties, +0.2% and +0.7%) and 127/130 vs AR.
Motion = persistence error averaged over k (the episode's mean true feature change), as in winmap.py.

Usage (repo root): .venv/bin/python reviews/fig5_candidates/make_fig5_candidates.py
Writes reviews/fig5_candidates/cand_{A,B,C,D}.{pdf,png} and prints the skill's layout audit + the numbers used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))
from layout_quality import audit_figure, issue_message  # noqa: E402

# ---- paper palette (paper/submission_folder/figures/src/make_figures.py) ----------------------------------------
GREEN, BLUE, ORANGE = "#009E73", "#0072B2", "#D55E00"
INK, MUTED, GRID = "#243447", "#8A8F98", "#E6E8EB"
LOSS = "#B8433A"
FS_TITLE, FS_LABEL, FS_TICK, FS_NOTE = 7.5, 6.8, 6.2, 6.2
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.0, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
WRAP = (2.31, 1.94)   # 0.42\linewidth wrapfigure slot

# ---- data -------------------------------------------------------------------------------------------------------
S = json.loads((ROOT / "results/v2/analysis/winmap/summary.json").read_text())
Z = np.load(ROOT / "results/v2/analysis/winmap/winmap.npz", allow_pickle=True)
BASE = ROOT / "results/v2s/droid/dinov2s"
EV = {a: np.load(BASE / a / "s0/eval_test.npz", allow_pickle=True) for a in ("shiftwm", "direct", "ar", "persistence")}
ORDER = np.argsort(EV["persistence"]["mse"].mean(1), kind="stable")
assert (EV["shiftwm"]["episodes"][ORDER] == Z["episodes"]).all()
N = len(ORDER)
MOTION = Z["motion"]
ERR = {a: EV[a]["mse"].mean(1)[ORDER] for a in ("shiftwm", "direct", "ar")}
REL = {b: -100 * Z[f"rel_{b}"].mean(1) for b in ("direct", "ar")}          # % error reduction (paper metric), >0 good
WIN = {b: int((REL[b] > 0).sum()) for b in REL}
assert WIN["direct"] == N - S["direct"]["n_lose_avg"] and WIN["ar"] == N - S["ar"]["n_lose_avg"]
ABS = {b: ERR[b] - ERR["shiftwm"] for b in ("direct", "ar")}               # absolute reduction (MSE), >0 good
WIN_ABS = {b: int((ABS[b] > 0).sum()) for b in ABS}
BASES = (("direct", "vs. Direct", BLUE, "D"), ("ar", "vs. AR", ORANGE, "^"))


def numbers():
    out = {"episodes": N, "seed": 0}
    for b in ("direct", "ar"):
        q = np.array_split(np.arange(N), 4)
        out[b] = {"win_rel": WIN[b], "win_abs": WIN_ABS[b], "median_rel_%": float(np.median(REL[b])),
                  "median_abs_by_motion_quartile": [float(np.median(ABS[b][i])) for i in q],
                  "median_rel_by_motion_quartile_%": [float(np.median(REL[b][i])) for i in q],
                  "spearman_motion_abs": float(spearmanr(MOTION, ABS[b])[0]),
                  "loss_motion_ranks_rel": np.where(REL[b] <= 0)[0].tolist(),
                  "loss_motion_ranks_abs": np.where(ABS[b] <= 0)[0].tolist()}
    return out


def finish(fig, name, width):
    issues = audit_figure(fig, min_font_pt=6.0, display_width_inches=width)
    fig.savefig(HERE / f"cand_{name}.pdf"); fig.savefig(HERE / f"cand_{name}.png", dpi=400)
    print(f"[{name}] layout issues: {len(issues)}")
    for it in issues:
        print("   ", issue_message(it))
    plt.close(fig)


def ticks(ax):
    ax.tick_params(labelsize=FS_TICK, length=2, pad=1.5)


# ================================================================== A: paired scatter (absolute errors)
def cand_A():
    fig, ax = plt.subplots(figsize=WRAP)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.18)
    lo, hi = 0.011, 0.6
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=GREEN, alpha=0.10, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.7, zorder=1)
    for b, lab, col, mk in BASES[::-1]:
        ax.scatter(ERR[b], ERR["shiftwm"], s=6, marker=mk, c=col, alpha=0.8, edgecolors="none", zorder=3,
                   label=f"{lab}: {WIN_ABS[b]}/{N}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    tv = [0.02, 0.05, 0.1, 0.2, 0.5]
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(FixedLocator(tv)); a.set_major_formatter(FixedFormatter([f"{t:g}" for t in tv]))
        a.set_minor_formatter(NullFormatter())
    ax.set_xlabel("baseline error (per episode)", fontsize=FS_NOTE, labelpad=1)
    ax.set_ylabel("ShiftWM error", fontsize=FS_NOTE, labelpad=1)
    ax.text(0.30, 0.021, "ShiftWM\nlower error", fontsize=FS_NOTE, color="#00664A", ha="center", va="center",
            linespacing=0.95)
    ax.text(0.33, 0.45, "$y{=}x$", fontsize=FS_NOTE, color=MUTED, ha="right", va="bottom")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=FS_NOTE, loc="upper left", bbox_to_anchor=(0.0, 0.93), handletextpad=0.1,
              markerscale=1.8, borderaxespad=0.2, title="episodes below $y{=}x$", title_fontsize=FS_NOTE)
    ticks(ax)
    finish(fig, "A", WRAP[0])


# ================================================================== B: sorted per-episode gain ("waterfall")
def waterfall(ax, long_note=True):
    x = np.arange(1, N + 1)
    top = 48
    ax.axhspan(0, 100, color=GREEN, alpha=0.08, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    for b, lab, col, mk in BASES:
        v = np.sort(REL[b])[::-1]
        ax.step(x, v, where="mid", color=col, lw=1.1, zorder=3, label=f"{lab}: median {np.median(REL[b]):.0f}%")
        if v[0] > top:                                         # clipped value: say so, with its number
            ax.text(x[0] + 5, 29.5, f"(top: {v[0]:.0f}%, clipped)", fontsize=FS_NOTE, color=col, ha="left", va="center")
    w = WIN["direct"]
    assert w == WIN["ar"]
    ax.axvline(w + 0.5, color=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=2)
    txt = f"{w}/{N} episodes ({100 * w / N:.0f}%)\nShiftWM lower error" if long_note else f"{w}/{N} ({100 * w / N:.0f}%)"
    ax.annotate(txt, xy=(w + 0.5, 25), xytext=(w - 4, 25), fontsize=FS_NOTE, color=INK, ha="right", va="center",
                linespacing=1.0, arrowprops=dict(arrowstyle="->", lw=0.6, color=INK, shrinkA=0, shrinkB=0))
    ax.text(w - 3, -10, f"{N - w} not lower", fontsize=FS_NOTE, color=MUTED, ha="right", va="center")
    ax.set_xlim(0, N + 2); ax.set_ylim(-20, top)
    ax.set_xticks([1, 65, 130]); ax.set_yticks([-20, 0, 20, 40])
    ax.set_xlabel("test episode, ranked by gain", fontsize=FS_NOTE, labelpad=1)
    ax.set_ylabel("error reduction (%)", fontsize=FS_NOTE, labelpad=1)
    ax.legend(fontsize=FS_NOTE, loc="upper center", bbox_to_anchor=(0.55, 1.0), handlelength=1.2, handletextpad=0.3,
              borderaxespad=0.1)
    ax.grid(axis="x", visible=False)
    ticks(ax)


def cand_B():
    fig, ax = plt.subplots(figsize=WRAP)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.18)
    waterfall(ax)
    finish(fig, "B", WRAP[0])


# ================================================================== C: absolute gain vs motion, with trend
def binned(x, y, nb=5):
    q = np.array_split(np.argsort(x), nb)
    return np.array([np.median(x[i]) for i in q]), np.array([np.median(y[i]) for i in q])


def gain_vs_motion(ax, note=True):
    ax.axhspan(0, 100, color=GREEN, alpha=0.08, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    for b, lab, col, mk in BASES:
        y = 100 * ABS[b]
        ax.scatter(MOTION, y, s=5, marker=mk, c=col, alpha=0.45, edgecolors="none", zorder=3)
        bx, by = binned(MOTION, y)
        ax.plot(bx, by, color=col, lw=1.4, zorder=4, marker=mk, ms=3.2, mec="white", mew=0.4,
                label=f"{lab}: {WIN_ABS[b]}/{N} episodes")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    tv = [0.01, 0.03, 0.1, 0.3]
    ax.xaxis.set_major_locator(FixedLocator(tv)); ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tv]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-1.5, 6.5)
    ax.set_xlabel("episode motion (true feature change)", fontsize=FS_NOTE, labelpad=1)
    ax.set_ylabel("abs. error reduction (×0.01)", fontsize=FS_NOTE, labelpad=1)
    ticks(ax)


def cand_C():
    fig, ax = plt.subplots(figsize=WRAP)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.18)
    gain_vs_motion(ax)
    ax.legend(fontsize=FS_NOTE, loc="upper left", handlelength=1.3, handletextpad=0.3, borderaxespad=0.1,
              markerscale=1.0, title="ShiftWM lower error on", title_fontsize=FS_NOTE)
    ax.get_legend()._legend_box.align = "left"
    ax.text(0.03, 0.03, "lines: medians of\nmotion quintiles", transform=ax.transAxes, fontsize=FS_NOTE, color=MUTED,
            ha="left", va="bottom", linespacing=1.0)
    finish(fig, "C", WRAP[0])


# ================================================================== D: full-width strip, B + C side by side
def cand_D():
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.62), gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.21, wspace=0.28)
    waterfall(axs[0], long_note=False)
    axs[0].set_title("(a) ShiftWM has lower error on almost every episode", fontsize=FS_LABEL, loc="left", pad=3,
                     fontweight="bold", color=INK)
    ax = axs[1]
    gain_vs_motion(ax)
    h, _ = ax.get_legend_handles_labels()
    ax.legend(h, ["vs. Direct", "vs. AR"], fontsize=FS_NOTE, loc="upper left", handlelength=1.3, handletextpad=0.3,
              borderaxespad=0.1)
    ax.text(0.02, 0.03, "lines: medians of motion quintiles", transform=ax.transAxes, fontsize=FS_NOTE, color=MUTED,
            ha="left", va="bottom")
    ax.set_title("(b) and its absolute gain grows with motion", fontsize=FS_LABEL, loc="left", pad=3,
                 fontweight="bold", color=INK)
    finish(fig, "D", 5.5)


if __name__ == "__main__":
    print(json.dumps(numbers(), indent=1))
    (HERE / "numbers.json").write_text(json.dumps(numbers(), indent=1))
    for c in sys.argv[1:] or "ABCD":
        globals()[f"cand_{c}"]()
