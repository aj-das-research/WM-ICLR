"""Main-text Fig. 4: each baseline's error relative to ShiftWM's (ShiftWM = 100%) vs. forecast horizon, one panel per benchmark with results.

Per panel and baseline b: 100 * err_b(k) / err_ShiftWM(k) (plotted); text macros keep the gain 100 * (1 - err_ShiftWM/err_b), k = 1..10, from the seed-mean per-episode feature MSE
in results/v2s/<ds>/dinov2s/<arm>/s*/eval_test.npz (same loader and root rule as the tables).
Bands: 95% paired bootstrap (10k resamples) of that ratio, resampling recording sessions for DROID (as for
\\droidCILo in scripts/v2/make_tables.py) and episodes elsewhere. Panels without ShiftWM results are left out.
Also writes tables/generated/gain_numbers.tex (gain at k=1 and k=10 per benchmark and baseline).

Usage (repo root):  PYTHONPATH=src python paper/submission_folder/figures/src/make_gain_horizon.py
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

BENCH = [("droid", "DROID", "Droid"), ("openh_hamlyn", "Open-H Hamlyn", "Ham"), ("language_table", "Language-Table", "Lt")]
BASES = [("ar_tf", "AR-TF"), ("ar", "AR"), ("direct", "Direct")]
N_BOOT = 10000


def groups_for(ds, episodes):
    if ds != "droid":
        return list(episodes)
    f = mf.ROOT / "data/v2/features/droid/dinov2s/manifest.json"
    sess = {r["id"]: r.get("session", r["id"]) for r in json.loads(f.read_text())["episodes"]} if f.exists() else {}
    return [sess.get(e, e) for e in episodes]


def gain_curves(ds):
    sw = mf.load_eval(ds, "dinov2s", "shiftwm")
    if sw is None:
        return None
    g = groups_for(ds, sw["episodes"]); uniq = sorted(set(g)); gi = np.array([uniq.index(x) for x in g])
    rng = np.random.default_rng(0); idx = rng.integers(0, len(uniq), (N_BOOT, len(uniq)))
    agg = lambda a: np.stack([np.bincount(gi, a[:, k], len(uniq)) for k in range(a.shape[1])], 1)   # [G, K] sums
    s = agg(sw["mse"].mean(0)); out = {"seeds": sw["seeds"], "curves": {}}
    for arm, _ in BASES:
        b = mf.load_eval(ds, "dinov2s", arm)
        if b is None:
            continue
        assert list(b["episodes"]) == list(sw["episodes"]), (ds, arm)
        bs = agg(b["mse"].mean(0))
        mean = 100 * (1 - s.sum(0) / bs.sum(0))                                     # gain (text macros)
        ratio = 100 * bs.sum(0) / s.sum(0)                                          # baseline error, ShiftWM = 100
        boots = 100 * bs[idx].sum(1) / s[idx].sum(1)                                # [N, K]
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        out["curves"][arm] = (mean, ratio, lo, hi)
    return out


def main():
    data = [(ds, t, tag, gain_curves(ds)) for ds, t, tag in BENCH]
    data = [d for d in data if d[3] is not None]
    fig, axes = plt.subplots(1, len(data), figsize=(5.5, 1.3), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.915, top=0.8, bottom=0.235, wspace=0.4)
    GREEN = mf.METHODS["shiftwm"][1]
    fig.text(0.07, 0.975, "Each method's feature error relative to ShiftWM's (ShiftWM = 100%), per forecast step; "
             "lower is better", color=mf.INK, fontsize=6.3, va="top", ha="left")
    k = np.arange(1, 11); lines = []
    for j, (ax, (ds, title, tag, D)) in enumerate(zip(np.atleast_1d(axes), data)):
        ends = []
        for arm, lab in BASES:
            if arm not in D["curves"]:
                continue
            gain, ratio, lo, hi = D["curves"][arm]; _, col, _, mk = mf.METHODS[arm]
            ax.fill_between(k, lo, hi, color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(k, ratio, color=col, lw=1.1, marker=mk, ms=2.3, markevery=[0, 4, 9], zorder=3)
            ends.append([lab, float(ratio[-1]), col])
            nm = {"ar_tf": "ARTF", "ar": "AR", "direct": "Dir"}[arm]
            lines += [f"\\def\\gain{tag}{nm}One{{{gain[0]:.0f}}}", f"\\def\\gain{tag}{nm}Ten{{{gain[-1]:.0f}}}",
                      f"\\def\\gain{tag}{nm}Min{{{gain.min():.0f}}}", f"\\def\\gain{tag}{nm}Max{{{gain.max():.0f}}}"]
        # ShiftWM: the reference every other curve is measured against
        ax.plot(k, np.full(10, 100.0), color=GREEN, lw=2.0, marker="o", ms=2.6, markevery=[0, 4, 9], zorder=4)
        top = max(D["curves"][a][3].max() for a in D["curves"])
        rng_ = top - 100; ylo, yhi = 100 - 0.3 * rng_, top + 0.08 * rng_
        ax.set_ylim(ylo, yhi)
        ax.text(1.0, 100 - 0.07 * rng_, "ShiftWM (ours)", color=GREEN, fontsize=5.8, fontweight="bold", va="top",
                ha="left")                                                   # inside the panel, under its line
        # end labels, nudged apart so they never overlap (ShiftWM stays at 100)
        gap = 0.13 * (yhi - ylo); ends.sort(key=lambda t: t[1])
        ends[0][1] = max(ends[0][1], 100 + 0.6 * gap)                          # clear of the ShiftWM line
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
        for lab, y, col in ends:
            ax.annotate(lab, (10, y), xytext=(3, 0), textcoords="offset points", fontsize=5.4, color=col,
                        va="center", annotation_clip=False, fontweight="normal")
        ax.set_xlim(0.6, 10.4); ax.set_xticks([1, 4, 7, 10])
        ax.tick_params(labelsize=5.6, length=2, pad=1.5)
        ax.set_xlabel("forecast step $k$", fontsize=5.8, labelpad=1)
        if j == 0:
            ax.set_ylabel("error rel. to ShiftWM (%)", fontsize=5.8, labelpad=1)
        ax.set_title(f"({'abc'[j]}) {title}", fontsize=6.5, pad=2)
    fig.savefig(mf.FIG / "gain_vs_horizon.pdf"); fig.savefig(mf.FIG / "gain_vs_horizon_preview.png", dpi=300)
    (mf.ROOT / "paper/submission_folder/tables/generated/gain_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote gain_vs_horizon")


if __name__ == "__main__":
    main()
