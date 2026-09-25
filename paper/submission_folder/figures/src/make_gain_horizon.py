"""Main-text Fig. 4: relative error reduction of ShiftWM vs. forecast horizon, one panel per benchmark with results.

Per panel and baseline b: 100 * (1 - err_ShiftWM(k) / err_b(k)), k = 1..10, from the seed-mean per-episode feature MSE
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
        mean = 100 * (1 - s.sum(0) / bs.sum(0))
        boots = 100 * (1 - s[idx].sum(1) / bs[idx].sum(1))                          # [N, K]
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        out["curves"][arm] = (mean, lo, hi)
    return out


def main():
    data = [(ds, t, tag, gain_curves(ds)) for ds, t, tag in BENCH]
    data = [d for d in data if d[3] is not None]
    fig, axes = plt.subplots(1, len(data), figsize=(5.5, 1.3), sharex=True)
    fig.subplots_adjust(left=0.058, right=0.925, top=0.77, bottom=0.235, wspace=0.42)
    GREEN = mf.METHODS["shiftwm"][1]
    # header strip: ShiftWM is the subject of every curve
    fig.text(0.058, 0.975, "\u25CF", color=GREEN, fontsize=7.5, va="top", ha="left")
    fig.text(0.078, 0.972, "ShiftWM compared with each baseline, per forecast step:  above 0 = ShiftWM has lower error; "
             "higher = bigger gain", color=mf.INK, fontsize=6.3, va="top", ha="left", family="STIXGeneral")
    k = np.arange(1, 11); lines = []
    for j, (ax, (ds, title, tag, D)) in enumerate(zip(np.atleast_1d(axes), data)):
        ends = []
        for arm, lab in BASES:
            if arm not in D["curves"]:
                continue
            mean, lo, hi = D["curves"][arm]; _, col, _, mk = mf.METHODS[arm]
            ax.fill_between(k, lo, hi, color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(k, mean, color=col, lw=1.1, marker=mk, ms=2.3, markevery=[0, 4, 9], zorder=3)
            ends.append([lab, float(mean[-1])])
            nm = {"ar_tf": "ARTF", "ar": "AR", "direct": "Dir"}[arm]
            lines += [f"\\def\\gain{tag}{nm}One{{{mean[0]:.0f}}}", f"\\def\\gain{tag}{nm}Ten{{{mean[-1]:.0f}}}",
                      f"\\def\\gain{tag}{nm}Min{{{mean.min():.0f}}}", f"\\def\\gain{tag}{nm}Max{{{mean.max():.0f}}}"]
        top = max(max(D["curves"][a][2].max() for a in D["curves"]), 1.0)
        bot = min(min(D["curves"][a][1].min() for a in D["curves"]), 0.0)
        rng_ = top - bot; ylo, yhi = bot - 0.2 * rng_, top + 0.08 * rng_
        ax.set_ylim(ylo, yhi)
        ax.axhspan(0, yhi, color=GREEN, alpha=0.07, lw=0, zorder=0)       # ShiftWM-better region
        ax.axhline(0, color=mf.INK, lw=0.7, zorder=2.5)
        if j == 0:
            ax.text(0.8, 0 - 0.03 * rng_, "0 = baseline's error", fontsize=5.0, color=mf.INK, va="top", ha="left",
                    style="italic")
            ax.text(0.8, yhi - 0.04 * rng_, "ShiftWM better $\\uparrow$", fontsize=5.6, color=GREEN, va="top",
                    ha="left", fontweight="bold")
        # direct end labels, nudged apart so they never overlap
        gap = 0.13 * (yhi - ylo); ends.sort(key=lambda t: t[1])
        ends[0][1] = max(ends[0][1], 0.45 * gap)                             # keep the lowest label off the zero line
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
        for lab, y in ends:
            col = mf.METHODS[{v: a for a, v in BASES}[lab]][1]
            ax.annotate("vs. " + lab, (10, y), xytext=(3, 0), textcoords="offset points", fontsize=5.4, color=col,
                        va="center", annotation_clip=False)
        ax.set_xlim(0.6, 10.4); ax.set_xticks([1, 4, 7, 10])
        ax.tick_params(labelsize=5.6, length=2, pad=1.5)
        ax.set_xlabel("forecast step $k$", fontsize=5.8, labelpad=1)
        if j == 0:
            ax.set_ylabel("ShiftWM gain (%)", fontsize=5.8, labelpad=1)
        seeds = f"{D['seeds']} seed{'s' if D['seeds'] > 1 else ''}"
        ax.set_title(f"({'abc'[j]}) {title} ({seeds})", fontsize=6.5, pad=2)
    fig.savefig(mf.FIG / "gain_vs_horizon.pdf"); fig.savefig(mf.FIG / "gain_vs_horizon_preview.png", dpi=300)
    (mf.ROOT / "paper/submission_folder/tables/generated/gain_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote gain_vs_horizon")


if __name__ == "__main__":
    main()
