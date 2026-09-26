"""Forecast skill vs. action-ranking error on every benchmark, plus the gain of ShiftWM over Direct by episode motion.

Inputs (held-out test split, the same runs as the main tables, via make_figures.root_for):
  results/v2s/<ds>/dinov2s/<arm>/s*/eval_test.npz     per-episode x horizon 'mse' and 'rank_ok'
  results/v2s/<ds>/dinov2s/persistence/s0/...          persistence error (the skill reference)
  results/v2s/<ds>/dinov2s/ablations/<name>/s*/...     ablations of the same recipe (muted points)

(a-c) x = skill = % of persistence error removed (mean over horizons); y = action-ranking error = share of test
      windows in which another episode's actions give a LOWER error than the true actions (log scale, so saturated
      benchmarks stay readable). Large marker = seed mean, crosshair = 95% bootstrap CI (recording sessions for DROID,
      episodes elsewhere; seed-mean per-episode values), faint dots = individual seeds.
(d)   Error reduction of ShiftWM over Direct in quintiles of per-episode motion (persistence error of the episode),
      95% bootstrap CI within each quintile.
Planning success is not encoded: no planning run of these predictors exists yet (results/v2/planning holds only the
LeWM reference and a random planner). Nothing is interpolated; a missing arm is simply absent.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

DATASETS = (("droid", "DROID"), ("openh_hamlyn", "Hamlyn (surgical)"), ("language_table", "Language-Table"))
ARMS = ("ar_tf", "ar", "direct", "shiftwm")
SHORT = {"ar_tf": "AR-TF", "ar": "AR", "direct": "Direct", "shiftwm": "ShiftWM"}
ABL = {"shiftwm_cv": "ShiftWM variant", "shiftwm_dil": "ShiftWM variant", "shiftwm_it2": "ShiftWM variant",
       "shiftwm_dil_it2": "ShiftWM variant", "artf_res": "AR-TF + residual"}
GREEN = mf.METHODS["shiftwm"][1]
NB = 10000


def groups(ds, episodes):
    if ds != "droid":
        return np.arange(len(episodes))
    f = mf.ROOT / "data/v2/features/droid/dinov2s/manifest.json"
    ses = {r["id"]: r.get("session", r["id"]) for r in json.loads(f.read_text())["episodes"]} if f.exists() else {}
    g = [ses.get(e, e) for e in episodes]
    u = {s: i for i, s in enumerate(sorted(set(g)))}
    return np.array([u[s] for s in g])


def boot_idx(gi, n=NB, seed=0):
    """Cluster bootstrap: [n, n_groups] resampled group ids."""
    return np.random.default_rng(seed).integers(0, gi.max() + 1, (n, gi.max() + 1))


def gsum(v, gi):
    return np.bincount(gi, v, gi.max() + 1)


def runs(ds, arm, sub=""):
    base = mf.root_for(ds)
    return [np.load(p) for p in sorted((base / ds / "dinov2s" / sub / arm).glob("s*/eval_test.npz"))]


def point(ds, zs, pers, gi, B):
    """Seed-mean point, 95% CI (cluster bootstrap) and per-seed values of (skill %, ranking error %)."""
    m = np.mean([z["mse"].mean(1) for z in zs], 0); r = np.mean([1 - z["rank_ok"].mean(1) for z in zs], 0)
    P = pers.sum()
    sk = 100 * (1 - m.sum() / P); er = 100 * r.mean()
    gm, gp, gr, gc = gsum(m, gi), gsum(pers, gi), gsum(r, gi), gsum(np.ones_like(m), gi)
    bs = 100 * (1 - gm[B].sum(1) / gp[B].sum(1)); be = 100 * gr[B].sum(1) / gc[B].sum(1)
    seeds = [(100 * (1 - z["mse"].mean(1).sum() / P), 100 * (1 - z["rank_ok"].mean())) for z in zs]
    return dict(x=sk, y=er, xci=np.percentile(bs, [2.5, 97.5]), yci=np.percentile(be, [2.5, 97.5]), seeds=seeds, n=len(zs))


def collect():
    out = {}
    for ds, _ in DATASETS:
        pz = runs(ds, "persistence")
        if not pz:
            continue
        pers = pz[0]["mse"].mean(1); eps = pz[0]["episodes"]
        gi = groups(ds, eps); B = boot_idx(gi)
        d = {"arms": {}, "abl": {}, "pers": pers, "gi": gi}
        for arm in ARMS:
            zs = runs(ds, arm)
            if zs and all(np.array_equal(z["episodes"], eps) for z in zs):
                d["arms"][arm] = point(ds, zs, pers, gi, B)
        for name in ABL:
            zs = runs(ds, name, "ablations")
            if zs and all(np.array_equal(z["episodes"], eps) for z in zs):
                d["abl"][name] = point(ds, zs, pers, gi, B)
        out[ds] = d
    return out


def pct(v, _=None):
    return f"{v:g}%"


def scatter(ax, d, title, first):
    for name, p in d["abl"].items():
        ax.scatter(p["x"], p["y"], s=11, marker="o", facecolor="white", edgecolor=mf.MUTED, lw=0.6, zorder=2)
        if name == "artf_res":
            ax.annotate("AR-TF + res.", (p["x"], p["y"]), xytext=(-3, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=4.8, color=mf.MUTED)
    for arm, p in d["arms"].items():
        _, col, _, mk = mf.METHODS[arm]
        if p["n"] > 1:
            sx, sy = zip(*p["seeds"])
            ax.scatter(sx, sy, s=6, marker=mk, color=col, alpha=0.35, lw=0, zorder=3)
        ax.errorbar(p["x"], p["y"], xerr=[[p["x"] - p["xci"][0]], [p["xci"][1] - p["x"]]],
                    yerr=[[p["y"] - p["yci"][0]], [p["yci"][1] - p["y"]]], fmt="none", ecolor=col, elinewidth=0.8,
                    capsize=0, alpha=0.9, zorder=4)
        ax.scatter(p["x"], p["y"], s=34 if arm == "shiftwm" else 24, marker=mk, color=col, edgecolor="white", lw=0.6,
                   zorder=5)
    ax.set_yscale("log")
    ys = [v for p in list(d["arms"].values()) + list(d["abl"].values()) for v in p["yci"]]
    xs = [v for p in list(d["arms"].values()) + list(d["abl"].values()) for v in p["xci"]]
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo / 1.18, hi * 1.18)
    xr = max(xs) - min(xs); ax.set_xlim(min(xs) - 0.08 * xr, max(xs) + 0.08 * xr)
    cands = [0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    tk = [t for t in cands if lo / 1.18 <= t <= hi * 1.18]
    if len(tk) > 6:
        tk = tk[::2]
    ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(pct)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(labelsize=5.6, length=2, pad=1.5)
    if first:
        ax.set_ylabel("action-ranking error $\\downarrow$ (log)", fontsize=5.8, labelpad=1)
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.grid(True, which="major", lw=0.5)
    # best-in-panel annotation: which arm has the lowest ranking error / highest skill
    a = d["arms"]
    bx, by = max(a, key=lambda k: a[k]["x"]), min(a, key=lambda k: a[k]["y"])
    txt = (f"best skill: {SHORT[bx]} {a[bx]['x']:.1f}%\nfewest rank errors: {SHORT[by]} {a[by]['y']:.1f}%")
    ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=4.9, color=mf.INK, va="bottom", ha="left",
            linespacing=1.15, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6), zorder=6)


def gain_panel(ax, data):
    styles = {"droid": ("-", "o", "DROID"), "openh_hamlyn": ((0, (4, 1.5)), "s", "Hamlyn"),
              "language_table": ((0, (1.2, 1.2)), "^", "Lang.-Table")}
    shades = {"droid": GREEN, "openh_hamlyn": "#00543D", "language_table": "#5BBE9C"}
    q = np.arange(1, 6); out = {}
    for ds, d in data.items():
        s, di = runs(ds, "shiftwm"), runs(ds, "direct")
        if not s or not di:
            continue
        S = np.mean([z["mse"].mean(1) for z in s], 0); D = np.mean([z["mse"].mean(1) for z in di], 0)
        mot = d["pers"]; e = np.quantile(mot, np.linspace(0, 1, 6)); b = np.clip(np.searchsorted(e, mot, "right") - 1, 0, 4)
        g, lo, hi = [], [], []
        for i in range(5):
            sel = b == i; gi = d["gi"][sel]; _, gi = np.unique(gi, return_inverse=True)
            gs_, gd_ = gsum(S[sel], gi), gsum(D[sel], gi); B = boot_idx(gi, seed=i)
            bt = 100 * (1 - gs_[B].sum(1) / gd_[B].sum(1))
            g.append(100 * (1 - S[sel].sum() / D[sel].sum())); lo.append(np.percentile(bt, 2.5)); hi.append(np.percentile(bt, 97.5))
        ls, mk, lab = styles[ds]; c = shades[ds]
        ax.fill_between(q, lo, hi, color=c, alpha=0.13, lw=0)
        ax.plot(q, g, color=c, ls=ls, marker=mk, ms=2.6, lw=1.2, label=lab)
        out[ds] = g
    ax.axhline(0, color=mf.INK, lw=0.5)
    ax.set_xlim(0.7, 5.3); ax.set_xticks(q); ax.set_xticklabels(["1", "2", "3", "4", "5"])
    lo_, hi_ = ax.get_ylim(); lo_ = min(0, lo_)
    ax.set_ylim(lo_, hi_ + 0.22 * (hi_ - lo_))  # headroom: legend sits inside the panel, above the data
    ax.legend(loc="upper center", ncol=3, fontsize=4.8, frameon=False, handlelength=1.0, handletextpad=0.25,
              columnspacing=0.4, borderaxespad=0.3, bbox_to_anchor=(0.53, 1.0), markerscale=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+g}%" if v else "0"))
    ax.tick_params(labelsize=5.6, length=2, pad=1.5); ax.grid(axis="x", visible=False)
    ax.set_xlabel("episode motion quintile $\\rightarrow$", fontsize=5.8, labelpad=1)
    ax.set_ylabel("error reduction vs. Direct", fontsize=5.8, labelpad=0)
    ax.set_title("(d) ShiftWM vs. Direct", fontsize=6.5, pad=2)
    return out


def main():
    data = collect()
    fig = plt.figure(figsize=(5.5, 1.78))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3, left=0.062, right=0.985, top=0.78, bottom=0.19)
    for j, (ds, name) in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, j])
        if ds not in data or not data[ds]["arms"]:
            mf.pending(ax, name); continue
        n = max(p["n"] for p in data[ds]["arms"].values())
        scatter(ax, data[ds], f"({'abc'[j]}) {name}" + (f", {n} seeds" if n > 1 else ", 1 seed"), j == 0)
    gains = gain_panel(fig.add_subplot(gs[0, 3]), data)
    l, r = fig.axes[0].get_position().x0, fig.axes[2].get_position().x1
    fig.text((l + r) / 2, 0.035, "skill: % of persistence error removed (mean over $k$) $\\rightarrow$ better", ha="center",
             va="bottom", fontsize=5.8, color=mf.INK)
    # shared legend (methods, seeds, ablations) in the header
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mf.METHODS[a][3], color=mf.METHODS[a][1], ls="none", ms=4 if a == "shiftwm" else 3.4,
                mec="white", mew=0.5, label=SHORT[a] + (" (ours)" if a == "shiftwm" else "")) for a in ("shiftwm", "direct", "ar", "ar_tf")]
    h += [Line2D([], [], marker="o", color=mf.MUTED, ls="none", ms=2.2, alpha=0.5, label="single seed"),
          Line2D([], [], marker="o", mfc="white", mec=mf.MUTED, ls="none", ms=3, mew=0.6, label="ablation (DROID)"),
          Line2D([], [], color=mf.INK, lw=0.8, marker="+", ms=5, mew=0.8, ls="none", label="95% CI")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=5.6, frameon=False,
               handletextpad=0.25, columnspacing=1.1)
    fig.savefig(mf.FIG / "tradeoff.pdf"); fig.savefig(mf.FIG / "tradeoff_preview.png", dpi=200); plt.close(fig)
    # numbers for the text
    lines = []
    for ds, key in (("droid", "Droid"), ("openh_hamlyn", "Ham"), ("language_table", "LT")):
        for arm, p in data.get(ds, {}).get("arms", {}).items():
            a = {"ar_tf": "ARTF", "ar": "AR", "direct": "Direct", "shiftwm": "Shift"}[arm]
            lines.append(f"\\def\\to{key}Err{a}{{{p['y']:.{2 if p['y'] < 3 else 1}f}}}")
            lines.append(f"\\def\\to{key}Skill{a}{{{p['x']:.1f}}}")
        if ds in gains:
            g = gains[ds]
            lines += [f"\\def\\to{key}GainStill{{{g[0]:.1f}}}", f"\\def\\to{key}GainFast{{{g[-1]:.1f}}}",
                      f"\\def\\to{key}GainMin{{{min(g):.1f}}}", f"\\def\\to{key}GainMax{{{max(g):.1f}}}"]
    (mf.ROOT / "paper/submission_folder/tables/generated/tradeoff_numbers.tex").write_text("\n".join(lines) + "\n")
    for ds, d in data.items():
        for arm, p in d["arms"].items():
            print(ds, arm, f"skill {p['x']:.1f} [{p['xci'][0]:.1f},{p['xci'][1]:.1f}]  err {p['y']:.2f} [{p['yci'][0]:.2f},{p['yci'][1]:.2f}]")
        for arm, p in d["abl"].items():
            print(ds, "abl", arm, f"skill {p['x']:.1f} err {p['y']:.2f}")
    print("wrote tradeoff")


if __name__ == "__main__":
    main()
