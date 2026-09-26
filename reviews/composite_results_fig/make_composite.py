"""Composite results figure: "Where and why ShiftWM wins" (replaces Figs. 5 and 6 and panels b-d of Fig. 4).

  main  composite_main.{pdf,png}  5.5 x 3.0 in, 2 rows x 3 panels
        (a) per-episode error reduction, ranked (127/130 vs Direct and AR; paper's relative metric)
        (b) absolute gain vs episode motion (quintile medians + IQR; faint points = episodes)
        (c) moving-patch error at k=10: persistence / AR / Direct / ShiftWM / oracle move / unrelated-episode move
        (d) gate knock-out and action swap, moving vs static patches
        (e) ablations (DROID val., seed 0): removals vs neutral design alternatives, seed-noise band
        (f) gain over Direct per horizon (4 datasets, 95% CI) + per-task strip (7 Open-H + 3 IWS, 95% CI)
  alt   composite_alt.{pdf,png}   5.5 x 1.9 in, 1 row: (a), (c), (e), (f)

Data (never hand-typed; all loaders reused):
  (a,b)  reviews/fig5_candidates/make_fig5_candidates.py  (results/v2/analysis/winmap + results/v2s/droid/.../eval_test.npz)
  (c)    results/v2/analysis/geometry/summary.json (k=10, moving patches) + oracle_null/summary_any_seed0.json (null move)
  (d)    results/v2/analysis/interpret/summary.json (knock-out, steering)
  (e,f)  reviews/fig6_candidates/make_fig6_candidates.py  (make_tables.ablation_data / make_tables.load, paired bootstrap)
Hand-set constant: NOISE = 0.3 (% seed-noise threshold stated in the paper).

Usage (repo root): PYTHONPATH=src .venv/bin/python reviews/composite_results_fig/make_composite.py [main] [alt]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reviews/fig5_candidates"))
sys.path.insert(0, str(ROOT / "reviews/fig6_candidates"))
import make_fig5_candidates as F5  # noqa: E402  (loads per-episode data at import)
import make_fig6_candidates as F6  # noqa: E402  (imports make_figures -> shared rcParams/palette)

mf = F6.mf
M = mf.METHODS
GREEN, BLUE, ORANGE, GREY = M["shiftwm"][1], M["direct"][1], M["ar"][1], M["persistence"][1]
INK, MUTED = mf.INK, mf.MUTED
FS, FT = 6.2, 7.2                     # text / panel-title size at print (figures authored at print width)
NOISE = F6.NOISE
LIGHT = "#C9CED6"
plt.rcParams.update({"font.size": FS, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "grid.linewidth": 0.5, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS})
LEDGER: dict = {}
LEDGER_C_PCT: dict = {}


# ------------------------------------------------------------------------------------------------ helpers
def axes_in(fig, x, y, w, h):
    W, H = fig.get_figwidth(), fig.get_figheight()
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.tick_params(labelsize=FS, length=2, pad=1.2)
    return a


def title(fig, x, y, s):
    W, H = fig.get_figwidth(), fig.get_figheight()
    fig.text(x / W, y / H, s, fontsize=FT, fontweight="bold", color=INK, ha="left", va="top")


def pct(v):
    return f"{v:+.0f}%" if abs(v) >= 1 else f"{v:+.1f}%"


# ------------------------------------------------------------------------------------------------ (a) episodes
def panel_episodes(ax, legend=True, short=False):
    N = F5.N
    x = np.arange(1, N + 1)
    top = 40
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    for b, lab, col in (("ar", "vs. AR", ORANGE), ("direct", "vs. Direct", BLUE)):
        v = np.sort(F5.REL[b])[::-1]
        ax.step(x, np.minimum(v, top), where="mid", color=col, lw=1.1, zorder=3,
                label=lab if short else f"{lab} (median {np.median(F5.REL[b]):.0f}%)")
        LEDGER.setdefault("a", {})[b] = {"wins": F5.WIN[b], "median_pct": float(np.median(F5.REL[b])),
                                         "max_pct": float(v[0]), "min_pct": float(v[-1])}
    w = F5.WIN["direct"]
    assert w == F5.WIN["ar"]
    ax.fill_between([0.5, w + 0.5], 0, top, color=GREEN, alpha=0.09, lw=0, zorder=0)
    ax.text(w * 0.5, 30, f"{w}/{N} episodes\n({100 * w / N:.0f}%) lower error" if not short else f"{w}/{N} ({100 * w / N:.0f}%)\nlower error", fontsize=FS, color="#00664A",
            ha="center", va="center", linespacing=1.0, fontweight="bold")
    ax.annotate(f"{N - w} not lower" if not short else f"{N - w} not", xy=(N - 1, -6), xytext=(N - 12, -13), fontsize=FS, color=MUTED, ha="right",
                va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0, shrinkB=1))
    ax.set_xlim(0, N + 2); ax.set_ylim(-20, top)
    ax.set_xticks([1, 65, 130]); ax.set_yticks([-20, 0, 20, 40])
    ax.set_yticklabels(["−20", "0", "20", "≥40"])
    ax.set_xlabel("test episode (ranked)", labelpad=0.5)
    ax.set_ylabel("error reduction (%)", labelpad=1)
    ax.grid(axis="x", visible=False)
    if legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], fontsize=FS, loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.0,
                  handletextpad=0.3, borderaxespad=0.15, labelspacing=0.15)


# ------------------------------------------------------------------------------------------------ (b) motion
def panel_motion(ax):
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    mot = F5.MOTION
    LEDGER["b"] = {}
    for b, col, mk in (("direct", BLUE, "D"), ("ar", ORANGE, "^")):
        y = 100 * F5.ABS[b]
        ax.scatter(mot, y, s=3, marker=mk, c=col, alpha=0.28, edgecolors="none", zorder=3)
        q = np.array_split(np.argsort(mot), 5)
        bx = np.array([np.median(mot[i]) for i in q])
        by = np.array([np.median(y[i]) for i in q])
        lo = np.array([np.percentile(y[i], 25) for i in q]); hi = np.array([np.percentile(y[i], 75) for i in q])
        ax.fill_between(bx, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(bx, by, color=col, lw=1.2, marker=mk, ms=2.8, mec="white", mew=0.4, zorder=4)
        rho = spearmanr(mot, F5.ABS[b])[0]
        LEDGER["b"][b] = {"quintile_median_x100": by.tolist(), "spearman": float(rho)}
    ax.set_xscale("log")
    tv = [0.01, 0.03, 0.1, 0.3]
    ax.xaxis.set_major_locator(mt.FixedLocator(tv)); ax.xaxis.set_major_formatter(mt.FixedFormatter([f"{t:g}" for t in tv]))
    ax.xaxis.set_minor_formatter(mt.NullFormatter()); ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xlim(0.009, mot.max() * 1.12); ax.set_ylim(-1.5, 5.6)
    ax.set_yticks([0, 2, 4])
    ax.set_xlabel("episode motion (true feature change)", labelpad=0.5)
    ax.set_ylabel("abs. reduction (×0.01)", labelpad=1)
    r = LEDGER["b"]
    ax.text(0.03, 0.97, "Spearman $\\rho$", transform=ax.transAxes, fontsize=FS, color=INK, ha="left", va="top")
    ax.text(0.03, 0.84, f"vs. Direct {r['direct']['spearman']:.2f}", transform=ax.transAxes, fontsize=FS,
            color=BLUE, ha="left", va="top")
    ax.text(0.03, 0.71, f"vs. AR {r['ar']['spearman']:.2f}", transform=ax.transAxes, fontsize=FS,
            color=ORANGE, ha="left", va="top")
    ax.text(0.97, 0.04, "line: quintile median, band: IQR", transform=ax.transAxes, fontsize=FS, color=MUTED,
            ha="right", va="bottom")


# ------------------------------------------------------------------------------------------------ (c) oracle ladder
def oracle_rows():
    g = json.loads((mf.RES / "analysis/geometry/summary.json").read_text())["regions"]["moving"]
    n = json.loads((mf.RES / "analysis/oracle_null/summary_any_seed0.json").read_text())["moving"]
    assert abs(n["persistence"] - g["persistence"]["k10"]) < 1e-4 and abs(n["oracle"] - g["oracle"]["k10"]) < 1e-4
    rows = [("move (other ep.)", n["null"], "null"), ("persistence", g["persistence"]["k10"], "persistence"),
            ("AR", g["ar"]["k10"], "ar"), ("Direct", g["direct"]["k10"], "direct"), ("ShiftWM", g["shiftwm"]["k10"], "shiftwm"),
            ("move (oracle)", g["oracle"]["k10"], "oracle")]
    LEDGER["c"] = {k: v for _, v, k in rows} | {"oracle_red_pct": n["oracle_red_pct"], "null_red_pct": n["null_red_pct"],
                                                "share_shiftwm": g["oracle_gain_share"]["shiftwm"]}
    return rows


def panel_oracle(ax, compact=False, xmax=2.2):
    rows = oracle_rows()
    c = LEDGER["c"]
    global LEDGER_C_PCT
    LEDGER_C_PCT = {"oracle": f"−{c['oracle_red_pct']:.0f}%", "null": f"+{-c['null_red_pct']:.0f}%"}
    col = {"null": LIGHT, "persistence": GREY, "ar": ORANGE, "direct": BLUE, "shiftwm": GREEN, "oracle": INK}
    y = np.arange(len(rows))[::-1]
    for (lab, v, k), yy in zip(rows, y):
        ax.barh(yy, v, height=0.62, color=col[k], ec="white" if k == "null" else "none", lw=0, zorder=2,
                hatch="/////" if k == "null" else None)
        per = LEDGER_C_PCT.get(k)
        txt = (per if k == "null" else f"{v:.2f} ({per})") if (per and compact) else f"{v:.2f}" + (f" ({per})" if per else "")
        ax.text(v + 0.03, yy, txt, fontsize=FS, va="center", ha="left",
                color="#00664A" if k == "shiftwm" else INK, fontweight="bold" if k in ("shiftwm", "oracle") else "normal")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    for t, (_, _, k) in zip(ax.get_yticklabels(), rows):
        if k == "shiftwm":
            t.set_color("#00664A"); t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, xmax); ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_xticklabels(["0", "0.5", "1", "1.5"])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("moving-patch error at $k{=}10$", labelpad=0.5)


# ------------------------------------------------------------------------------------------------ (d) knock-out + steering
def panel_transport(fig, x, y, w, h, compact=False):
    S = json.loads((mf.RES / "analysis/interpret/summary.json").read_text())
    LEDGER["d"] = {k: S[k] for k in ("knockout_increase_moving", "knockout_increase_highgate", "knockout_increase_static",
                                     "steer_moving", "steer_static", "steer_ratio_moving_over_static")}
    lab_w, hdr_h, gap = (0.33 if compact else 0.36), 0.14, 0.1
    hh = (h - 2 * hdr_h - gap) / 2
    r = S["steer_ratio_moving_over_static"]
    if compact:
        specs = [
            ("gate off ($g{=}0$): error rises", [("moving", S["knockout_increase_moving"]), ("static", S["knockout_increase_static"])],
             85, lambda v, lab: f"+{v:.0f}%"),
            ("other actions: shift (patches)", [("moving", S["steer_moving"]), ("static", S["steer_static"])],
             1.0, lambda v, lab: f"{v:.2f}" + (f" ({r:.1f}×)" if lab == "moving" else "")),
        ]
    else:
      specs = [
        ("turn off transport ($g{=}0$): error rises", [("moving", S["knockout_increase_moving"]), ("static", S["knockout_increase_static"])],
         70, lambda v, lab: f"+{v:.0f}%"),
        ("swap in other actions: shift (patches)", [("moving", S["steer_moving"]), ("static", S["steer_static"])],
         0.78, lambda v, lab: f"{v:.2f}" + (f"  ({r:.1f}×)" if lab == "moving" else "")),
    ]
    axs = []
    for i, (hdr, rows, xmax, fmt) in enumerate(specs):
        yy = y + h - (i + 1) * (hdr_h + hh) - i * gap
        a = axes_in(fig, x + lab_w, yy, w - lab_w, hh)
        for j, (lab, v) in enumerate(rows):
            a.barh(1 - j, v, height=0.66, color=INK if lab == "moving" else LIGHT, zorder=2)
            a.text(v + xmax * 0.03, 1 - j, fmt(v, lab), fontsize=FS, va="center", ha="left", color=INK,
                   fontweight="bold" if lab == "moving" else "normal")
        a.set_yticks([1, 0]); a.set_yticklabels([q[0] for q in rows]); a.tick_params(axis="y", length=0, pad=1.5)
        a.set_ylim(-0.5, 1.5); a.set_xlim(0, xmax); a.set_xticks([])
        a.grid(visible=False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_visible(False)
        a.axvline(0, color=MUTED, lw=0.6)
        W_, H_ = fig.get_figwidth(), fig.get_figheight()
        fig.text((x + 0.02) / W_, (yy + hh + 0.02) / H_, hdr, fontsize=FS, color=INK, ha="left", va="bottom",
                 style="italic")
        axs.append(a)
    return axs


# ------------------------------------------------------------------------------------------------ (e) ablations
ABL_LAB = {"actfree": "no actions", "w1": "no move ($w{=}1$)", "direct": "no transport (Direct)",
           "s1": "one frame ($S{=}1$)", "nocorr": "no correction",
           "w5": "window $w{=}5$", "ctr": "+ contrastive loss", "tanh": "tanh correction", "global": "global window"}


def panel_ablations(ax, compact=False, stacked_notes=False, gap_note=False):
    full, groups = F6.abl_rows()
    var = [r for _, rs in groups for r in rs]
    rem = sorted([r for r in var if F6.tag(r) in ("actfree", "w1", "direct", "s1", "nocorr")], key=lambda r: -r["d"])
    alt = sorted([r for r in var if r not in rem], key=lambda r: -r["d"])
    LEDGER["e"] = {F6.tag(r): {"d_pct": r["d"], "rank": r["rank"]} for r in var} | {"full_rank": full["rank"]}
    ys, y = [], 0.0
    for i, r in enumerate(rem + alt):
        if i == len(rem):
            y += 0.9
        ys.append(y); y += 1
    ys = np.array(ys)
    ax.axvspan(-NOISE, NOISE, color="#DADDE2", lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.6, zorder=1)
    for r, yy in zip(rem + alt, ys):
        t = F6.tag(r)
        c = BLUE if t == "direct" else (INK if abs(r["d"]) > 1 else MUTED)
        ax.plot([0, r["d"]], [yy, yy], color=c, lw=1.1, alpha=0.6, solid_capstyle="butt", zorder=2)
        ax.scatter([r["d"]], [yy], s=11, color=c, marker="D" if t == "direct" else "o", zorder=3, edgecolors="white",
                   linewidths=0.3)
        xt = max(r["d"], NOISE) + 0.45
        ax.text(xt, yy, f"{r['d']:+.1f}".replace("-", "−"), fontsize=FS, va="center", ha="left", color=c)
    ax.set_yticks(ys); ax.set_yticklabels([ABL_LAB[F6.tag(r)] for r in rem + alt])
    for tl, r in zip(ax.get_yticklabels(), rem + alt):
        tl.set_color(BLUE if F6.tag(r) == "direct" else (INK if r in rem else MUTED))
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    ax.set_xlim(-2.2, 13.5); ax.xaxis.set_major_locator(mt.MultipleLocator(5))
    ax.set_xlabel("val. error increase (%)", labelpad=0.5)
    ym = (ys[len(rem) - 1] + ys[len(rem)]) / 2
    ax.axhline(ym, color="#E3E6EA", lw=0.5, zorder=0)
    if compact:
        return
    if gap_note:
        ax.text(ax.get_xlim()[1], ym, f"design alternatives (all within {max(abs(r['d']) for r in alt):.1f}%)",
                fontsize=FS, color=MUTED, ha="right", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.3))
        ax.text(ax.get_xlim()[1], ys[-2], "grey band:", fontsize=FS, color=MUTED, ha="right", va="center")
        ax.text(ax.get_xlim()[1], ys[-1], "seed noise", fontsize=FS, color=MUTED, ha="right", va="center")
        return
    if stacked_notes:
        notes = ("alternatives:", f"within {max(abs(r['d']) for r in alt):.1f}%", "grey band:", "seed noise")
        for yy, t in zip(ys[len(rem):], notes):
            ax.text(13.5, yy, t, fontsize=FS, color=MUTED, ha="right", va="center")
        return
    ax.text(13.5, ys[len(rem)] - 0.05, f"alternatives:\nwithin {max(abs(r['d']) for r in alt):.1f}%", fontsize=FS, color=MUTED, ha="right", va="top",
            linespacing=0.95)
    ax.text(13.5, ys[-1] + 0.35, "grey: seed noise", fontsize=FS, color=MUTED, ha="right", va="bottom")


# ------------------------------------------------------------------------------------------------ (f) horizons + tasks
DS_BLUE = {"DROID": ("#08306B", "-", "o"), "Language-Table": ("#2171B5", (0, (4, 1.5)), "^"),
           "Open-H": ("#4292C6", "-", "s"), "IWS (3 tasks)": ("#5E9FD0", (0, (1.5, 1.2)), "D")}
DS_SHORT = {"Language-Table": "Lang.-T.", "IWS (3 tasks)": "IWS", "DROID": "DROID", "Open-H": "Open-H"}


def panel_horizon(fig, x, y, w, h, strip_w=0.3, tasks=True, styles=None, ylabel=None):
    ST = styles or DS_BLUE
    a = axes_in(fig, x, y, w - (strip_w if tasks else 0) - 0.34, h)
    curves = F6.horizon_curves()
    LEDGER["f"] = {"horizon": {}, "tasks": {}}
    ends = []
    for lab, seeds, g, lo, hi in curves:
        c, ls, mk = ST[lab][:3]; k = np.arange(1, len(g) + 1)
        a.fill_between(k, lo, hi, color=c, alpha=0.14, lw=0)
        a.plot(k, g, color=c, ls=ls, lw=1.0, marker=mk, ms=2.0, markevery=[0, len(g) - 1])
        ends.append([lab, len(g), float(g[-1]), ST[lab][3] if len(ST[lab]) > 3 else c])
        LEDGER["f"]["horizon"][lab] = {"seeds": int(seeds), "gain": g.tolist(), "ci_lo_min": float(lo.min())}
    a.axhline(0, color=INK, lw=0.6)
    a.set_xlim(0.5, 12.5); a.set_xticks([1, 5, 10]); a.set_ylim(0, 10)
    a.yaxis.set_major_locator(mt.MultipleLocator(2.5)); a.yaxis.set_major_formatter(mt.FormatStrFormatter("%g"))
    a.set_xlabel("forecast step $k$", labelpad=0.5)
    a.set_ylabel(ylabel or "reduction vs. Direct (%)", labelpad=1)
    ends.sort(key=lambda e: e[2]); gp = 1.15
    for i in range(1, len(ends)):
        ends[i][2] = max(ends[i][2], ends[i - 1][2] + gp)
    for lab, kx, yv, c in ends:
        a.annotate(DS_SHORT[lab], (kx, yv), xytext=(2.0, 0), textcoords="offset points", fontsize=FS, color=c,
                   va="center", annotation_clip=False)
    if not tasks:
        rows = F6.task_gains()
        LEDGER["f"]["tasks"] = {r["label"]: {"gain": float(r["direct"][0]), "lo": float(r["direct"][1])} for r in rows}
        assert all(v["lo"] > 0 for v in LEDGER["f"]["tasks"].values())
        return a, None
    # per-task strip, same y scale
    b = axes_in(fig, x + w - strip_w, y, strip_w, h)
    rows = F6.task_gains()
    xs = {"Open-H": np.linspace(0, 1.5, 7), "IWS": np.linspace(2.1, 2.7, 3)}
    idx = {"Open-H": 0, "IWS": 0}
    for r in rows:
        g, lo, hi = r["direct"]
        c, _, mk = DS_BLUE["Open-H" if r["grp"] == "Open-H" else "IWS (3 tasks)"]
        xx = xs[r["grp"]][idx[r["grp"]]]; idx[r["grp"]] += 1
        b.plot([xx, xx], [lo, hi], color=c, lw=0.8, solid_capstyle="butt")
        b.scatter([xx], [g], s=7, color=c, marker=mk, zorder=3, edgecolors="white", linewidths=0.25)
        LEDGER["f"]["tasks"][r["label"]] = {"gain": float(g), "lo": float(lo), "hi": float(hi)}
    b.axhline(0, color=INK, lw=0.6)
    b.set_ylim(0, 10); b.set_xlim(-0.35, 3.15)
    b.yaxis.set_major_locator(mt.MultipleLocator(2.5)); b.set_yticklabels([]); b.tick_params(axis="y", length=0)
    b.set_xticks([]); b.tick_params(axis="x", length=0)
    b.grid(axis="x", visible=False); b.spines["left"].set_visible(False)
    b.set_xlabel("tasks", labelpad=0.5)
    b.xaxis.set_label_coords(0.5, -0.1)
    assert all(v["lo"] > 0 for v in LEDGER["f"]["tasks"].values())
    return a, b


# ------------------------------------------------------------------------------------------------ figures
def build_main():
    W, H = 5.5, 3.0
    fig = plt.figure(figsize=(W, H))
    r1y, r1h, r2y, r2h = 1.85, 0.92, 0.29, 1.02
    t1, t2 = 2.975, 1.465
    title(fig, 0.0, t1, "(a) Lower error on 127 of 130 episodes")
    panel_episodes(axes_in(fig, 0.33, r1y, 1.52, r1h))
    title(fig, 2.00, t1, "(b) The gain grows with motion")
    panel_motion(axes_in(fig, 2.25, r1y, 1.32, r1h))
    title(fig, 3.76, t1, "(c) The future is the present, moved")
    panel_oracle(axes_in(fig, 4.44, r1y, 1.0, r1h))
    title(fig, 0.0, t2, "(d) Moving content goes via transport")
    panel_transport(fig, 0.0, r2y - 0.05, 1.62, r2h + 0.05)
    title(fig, 1.83, t2, "(e) Each core component matters")
    panel_ablations(axes_in(fig, 2.68, r2y, 0.95, r2h))
    title(fig, 3.76, t2, "(f) Holds at every horizon and task")
    panel_horizon(fig, 3.98, r2y, 1.49, r2h)
    _finish(fig, "composite_main", W)


TASK_SHORT = {"needle hand.": "needle", "retraction": "retraction", "tissue lift": "tissue lift"}


def panel_tasks(ax):
    """v2: labelled per-task gain over Direct (mean over k, 95% CI), Open-H then IWS, each sorted by gain."""
    rows = F6.task_gains()
    grp = {g: sorted([r for r in rows if r["grp"] == g], key=lambda r: -r["direct"][0]) for g in ("Open-H", "IWS")}
    order, ys, y = [], [], 0.0
    for g in ("Open-H", "IWS"):
        if order:
            y += 0.55
        for r in grp[g]:
            order.append(r); ys.append(y); y += 1
    ys = np.array(ys)
    for r, yy in zip(order, ys):
        c, _, mk = DS_BLUE["Open-H" if r["grp"] == "Open-H" else "IWS (3 tasks)"]
        g, lo, hi = r["direct"]
        ax.plot([lo, hi], [yy, yy], color=c, lw=0.9, solid_capstyle="butt")
        ax.scatter([g], [yy], s=8, color=c, marker=mk, zorder=3, edgecolors="white", linewidths=0.25)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(ys); ax.set_yticklabels([TASK_SHORT.get(r["label"], r["label"]) for r in order])
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.set_ylim(ys[-1] + 0.6, -0.6); ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 8.2); ax.set_xticks([0, 4, 8])
    ax.set_xlabel("per task (%)", labelpad=0.5)
    for g, key in (("Open-H", "Open-H"), ("IWS", "IWS (3 tasks)")):
        sel = [i for i, r in enumerate(order) if r["grp"] == g]
        yc = ys[sel].mean() + (0.5 if g == "Open-H" else 0.0)
        ax.text(8.1, yc, g, fontsize=FS, color=DS_BLUE[key][0], ha="right", va="center", style="italic", rotation=90)
    assert all(r["direct"][1] > 0 for r in order)


def build_main_v2():
    """As build_main, but (f) gets a labelled per-task panel; (d)/(e) are narrowed to make room."""
    W, H = 5.5, 3.0
    fig = plt.figure(figsize=(W, H))
    r1y, r1h, r2y, r2h = 1.85, 0.92, 0.29, 1.02
    t1, t2 = 2.975, 1.465
    title(fig, 0.0, t1, "(a) Lower error on 127 of 130 episodes")
    panel_episodes(axes_in(fig, 0.33, r1y, 1.52, r1h))
    title(fig, 2.00, t1, "(b) The gain grows with motion")
    panel_motion(axes_in(fig, 2.25, r1y, 1.32, r1h))
    title(fig, 3.76, t1, "(c) The future is the present, moved")
    panel_oracle(axes_in(fig, 4.44, r1y, 1.0, r1h))
    title(fig, 0.0, t2, "(d) Moving parts use transport")
    panel_transport(fig, 0.0, r2y - 0.05, 1.36, r2h + 0.05)
    title(fig, 1.47, t2, "(e) Each core component matters")
    panel_ablations(axes_in(fig, 2.33, r2y, 0.80, r2h), stacked_notes=True)
    title(fig, 3.24, t2, "(f) Holds at every horizon and task")
    panel_horizon(fig, 3.45, r2y, 0.80 + 0.34, r2h, tasks=False)
    panel_tasks(axes_in(fig, 5.08, r2y, 0.40, r2h + 0.02))
    _finish(fig, "composite_main_v2", W)


# v3: Open-H / IWS get clearly different tints (bars in (f2) coloured by group); (line, dash, marker, label colour)
DS_V3 = {"DROID": ("#08306B", "-", "o"), "Language-Table": ("#2171B5", (0, (4, 1.5)), "^"),
         "Open-H": ("#4A90C8", "-", "s"), "IWS (3 tasks)": ("#9CC6E6", (0, (1.5, 1.2)), "D", "#5E9FD0")}
GROUP_V3 = {"Open-H": "#4A90C8", "IWS": "#9CC6E6"}
TASK_FULL = {"knot tying": "knot tying", "needle hand.": "needle handover", "peg transfer": "peg transfer",
             "suturing 1": "suturing 1", "suturing 2": "suturing 2", "tissue lift": "tissue lifting",
             "retraction": "tissue retraction", "PushT": "PushT", "Box": "Box", "Rope": "Rope"}


def panel_task_bars(ax):
    """(f2): one horizontal bar per task = gain over Direct (mean over k), 95% CI whisker; grouped, sorted."""
    rows = F6.task_gains()
    ys, labels, y = [], [], 0.0
    hdr = []
    for g in ("Open-H", "IWS"):
        rs = sorted([r for r in rows if r["grp"] == g], key=lambda r: -r["direct"][0])
        hdr.append((g, y)); y += 0.95
        for r in rs:
            gain, lo, hi = r["direct"]
            ax.barh(y, gain, height=0.72, color=GROUP_V3[g], ec="none", zorder=2)
            ax.plot([lo, hi], [y, y], color=INK, lw=0.6, zorder=3, solid_capstyle="butt")
            ax.plot([lo, lo], [y - 0.18, y + 0.18], color=INK, lw=0.5, zorder=3)
            ax.plot([hi, hi], [y - 0.18, y + 0.18], color=INK, lw=0.5, zorder=3)
            ys.append(y); labels.append(TASK_FULL[r["label"]]); y += 1
            assert lo > 0
        y += 0.15
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0, pad=1.5)
    for g, yy in hdr:
        ax.text(-0.04, yy, g, transform=ax.get_yaxis_transform(), fontsize=FS, fontweight="bold",
                color="#1F5F99" if g == "Open-H" else "#4F8FC0", ha="right", va="center")
    ax.set_ylim(y - 0.15 - 0.4, -0.5)
    ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    ax.axvline(0, color=INK, lw=0.6, zorder=3)
    ax.set_xlim(0, 7.6); ax.set_xticks([0, 2, 4, 6])
    ax.set_xlabel("gain over Direct (%)", labelpad=0.5, loc="right")


def build_main_v3():
    """As v2; (d) narrowed, (f) = (f1) per-step curves + (f2) per-task bar chart."""
    W, H = 5.5, 3.0
    fig = plt.figure(figsize=(W, H))
    r1y, r1h, r2y, r2h = 1.85, 0.92, 0.29, 1.02
    t1, t2 = 2.975, 1.465
    title(fig, 0.0, t1, "(a) Lower error on 127 of 130 episodes")
    panel_episodes(axes_in(fig, 0.33, r1y, 1.52, r1h))
    title(fig, 2.00, t1, "(b) The gain grows with motion")
    panel_motion(axes_in(fig, 2.25, r1y, 1.32, r1h))
    title(fig, 3.76, t1, "(c) The future is the present, moved")
    panel_oracle(axes_in(fig, 4.44, r1y, 1.0, r1h))
    title(fig, 0.0, t2, "(d) Motion uses transport")
    panel_transport(fig, 0.0, r2y - 0.05, 1.18, r2h + 0.05, compact=True)
    title(fig, 1.30, t2, "(e) Each core component matters")
    panel_ablations(axes_in(fig, 2.07, r2y, 0.74, r2h), gap_note=True)
    title(fig, 2.95, t2, "(f) Every step and every task: gain over Direct")
    panel_horizon(fig, 3.16, r2y, 0.66 + 0.34, r2h, tasks=False, styles=DS_V3, ylabel="gain over Direct (%)")
    panel_task_bars(axes_in(fig, 4.92, r2y, 0.56, r2h + 0.05))
    _finish(fig, "composite_main_v3", W)


def build_alt():
    W, H = 5.5, 1.9
    fig = plt.figure(figsize=(W, H))
    py, ph, t = 0.29, 1.40, 1.875
    title(fig, 0.0, t, "(a) Lower error, 127/130 ep.")
    panel_episodes(axes_in(fig, 0.31, py, 0.94, ph), short=True)
    title(fig, 1.34, t, "(b) Future = present, moved")
    panel_oracle(axes_in(fig, 1.99, py, 0.80, ph), compact=True, xmax=2.45)
    title(fig, 2.90, t, "(c) Each component matters")
    panel_ablations(axes_in(fig, 3.72, py, 0.64, ph), compact=True)
    title(fig, 4.47, t, "(d) At every horizon")
    panel_horizon(fig, 4.70, py, 0.80, ph, tasks=False)
    _finish(fig, "composite_alt", W)


def _finish(fig, name, W):
    iss = mf.qa(fig, name, W)
    fig.savefig(HERE / f"{name}.pdf"); fig.savefig(HERE / f"{name}.png", dpi=400)
    plt.close(fig)
    print("wrote", name, "qa issues:", len(iss))


if __name__ == "__main__":
    for k in sys.argv[1:] or ["alt", "v2", "v3", "main"]:  # main last: its ledger is the complete one
        {"main": build_main, "alt": build_alt, "v2": build_main_v2, "v3": build_main_v3}[k]()
    (HERE / "ledger.json").write_text(json.dumps(LEDGER, indent=1, default=float))
