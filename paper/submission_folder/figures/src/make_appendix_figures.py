"""Compact multi-panel figures of the appendix (one row of panels each, 5.5 in wide).

  appx_horizon.pdf   fig:horizon         row 1: each baseline's error relative to ShiftWM's per forecast step
                                          (DROID, Open-H Hamlyn, Language-Table, IWS macro over its three tasks);
                                          row 2: the same ratio (mean over k) per Open-H surgical task.
                     Inputs: results/v2s/<ds>/dinov2s/<arm>/s*/eval_test.npz via make_figures.load_eval
                     (seed-mean per-episode MSE, the rows of Table 1). Bands: 95% paired bootstrap (10k) of the
                     ratio, resampling recording sessions for DROID and episodes (handles) elsewhere, within task
                     for IWS. Uses make_gain_horizon.gain_curves for DROID/Hamlyn/Language-Table (no macros written).
  appx_geometry.pdf  fig:geometry        (a,b) gain maps horizon x motion decile vs Direct / AR, (c) share of the
                                          oracle-move gain recovered on moving patches, (d) per-patch error
                                          difference to Direct at k=10.
                     Inputs: results/v2/analysis/anatomy/summary.json, results/v2/analysis/geometry/{summary.json,
                     examples.npz} (scripts/v2/anatomy.py, scripts/v2/geometry.py).
  appx_plugin.pdf    fig:vjepa-plugin    V-JEPA 2-AC validation error and gate during fine-tuning, error change per
                                          test horizon, and DINO-WM open-loop / teacher-forced error change
                                          (PushT, Wall). Inputs as make_vjepa_plugin.py.
  appx_planning.pdf  fig:planning        PushT closed-loop planning (official LeWM protocol): success, steps to first
                                          success, final position error, for the released LeWM, random actions and
                                          the matched predictors (make_tables.planning_data / planning_extra_data).

Nothing is interpolated or typed in; every value is read from the files above.
Usage (repo root):  PYTHONPATH=src python paper/submission_folder/figures/src/make_appendix_figures.py [--only ...]
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402
import make_gain_horizon as gh  # noqa: E402
import make_vjepa_plugin as vp  # noqa: E402
import make_anatomy as an  # noqa: E402

sys.path.insert(0, str(mf.ROOT / "scripts/v2"))
import make_tables as T  # noqa: E402

M = mf.METHODS
W = 5.5
GREEN, BLUE, ORANGE, PURPLE = M["shiftwm"][1], M["direct"][1], M["ar"][1], M["ar_tf"][1]
BASES = [("ar_tf", "AR-TF"), ("ar", "AR"), ("direct", "Direct")]
FT, FL, FK = 6.8, 6.2, 6.0          # panel title, label, tick (pt at print size)
N_BOOT = 10000
IWS = ("iws_pusht", "iws_box", "iws_rope")


def style(ax, xlabel=None, ylabel=None, title=None):
    ax.tick_params(labelsize=FK, length=2, pad=1.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FL, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FL, labelpad=1)
    if title:
        ax.set_title(title, fontsize=FT, pad=3, loc="left")


# ============================================================================================== per-step / per-task
def iws_curves():
    """Equal-weight macro over the IWS tasks: ratio(k) = sum_t mean err_b,t(k) / sum_t mean err_S,t(k) (x100),
    bootstrap over handles within each task."""
    sw = [mf.load_eval(t, "dinov2s", "shiftwm") for t in IWS]
    if any(s is None for s in sw):
        return None
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, s["mse"].shape[1], (N_BOOT, s["mse"].shape[1])) for s in sw]
    S = [s["mse"].mean(0) for s in sw]                                     # [E, K] per task
    s_mean = sum(x.mean(0) for x in S); s_boot = sum(x[i].mean(1) for x, i in zip(S, idx))
    out = {"curves": {}}
    for arm, _ in BASES:
        B = [mf.load_eval(t, "dinov2s", arm)["mse"].mean(0) for t in IWS]
        b_mean = sum(x.mean(0) for x in B); b_boot = sum(x[i].mean(1) for x, i in zip(B, idx))
        ratio = 100 * b_mean / s_mean
        lo, hi = np.percentile(100 * b_boot / s_boot, [2.5, 97.5], axis=0)
        out["curves"][arm] = (None, ratio, lo, hi)
    return out


def hamlyn_task_ratios():
    ev = {a: T.load("openh_hamlyn", a) for a in ("shiftwm",) + tuple(a for a, _ in BASES)}
    tasks = np.array(ev["shiftwm"]["tasks"])
    names = [("knot_tying", "knot tying"), ("needle_grasp_and_handover", "needle"), ("peg_transfer", "peg transfer"),
             ("suturing_1", "suturing 1"), ("suturing_2", "suturing 2"), ("tissue_lifting", "tissue lift"),
             ("tissue_retraction", "retraction")]
    s = ev["shiftwm"]["mse"].mean(1)
    out = {}
    for arm, _ in BASES:
        b = ev[arm]["mse"].mean(1)
        assert list(ev[arm]["episodes"]) == list(ev["shiftwm"]["episodes"])
        out[arm] = [100 * b[tasks == t].mean() / s[tasks == t].mean() for t, _ in names]
    n = [int((tasks == t).sum()) for t, _ in names]
    return [lab for _, lab in names], n, out


def fig_horizon():
    panels = [("droid", "DROID"), ("openh_hamlyn", "Open-H Hamlyn"), ("language_table", "Language-Table"),
              ("iws", "IWS (3 tasks)")]
    data = {ds: (iws_curves() if ds == "iws" else gh.gain_curves(ds)) for ds, _ in panels}
    fig = plt.figure(figsize=(W, 2.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.78], wspace=0.34, hspace=0.78, left=0.075, right=0.975,
                          top=0.87, bottom=0.1)
    for j, (ds, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j]); D = data[ds]
        K = len(next(iter(D["curves"].values()))[1]); k = np.arange(1, K + 1)
        for arm, lab in BASES:
            _, ratio, lo, hi = D["curves"][arm]; _, col, _, mk = M[arm]
            ax.fill_between(k, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(k, ratio, color=col, lw=1.1, marker=mk, ms=2.3, markevery=[0, K // 2 - 1, K - 1], zorder=3)
        ax.axhline(100, color=GREEN, lw=1.8, zorder=4)
        top = max(D["curves"][a][3].max() for a, _ in BASES); bot = min(D["curves"][a][2].min() for a, _ in BASES)
        r = top - min(100, bot); ax.set_ylim(min(100, bot) - 0.1 * r, top + 0.08 * r)
        ax.set_xlim(0.6, K + 0.4); ax.set_xticks([1, K // 2, K] if K != 10 else [1, 5, 10])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        style(ax, "forecast step $k$", "error rel. to ShiftWM (%)" if j == 0 else None, f"({'abcd'[j]}) {title}")
    # row 2: Hamlyn per task
    labs, n, R = hamlyn_task_ratios()
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(labs)); off = {"ar_tf": -0.18, "ar": 0.0, "direct": 0.18}
    for arm, lab in BASES:
        _, col, _, mk = M[arm]
        ax.scatter(x + off[arm], R[arm], s=13, marker=mk, color=col, edgecolors="white", linewidths=0.4, zorder=3)
        ax.vlines(x + off[arm], 100, R[arm], color=col, lw=0.8, alpha=0.5, zorder=2)
    ax.axhline(100, color=GREEN, lw=1.3, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f"{l_} ({n_})" for l_, n_ in zip(labs, n)])
    ax.set_xlim(-0.5, len(labs) - 0.5); ax.grid(axis="x", visible=False)
    ax.set_ylim(92, max(max(v) for v in R.values()) * 1.04); ax.yaxis.set_major_locator(MaxNLocator(4))
    style(ax, None, "rel. error (%)", "(e) Open-H Hamlyn per surgical task (test episodes), mean over $k$")
    h = [Line2D([], [], color=GREEN, lw=1.8, label="ShiftWM (ours) = 100%")]
    h += [Line2D([], [], color=M[a][1], marker=M[a][3], ms=3, lw=1.0, label=lab) for a, lab in BASES]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=FL, frameon=False,
               handlelength=1.6, columnspacing=1.4)
    mf.qa(fig, "appx_horizon", W)
    fig.savefig(mf.FIG / "appx_horizon.pdf"); fig.savefig(mf.FIG / "appx_horizon_preview.png", dpi=200); plt.close(fig)


# ============================================================================================== geometry / anatomy
def fig_geometry():
    A = json.loads((mf.RES / "analysis/anatomy/summary.json").read_text())
    G = json.loads((mf.RES / "analysis/geometry/summary.json").read_text())
    X = np.load(mf.RES / "analysis/geometry/examples.npz")
    from scipy.ndimage import zoom
    fig = plt.figure(figsize=(W, 1.6))
    gs = fig.add_gridspec(1, 4, wspace=0.42, left=0.075, right=0.975, top=0.83, bottom=0.22)
    for j, (base, title) in enumerate((("direct", "(a) Gain over Direct (%)"), ("ar", "(b) Gain over AR (%)"))):
        m = np.asarray(A["gain_map"][base])                                 # [decile, k]
        f = zoom(m, 6, order=3, mode="nearest")
        xx = np.linspace(1, 10, f.shape[1]); yy = np.linspace(1, 10, f.shape[0])
        lv = np.linspace(0, np.ceil(f.max() / 5) * 5, 11) if base == "ar" else np.linspace(0, np.ceil(f.max()), 8)
        ax = fig.add_subplot(gs[0, j])
        ax.contourf(xx, yy, f, levels=lv, cmap=an.CMAP, extend="both")
        cl = ax.contour(xx, yy, f, levels=lv[1::2], colors="white", linewidths=0.5, alpha=0.9)
        for t_ in ax.clabel(cl, fmt="%d", fontsize=6.0, inline=True, inline_spacing=2):
            x_, y_ = t_.get_position()
            if not (1.8 < x_ < 9.2 and 1.8 < y_ < 8.2):
                t_.set_visible(False)
        ax.set_xticks([1, 5, 10]); ax.set_yticks([1, 10])
        ax.set_yticklabels(["static", "fast"] if j == 0 else ["", ""]); ax.grid(False)
        ax.text(0.96, 0.95, f"+{m.min():.0f} to +{m.max():.0f}", transform=ax.transAxes, fontsize=6.0, color=mf.INK,
                ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
        style(ax, "horizon $k$", "true motion (decile)" if j == 0 else None, title)
    # (c) share of oracle-move gain recovered, moving patches
    ax = fig.add_subplot(gs[0, 2]); mc = A["moving_curves"]; k = np.arange(1, 11)
    st, orc = np.array(mc["persistence"]["mean"]), np.array(mc["oracle"]["mean"])
    rec = {a: 100 * (st - np.array(mc[a]["mean"])) / (st - orc) for a in ("shiftwm", "direct", "ar")}
    for a, lab in (("ar", "AR"), ("direct", "Direct"), ("shiftwm", "ShiftWM")):
        _, col, _, mk = M[a]
        ax.plot(k, rec[a], color=col, lw=1.6 if a == "shiftwm" else 1.0, marker=mk, ms=2.2, markevery=[0, 4, 9], label=lab)
        ax.text(10.35, rec[a][-1], f"{rec[a][-1]:.0f}%", fontsize=6.0, va="center", color=col,
                fontweight="bold" if a == "shiftwm" else "normal")
    ax.set_xlim(0.7, 12.2); ax.set_xticks([1, 5, 10]); ax.set_ylim(20, 88)
    ax.legend(fontsize=6.0, frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0.2, bbox_to_anchor=(0.9, 0))
    style(ax, "horizon $k$", "oracle gain recovered (%)", "(c) Moving patches")
    # (d) per-patch difference to Direct at k=10
    ax = fig.add_subplot(gs[0, 3])
    bins = X["bins"]; c = (bins[:-1] + bins[1:]) / 2; h = X["diff_direct"] / X["diff_direct"].sum()
    ax.bar(c, h, width=bins[1] - bins[0], color=np.where(c > 0, GREEN, "#E3A79F"), edgecolor="none")
    ax.axvline(0, color=mf.INK, lw=0.6)
    wr = G["moving_patch_winrate"]["vs_direct"]
    ax.text(0.97, 0.95, f"ShiftWM\nlower on\n{100 * wr:.0f}%", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.0, color=GREEN, fontweight="bold")
    ax.set_ylim(0, h.max() * 1.35); ax.set_xlim(-1.2, 1.2); ax.set_yticks([]); ax.grid(False)
    style(ax, "Direct err $-$ ShiftWM err", "share of patches", "(d) Per moving patch")
    mf.qa(fig, "appx_geometry", W)
    fig.savefig(mf.FIG / "appx_geometry.pdf"); fig.savefig(mf.FIG / "appx_geometry_preview.png", dpi=200); plt.close(fig)


# ============================================================================================== plug-in dynamics
def fig_plugin():
    fig = plt.figure(figsize=(W, 1.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.85, 1, 1.25], wspace=0.5, left=0.07, right=0.985, top=0.8, bottom=0.2)
    BB, GREY = vp.BLUE, vp.GREY
    ax = fig.add_subplot(gs[0, 0])
    for arm, col in (("finetune", BB), ("finetune_shiftwm", GREEN)):
        st, m, _, pers = vp.curves(arm)
        ax.plot(st / 1000, m.mean(0), color=col, lw=1.4 if col == GREEN else 1.1, marker="o", ms=1.6, zorder=3)
        ax.fill_between(st / 1000, m.min(0), m.max(0), color=col, alpha=0.2, lw=0)
    ax.axhline(pers, color=GREY, lw=0.7, ls=(0, (3, 2)))
    ax.text(st[-1] / 1000, pers, "persistence", fontsize=6.0, color=GREY, ha="right", va="bottom")
    ax.set_ylim(0.3, pers * 1.1)
    style(ax, "step ($\\times$1000)", "val. latent error", "(a) V-JEPA 2-AC")
    ax = fig.add_subplot(gs[0, 1])
    st, _, g, _ = vp.curves("finetune_shiftwm")
    ax.plot(st / 1000, g.mean(0), color=GREEN, lw=1.4, marker="o", ms=1.6)
    ax.fill_between(st / 1000, np.nanmin(g, 0), np.nanmax(g, 0), color=GREEN, alpha=0.2, lw=0)
    ax.set_ylim(0, 1)
    style(ax, "step ($\\times$1000)", "mean gate $g$", "(b) Gate")
    ax = fig.add_subplot(gs[0, 2])
    B, C = vp.vjepa_test("finetune"), vp.vjepa_test("finetune_shiftwm")
    d = 100 * (C.mean(0) / B.mean(0) - 1)
    pairs = np.array([100 * (c / b - 1) for c in C for b in B]).T
    x = np.arange(1, len(d) + 1)
    ax.bar(x, d, width=0.7, color=GREEN, zorder=3)
    ax.vlines(x, pairs.min(1), pairs.max(1), color=mf.INK, lw=0.6, zorder=4)
    ax.axhline(0, color=mf.INK, lw=0.6); ax.set_xticks([1, 5, 10]); ax.grid(axis="x", visible=False)
    ax.set_ylim(min(pairs.min(), d.min()) * 1.2, 2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "−") if round(v) else "0"))
    style(ax, "test horizon $k$", "error change", "(c) V-JEPA 2-AC, test")
    ax = fig.add_subplot(gs[0, 3])
    envs = [("pusht", "PushT"), ("wall", "Wall")]
    labels = [str(i) for i in range(1, 6)] + ["TF"]
    w = 0.38
    for i, (env, name) in enumerate(envs):
        b, s = vp.dinowm(env, "dinowm"), vp.dinowm(env, "dinowm_shiftwm")
        n = min(len(b["ol"]), len(s["ol"]))
        dd = np.array(list(100 * (s["ol"][:n] / b["ol"][:n] - 1)) + [100 * (s["tf"] / b["tf"] - 1)])
        xs = np.arange(len(dd)) + (i - 0.5) * w
        ax.bar(xs, dd, width=w, color=[GREEN if v <= 0 else vp.WORSE for v in dd], edgecolor="white", lw=0.3,
               hatch=None if env == "pusht" else "////", zorder=3)
    ax.axhline(0, color=mf.INK, lw=0.6); ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "−") if round(v) else "0"))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#9AA3AE", label="PushT"), Patch(fc="#9AA3AE", hatch="////", ec="white", label="Wall")],
              fontsize=6.0, frameon=False, loc="upper center", ncol=2, handlelength=1.2, borderaxespad=0.1,
              columnspacing=1.0)
    ax.set_ylim(None, ax.get_ylim()[1] * 1.45)
    style(ax, "open-loop step / teacher-forced", None, "(d) DINO-WM, validation")
    h = [Line2D([], [], color=BB, lw=1.1, marker="o", ms=2, label="backbone alone"),
         Line2D([], [], color=GREEN, lw=1.4, marker="o", ms=2, label="+ ShiftWM head (bars: change from adding it)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=FL, frameon=False,
               handlelength=1.6, columnspacing=1.6)
    mf.qa(fig, "appx_plugin", W)
    fig.savefig(mf.FIG / "appx_plugin.pdf"); fig.savefig(mf.FIG / "appx_plugin_preview.png", dpi=200); plt.close(fig)


# ============================================================================================== planning (PushT)
PLAN = [("random", "random actions", mf.MUTED, "x"), ("lewm", "LeWM (released)", mf.BACKBONE, "*"),
        ("v2_ar_tf_s0", "AR-TF", PURPLE, "s"), ("v2_ar_s0", "AR", ORANGE, "^"),
        ("v2_direct_s0", "Direct", BLUE, "D"), ("v2_shiftwm_s0", "ShiftWM", GREEN, "o")]


def fig_planning():
    P, X = T.planning_data(), T.planning_extra_data()
    rows = [r for r in PLAN if r[0] in P and P[r[0]]["pusht"] is not None]
    fig = plt.figure(figsize=(W, 1.05))
    gs = fig.add_gridspec(1, 3, wspace=0.12, left=0.155, right=0.975, top=0.83, bottom=0.2)
    specs = [("success (%) $\\uparrow$", lambda k: P[k]["pusht"], (0, 100), "{:.1f}"),
             ("steps to first success $\\downarrow$", lambda k: X[k]["pusht"][0], (0, 55), "{:.1f}"),
             ("final position error (px, log) $\\downarrow$", lambda k: X[k]["pusht"][1], (8, 600), "{:.1f}")]
    y = np.arange(len(rows))[::-1]
    for j, (title, get, xl, fmt) in enumerate(specs):
        ax = fig.add_subplot(gs[0, j]); vs = [get(k) for k, *_ in rows]
        if xl is None:
            xl = (0, max(vs) * 1.3)
        for (key, lab, c, mk), yy, v in zip(rows, y, vs):
            ax.plot([xl[0], v], [yy, yy], color=c, lw=1.0, alpha=0.35, solid_capstyle="butt")
            ax.scatter([v], [yy], s=20 if mk == "*" else 13, marker=mk, color=c, zorder=3, linewidths=0.8)
            frac = (np.log(v / xl[0]) / np.log(xl[1] / xl[0])) if j == 2 else (v - xl[0]) / (xl[1] - xl[0])
            inside = frac > 0.7
            ax.annotate(fmt.format(v), (v, yy), xytext=(-5 if inside else 5, 0), textcoords="offset points",
                        fontsize=6.0, color=c, va="center", ha="right" if inside else "left")
        if j == 2:
            ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(["10", "30", "100", "300"])
            ax.minorticks_off()
        ax.set_xlim(*xl); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.axhline(len(rows) - 2.5, color=mf.PANEL_EDGE, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab, *_ in rows] if j == 0 else [])
        if j == 0:
            for t_, (_, _, c, _) in zip(ax.get_yticklabels(), rows):
                t_.set_color(c)
        ax.grid(axis="y", visible=False)
        if j != 2:
            ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=FK, length=2, pad=1.5)
        ax.set_title(title, fontsize=FL, pad=3)
    mf.qa(fig, "appx_planning", W)
    fig.savefig(mf.FIG / "appx_planning.pdf"); fig.savefig(mf.FIG / "appx_planning_preview.png", dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--only", nargs="*"); a = p.parse_args()
    jobs = {"horizon": fig_horizon, "geometry": fig_geometry, "plugin": fig_plugin, "planning": fig_planning}
    for k, fn in jobs.items():
        if not a.only or k in a.only:
            fn(); print("wrote", k)


if __name__ == "__main__":
    main()
