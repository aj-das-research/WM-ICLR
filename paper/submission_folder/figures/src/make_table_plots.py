"""Plots that replace former tables. Every value comes from the same generator as the old table rows:

  ablations.pdf        fig:ablations   scripts/v2/make_tables.py ablation_data()  (results/v2s/.../summary.json, val split)
  hamlyn_tasks.pdf     fig:horizon (c) make_tables.load("openh_hamlyn", arm)      (eval_test.npz, per task)
  planning.pdf         fig:planning    make_tables.planning_data() / planning_extra_data()  (results/v2/planning)
  segmentation.pdf     fig:segments    results/v2/analysis/segments/<ds>/summary.json (same as make_segments.py macros)

Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_table_plots.py [--only ...]
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

sys.path.insert(0, str(mf.ROOT / "scripts/v2"))
import make_tables as T  # noqa: E402

M = mf.METHODS
W = 5.5


def _clean(label):
    return (label.replace(r"\ours{}", "ShiftWM").replace(r"\citep{maes2026lewm}", "").replace("(ours)", "").strip())


def _canvas(h):
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def _axes(fig, x0, y0, w, h):
    H = fig.get_figheight()
    a = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    a.tick_params(labelsize=mf.FS_TICK, length=2, pad=1)
    a.grid(axis="y", visible=False); a.grid(axis="x", color="#EEF0F3", lw=0.5)
    return a


# --------------------------------------------------------------------------------------------- ablations (main text)
ABL_SHORT = {  # compact display names (same variants and order source as the table)
    "v2s/droid/dinov2s/shiftwm/s0": "full model (window 7, 3 sources)",
    "v2s/droid/dinov2s/direct/s0": "no transport head (= Direct)",
    "v2s/droid/dinov2s/ablations/w1/s0": "window $w{=}1$ (gate + correction, no move)",
    "v2s/droid/dinov2s/ablations/w5/s0": "window $w{=}5$",
    "v2s/droid/dinov2s/ablations/global/s0": "global window",
    "v2s/droid/dinov2s/ablations/s1/s0": "last frame only ($S{=}1$)",
    "v2s/droid/dinov2s/ablations/tanh/s0": "tanh-bounded correction",
    "v2s/droid/dinov2s/ablations/nocorr/s0": "no correction (pure transport)",
    "v2s/droid/dinov2s/ablations/actfree/s0": "action-free",
    "v2s/droid/dinov2s/ablations/ctr/s0": "+ action-contrastive loss ($\\lambda{=}0.5$)",
    "v2s/droid/dinov2s/ablations/dinov2b_shiftwm/s0": "DINOv2-B/14 encoder",
}


def fig_ablations():
    rows = T.ablation_data()
    full = next(r for r in rows if r["full"])
    var = sorted([r for r in rows if not r["full"]], key=lambda r: -r["d"])      # largest harm first
    order = [full] + var
    n = len(order)
    pitch, top_pad, bot = 0.128, 0.2, 0.28
    H = top_pad + n * pitch + bot
    fig, ax = _canvas(H)
    ys = [H - top_pad - (i + 0.5) * pitch for i in range(n)]
    xg, xl = 0.02, 0.62                                # group column, variant column
    xa, wa = 2.68, 1.32                                # Delta panel
    xb, wb = 4.12, 0.5                                 # action-ranking panel
    xr = 4.66                                          # rank value column
    xc = 4.95                                          # val. MSE text column
    for x, t, ha in ((xg, "component", "left"), (xl, "variant", "left"), (xa + wa / 2, "$\\Delta$ val. error (%)", "center"),
                     (xb + (xr + 0.25 - xb) / 2, "rank acc. (%)", "center"), (xc + 0.27, "MSE / $k{=}K$", "center")):
        ax.text(x, H - 0.03, t, fontsize=mf.FS_NOTE, color=mf.MUTED, ha=ha, va="top", fontweight="bold")
    for r, y in zip(order, ys):
        if r["full"]:
            ax.add_patch(matplotlib.patches.Rectangle((0.0, y - pitch / 2), W, pitch, fc="#E8F5EF", ec="none", zorder=0))
        ax.text(xg, y, "" if r["full"] else r["grp"], fontsize=mf.FS_NOTE, color=mf.MUTED, va="center")
        ax.text(xl, y, ABL_SHORT.get(r["rel"], _clean(r["label"])), fontsize=mf.FS_NOTE, va="center",
                color=M["shiftwm"][1] if r["full"] else mf.INK, fontweight="bold" if r["full"] else "normal")
        ax.text(xc + 0.27, y, f"{r['avg']:.3f} / {r['end']:.3f}", fontsize=mf.FS_NOTE, va="center", ha="center",
                color=mf.INK)
        ax.text(xr + 0.25, y, "n/a" if r["rank"] is None else f"{r['rank']:.1f}", fontsize=mf.FS_NOTE, va="center",
                ha="right", color=mf.MUTED if r["rank"] is None else (M["shiftwm"][1] if r["full"] else mf.INK))
    ylo, yhi = ys[-1] - pitch / 2, ys[0] + pitch / 2
    # Delta panel
    a = _axes(fig, xa, ylo, wa, yhi - ylo); a.patch.set_alpha(0)
    lo, hi = min(-2.0, min(r["d"] for r in var) - 0.5), max(r["d"] for r in var) * 1.3
    a.set_xlim(lo, hi); a.set_ylim(ylo, yhi); a.set_yticks([])
    a.axvline(0, color=mf.INK, lw=0.6)
    for r, y in zip(order, ys):
        if r["full"]:
            a.scatter([0], [y], s=22, color=M["shiftwm"][1], zorder=3, edgecolors="white", linewidths=0.5)
            continue
        c = M["direct"][1] if "direct/s0" in r["rel"] else (mf.LOSS if r["d"] > 1 else mf.MUTED)
        a.plot([0, r["d"]], [y, y], color=c, lw=1.2, alpha=0.5, solid_capstyle="butt")
        a.scatter([r["d"]], [y], s=16, color=c, zorder=3, marker="D" if "direct/s0" in r["rel"] else "o")
        a.text(r["d"] + (0.5 if r["d"] >= 0 else -0.5), y, f"{r['d']:+.1f}".replace("-", "−"), fontsize=mf.FS_NOTE,
               va="center", ha="left" if r["d"] >= 0 else "right", color=c)
    a.set_xlabel("worse $\\rightarrow$", fontsize=mf.FS_NOTE, labelpad=1, loc="right")
    for s_ in ("left",):
        a.spines[s_].set_visible(False)
    # action-ranking panel
    b = _axes(fig, xb, ylo, wb, yhi - ylo); b.patch.set_alpha(0)
    rk = [r["rank"] for r in order if r["rank"] is not None]
    b.set_xlim(min(rk) - 0.4, max(rk) + 0.4); b.set_ylim(ylo, yhi); b.set_yticks([]); b.spines["left"].set_visible(False)
    b.axvline(full["rank"], color=M["shiftwm"][1], lw=0.6, ls=(0, (2, 2)))
    for r, y in zip(order, ys):
        if r["rank"] is None:
            continue
        c = M["shiftwm"][1] if r["full"] else mf.INK
        b.scatter([r["rank"]], [y], s=14, color=c, zorder=3)
    b.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    b.set_xlabel("better $\\rightarrow$", fontsize=mf.FS_NOTE, labelpad=1, loc="right")
    mf.qa(fig, "ablations", W)
    fig.savefig(mf.FIG / "ablations.pdf"); plt.close(fig)


# --------------------------------------------------------------------------------------------- Hamlyn per task
TASKS = [("knot_tying", "knot tying"), ("needle_grasp_and_handover", "needle"), ("peg_transfer", "peg transfer"),
         ("suturing_1", "suturing 1"), ("suturing_2", "suturing 2"), ("tissue_lifting", "tissue lift"),
         ("tissue_retraction", "retraction")]
ARMS = [("persistence", "Persistence"), ("ar_tf", "AR-TF"), ("ar", "AR"), ("direct", "Direct"), ("shiftwm", "ShiftWM")]


def fig_hamlyn_tasks():
    vals = {}
    for arm, _ in ARMS:
        ev = T.load("openh_hamlyn", arm)
        t = np.array(ev["tasks"]); m = ev["mse"].mean(1)
        vals[arm] = [float(m[t == k].mean()) for k, _ in TASKS]
    H = 1.3
    fig, ax = _canvas(H)
    a = _axes(fig, 0.42, 0.3, W - 0.52, H - 0.55)
    x = np.arange(len(TASKS))
    off = np.linspace(-0.24, 0.24, len(ARMS))
    for (arm, lab), o in zip(ARMS, off):
        _, c, _, mk = M[arm]
        a.scatter(x + o, vals[arm], s=20 if arm == "shiftwm" else 13, marker=mk or "_", color=c, zorder=3,
                  linewidths=1.2 if arm == "persistence" else 0.5, edgecolors="white" if mk else None, label=lab)
    for i in range(len(TASKS)):
        a.text(i + off[-1] + 0.07, vals["shiftwm"][i], f"{vals['shiftwm'][i]:.3f}", fontsize=mf.FS_NOTE,
               color=M["shiftwm"][1], va="center", ha="left")
    a.set_xticks(x); a.set_xticklabels([t for _, t in TASKS], fontsize=mf.FS_TICK)
    a.set_xlim(-0.5, len(TASKS) - 0.3); a.grid(axis="x", visible=False); a.grid(axis="y", color="#EEF0F3", lw=0.5)
    a.set_ylabel("test MSE", fontsize=mf.FS_NOTE, labelpad=1)
    ax.text(0.03, H - 0.03, "Open-H Hamlyn: test MSE per surgical task", fontsize=mf.FS_TITLE, fontweight="bold", color=mf.INK,
            va="top")
    a.legend(ncol=5, fontsize=mf.FS_NOTE, loc="lower right", bbox_to_anchor=(1.0, 1.02), frameon=False,
             handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    mf.qa(fig, "hamlyn_tasks", W)
    fig.savefig(mf.FIG / "hamlyn_tasks.pdf"); plt.close(fig)


# --------------------------------------------------------------------------------------------- planning
PLAN = [("random", "random actions", mf.MUTED, "x"), ("lewm", "LeWM (released)", mf.BACKBONE, "*"),
        ("v2_ar_tf_s0", "AR-TF", M["ar_tf"][1], M["ar_tf"][3]), ("v2_ar_s0", "AR", M["ar"][1], M["ar"][3]),
        ("v2_direct_s0", "Direct", M["direct"][1], M["direct"][3]), ("v2_shiftwm_s0", "ShiftWM", M["shiftwm"][1], "o")]
ENVS = [("pusht", "PushT"), ("tworoom", "TwoRoom"), ("reacher", "Reacher")]


def fig_planning():
    P, X = T.planning_data(), T.planning_extra_data()
    rows = [r for r in PLAN if r[0] in P]
    n = len(rows)
    pitch, top, bot = 0.14, 0.36, 0.42
    H = top + n * pitch + bot
    fig, ax = _canvas(H)
    ys = [H - top - (i + 0.5) * pitch for i in range(n)]
    lab_w = 0.78
    for (key, lab, c, _), y in zip(rows, ys):
        ax.text(lab_w - 0.04, y, lab, fontsize=mf.FS_NOTE, ha="right", va="center", color=c,
                fontweight="bold" if key == "v2_shiftwm_s0" else "normal")
    ylo, yhi = ys[-1] - pitch / 2, ys[0] + pitch / 2
    ax.plot([0.05, W - 0.05], [(ys[1] + ys[2]) / 2] * 2, color=mf.PANEL_EDGE, lw=0.6)
    # panel specs: (x0, width, title, getter, fmt, xlim)
    def succ(env):
        return lambda k: P[k][env]
    def steps(env):
        return lambda k: (X.get(k, {}).get(env) if X.get(k, {}).get(env) != "notrun" else None) and X[k][env][0]
    def err(env):
        return lambda k: (X.get(k, {}).get(env) if X.get(k, {}).get(env) != "notrun" else None) and X[k][env][1]
    panels = [(lab_w, 0.95, "success, PushT (%)", succ("pusht"), "{:.1f}", (0, 100)),
              (lab_w + 1.05, 0.62, "TwoRoom (%)", succ("tworoom"), "{:.1f}", (0, 100)),
              (lab_w + 1.77, 0.62, "Reacher (%)", succ("reacher"), "{:.1f}", (0, 100)),
              (lab_w + 2.49, 0.62, "PushT steps", steps("pusht"), "{:.1f}", (0, 60)),
              (lab_w + 3.21, 0.62, "PushT error (px)", err("pusht"), "{:.1f}", None),
              (lab_w + 3.93, 0.62, "s / plan", lambda k: P[k]["spp"], "{:.2f}", (0, 8))]
    for x0, w, title, get, fmt, xl in panels:
        a = _axes(fig, x0, ylo, w, yhi - ylo)
        a.set_ylim(ylo, yhi); a.set_yticks([]); a.spines["left"].set_visible(False)
        vs = [get(k) for k, *_ in rows]
        if xl is None:
            xl = (0, max(v for v in vs if v is not None) * 1.45)
        a.set_xlim(*xl)
        for (key, lab, c, mk), y, v in zip(rows, ys, vs):
            if v is None:
                a.text(xl[0] + 0.03 * (xl[1] - xl[0]), y, "n/a" if key == "random" else "not run", fontsize=mf.FS_NOTE, color="#B8BEC7", va="center")
                continue
            a.plot([xl[0], v], [y, y], color=c, lw=1.0, alpha=0.35, solid_capstyle="butt")
            a.scatter([v], [y], s=18 if mk == "*" else 13, marker=mk, color=c, zorder=3, linewidths=0.8)
            right = v > xl[0] + 0.62 * (xl[1] - xl[0])
            a.text(v + (-0.07 if right else 0.07) * (xl[1] - xl[0]), y + (0.0 if not right else 0), fmt.format(v),
                   fontsize=mf.FS_NOTE, color=c, va="center", ha="right" if right else "left",
                   bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.8) if right else None)
        a.set_title(title, fontsize=mf.FS_NOTE, pad=2, color=mf.INK, fontweight="bold")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.text(0.03, H - 0.03, "success $\\uparrow$ (50 episodes $\\times$ 3 planner seeds);  steps to first success, "
            "final goal error, time per plan $\\downarrow$", fontsize=mf.FS_NOTE, color=mf.MUTED, va="top")
    # TwoRoom / Reacher steps and errors exist for the references only: printed as a compact key (same generator)
    ref = []
    for key, lab in (("lewm", "LeWM"), ("random", "random")):
        for env, en, u, f_ in (("tworoom", "TwoRoom", "px", "{:.1f}"), ("reacher", "Reacher", "rad", "{:.3f}")):
            st, er = X[key][env]
            ref.append(f"{lab} {en}: {st:.1f} steps, {f_.format(er)} {u}")
    ax.text(0.03, 0.03, "References, steps to success and final error:  " + ";  ".join(ref[:2]) + ";\n" + ";  ".join(ref[2:]),
            fontsize=mf.FS_NOTE, color=mf.MUTED, va="bottom", linespacing=1.05)
    mf.qa(fig, "planning", W)
    fig.savefig(mf.FIG / "planning.pdf"); plt.close(fig)


# --------------------------------------------------------------------------------------------- segmentation
def _seg(ds):
    S = json.loads((mf.RES / "analysis/segments" / ds / "summary.json").read_text())
    return S, S["results"][S["primary_labeller"]], S["placement"]["px"], S["horizon"]


def fig_segmentation():
    H = 1.8
    fig, ax = _canvas(H)
    ax.text(0.03, 0.47, "moving,\nmean\nover $k$:", fontsize=mf.FS_NOTE, color=mf.MUTED, va="top", linespacing=1.0)
    specs = [("droid", "iou", "DROID IoU gain (pts)"), ("droid", "place", "DROID placement gain (px)"),
             ("openh_hamlyn", "iou", "Hamlyn IoU gain (pts)"), ("openh_hamlyn", "place", "Hamlyn place. gain (px)")]
    x0s = [0.42, 1.74, 3.06, 4.38]
    for (ds, kind, title), x0 in zip(specs, x0s):
        S, R, P, K = _seg(ds)
        a = _axes(fig, x0, 0.72, 0.98, H - 1.03)
        k = np.arange(1, K + 1)
        a.axhline(0, color=mf.INK, lw=0.6)
        for base in ("direct", "ar"):
            _, c, _, mk = M[base]
            for sub, ls, alpha in (("moving", "-", 0.18), ("all", (0, (2, 1.5)), 0)):
                src = R[sub][base] if kind == "iou" else P[sub][base]
                d = src["diff_sw_minus"]
                sc = 100 if kind == "iou" else -1                          # IoU gain in points; placement: error removed
                mean, lo, hi = sc * np.array(d["mean"]), sc * np.array(d["lo"]), sc * np.array(d["hi"])
                if sc < 0:
                    lo, hi = hi, lo
                a.plot(k, mean, color=c, lw=1.2 if sub == "moving" else 0.8, ls=ls, marker=mk if sub == "moving" else None,
                       ms=2.2, markevery=3)
                if alpha:
                    a.fill_between(k, lo, hi, color=c, alpha=alpha, lw=0)
        a.set_xticks([1, 5, K]); a.set_xlim(1, K)
        a.set_xlabel("horizon $k$", fontsize=mf.FS_NOTE, labelpad=0)
        a.set_title(title, fontsize=mf.FS_NOTE, pad=2, fontweight="bold", color=mf.INK)
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        # moving-window averages over k, with paired 95% CIs (the values the text cites)
        lines = []
        for base, bl in (("direct", "Direct"), ("ar", "AR")):
            if kind == "iou":
                d = R["moving"][base]["avg"]["diff_sw_minus"]
                lines.append(f"vs {bl} {100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]")
            else:
                d = P["moving"][base]["avg"]["diff_sw_minus"]
                lines.append(f"vs {bl} {-d['mean']:+.1f} [{-d['hi']:+.1f}, {-d['lo']:+.1f}]")
        if kind == "place":
            lines.append("px: " + ", ".join(f"{n_} {P['moving'][a_]['avg']['mean']:.1f}"
                                            for a_, n_ in (("shiftwm", "S"), ("direct", "D"), ("ar", "A"))))
        ax.text(x0 + 0.5, 0.42, "\n".join(lines).replace("-", "\u2212"), fontsize=mf.FS_NOTE, color=mf.INK, ha="center", va="top",
                linespacing=1.05)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=M["direct"][1], lw=1.2, marker="D", ms=2.5), Line2D([], [], color=M["ar"][1], lw=1.2, marker="^", ms=2.5),
          Line2D([], [], color=mf.INK, lw=1.2), Line2D([], [], color=mf.INK, lw=0.8, ls=(0, (2, 1.5)))]
    fig.legend(hs, ["ShiftWM vs Direct", "ShiftWM vs AR", "moving windows (band: paired 95% CI)", "all windows"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=mf.FS_NOTE, frameon=False,
               handlelength=1.6, columnspacing=1.2)
    mf.qa(fig, "segmentation", W)
    fig.savefig(mf.FIG / "segmentation.pdf"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--only", nargs="*"); a = p.parse_args()
    jobs = {"ablations": fig_ablations, "hamlyn_tasks": fig_hamlyn_tasks, "planning": fig_planning,
            "segmentation": fig_segmentation}
    for k, fn in jobs.items():
        if not a.only or k in a.only:
            fn(); print("wrote", k)


if __name__ == "__main__":
    main()
