"""Plug-in heads across backbones: V-JEPA 2-AC (DROID) and DINO-WM (PushT, Wall), each with and without the ShiftWM head.

Inputs
  results/v2/external/vjepa2ac_plugin/{finetune,finetune_shiftwm}/s*/{curve,test_summary}.json   (scripts/v2/vjepa_finetune.py)
  results/v2/external/dinowm_plugin/{pusht,wall}/{dinowm,dinowm_shiftwm}/
      metrics.jsonl  per-epoch validation metrics of the official DINO-WM trainer (val_z_visual_err_pred)
      openloop.json  teacher-forced and open-loop (rollout_H5) latent error on the validation split
      plan_*_seed*/final.json  planning success (official planner, reduced budget), where finished

Row 1: validation error during training, without (blue) and with (green) the head; grey dashed = persistence
       (copy the last observed frame). V-JEPA: 2 seeds (line = mean, band = range); DINO-WM: 1 seed.
       Last panel: mean gate of the V-JEPA head (the only backbone whose gate is logged).
Row 2: change in latent error from adding the head, per horizon (V-JEPA: test, k = 1..10; DINO-WM: validation,
       open-loop rollout steps 1..5 and teacher-forced "TF"). Green = lower error with the head, red = higher.
       Last panel: planning success where it exists; unfinished cells are drawn as empty outlines marked "queued".
Nothing is interpolated or estimated.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.ROOT / "results/v2/external/vjepa2ac_plugin"
DW = mf.ROOT / "results/v2/external/dinowm_plugin"
GREEN, BLUE, GREY = mf.METHODS["shiftwm"][1], mf.BACKBONE, "#8C95A1"   # BLUE = backbone alone (neutral slate, not Direct)
WORSE = "#D9776A"
FS_T, FS_L, FS_K = 7.0, mf.FS_NOTE, mf.FS_TICK
FS_S = mf.FS_NOTE                     # small annotations


# ------------------------------------------------------------------------------------------------ data
def curves(arm):
    cs = [json.loads(p.read_text()) for p in sorted((D / arm).glob("s*/curve.json"))]
    step = np.array([r["step"] for r in cs[0]["curve"]])
    mse = np.array([[r["val_mse"] for r in c["curve"]] for c in cs])
    gate = np.array([[r.get("val_gate_mean", np.nan) for r in c["curve"]] for c in cs])
    pers = cs[0]["curve"][0].get("val_pers_mse")
    return step, mse, gate, pers


def vjepa_test(arm):
    return np.array([json.loads(p.read_text())["per_horizon"]["model_mse"] for p in sorted((D / arm).glob("s*/test_summary.json"))])


def dinowm(env, arm):
    d = DW / env / arm
    if not (d / "metrics.jsonl").exists():
        return None
    rows = [json.loads(l.replace("NaN", "null")) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    ol = json.loads((d / "openloop.json").read_text()) if (d / "openloop.json").exists() else {}
    wi = json.loads((d / "wrapper_info.json").read_text()) if (d / "wrapper_info.json").exists() else {}
    succ = []
    for f in sorted(d.glob("plan_*_seed*/final.json")):
        s = json.loads(f.read_text().strip().splitlines()[-1]).get("final_eval/success_rate")
        if s is not None:
            succ.append(100 * s)
    r5 = ol.get("rollout_H5", {})
    return dict(epoch=np.array([int(r["epoch"]) for r in rows]), val=np.array([r["val_z_visual_err_pred"] for r in rows]),
                ol=np.array([r5[f"z_visual_err_step{i}"] for i in range(1, 6) if f"z_visual_err_step{i}" in r5]),
                tf=ol.get("teacher_forced", {}).get("z_visual_err_pred"),
                pers=ol.get("teacher_forced", {}).get("z_visual_err_persistence"),
                ol_epoch=ol.get("epoch"), target=wi.get("target_epochs"), succ=succ)


# ------------------------------------------------------------------------------------------------ drawing
def style(ax, xlabel=None, ylabel=None, title=None):
    ax.tick_params(labelsize=FS_K, length=2, pad=1.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_L, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_L, labelpad=1)
    if title:
        ax.set_title(title, fontsize=FS_T, pad=5)


def pers_line(ax, y, x_text, ha="right"):
    ax.axhline(y, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(x_text, y, "persistence", fontsize=FS_S, color=GREY, ha=ha, va="bottom")


def delta_bars(ax, labels, d, spread=None):
    x = np.arange(len(d))
    cols = [GREEN if v <= 0 else WORSE for v in d]
    ax.bar(x, d, width=0.68, color=cols, edgecolor="none", zorder=3)
    if spread is not None:                    # min-max over seed pairings
        ax.vlines(x, [np.min(p) for p in spread], [np.max(p) for p in spread], color=mf.INK, lw=0.6, zorder=4)
    ax.axhline(0, color=mf.INK, lw=0.6, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)
    lim = max(abs(v) for v in d) * 1.35
    lo = min(0, min(d) * 1.35) if min(d) < 0 else -0.12 * lim
    hi = max(0, max(d) * 1.35) if max(d) > 0 else 0.12 * lim
    if spread is not None:
        lo = min(lo, min(np.min(p) for p in spread) * 1.3)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "\u2212") if round(v) else "0"))
    for xi, v in zip(x, d):                   # value on the extreme bars only (first and last horizon)
        if xi in (0, len(d) - 1):
            ext = v if spread is None else (np.max(spread[xi]) if v > 0 else np.min(spread[xi]))
            ax.text(xi, ext + (0.04 * (hi - lo) if v > 0 else -0.04 * (hi - lo)), f"{v:+.0f}".replace("-", "\u2212"), ha="center",
                    va="bottom" if v > 0 else "top", fontsize=FS_S, color=mf.INK)


def main():
    if not (D / "finetune_shiftwm").exists():
        fig, ax = plt.subplots(figsize=(5.5, 1.7)); mf.pending(ax, "plug-in heads"); fig.savefig(mf.FIG / "vjepa_plugin.pdf"); return
    fig = plt.figure(figsize=(5.5, 2.95))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.45, hspace=0.68, left=0.08, right=0.975,
                          top=0.8, bottom=0.1)
    heads = [("V-JEPA 2-AC $\\cdot$ DROID", "ViT-g, 2 seeds, test"), ("DINO-WM $\\cdot$ PushT", None), ("DINO-WM $\\cdot$ Wall", None)]
    dw = {env: {arm: dinowm(env, arm) for arm in ("dinowm", "dinowm_shiftwm")} for env in ("pusht", "wall")}
    for j, env in enumerate(("pusht", "wall"), start=1):
        b = dw[env]["dinowm"]
        heads[j] = (heads[j][0], f"ViT-S, 1 seed, {b['target']} epochs, val.")

    # ---- row 1: validation curves
    ax = fig.add_subplot(gs[0, 0])
    for arm, lab, col in (("finetune", "backbone", BLUE), ("finetune_shiftwm", "+ ShiftWM head", GREEN)):
        st, m, _, pers = curves(arm)
        ax.plot(st / 1000, m.mean(0), color=col, lw=1.4 if col == GREEN else 1.1, marker="o", ms=1.8, label=lab, zorder=3)
        ax.fill_between(st / 1000, m.min(0), m.max(0), color=col, alpha=0.2, lw=0)
    pers_line(ax, pers, st[-1] / 1000)
    ax.set_ylim(0.3, pers * 1.08)
    style(ax, "fine-tuning step ($\\times$1000)", "val. latent error", "(a) Validation error")
    for j, env in enumerate(("pusht", "wall"), start=1):
        ax = fig.add_subplot(gs[0, j])
        for arm, col in (("dinowm", BLUE), ("dinowm_shiftwm", GREEN)):
            r = dw[env][arm]
            ax.plot(r["epoch"], r["val"], color=col, lw=1.4 if col == GREEN else 1.1, marker="o", ms=1.8, zorder=3)
        p = dw[env]["dinowm"]["pers"]
        e = dw[env]["dinowm"]["epoch"]
        if env == "wall":
            ax.set_yscale("log"); ax.yaxis.set_major_locator(LogLocator(subs=(1, 3)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if env == "wall":
            pers_line(ax, p, e[-1])
        else:
            ax.text(0.97, 0.97, f"persistence {p:.2f}\n(off scale)", transform=ax.transAxes, ha="right", va="top",
                    fontsize=FS_S, color=GREY)
        ax.set_xticks(e if len(e) <= 5 else [1, 5, 10, 15])
        style(ax, "epoch", None, f"({'bc'[j - 1]}) Validation error")
    # gate
    ax = fig.add_subplot(gs[0, 3])
    st, _, g, _ = curves("finetune_shiftwm")
    ax.plot(st / 1000, g.mean(0), color=GREEN, lw=1.4, marker="o", ms=1.8)
    ax.fill_between(st / 1000, np.nanmin(g, 0), np.nanmax(g, 0), color=GREEN, alpha=0.2, lw=0)
    ax.axhline(g[:, 0].mean(), color=GREY, lw=0.6, ls=(0, (2, 2)))
    ax.text(st[-1] / 1000, g[:, 0].mean() + 0.04, f"init {g[:, 0].mean():.3f}", ha="right", fontsize=FS_S, color=GREY)
    ax.set_ylim(0, 1)
    style(ax, "fine-tuning step ($\\times$1000)", "mean gate $g$", "(d) V-JEPA head gate")

    # ---- row 2: error change per horizon
    ax = fig.add_subplot(gs[1, 0])
    B, C = vjepa_test("finetune"), vjepa_test("finetune_shiftwm")
    d = 100 * (C.mean(0) / B.mean(0) - 1)
    pairs = np.array([100 * (c / b - 1) for c in C for b in B]).T          # every seed pairing, for spread
    delta_bars(ax, [str(k) if k in (1, 5, 10) else "" for k in range(1, 11)], d, spread=pairs)
    style(ax, "horizon $k$ (test)", "$\\Delta$ error with head", "(e) Error change per $k$")
    for j, env in enumerate(("pusht", "wall"), start=1):
        ax = fig.add_subplot(gs[1, j])
        b, s = dw[env]["dinowm"], dw[env]["dinowm_shiftwm"]
        n = min(len(b["ol"]), len(s["ol"]))
        dd = list(100 * (s["ol"][:n] / b["ol"][:n] - 1)) + [100 * (s["tf"] / b["tf"] - 1)]
        delta_bars(ax, [str(i) for i in range(1, n + 1)] + ["TF"], dd)
        ax.axvline(n - 0.5, color=GRID_C, lw=0.6)
        style(ax, "open-loop step (val.)", None, f"({'fg'[j - 1]}) Error change per step")
    # planning
    ax = fig.add_subplot(gs[1, 3])
    cells = [("PushT", dw["pusht"]["dinowm"]["succ"], dw["pusht"]["dinowm_shiftwm"]["succ"]),
             ("Wall", dw["wall"]["dinowm"]["succ"], dw["wall"]["dinowm_shiftwm"]["succ"])]
    for i, (env, sb, ss) in enumerate(cells):
        for off, v, col in ((-0.2, sb, BLUE), (0.2, ss, GREEN)):
            if v:
                ax.bar(i + off, np.mean(v), width=0.36, color=col, zorder=3)
                ax.text(i + off, np.mean(v) + 2, f"{np.mean(v):.0f}", ha="center", va="bottom", fontsize=FS_S, color=mf.INK)
            else:                                   # run did not finish within the budget (see App. text): no value
                ax.plot([i + off - 0.17, i + off + 0.17], [0.6, 0.6], color=col, lw=1.2, zorder=3)
                ax.text(i + off, 4, "not finished", rotation=90, ha="center", va="bottom", fontsize=FS_S, color=GREY)
    ax.set_xticks([0, 1]); ax.set_xticklabels([c[0] for c in cells]); ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    style(ax, "DINO-WM task (MPC-CEM)", "success (%)", "(h) Planning")

    # column headers (backbone / task) above row 1
    for j, (h, sub) in enumerate(heads):
        pos = fig.axes[j].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.957, h, ha="center", va="bottom", fontsize=mf.FS_TITLE, fontweight="bold", color=mf.INK)
        fig.text((pos.x0 + pos.x1) / 2, 0.922, sub, ha="center", va="bottom", fontsize=FS_S, color=mf.MUTED)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, lw=1.1, marker="o", ms=2, label="backbone alone"),
         Line2D([], [], color=GREEN, lw=1.4, marker="o", ms=2, label="+ ShiftWM head"),
         Line2D([], [], color=GREY, lw=0.7, ls=(0, (3, 2)), label="persistence")]
    pos = fig.axes[3].get_position()
    fig.legend(handles=h, loc="lower right", bbox_to_anchor=(0.985, 0.875), ncol=1, fontsize=FS_S,
               frameon=False, handlelength=1.6, borderaxespad=0, labelspacing=0.25)
    mf.qa(fig, "vjepa_plugin", 5.5)
    fig.savefig(mf.FIG / "vjepa_plugin.pdf"); fig.savefig(mf.FIG / "vjepa_plugin_preview.png", dpi=200); plt.close(fig)
    print("V-JEPA dErr per k", d.round(1), " skill gain per k",
          (100 * (B.mean(0) - C.mean(0)) / np.array(json.loads(sorted((D / 'finetune').glob('s*/test_summary.json'))[0].read_text())["per_horizon"]["persistence_mse"])).round(1))
    for env in ("pusht", "wall"):
        b, s = dw[env]["dinowm"], dw[env]["dinowm_shiftwm"]
        print(env, "epochs", b["ol_epoch"], s["ol_epoch"], "target", b["target"], s["target"], "TF", b["tf"], s["tf"],
              f"{100 * (s['tf'] / b['tf'] - 1):+.1f}%", "OL", (100 * (s["ol"] / b["ol"] - 1)).round(1), "succ", b["succ"], s["succ"])
    print("wrote vjepa_plugin")


GRID_C = "#C9CED6"

if __name__ == "__main__":
    main()
