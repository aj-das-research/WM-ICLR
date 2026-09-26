"""Figure 4 (round 2): ShiftWM as a plug-in head -- paired bars "without head / with head".

Numbers: scripts/v2/make_tables.py vjepa_rows() / dinowm_rows() (the generator of the plug-in table rows), which read
results/v2/external/vjepa2ac_plugin/*/[s*/]test_summary.json and
results/v2/external/dinowm_plugin/pusht/{dinowm,dinowm_shiftwm}/openloop.json (teacher-forced). Nothing is hand-typed.

Panels (one row, half width, 2.7 x 1.62 in):
  V-JEPA 2-AC, DROID test (base = fine-tuned, 2 seeds; + head = same fine-tuning with the head, 2 seeds)
    error: MSE (all patches), moving, static -- absolute values, bars from 0; label = relative change.
    skill: % of persistence error removed; zero-shot (released model) / fine-tuned / + head.
  DINO-WM, PushT validation (official code, 1 seed): teacher-forced latent error.
Colour: grey = the same model without the head, green = + ShiftWM head (paper palette).

Usage (repo root): PYTHONPATH=src .venv/bin/python reviews/main_figs_round2/make_fig_plugin.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper/submission_folder/figures/src"))
import make_figures as mf  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts/v2"))
import make_tables as T  # noqa: E402

GREEN, BASE, ZS = mf.METHODS["shiftwm"][1], "#9AA3AE", "#D5D9DF"
FS, FT = mf.FS_NOTE, mf.FS_TICK
MINUS = "−"


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def load():
    _, vj = T.vjepa_rows()
    _, dw = T.dinowm_rows()
    ft, hd, zs = vj["finetune"], vj["finetune_shiftwm"], vj["zeroshot"]
    b, o = dw["pusht"]["dinowm"], dw["pusht"]["dinowm_shiftwm"]
    L = {"vjepa_error": [dict(name=n, base=ft[k], head=hd[k], d=pct(hd[k], ft[k]))
                         for k, n in (("mse", "all"), ("moving", "moving"), ("static", "static"))],
         "vjepa_skill": dict(zeroshot=zs["skill"], base=ft["skill"], head=hd["skill"]),
         "vjepa_seeds": dict(base=ft["seeds"], head=hd["seeds"]),
         "dinowm_pusht": dict(base=b["err"], head=o["err"], d=pct(o["err"], b["err"]))}
    return L


def style(a, ylab):
    a.set_ylabel(ylab, fontsize=FS, labelpad=1.5)
    a.tick_params(axis="y", labelsize=FT, length=2, pad=1)
    a.tick_params(axis="x", length=0, pad=2, labelsize=FS)
    a.grid(axis="x", visible=False); a.grid(axis="y", color="#EEF0F3", lw=0.5)
    a.spines["left"].set_color(mf.MUTED); a.spines["bottom"].set_color(mf.MUTED)
    a.set_axisbelow(True)


def fmt(d):
    return f"{d:+.0f}%".replace("-", MINUS)


def main():
    L = load()
    W, H = 2.7, 1.62
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.20, 1.29                     # axes span (in)
    bw = 0.36
    # --- (1) V-JEPA error: 3 pairs
    x0, w0 = 0.30, 0.92
    a = fig.add_axes([x0 / W, bot / H, w0 / W, (top - bot) / H])
    E = L["vjepa_error"]
    for i, r in enumerate(E):
        a.bar(i - bw / 2, r["base"], bw, color=BASE, lw=0)
        a.bar(i + bw / 2, r["head"], bw, color=GREEN, lw=0)
        a.text(i, max(r["base"], r["head"]) + 0.015, fmt(r["d"]), ha="center", va="bottom", fontsize=FS,
               color=GREEN, fontweight="bold")
    a.set_xticks(range(len(E))); a.set_xticklabels([r["name"] for r in E])
    a.set_xlim(-0.6, len(E) - 0.4); a.set_ylim(0, 0.84); a.set_yticks([0, 0.4, 0.8]); a.set_yticklabels(["0", "0.4", "0.8"])
    style(a, r"feature error $\downarrow$")
    # --- (2) V-JEPA skill: base vs + head; zero-shot (released model) as a dashed reference
    x1, w1 = x0 + w0 + 0.31, 0.40
    b = fig.add_axes([x1 / W, bot / H, w1 / W, (top - bot) / H])
    S = L["vjepa_skill"]
    b.bar(-0.2, S["base"], 0.36, color=BASE, lw=0); b.bar(0.2, S["head"], 0.36, color=GREEN, lw=0)
    b.text(-0.2, S["base"] + 0.8, f"{S['base']:.1f}", ha="center", va="bottom", fontsize=FS, color=mf.INK)
    b.text(0.2, S["head"] + 0.8, f"{S['head']:.1f}", ha="center", va="bottom", fontsize=FS, color=GREEN,
           fontweight="bold")
    b.axhline(S["zeroshot"], color=mf.INK, lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    b.text(0.5, S["zeroshot"] + 0.8, f"0-shot {S['zeroshot']:.1f}", ha="right", va="bottom", fontsize=FS,
           color=mf.INK, zorder=4, bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))
    b.set_xticks([0]); b.set_xticklabels([r"skill (%) $\uparrow$"]); b.set_xlim(-0.5, 0.5); b.set_ylim(0, 44)
    b.set_yticks([0, 20, 40])
    style(b, "")
    # --- (3) DINO-WM PushT latent error
    x2, w2 = x1 + w1 + 0.30, 0.40
    c = fig.add_axes([x2 / W, bot / H, w2 / W, (top - bot) / H])
    P = L["dinowm_pusht"]
    c.bar(-0.2, P["base"], 0.36, color=BASE, lw=0); c.bar(0.2, P["head"], 0.36, color=GREEN, lw=0)
    c.text(0.0, P["base"] + 0.002, fmt(P["d"]), ha="center", va="bottom", fontsize=FS, color=GREEN, fontweight="bold")
    c.set_xticks([0]); c.set_xticklabels([r"latent err. $\downarrow$"]); c.set_xlim(-0.5, 0.5); c.set_ylim(0, 0.13)
    c.set_yticks([0, 0.05, 0.10]); c.set_yticklabels(["0", "0.05", "0.10"])
    style(c, "")
    # --- group headings and key (figure inches -> fraction)
    ty = top + 0.07
    fig.text((x0 - 0.02) / W, ty / H, "V-JEPA 2-AC, DROID test", fontsize=mf.FS_LABEL, fontweight="bold",
             color=mf.INK, va="bottom")
    fig.text((W - 0.01) / W, ty / H, "DINO-WM, PushT", fontsize=mf.FS_LABEL, fontweight="bold", color=mf.INK,
             va="bottom", ha="right")
    fig.legend(handles=[Patch(color=BASE, label="without head"), Patch(color=GREEN, label="+ ShiftWM head")],
               loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=FS, frameon=False,
               handlelength=0.9, handleheight=0.65, handletextpad=0.35, columnspacing=1.0, borderaxespad=0.1)
    ledger = {**L, "note": "base = fine-tuned V-JEPA 2-AC (2 seeds) / DINO-WM (1 seed); d = 100*(head/base-1)"}
    (HERE / "ledger_plugin.json").write_text(json.dumps(ledger, indent=1))
    mf.qa(fig, "fig_plugin", display_width=W)
    fig.savefig(HERE / "fig_plugin.pdf"); fig.savefig(HERE / "fig_plugin.png", dpi=400)
    print(json.dumps(ledger, indent=1)); print("size", W, H)


if __name__ == "__main__":
    main()
