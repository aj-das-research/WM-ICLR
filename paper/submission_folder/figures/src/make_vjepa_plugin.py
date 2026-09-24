"""V-JEPA 2-AC plug-in dynamics from results/v2/external/vjepa2ac_plugin (scripts/v2/vjepa_finetune.py), DROID.

(a) Validation MSE during fine-tuning: released predictor fine-tuned alone (B) vs. with the ShiftWM head (C); 2 seeds each
    (line = mean, band = range); step 0 = the released model.
(b) Mean gate of the head on validation windows during fine-tuning (initialised at sigmoid(-4) = 0.018).
(c) Test skill (fraction of persistence error removed) per horizon: zero-shot, B, C.
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

D = mf.ROOT / "results/v2/external/vjepa2ac_plugin"
GREEN, BLUE, GREY = mf.METHODS["shiftwm"][1], mf.METHODS["direct"][1], "#8C95A1"


def curves(arm):
    cs = [json.loads(p.read_text()) for p in sorted((D / arm).glob("s*/curve.json"))]
    step = np.array([r["step"] for r in cs[0]["curve"]])
    mse = np.array([[r["val_mse"] for r in c["curve"]] for c in cs])
    gate = np.array([[r.get("val_gate_mean", np.nan) for r in c["curve"]] for c in cs])
    return step, mse, gate


def skill(arm):
    out = []
    for p in sorted((D / arm).glob("s*/test_summary.json")) or [D / arm / "test_summary.json"]:
        s = json.loads(p.read_text())["per_horizon"]
        out.append(1 - np.array(s["model_mse"]) / np.array(s["persistence_mse"]))
    return 100 * np.array(out)


def main():
    if not (D / "finetune_shiftwm").exists():
        fig, ax = plt.subplots(figsize=(5.5, 1.7)); mf.pending(ax, "vjepa plug-in"); fig.savefig(mf.FIG / "vjepa_plugin.pdf"); return
    fig = plt.figure(figsize=(5.5, 1.75))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.99, top=0.86, bottom=0.22)
    ax = fig.add_subplot(gs[0, 0])
    for arm, lab, col in (("finetune", "fine-tuned", BLUE), ("finetune_shiftwm", "fine-tuned + ShiftWM", GREEN)):
        st, m, _ = curves(arm)
        ax.plot(st, m.mean(0), color=col, lw=1.4 if col == GREEN else 1.1, marker="o", ms=2, label=lab)
        ax.fill_between(st, m.min(0), m.max(0), color=col, alpha=0.2, lw=0)
    ax.set_xlabel("fine-tuning step", fontsize=5.8, labelpad=1); ax.set_ylabel("val. feature MSE", fontsize=5.8, labelpad=1)
    ax.tick_params(labelsize=5.6, length=2); ax.legend(fontsize=5.2, frameon=False, loc="upper right")
    ax.set_title("(a) validation error", fontsize=6.5, pad=2)
    ax = fig.add_subplot(gs[0, 1])
    st, _, g = curves("finetune_shiftwm")
    ax.plot(st, g.mean(0), color=GREEN, lw=1.4, marker="o", ms=2)
    ax.fill_between(st, np.nanmin(g, 0), np.nanmax(g, 0), color=GREEN, alpha=0.2, lw=0)
    ax.axhline(g[:, 0].mean(), color=GREY, lw=0.6, ls=(0, (2, 2)))
    ax.text(st[-1], g[:, 0].mean() + 0.03, f"init {g[:, 0].mean():.3f}", ha="right", fontsize=5.3, color=GREY)
    ax.set_ylim(0, 1); ax.set_xlabel("fine-tuning step", fontsize=5.8, labelpad=1); ax.set_ylabel("mean gate $g$", fontsize=5.8, labelpad=1)
    ax.tick_params(labelsize=5.6, length=2); ax.set_title("(b) the backbone adopts transport", fontsize=6.5, pad=2)
    ax = fig.add_subplot(gs[0, 2]); k = np.arange(1, 11)
    for arm, lab, col, ls in (("zeroshot", "zero-shot", GREY, (0, (3, 1.5))), ("finetune", "fine-tuned", BLUE, "-"),
                              ("finetune_shiftwm", "+ ShiftWM", GREEN, "-")):
        s = skill(arm)
        ax.plot(k, s.mean(0), color=col, ls=ls, lw=1.4 if col == GREEN else 1.1, marker="o", ms=2, label=lab)
        if len(s) > 1:
            ax.fill_between(k, s.min(0), s.max(0), color=col, alpha=0.2, lw=0)
    ax.axhline(0, color=mf.INK, lw=0.5)
    ax.set_xticks([1, 5, 10]); ax.set_xlabel("horizon $k$", fontsize=5.8, labelpad=1); ax.set_ylabel("skill (%)", fontsize=5.8, labelpad=1)
    ax.tick_params(labelsize=5.6, length=2); ax.legend(fontsize=5.2, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36))
    ax.set_title("(c) test skill per horizon", fontsize=6.5, pad=2)
    fig.savefig(mf.FIG / "vjepa_plugin.pdf"); fig.savefig(mf.FIG / "vjepa_plugin_preview.png", dpi=200)
    sb, sc = skill("finetune"), skill("finetune_shiftwm")
    print("skill per k B", sb.mean(0).round(1), "C", sc.mean(0).round(1))
    print("wrote vjepa_plugin")


if __name__ == "__main__":
    main()
