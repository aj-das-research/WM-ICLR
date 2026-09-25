"""Evaluation wheel for Sec. 4: inner ring = evaluation axis, outer ring = benchmark / analysis, labelled with its metrics.
One equal slice per item. Datasets and tasks are those described in Appendix C (tab:tasks)."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

AXES = [
    ("Fore-\ncasting", "#0B7A55", [("DROID", "MSE, skill"), ("DROID cam 2", "zero-shot"), ("Language-Table", "MSE, skill"),
                                   ("Hamlyn, 7 tasks", "MSE, skill")]),
    ("Plug-in\nhead", "#D98E00", [("V-JEPA 2-AC", "MSE, skill"), ("DINO-WM PushT", "latent err."),
                                  ("DINO-WM Wall", "latent err.")]),
    ("Plan-\nning", "#6D4BA8", [("LeWM PushT", "success")]),
    ("Analysis", "#2B6CB0", [("arm placement", "px, IoU"), ("decoded pixels", "PSNR, LPIPS"), ("flow warp", "MSE"),
                             ("transport vs flow", "EPE"), ("causal knockout", "error change")]),
]


def main():
    fig, ax = plt.subplots(figsize=(2.75, 2.05))
    ax.set_aspect("equal"); ax.axis("off")
    outer_v, outer_c, labels, inner_v, inner_c = [], [], [], [], []
    for name, col, items in AXES:
        base = np.array(matplotlib.colors.to_rgb(col))
        inner_v.append(len(items)); inner_c.append(col)
        for i, it in enumerate(items):
            t = 0.4 + 0.35 * (i % 2)
            outer_v.append(1); outer_c.append(tuple(base + (1 - base) * t)); labels.append((it, col))
    wo, _ = ax.pie(outer_v, radius=1.0, colors=outer_c, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.28, edgecolor="white", linewidth=0.8))
    wi, _ = ax.pie(inner_v, radius=0.71, colors=inner_c, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.3))
    for w_, (name, col, _) in zip(wi, AXES):
        a = np.deg2rad((w_.theta1 + w_.theta2) / 2)
        ax.text(0.54 * np.cos(a), 0.54 * np.sin(a), name, ha="center", va="center", fontsize=4.0, color="white",
                fontweight="bold", linespacing=0.9)
    pole_count = {1: 0, -1: 0}                                                         # alternate radii near each pole
    for w_, ((n, m), col) in zip(wo, labels):
        r = np.deg2rad((w_.theta1 + w_.theta2) / 2); c_, s_ = np.cos(r), np.sin(r)
        stag = {"DROID": 0.2, "causal knockout": 0.2, "DINO-WM PushT": 0.28, "DINO-WM Wall": 0.3}.get(n, 0.0)
        rr = 1.08 + stag
        dy = 0.0
        ax.plot([1.0 * c_, (rr - 0.03) * c_], [1.0 * s_, (rr - 0.03) * s_], color="#9AA3AE", lw=0.4)
        ha = "right" if n == "DINO-WM Wall" else ("left" if c_ >= 0 else "right")
        ax.text(rr * c_, rr * s_ + dy + 0.06, n, ha=ha, va="center", fontsize=5.4, color=mf.INK)
        ax.text(rr * c_, rr * s_ + dy - 0.08, m, ha=ha, va="center", fontsize=4.7, color=col, style="italic")
    n_items = sum(len(it) for _, _, it in AXES)
    ax.text(0, 0.07, f"{n_items}", ha="center", va="center", fontsize=12, fontweight="bold", color=mf.INK)
    ax.text(0, -0.15, "evaluations\n4 axes", ha="center", va="center", fontsize=5.6, color=mf.MUTED, linespacing=1.0)
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.5, 1.48)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(mf.FIG / "benchmarks.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(mf.FIG / "benchmarks_preview.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("wrote evaluation wheel", n_items)


if __name__ == "__main__":
    main()
