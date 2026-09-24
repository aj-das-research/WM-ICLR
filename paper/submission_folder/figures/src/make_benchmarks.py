"""Benchmark wheel (sunburst) for Sec. 4.1: domains (inner ring) and datasets/tasks (outer ring), one equal slice per benchmark, episode counts in the labels.

Episode counts are read from the data manifests where available (data/v2/features/<ds>/dinov2s/manifest.json,
IWS without the diagnostic test_recordings) and from the DINO-WM release tensors for PushT / Wall.
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

V2 = mf.ROOT / "data/v2/features"


def n_eps(ds, drop=("test_recordings",)):
    p = V2 / ds / "dinov2s/manifest.json"
    if not p.exists():
        return None
    return sum(1 for e in json.loads(p.read_text())["episodes"] if e["split"] not in drop)


def hamlyn_tasks():
    p = V2 / "openh_hamlyn/dinov2s/manifest.json"
    from collections import Counter
    c = Counter(e["task"] for e in json.loads(p.read_text())["episodes"])
    names = {"knot_tying": "knot tying", "needle_grasp_and_handover": "needle handover", "peg_transfer": "peg transfer",
             "suturing_1": "suturing 1", "suturing_2": "suturing 2", "tissue_lifting": "tissue lifting",
             "tissue_retraction": "tissue retraction"}
    return [(names[k], v) for k, v in sorted(c.items())]


def dinowm_eps(name):
    import torch
    base = mf.ROOT / "data/dinowm"
    try:
        if name == "wall":
            return int(torch.load(base / "wall_single/actions.pth", map_location="cpu").shape[0])
        return sum(int(torch.load(base / f"pusht_noise/{s}/rel_actions.pth", map_location="cpu").shape[0]) for s in ("train", "val"))
    except Exception:
        return None


def main():
    iws = sum(n_eps(f"iws_{t}") or 0 for t in ("pusht", "box", "rope"))
    domains = [
        ("Real\nrobots", "#2B6CB0", [("DROID", n_eps("droid")), ("RT-1", n_eps("fractal")), ("BridgeData V2", n_eps("bridge")),
                                     ("Language-Table", n_eps("language_table")), ("IWS (3 tasks)", iws)]),
        ("Surgical\n(dVRK)", "#C2185B", hamlyn_tasks()),
        ("Simulated\nplanning", "#D98E00", [("LeWM PushT", n_eps("plan_pusht")), ("TwoRoom", n_eps("plan_tworoom")),
                                             ("Reacher", n_eps("plan_reacher")), ("DINO-WM PushT", dinowm_eps("pusht")),
                                             ("DINO-WM Wall", dinowm_eps("wall"))]),
    ]
    domains = [(d, c, [(n, e) for n, e in items if e]) for d, c, items in domains]
    total = sum(e for _, _, items in domains for _, e in items)
    nb = sum(len(items) for _, _, items in domains)
    fig, ax = plt.subplots(figsize=(2.9, 1.95))
    ax.set_aspect("equal"); ax.axis("off")
    size = lambda e: 1.0                                   # equal slice per benchmark; counts are in the labels
    outer_vals, outer_cols, outer_labels = [], [], []
    inner_vals, inner_cols = [], []
    for d, c, items in domains:
        base = np.array(matplotlib.colors.to_rgb(c))
        inner_vals.append(sum(size(e) for _, e in items)); inner_cols.append(c)
        for i, (n, e) in enumerate(items):
            t = 0.35 + 0.45 * (i % 2)
            outer_vals.append(size(e)); outer_cols.append(tuple(base + (1 - base) * t)); outer_labels.append((n, e))
    start = 90
    kw = dict(startangle=start, counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=1.0))
    wo, _ = ax.pie(outer_vals, radius=1.0, colors=outer_cols, wedgeprops=dict(width=0.3, edgecolor="white", linewidth=0.8),
                   startangle=start, counterclock=False)
    wi, _ = ax.pie(inner_vals, radius=0.69, colors=inner_cols, wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.2),
                   startangle=start, counterclock=False)
    # inner labels (horizontal, on the ring)
    for w_, (d, c, _) in zip(wi, domains):
        a = np.deg2rad((w_.theta1 + w_.theta2) / 2)
        ax.text(0.52 * np.cos(a), 0.52 * np.sin(a), d, ha="center", va="center", fontsize=4.7, color="white", fontweight="bold",
                linespacing=0.9)
    # outer labels: horizontal text outside the ring with a short leader
    for w_, (n, e) in zip(wo, outer_labels):
        r = np.deg2rad((w_.theta1 + w_.theta2) / 2); c_, s_ = np.cos(r), np.sin(r)
        lab = f"{n}  {e / 1000:.1f}k" if e >= 1000 else f"{n}  {e}"
        rr = 1.11 + (0.1 if abs(c_) < 0.3 and (round(np.rad2deg(r)) // 20) % 2 else 0.0)   # stagger near the poles
        ax.plot([1.0 * c_, (rr - 0.03) * c_], [1.0 * s_, (rr - 0.03) * s_], color="#9AA3AE", lw=0.4)
        ax.text(rr * c_, rr * s_, lab, ha="left" if c_ >= 0 else "right", va="center", fontsize=4.9, color=mf.INK)
    ax.text(0, 0.06, f"{nb}", ha="center", va="center", fontsize=10, fontweight="bold", color=mf.INK)
    ax.text(0, -0.13, f"benchmarks\n{total / 1000:.0f}k episodes", ha="center", va="center", fontsize=5.0, color=mf.MUTED,
            linespacing=1.0)
    ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.38, 1.3)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(mf.FIG / "benchmarks.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(mf.FIG / "benchmarks_preview.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("wrote benchmarks", nb, total)


if __name__ == "__main__":
    main()
