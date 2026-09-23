"""Generate all data-driven paper figures from completed runs in results/v2.

Every panel is computed from logged evaluation files or real dataset frames. Panels whose
inputs do not exist yet are drawn as explicit "pending" boxes -- nothing is invented.

Usage (repo root):  PYTHONPATH=src python paper/submission_folder/figures/src/make_figures.py [--device cpu]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FIG = ROOT / "paper/submission_folder/figures"
RES = ROOT / "results/v2"

# Palette validated with the dataviz validator (light mode, all checks pass; AR-TF/Direct protan
# all-pairs dE 6.4 -> always paired with distinct markers/linestyles + direct labels).
METHODS = {  # key: (label, colour, linestyle, marker)
    "persistence": ("Persistence", "#6B7280", (0, (3, 2)), None),
    "ar_tf": ("AR-TF (DINO-WM-style)", "#AA4499", "-", "s"),
    "ar": ("AR", "#D55E00", "-", "^"),
    "direct": ("Direct", "#0072B2", "-", "D"),
    "shiftwm": ("ShiftWM (ours)", "#009E73", "-", "o"),
}
INK, MUTED, GRID = "#243447", "#8A8F98", "#E6E8EB"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4, "pdf.fonttype": 42,
})


def pending(ax, text):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0.01,rounding_size=0.03",
                                transform=ax.transAxes, fc="white", ec=MUTED, lw=0.8, ls=(0, (3, 2))))
    ax.text(0.5, 0.5, "pending\n" + text, ha="center", va="center", color=MUTED, fontsize=6.5,
            transform=ax.transAxes, wrap=True)


def load_eval(dataset, encoder, arm, split="test"):
    """Stack per-seed per-episode arrays: returns dict metric -> [seeds, episodes, K] or None."""
    runs = sorted((RES / dataset / encoder / arm).glob("s*/eval_%s.npz" % split))
    if not runs:
        return None
    arrs = [np.load(r, allow_pickle=True) for r in runs]
    out = {m: np.stack([a[m] for a in arrs]) for m in ("mse", "cos", "rank_ok", "sens") if m in arrs[0].files}
    out["seeds"] = len(runs)
    out["episodes"] = arrs[0]["episodes"]
    return out


# ------------------------------------------------------------------------------------------ plots
def plot_horizon(ax, dataset, encoder="dinov2s", title=None):
    drawn = False
    for arm, (label, color, ls, marker) in METHODS.items():
        ev = load_eval(dataset, encoder, arm)
        if ev is None:
            continue
        per_ep = ev["mse"].mean(0)                         # seeds -> [E, K]
        mean = per_ep.mean(0)
        rng = np.random.default_rng(0)
        boots = np.stack([per_ep[rng.integers(0, len(per_ep), len(per_ep))].mean(0) for _ in range(1000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        k = np.arange(1, len(mean) + 1)
        ax.plot(k, mean, color=color, ls=ls, marker=marker, markevery=3, label=label,
                lw=2.0 if arm == "shiftwm" else 1.4, zorder=3 if arm == "shiftwm" else 2)
        ax.fill_between(k, lo, hi, color=color, alpha=0.15, lw=0)
        ax.annotate(label.split(" (")[0], (k[-1], mean[-1]), xytext=(3, 0), textcoords="offset points",
                    fontsize=6, color=INK, va="center")
        drawn = True
    if not drawn:
        pending(ax, f"{dataset}: error vs horizon")
        return False
    ax.set_xlabel("forecast step $k$")
    ax.set_ylabel("feature MSE")
    ax.set_title(title or dataset)
    ax.margins(x=0.18)
    return True


def fig_error_vs_horizon():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), constrained_layout=True)
    for ax, (ds, t) in zip(axes, [("droid", "DROID (test, cam 1)"), ("openh_hamlyn", "Open-H Hamlyn (test)"),
                                  ("iws", "IWS (official handles)")]):
        plot_horizon(ax, ds, title=t)
    fig.savefig(FIG / "error_vs_horizon.pdf")
    plt.close(fig)


def relative_to_persistence(dataset, encoder="dinov2s"):
    base = load_eval(dataset, encoder, "persistence")
    if base is None:
        return None
    b = base["mse"].mean((0, 2))                          # per episode
    out = {}
    for arm in ("ar_tf", "ar", "direct", "shiftwm"):
        ev = load_eval(dataset, encoder, arm)
        if ev is not None:
            out[arm] = 100 * (1 - ev["mse"].mean((0, 2)).mean() / b.mean())
    return out


# ------------------------------------------------------------------------------------------ teaser
def droid_frames(episode_id, camera="exterior_image_1_left", steps=(0, 10)):
    root = ROOT / "data/real_video/droid_selected/processed"
    man = json.loads((root / "manifest.json").read_text())
    row = next(r for r in man["episodes"] if r["episode_id"] == episode_id)
    with np.load(root / row["cameras"][camera]["file"]) as z:
        return [z["images"][s] for s in steps]


def pick_teaser_episode():
    """Deterministic rule: among DROID test episodes, the one with the largest true feature change
    between the 3rd frame and 10 steps later (visible motion for illustration)."""
    root = ROOT / "data/v2/features/droid/dinov2s"
    man = json.loads((root / "manifest.json").read_text())
    best, best_id = -1, None
    for r in man["episodes"]:
        if r["split"] != "test" or r["T"] < 14:
            continue
        with np.load(root / r["file"]) as z:
            f = z["features"]
            d = float(np.mean((f[12].astype(np.float32) - f[2].astype(np.float32)) ** 2))
        if d > best:
            best, best_id = d, r["id"]
    return best_id


def transport_arrows(ckpt, episode_id, k=10, device="cpu"):
    """Expected source displacement (in patches) per query patch from a trained ShiftWM checkpoint."""
    import torch
    from shiftwm.v2.models import V2WorldModel
    root = ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    with np.load(root / f"episodes/{episode_id}.npz") as z:
        f, a = z["features"].astype(np.float32), z["actions"]
    st = torch.load(ckpt, map_location=device)
    model = V2WorldModel(st["config"]).to(device).eval()
    model.load_state_dict(st["model"])
    fm, fs = np.array(stats["feature_mean"]), np.array(stats["feature_std"])
    am, ast = np.array(stats["action_mean"]), np.array(stats["action_std"])
    x = torch.tensor(((f[:3] - fm) / fs).reshape(1, 3, -1, f.shape[-1]), dtype=torch.float32)
    an = (a - am) / ast
    past = torch.tensor(an[None, :2], dtype=torch.float32)
    fut = torch.tensor(an[None, 2:2 + k], dtype=torch.float32)
    with torch.no_grad():
        _, det = model(x, past, fut, return_details=True)
    w = det["weights"][0, k - 1].numpy()                 # [N, S*w*w]
    gate = det["gate"][0, k - 1, :, 0].numpy()
    gates = {kk: det["gate"][0, kk - 1, :, 0].numpy().reshape(model.config.grid, -1) for kk in (1, 5, k)}
    cfg = model.config
    g, win, s = cfg.grid, cfg.window, cfg.sources
    r = win // 2
    oy, ox = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    oy, ox = np.tile(oy.ravel(), s), np.tile(ox.ravel(), s)
    dy, dx = w @ oy, w @ ox                                # expected source offset
    return dx.reshape(g, g), dy.reshape(g, g), gate.reshape(g, g), gates


def fig_teaser(device="cpu"):
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 0.1, 2.1, 0.32, 1.3, 1.3], wspace=0.1, hspace=0.28,
                          left=0.01, right=0.99, top=0.86, bottom=0.08)
    # (a) the problem: real frames persist
    ep = None
    try:
        ep = pick_teaser_episode()
        frames = droid_frames(ep, steps=(2, 12))
    except Exception:
        frames = None
    ax_a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    if frames:
        for ax, im, t in zip(ax_a, frames, ("observed $t$", "$t$ + 3.3 s")):
            ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=6.5, fontweight="normal")
            ax.grid(False)
    else:
        for ax in ax_a:
            pending(ax, "DROID frames")
    surg = None
    try:
        sroot = ROOT / "data/v2/frames/openh_hamlyn"
        sm = json.loads((sroot / "manifest.json").read_text())
        srow = next(r for r in sm["episodes"] if r["split"] == "test" and r["task"] == "suturing_1" and r["T"] >= 14)
        with np.load(sroot / srow["file"]) as z:
            surg = [z["images"][2], z["images"][12]]
    except Exception:
        pass
    ax_s = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, im in zip(ax_s, surg or [None, None]):
        if im is None:
            pending(ax, "surgical frames")
        else:
            ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.text(ax_a[0].get_position().x0, 0.955, "(a) Scenes move, rarely change",
             fontsize=7.5, fontweight="bold", color=INK)
    # (b) mechanism: learned transport field on the real frame
    ax_b = fig.add_subplot(gs[0, 3])
    sub = gs[1, 3].subgridspec(1, 3, wspace=0.05)
    ax_g = [fig.add_subplot(sub[0, i]) for i in range(3)]
    ckpts = sorted((RES / "droid/dinov2s/shiftwm").glob("s*/best.pt"))
    if frames and ckpts:
        dx, dy, gate, gates = transport_arrows(ckpts[0], ep, device=device)
        vmax = max(float(v.max()) for v in gates.values())
        for axg, (kk, gm) in zip(ax_g, gates.items()):
            axg.imshow(gm, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            axg.set_xticks([]); axg.set_yticks([]); axg.grid(False)
            axg.set_title(f"gate $k={kk}$", fontsize=6, fontweight="normal", pad=2)
        img = frames[0]
        h, w_ = img.shape[:2]
        ax_b.imshow(img, alpha=0.9, aspect="auto")
        g = dx.shape[0]
        ys, xs = (np.arange(g) + 0.5) * h / g, (np.arange(g) + 0.5) * w_ / g
        X, Y = np.meshgrid(xs, ys)
        # arrow from the expected source location to the query patch (content motion), true scale
        mask = gate > np.quantile(gate, 0.7)
        sx, sy = X + dx * w_ / g, Y + dy * h / g
        ax_b.quiver(sx[mask], sy[mask], (X - sx)[mask], (Y - sy)[mask], gate[mask], cmap="viridis",
                    angles="xy", scale_units="xy", scale=1, width=0.007, headwidth=3.2, headlength=3.5)
        ax_b.set_xlim(-0.5, w_ - 0.5); ax_b.set_ylim(h - 0.5, -0.5)
        for sp in ax_b.spines.values():
            sp.set_visible(False)
        for gx in np.linspace(0, w_, g + 1):
            ax_b.axvline(gx, color="white", lw=0.25, alpha=0.35)
        for gy in np.linspace(0, h, g + 1):
            ax_b.axhline(gy, color="white", lw=0.25, alpha=0.35)
        ax_b.set_xticks([]); ax_b.set_yticks([]); ax_b.grid(False)
        ax_b.set_title("transport at $k{=}10$: source $\\rightarrow$ patch", fontsize=6.5, fontweight="normal")
    else:
        pending(ax_b, "learned transport field\non a DROID test frame")
        for axg in ax_g:
            pending(axg, "gate")
    fig.text(ax_b.get_position().x0, 0.955, "(b) ShiftWM moves what it saw", fontsize=7.5, fontweight="bold", color=INK)
    # (c) evidence
    ax_c1 = fig.add_subplot(gs[:, 5])
    rels = {ds: relative_to_persistence(ds) for ds in ("droid", "openh_hamlyn", "iws")}
    if any(rels.values()):
        names = [("droid", "DROID"), ("openh_hamlyn", "Surgical"), ("iws", "IWS")]
        arms = [a for a in ("ar_tf", "ar", "direct", "shiftwm") if any(rels[d] and a in rels[d] for d, _ in names)]
        width = 0.8 / max(len(arms), 1)
        for i, arm in enumerate(arms):
            vals = [rels[d].get(arm, np.nan) if rels[d] else np.nan for d, _ in names]
            x = np.arange(3) + (i - (len(arms) - 1) / 2) * width
            ax_c1.bar(x, vals, width * 0.88, color=METHODS[arm][1], label=METHODS[arm][0].split(" (")[0])
            if arm == "shiftwm":
                for xi, v in zip(x, vals):
                    if np.isfinite(v):
                        ax_c1.text(xi, v + 0.6, f"{v:.0f}%", ha="center", va="bottom", fontsize=6, color=INK)
        for j, (d, _) in enumerate(names):
            if not rels[d]:
                ax_c1.text(j, 1.0, "pending", ha="center", va="bottom", fontsize=6, color=MUTED, rotation=90)
        ax_c1.axhline(0, color=MUTED, lw=0.8)
        ax_c1.set_xticks(range(3)); ax_c1.set_xticklabels([n for _, n in names])
        ax_c1.set_ylabel("error reduction vs. persistence (%)", fontsize=6.5)
        ax_c1.legend(fontsize=5.8, loc="upper right", handlelength=1, borderaxespad=0.2)
        ax_c1.grid(axis="x", visible=False)
    else:
        pending(ax_c1, "held-out gain\nvs. persistence\n(DROID, surgical, IWS)")
    ax_c2 = fig.add_subplot(gs[:, 6])
    plan = RES / "planning_summary.json"
    if plan.exists():
        pass  # filled once planning runs complete (see fig_planning)
    pending(ax_c2, "planning success\n& action\nsensitivity")
    fig.text(ax_c1.get_position().x0, 0.955, "(c) Held-out evidence", fontsize=7.5, fontweight="bold", color=INK)
    fig.savefig(FIG / "teaser.pdf")
    fig.savefig(FIG / "teaser_preview.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    jobs = {"teaser": lambda: fig_teaser(a.device), "horizon": fig_error_vs_horizon}
    for name, fn in jobs.items():
        if not a.only or name in a.only:
            fn(); print("wrote", name)


if __name__ == "__main__":
    main()
