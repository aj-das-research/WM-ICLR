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
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "image.interpolation": "lanczos", "savefig.dpi": 300, "image.resample": True,
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


ROOTS = [ROOT / "results/v2s", ROOT / "results/v2"]
LEARNED = ("ar_tf", "ar", "direct", "shiftwm")


def root_for(dataset, encoder="dinov2s", split="test"):
    """Same rule as the tables: one recipe per dataset, the first root where every learned arm has finished."""
    for base in ROOTS:
        if all(list((base / dataset / encoder / a).glob(f"s*/eval_{split}.npz")) for a in LEARNED):
            return base
    return ROOTS[-1]


def load_eval(dataset, encoder, arm, split="test"):
    """Stack per-seed per-episode arrays: returns dict metric -> [seeds, episodes, K] or None."""
    base = root_for(dataset, encoder, split)
    runs = sorted((base / dataset / encoder / arm).glob("s*/eval_%s.npz" % split))
    if not runs and arm in ("persistence", "linear"):
        runs = sorted((ROOTS[-1] / dataset / encoder / arm).glob("s*/eval_%s.npz" % split))
    if arm in ("persistence", "linear"):
        runs = runs[:1]
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
    labels = []
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
        labels.append([label.split(" (")[0], k[-1], float(mean[-1])])
        drawn = True
    if not drawn:
        pending(ax, f"{dataset}: error vs horizon")
        return False
    # direct labels, spread vertically so they never overlap
    lo, hi = ax.get_ylim(); gap = 0.075 * (hi - lo)
    labels.sort(key=lambda t: t[2])
    orig = [t[2] for t in labels]
    clusters = [[0]]
    for i in range(1, len(labels)):
        (clusters[-1].append(i) if orig[i] - orig[clusters[-1][-1]] < gap else clusters.append([i]))
    for c in clusters:                     # spread each cluster evenly, centred on its lines' mean end value
        centre = float(np.mean([orig[i] for i in c]))
        for j, i in enumerate(c):
            labels[i][2] = centre + (j - (len(c) - 1) / 2) * gap
    for i in range(1, len(labels)):        # resolve any residual overlap between clusters
        labels[i][2] = max(labels[i][2], labels[i - 1][2] + gap)
    for name, x, y in labels:
        ax.annotate(name, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=6, color=INK, va="center")
    ax.set_xlabel("forecast step $k$")
    ax.set_ylabel("feature MSE")
    ax.set_title(title or dataset)
    ax.margins(x=0.18)
    return True


def fig_error_vs_horizon():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), constrained_layout=True)
    for ax, (ds, t) in zip(axes, [("droid", "DROID (test)"), ("openh_hamlyn", "Open-H surgical (test)"),
                                  ("bridge", "BridgeData V2 (test)")]):
        if plot_horizon(ax, ds, title=t):
            ax.set_xticks([1, 4, 7, 10])
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


def _chip(ax, x, y, w, color, edge="#243447", lw=0.6, grid=True):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, w, boxstyle="round,pad=0,rounding_size=0.012", fc=color, ec=edge, lw=lw))
    if grid:
        for t in np.linspace(x, x + w, 5)[1:-1]:
            ax.plot([t, t], [y, y + w], color="white", lw=0.3, alpha=0.7)
        for t in np.linspace(y, y + w, 5)[1:-1]:
            ax.plot([x, x + w], [t, t], color="white", lw=0.3, alpha=0.7)


def teaser_rollout_panel(ax):
    """(a) Recursive vs anchored forecasting; chips coloured by measured held-out MSE per horizon (DROID test)."""
    from matplotlib import cm, colors
    from matplotlib.patches import FancyArrowPatch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ar, sw, pe = (load_eval("droid", "dinov2s", a) for a in ("ar", "shiftwm", "persistence"))
    if ar is None or sw is None:
        pending(ax, "recursive vs anchored rollout\n(DROID test)"); return
    e_ar, e_sw = ar["mse"].mean((0, 1)), sw["mse"].mean((0, 1))
    ks = [1, 2, 4, 7, 10]
    norm = colors.Normalize(float(min(e_ar.min(), e_sw.min())) * 0.8, float(max(e_ar.max(), e_sw.max())) * 1.02)
    cmap = matplotlib.colormaps["Oranges"]
    w, x0, dx = 0.12, 0.2, 0.15
    lanes = [(0.63, "Recursive (AR)", e_ar, "feeds back its own forecast"),
             (0.22, "ShiftWM (ours)", e_sw, "reads observed features at every step")]
    for y, name, err, sub in lanes:
        _chip(ax, 0.02, y, w, "#9CC3E4")                       # observed Z0
        ax.text(0.02 + w / 2, y + w + 0.012, "$Z_0$", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.text(0.02, y + w + 0.15, name, fontsize=7.5, fontweight="bold", color=INK, va="bottom")
        ax.text(0.02, y + w + 0.085, sub, fontsize=6.5, color=MUTED, va="bottom")
        for i, k in enumerate(ks):
            x = x0 + i * dx
            _chip(ax, x, y, w, cmap(norm(err[k - 1])))
            ax.text(x + w / 2, y + w + 0.012, f"$\\hat Z_{{{k}}}$", ha="center", va="bottom", fontsize=6.5, color=INK)
        if name.startswith("Recursive"):
            xs = [0.02] + [x0 + i * dx for i in range(len(ks))]
            for xa, xb in zip(xs[:-1], xs[1:]):
                ax.add_patch(FancyArrowPatch((xa + w, y + w / 2), (xb, y + w / 2), arrowstyle="-|>",
                                             mutation_scale=5, lw=0.7, color="#D55E00"))
        else:
            for i, k in enumerate(ks):
                xb = x0 + i * dx + w / 2
                ax.add_patch(FancyArrowPatch((0.02 + w / 2, y), (xb, y), arrowstyle="-|>",
                                             connectionstyle=f"arc3,rad={0.22 + 0.02 * i}", mutation_scale=5,
                                             lw=0.6, color="#009E73", alpha=0.9))
        ax.text(x0 + (len(ks) - 1) * dx + w + 0.015, y + w / 2, f"{err[-1]:.3f}", fontsize=6.8, va="center",
                color=INK, fontweight="bold")
    # colour bar
    cax = ax.inset_axes([0.2, 0.0, 0.55, 0.03])
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.set_label("error per step (DROID test)", fontsize=6.3, color=MUTED, labelpad=1)


def teaser_mechanism_panel(fig, gs_cell, device):
    """(b) Real held-out frame: learned transport arrows, the true future and the gate."""
    sub = gs_cell.subgridspec(2, 2, height_ratios=[1.35, 1], hspace=0.12, wspace=0.06)
    ax_main = fig.add_subplot(sub[0, :]); ax_f = fig.add_subplot(sub[1, 0]); ax_g = fig.add_subplot(sub[1, 1])
    try:
        ep = pick_teaser_episode(); frames = droid_frames(ep, steps=(2, 12))
    except Exception:
        ep, frames = None, None
    ckpts = sorted((RES / "droid/dinov2s/shiftwm").glob("s*/best.pt"))
    if not frames or not ckpts:
        for ax in (ax_main, ax_f, ax_g):
            pending(ax, "transport field")
        return ax_main
    dx, dy, gate, gates = transport_arrows(ckpts[0], ep, device=device)
    img = frames[0]; h, w_ = img.shape[:2]; g = dx.shape[0]
    ys, xs = (np.arange(g) + 0.5) * h / g, (np.arange(g) + 0.5) * w_ / g
    X, Y = np.meshgrid(xs, ys)
    m = gate > np.quantile(gate, 0.72)
    sx, sy = X + dx * w_ / g, Y + dy * h / g
    gy, gx = np.unravel_index(np.argmax(gate), gate.shape)
    cx0 = int(np.clip(gx - 7, 0, g - 14)); cy0 = int(np.clip(gy - 5, 0, g - 10))
    x0, x1 = cx0 * w_ / g, (cx0 + 14) * w_ / g; y0, y1 = cy0 * h / g, (cy0 + 10) * h / g
    ax_main.imshow(img, aspect="auto")
    for gxl in np.linspace(0, w_, g + 1):
        ax_main.axvline(gxl, color="white", lw=0.35, alpha=0.45)
    for gyl in np.linspace(0, h, g + 1):
        ax_main.axhline(gyl, color="white", lw=0.35, alpha=0.45)
    mm = m & (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    mag = np.hypot(X - sx, Y - sy) * mm
    mm = mag >= np.sort(mag.ravel())[-14]                 # the 14 strongest predicted motions in view
    ax_main.quiver(sx[mm], sy[mm], (X - sx)[mm], (Y - sy)[mm], color="#FFE066", angles="xy", scale_units="xy",
                   scale=1, width=0.012, headwidth=3.6, headlength=3.8, edgecolor="#243447", linewidth=0.4)
    ax_main.set_xlim(x0, x1); ax_main.set_ylim(y1, y0)
    ax_main.text(0.02, 0.97, "predicted motion, $k{=}10$", transform=ax_main.transAxes, fontsize=6.5, color="white",
                 va="top", fontweight="bold", bbox=dict(fc="#243447", ec="none", alpha=0.6, pad=1.2))
    ax_f.imshow(frames[1], aspect="auto")
    ax_f.text(4, 8, "true, +3.3 s", fontsize=6.3, color="white", va="top", fontweight="bold",
              bbox=dict(fc="#243447", ec="none", alpha=0.55, pad=1))
    ax_g.imshow(img, aspect="auto")
    gm = np.kron(gates[10], np.ones((1, 1)))
    ax_g.imshow(gm, cmap="viridis", alpha=0.6, extent=(-0.5, w_ - 0.5, h - 0.5, -0.5), aspect="auto",
                vmin=0, vmax=float(gm.max()), interpolation="bicubic")
    ax_g.text(4, 8, "gate", fontsize=6.3, color="white", va="top", fontweight="bold",
              bbox=dict(fc="#243447", ec="none", alpha=0.55, pad=1))
    for ax in (ax_main, ax_f, ax_g):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    return ax_main


def _vjepa_skill(arm):
    base = RES / "external/vjepa2ac_plugin" / arm
    fs = sorted(base.glob("s*/test_summary.json"))
    v = [json.loads(f.read_text())["relative_gain_vs_persistence"]["mse"]["mean_over_horizons"] for f in fs]
    return 100 * float(np.mean(v)) if v else None


def _dinowm_errred(env):
    base = RES / "external/dinowm_plugin" / env
    e = {}
    for arm in ("dinowm", "dinowm_shiftwm"):
        f = base / arm / "openloop.json"
        if f.exists():
            e[arm] = json.loads(f.read_text()).get("teacher_forced", {}).get("z_visual_err_pred")
    return e if len(e) == 2 and None not in e.values() else None


def teaser_result_panel(ax):
    """(c) Skill (% of persistence error removed) without vs. with the ShiftWM head, on held-out data."""
    groups = []
    rel = relative_to_persistence("droid")
    if rel and "ar" in rel and "shiftwm" in rel:
        groups.append(("DROID\n(same backbone)", rel["ar"], rel["shiftwm"], "recursive AR"))
    ft, ours = _vjepa_skill("finetune"), _vjepa_skill("finetune_shiftwm")
    if ft is not None and ours is not None:
        groups.append(("V-JEPA 2-AC\n(1.3B, Meta)", ft, ours, "fine-tuned"))
    if not groups:
        pending(ax, "skill gains"); return
    x = np.arange(len(groups)); w = 0.36
    base_c, ours_c = "#B8BEC7", METHODS["shiftwm"][1]
    for i, (name, b, o, blab) in enumerate(groups):
        ax.bar(i - w / 2, b, w * 0.92, color=base_c)
        ax.bar(i + w / 2, o, w * 0.92, color=ours_c)
        ax.text(i - w / 2, b + 0.8, f"{b:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.text(i + w / 2, o + 0.8, f"{o:.0f}", ha="center", va="bottom", fontsize=6.8, color=INK, fontweight="bold")
        ax.annotate(f"+{o - b:.1f}", xy=(i + w / 2, o + 5.2), ha="center", fontsize=6.8, color=ours_c, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=6.3)
    ax.set_ylabel("skill: % of persistence error removed", fontsize=6.3, labelpad=1)
    ax.set_ylim(0, max(g[2] for g in groups) * 1.3); ax.tick_params(labelsize=6.3)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=base_c, label="without ShiftWM head"), Patch(color=ours_c, label="with ShiftWM head")],
              fontsize=5.8, loc="upper left", handlelength=1, borderaxespad=0.2)


def fig_teaser(device="cpu"):
    fig = plt.figure(figsize=(5.5, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.95], wspace=0.45, left=0.005, right=0.9,
                          top=0.87, bottom=0.14)
    ax_a = fig.add_subplot(gs[0, 0]); teaser_rollout_panel(ax_a)
    ax_b = teaser_mechanism_panel(fig, gs[0, 1], device)
    ax_c = fig.add_subplot(gs[0, 2]); teaser_result_panel(ax_c)
    for ax, t in ((ax_a, "(a) Move, don't regenerate"), (ax_b, "(b) Learned motion"),
                  (ax_c, "(c) Gains on held-out data")):
        x0 = ax.get_position().x0 if ax is not ax_c else ax.get_position().x0 - 0.06
        fig.text(max(x0, 0.005), 0.955, t, fontsize=8, fontweight="bold", color=INK)
    fig.savefig(FIG / "teaser.pdf")
    fig.savefig(FIG / "teaser_preview.png", dpi=200)
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
