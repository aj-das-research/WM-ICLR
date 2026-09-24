"""'Where and why ShiftWM wins' figure from results/v2/analysis/interpret (scripts/v2/interpret.py), held-out DROID, k=10.

(a) Two windows (rule in interpret.py: largest advantage over Direct among high-motion windows). Per window: the true
    future frame with the region of largest error boxed, then on that crop the per-patch error of Direct and of
    ShiftWM (same colour scale) and the win map (green: ShiftWM lower error).
(b) Causal knockout of transport: error before -> after setting the gate to 0, per region.
(c) Gain over Direct per decile of true motion.  (d) Action steering: change of the transport field, moving vs static.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/interpret"
GREEN, RED, GREY, BLUE = mf.METHODS["shiftwm"][1], "#C0392B", "#B8BEC7", mf.METHODS["direct"][1]
ROI = "#FFD23F"
ERR = LinearSegmentedColormap.from_list("err", [(1, 0.25, 0.1, 0.0), (1, 0.25, 0.1, 0.55), (0.75, 0.0, 0.1, 0.92)])


def frame_axes(ax, color="#C9CED6", lw=0.6):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(lw)


def corner(ax, text, fc):
    ax.text(0.96, 0.06, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="white",
            fontweight="bold", bbox=dict(fc=fc, ec="none", alpha=0.92, pad=0.9))


def win_rgba(win, lim):
    a = np.clip(np.abs(win) / lim, 0, 1) * 0.85
    col = np.where(win[..., None] > 0, np.array(to_rgb(GREEN)), np.array(to_rgb(BLUE)))
    return np.concatenate([col, a[..., None]], -1)


def gray(fr):
    return (fr.mean(-1, keepdims=True).repeat(3, -1) * 0.55 + fr * 0.45).astype(np.uint8)


def transport_for(ep, st, k=10):
    """Expected source displacement (patches) and gate at step k for the window starting at frame st (CPU)."""
    import torch
    from shiftwm.v2.models import V2WorldModel
    root = mf.ROOT / "data/v2/features/droid/dinov2s"; stats = json.loads((root / "stats.json").read_text())
    with np.load(root / f"episodes/{ep}.npz") as z:
        f, a = z["features"].astype(np.float32), z["actions"].astype(np.float32)
    fm, fs = np.array(stats["feature_mean"], np.float32), np.array(stats["feature_std"], np.float32)
    am, ast = np.array(stats["action_mean"], np.float32), np.array(stats["action_std"], np.float32)
    fz = (f.reshape(len(f), -1, f.shape[-1]) - fm) / fs; an = (a - am) / ast
    ck = sorted((mf.ROOT / "results/v2s/droid/dinov2s/shiftwm").glob("s*/best.pt"))[0]
    sd = torch.load(ck, map_location="cpu"); m = V2WorldModel(sd["config"]).eval(); m.load_state_dict(sd["model"])
    T = lambda x: torch.tensor(x[None], dtype=torch.float32)
    with torch.no_grad():
        _, det = m(T(fz[st:st + 3]), T(an[st:st + 2]), T(an[st + 2:st + 2 + k]), return_details=True)
    cfg = m.config; r = cfg.window // 2
    oy, ox = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    oy, ox = np.tile(oy.ravel(), cfg.sources), np.tile(ox.ravel(), cfg.sources)
    w = det["weights"][0, k - 1].numpy()
    return (w @ ox).reshape(16, 16), (w @ oy).reshape(16, 16), det["gate"][0, k - 1, :, 0].numpy().reshape(16, 16)


def main():
    if not (D / "summary.json").exists():
        fig, ax = plt.subplots(figsize=(5.5, 2.6)); mf.pending(ax, "interpretability"); fig.savefig(mf.FIG / "interpret.pdf"); return
    S = json.loads((D / "summary.json").read_text()); Z = np.load(D / "droid.npz", allow_pickle=True)
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(5.5, 2.45))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.22, left=0.015, right=0.985, top=0.86, bottom=0.12)
    left = outer[0, 0].subgridspec(2, 4, wspace=0.04, hspace=0.05)
    E_d, E_s = Z["err_di"][:2], Z["err_sw"][:2]
    vmax = float(np.quantile(np.concatenate([E_d.ravel(), E_s.ravel()]), 0.97))
    heads = ["observed $t$ + transport", "true $t{+}10$", "Direct error", "ShiftWM error"]
    for r in range(2):
        ep, st = str(Z["episode"][r]), int(Z["start"][r])
        obs, fut = mf.droid_frames(ep, steps=(st + 2, st + 2 + 10)); h, w = fut.shape[:2]
        ed, es = E_d[r].reshape(16, 16), E_s[r].reshape(16, 16)
        gy, gx = np.unravel_index(np.argmax(ed + es), ed.shape)
        side = min(h, 7 * w / 16) * 1.0                                        # square crop in pixels
        px, py = (gx + 0.5) * w / 16, (gy + 0.5) * h / 16
        x0 = float(np.clip(px - side / 2, 0, w - side)); y0 = float(np.clip(py - side / 2, 0, h - side))
        box = (x0 - 0.5, y0 - 0.5, side, side)
        ext = (-0.5, w - 0.5, h - 0.5, -0.5)
        cx0, cx1 = int(x0 // (w / 16)), int(np.ceil((x0 + side) / (w / 16))); cy0, cy1 = int(y0 // (h / 16)), int(np.ceil((y0 + side) / (h / 16)))
        yy, xx = slice(cy0, cy1), slice(cx0, cx1)

        def crop(a):
            a.set_xlim(box[0], box[0] + box[2]); a.set_ylim(box[1] + box[3], box[1])
        # 1: observed crop + transport arrows (source -> target), moving high-gate patches
        ax = fig.add_subplot(left[r, 0]); ax.imshow(obs, aspect="equal", interpolation="lanczos"); crop(ax); frame_axes(ax, ROI, 1.1)
        try:
            dx, dy, gate = transport_for(ep, st)
            mag = np.hypot(dx, dy); sx, sy = w / 16, h / 16
            sel = [(i, j) for i in range(cy0, cy1) for j in range(cx0, cx1) if gate[i, j] > 0.5 and mag[i, j] > 0.35]
            sel = sorted(sel, key=lambda ij: -mag[ij])[:10]
            import matplotlib.patheffects as pe
            for i, j in sel:
                tx, ty = (j + 0.5) * sx, (i + 0.5) * sy
                a_ = ax.annotate("", xy=(tx, ty), xytext=(tx + dx[i, j] * sx, ty + dy[i, j] * sy), annotation_clip=True,
                                 arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.18", color=GREEN, lw=1.1,
                                                 shrinkA=0, shrinkB=0, clip_on=True))
                a_.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
                a_.arrow_patch.set_clip_box(ax.bbox)
        except Exception as exc:  # checkpoint missing: frame only
            print("transport arrows skipped:", exc)
        ins = ax.inset_axes([0.02, 0.70, 0.46, 0.28]); ins.imshow(fut, aspect="equal", interpolation="lanczos")
        ins.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, ec=ROI, lw=0.8)); frame_axes(ins, "white", 0.6)
        # 2: true future crop
        ax = fig.add_subplot(left[r, 1]); ax.imshow(fut, aspect="equal", interpolation="lanczos"); crop(ax); frame_axes(ax, ROI, 1.1)
        # 3-4: errors on the same crop, shared scale
        for c, (name, e, col) in enumerate((("Direct", ed, BLUE), ("ShiftWM", es, GREEN))):
            ax = fig.add_subplot(left[r, 2 + c]); ax.imshow(gray(fut), aspect="equal", interpolation="lanczos")
            ax.imshow(np.clip(e / vmax, 0, 1), cmap=ERR, vmin=0, vmax=1, extent=ext, interpolation="bicubic", aspect="equal")
            crop(ax); frame_axes(ax, col, 1.6 if name == "ShiftWM" else 1.1)
            corner(ax, f"{float(e[yy, xx].mean()):.2f}", col)
    # column heads (placed above the first row)
    for c, t in enumerate(heads):
        pos = left[0, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.012, t, ha="center", va="bottom", fontsize=6.4, fontweight="bold",
                 color={2: BLUE, 3: GREEN}.get(c, mf.INK))
    lp, rp = outer[0, 0].get_position(fig), outer[0, 1].get_position(fig)
    for p_, t in ((lp, "(a) Anatomy of a win: where features move from, and where each model errs"), (rp, "")):
        fig.patches.append(FancyBboxPatch((p_.x0 - 0.008, p_.y0 - 0.02), p_.x1 - p_.x0 + 0.016, p_.y1 - p_.y0 + 0.095,
                                          boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                          fc="#F6F8FA", ec="#E1E5EA", lw=0.6, zorder=-10))
    fig.text(lp.x0, 0.955, "(a) Anatomy of a win (held-out DROID, $k{=}10$): where features move from, and where each model errs",
             fontsize=6.9, fontweight="bold", color=mf.INK)
    fig.legend(handles=[Patch(color=GREEN, label="transport (source $\\to$ target)"),
                        Patch(color=(0.85, 0.1, 0.1), alpha=0.75, label="forecast error, shared scale")],
               loc="lower left", bbox_to_anchor=(0.015, -0.01), ncol=2, fontsize=5.7, frameon=False, handlelength=1.2, columnspacing=1.4)
    right = outer[0, 1].subgridspec(3, 1, hspace=1.15, height_ratios=[1, 1, 0.8])
    ax = fig.add_subplot(right[0])
    regs = [("moving", "moving"), ("high gate", "highgate"), ("static", "static")]
    for y, (lab, key) in zip([2, 1, 0], regs):
        b, k_ = S[f"base_{key}"], S[f"ko_{key}"]; inc = S[f"knockout_increase_{key}"]
        col = GREEN if inc > 5 else GREY
        ax.plot([b, k_], [y, y], color=col, lw=1.6, alpha=0.6, solid_capstyle="round")
        ax.scatter([b], [y], s=18, c="white", edgecolors=col, linewidths=1.1, zorder=3)
        ax.scatter([k_], [y], s=18, c=col, zorder=3)
        ax.text(max(b, k_) + 0.03, y, f"+{inc:.0f}%", va="center", fontsize=5.9, color=col if inc > 5 else mf.INK,
                fontweight="bold" if inc > 5 else "normal")
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in regs], fontsize=5.9); ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0, max(S["ko_moving"], S["ko_highgate"]) * 1.35); ax.tick_params(labelsize=5.6, length=2); ax.grid(axis="y", visible=False)
    ax.set_xlabel("error with $\\circ$ / without $\\bullet$ transport", fontsize=5.6, labelpad=1)
    ax.set_title("(b) remove transport", fontsize=6.6, pad=3, loc="left", x=-0.3)
    ax = fig.add_subplot(right[1])
    g = np.array(S["gain_vs_direct_by_decile"]); x = np.arange(1, 11)
    cols = [plt.cm.Greens(0.45 + 0.5 * i / 9) for i in range(10)]
    ax.vlines(x, 0, g, colors=cols, lw=1.6); ax.scatter(x, g, s=16, c=cols, zorder=3)
    ax.axhline(0, color=mf.INK, lw=0.5)
    ax.set_xticks([1, 10]); ax.set_xticklabels(["static", "fast"], fontsize=5.8); ax.tick_params(labelsize=5.6, length=2)
    ax.set_ylim(0, g.max() * 1.35); ax.set_ylabel("gain (%)", fontsize=5.8, labelpad=1); ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.97, "> 0 in all 10 deciles", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=GREEN, fontweight="bold")
    ax.set_title("(c) gain over Direct by motion", fontsize=6.6, pad=3, loc="left", x=-0.3)
    ax = fig.add_subplot(right[2])
    st_ = [S["steer_static"], S["steer_moving"]]
    ax.barh([0, 1], st_, color=[GREY, GREEN], height=0.55)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["static", "moving"], fontsize=5.9)
    ax.text(st_[1] * 1.03, 1, f"{S['steer_ratio_moving_over_static']:.1f}$\\times$", va="center", fontsize=6.2, color=GREEN, fontweight="bold")
    ax.set_xlim(0, st_[1] * 1.35); ax.tick_params(labelsize=5.6, length=2); ax.grid(axis="y", visible=False)
    ax.set_xlabel("transport change, other actions (patches)", fontsize=5.4, labelpad=1)
    ax.set_title("(d) actions steer moving parts", fontsize=6.6, pad=3, loc="left", x=-0.3)
    fig.savefig(mf.FIG / "interpret.pdf"); fig.savefig(mf.FIG / "interpret_preview.png", dpi=200)
    print("wrote interpret")


if __name__ == "__main__":
    main()
