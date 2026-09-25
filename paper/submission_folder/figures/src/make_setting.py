"""Appendix 'problem setting' figure: what a world model sees and must predict, on real held-out episodes.

For DROID and Open-H Hamlyn: the H=3 observed frames, three future frames to be predicted, the commanded
end-effector trajectory (observed part vs. future action prefix) and the true per-patch feature change
||z_{t+10} - z_t||^2 over the 16x16 grid, which shows that most of the scene barely changes.
Episode rule (no manual picking): DROID uses the teaser episode (largest change, t0 = 2); Hamlyn uses the test
episode with the largest change at its middle window t0 = clip(T // 2, 2, T - 11).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

sys.path.insert(0, str(mf.ROOT / "src"))
from shiftwm.v2 import analysis as A  # noqa: E402

H, K = 3, 10
OBS, FUT = "#52627A", "#8A94A3"          # observed input: slate (as in Fig. 2); future: light dashed grey
ACT = ("#4B2D6B", "#7B4FA0", "#B07CC6")  # action components: the purple action shades of Fig. 2
NATIVE = {"droid": None, "openh_hamlyn": (848, 480)}


def pick(ds):
    root = A.cache_root(ds); man, stats = A.read_cache(root)
    best = (-1, None, None)
    for r in man["episodes"]:
        if r["split"] != "test" or r["T"] < H + K + 1:
            continue
        t0 = 2 if ds == "droid" else int(np.clip(r["T"] // 2, H - 1, r["T"] - 1 - K))
        with np.load(root / r["file"]) as z:
            f = z["features"][[t0, t0 + K]].astype(np.float32)
        d = float(np.mean((f[1] - f[0]) ** 2))
        if d > best[0]:
            best = (d, r, t0)
    return root, stats, best[1], best[2]


def frames(ds, root, ep, ts):
    im = A.FrameSource(root).load_native(ep, ts)
    if NATIVE[ds]:
        w, h = NATIVE[ds]; s = 224 / h
        im = np.stack([np.asarray(Image.fromarray(x).resize((int(w * s), 224), Image.LANCZOS)) for x in im])
    return im


def block(fig, gs, ds, name, letter):
    root, stats, row, t0 = pick(ds)
    with np.load(root / row["file"]) as z:
        f = z["features"].astype(np.float32); a = z["actions"].astype(np.float32)
    fs = np.array(stats["feature_std"], np.float32)
    ts_obs = [t0 - 2, t0 - 1, t0]; ts_fut = [t0 + 1, t0 + 5, t0 + 10]
    ims = frames(ds, root, row["id"], ts_obs + ts_fut)
    sub = gs.subgridspec(2, 8, width_ratios=[1, 1, 1, 0.22, 1, 1, 1, 1.0], height_ratios=[1, 0.95], hspace=0.62, wspace=0.05)
    labels = ["$t{-}2$", "$t{-}1$", "$t$", "$t{+}1$", "$t{+}5$", "$t{+}10$"]
    cols = [0, 1, 2, 4, 5, 6]
    for i, (c, im) in enumerate(zip(cols, ims)):
        ax = fig.add_subplot(sub[0, c]); ax.imshow(im, aspect="equal", interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        future = i >= 3
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(FUT if future else OBS); sp.set_linewidth(1.1)
            sp.set_linestyle((0, (2.5, 1.5)) if future else "-")
        if future:
            ax.imshow(np.ones_like(im) * 255, alpha=0.28, aspect="equal")
        ax.set_title(labels[i], fontsize=mf.FS_NOTE, pad=1.5, color=FUT if future else OBS)
        if i == 0:
            ax.text(0.0, 1.36, f"({letter}) {name}", transform=ax.transAxes, fontsize=mf.FS_TITLE, fontweight="bold",
                    color=mf.INK)
        if i == 1:
            ax.text(0.5, -0.08, "observed (input)", transform=ax.transAxes, fontsize=mf.FS_NOTE, ha="center", va="top",
                    color=OBS)
        if i == 4:
            ax.text(0.5, -0.08, "future frames (to predict)", transform=ax.transAxes, fontsize=mf.FS_NOTE, ha="center",
                    va="top", color=FUT)
    ax = fig.add_subplot(sub[0, 3]); ax.axis("off")
    ax.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=mf.INK, lw=0.9))
    # true feature change over the grid, on frame t
    chg = (((f[t0 + K] - f[t0]) / fs) ** 2).mean(-1)
    ax = fig.add_subplot(sub[0, 7]); fr = ims[2]; h, w = fr.shape[:2]
    ax.imshow(fr, aspect="equal", interpolation="lanczos")
    ax.imshow(chg, cmap="magma", alpha=0.72, extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="bicubic",
              vmin=0, vmax=np.quantile(chg, 0.98))
    for g in range(1, 16):
        ax.axvline(g * w / 16 - 0.5, color="white", lw=0.15, alpha=0.5); ax.axhline(g * h / 16 - 0.5, color="white", lw=0.15, alpha=0.5)
    srt = np.sort(chg.ravel())[::-1]; share = 100 * float(srt[: len(srt) // 4].sum() / srt.sum())
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("true change to $t{+}10$", fontsize=mf.FS_NOTE, pad=1.5, color=mf.INK)
    ax.text(0.5, -0.08, f"top 25% of patches\ncarry {share:.0f}% of it", transform=ax.transAxes, fontsize=mf.FS_NOTE,
            color=mf.INK, ha="center", va="top", linespacing=1.0)
    # commanded end-effector position (first command of each block; xyz, centred)
    ax = fig.add_subplot(sub[1, :])
    steps = np.arange(t0 - 2, t0 + K); xyz = a[steps, :3]; xyz = xyz - xyz.mean(0)
    ends = xyz[-1].copy(); gap = 0.24 * float(np.ptp(xyz))          # end labels, spread so they never overlap
    order = np.argsort(ends)
    for a_, b_ in zip(order[:-1], order[1:]):
        ends[b_] = max(ends[b_], ends[a_] + gap)
    for j, (c, lab, ls) in enumerate(zip(ACT, "xyz", ("-", "--", ":"))):
        ax.plot(steps - t0, xyz[:, j], color=c, lw=1.1, ls=ls, marker="o", ms=2.2)
        ax.annotate(lab, (steps[-1] - t0, ends[j]), xytext=(5, 0), textcoords="offset points", fontsize=mf.FS_NOTE,
                    color=c, va="center", fontweight="bold", annotation_clip=False)
    ax.axvspan(-2.4, -0.5, color=OBS, alpha=0.08, lw=0); ax.axvspan(-0.5, K - 0.6, color=FUT, alpha=0.06, lw=0)
    ax.axvline(-0.5, color=mf.INK, lw=0.6, ls=(0, (2, 2)))
    ax.text(-1.45, 1.03, "past actions", transform=ax.get_xaxis_transform(), fontsize=mf.FS_NOTE, ha="center", va="bottom",
            color=OBS)
    ax.text(4.5, 1.03, "future actions $\\mathbf{a}_{0:K-1}$ (given to the model)", transform=ax.get_xaxis_transform(),
            fontsize=mf.FS_NOTE, ha="center", va="bottom", color=mf.INK)
    ax.set_xlim(-2.4, K - 0.6); ax.set_xticks(range(-2, K))
    ax.set_xticklabels([f"{s:+d}".replace("-", "\u2212") if s else "0" for s in range(-2, K)], fontsize=mf.FS_TICK)
    ax.tick_params(axis="y", labelsize=mf.FS_TICK, length=2, pad=1); ax.tick_params(axis="x", length=2, pad=1)
    ax.set_ylabel("end-effector\n(centred)", fontsize=mf.FS_NOTE, labelpad=1, linespacing=1.0)
    ax.set_xlabel("model step (1 step = 1/3 s)", fontsize=mf.FS_NOTE, labelpad=1)
    ax.grid(alpha=0.25, lw=0.4)
    return row["id"], t0, share


def main():
    fig = plt.figure(figsize=(5.5, 3.5))
    gs = fig.add_gridspec(2, 1, hspace=0.55, left=0.085, right=0.97, top=0.92, bottom=0.075)
    info = [block(fig, gs[0], "droid", "DROID: real Franka manipulation", "a"),
            block(fig, gs[1], "openh_hamlyn", "Open-H Hamlyn: surgical dVRK training task", "b")]
    mf.qa(fig, "setting", 5.5)
    fig.savefig(mf.FIG / "setting.pdf"); fig.savefig(mf.FIG / "setting_preview.png", dpi=200)
    print("wrote setting", info)


if __name__ == "__main__":
    main()
