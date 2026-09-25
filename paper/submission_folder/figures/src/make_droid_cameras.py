"""Appendix data figure: DROID camera 1 (training) vs camera 2 (zero-shot transfer) under the current contract.

Same held-out episode and window as fig:setting (make_setting.pick, t0 = 2): H=3 observed frames t-2..t, the target
at t+K (K=10 blocks of 1/3 s, each a 35-D block of 5 commanded poses + gripper), both cameras recorded at the same
instants and driven by the same action blocks. The last column overlays the true per-patch change
||z_{t+10}-z_t||^2 (feature-std normalised) on the 16x16 DINOv2-S grid of each camera's own cache.
Frames come from the raw DROID processed episodes via the v2 caches (data/v2/features/droid{,_cam2}/dinov2s).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402
from make_setting import H, K, OBS, FUT, pick  # noqa: E402

sys.path.insert(0, str(mf.ROOT / "src"))
from shiftwm.v2 import analysis as A  # noqa: E402

CAMS = [("droid", "camera 1 (training)"), ("droid_cam2", "camera 2 (zero-shot transfer)")]


def grid(ax, w, h, n=16, color="white", alpha=0.45):
    for g in range(1, n):
        ax.axvline(g * w / n - 0.5, color=color, lw=0.15, alpha=alpha)
        ax.axhline(g * h / n - 0.5, color=color, lw=0.15, alpha=alpha)


def main():
    _, _, row, t0 = pick("droid")
    ts = [t0 - 2, t0 - 1, t0, t0 + K]
    fig = plt.figure(figsize=(5.5, 1.62))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 0.18, 1, 1], hspace=0.12, wspace=0.05,
                          left=0.075, right=0.995, top=0.84, bottom=0.01)
    shares = {}
    for r, (ds, lab) in enumerate(CAMS):
        root = A.cache_root(ds); man, stats = A.read_cache(root)
        rec = next(e for e in man["episodes"] if e["id"] == row["id"])
        assert rec["split"] == "test"
        with np.load(root / rec["file"]) as z:
            f = z["features"][[t0, t0 + K]].astype(np.float32)
        fs = np.array(stats["feature_std"], np.float32)
        ims = A.FrameSource(root).load_native(row["id"], ts)
        h, w = ims[0].shape[:2]
        for i, (c, im) in enumerate(zip([0, 1, 2, 4], ims)):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(im, aspect="equal", interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            fut = i == 3
            for sp in ax.spines.values():
                sp.set_edgecolor(FUT if fut else OBS); sp.set_linewidth(1.0)
                sp.set_linestyle((0, (2.5, 1.5)) if fut else "-")
            if fut:
                ax.imshow(np.ones_like(im) * 255, alpha=0.28, aspect="equal")
            if i == 2:
                grid(ax, w, h)
            if r == 0:
                ax.set_title(["$t{-}2$", "$t{-}1$", "$t$ (16$\\times$16 patches)", "$t{+}10$ (target)"][i],
                             fontsize=6.2, pad=1.5, color=FUT if fut else OBS)
            if i == 0:
                ax.set_ylabel(lab.replace(" (", "\n("), fontsize=6.2, labelpad=3, color=mf.INK)
        ax = fig.add_subplot(gs[r, 3]); ax.axis("off")
        ax.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=mf.INK, lw=0.8))
        chg = (((f[1] - f[0]) / fs) ** 2).mean(-1)
        ax = fig.add_subplot(gs[r, 5]); ax.imshow(ims[2], aspect="equal", interpolation="lanczos")
        ax.imshow(chg, cmap="magma", alpha=0.72, extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="bicubic",
                  vmin=0, vmax=np.quantile(chg, 0.98))
        grid(ax, w, h, alpha=0.35)
        ax.set_xticks([]); ax.set_yticks([])
        srt = np.sort(chg.ravel())[::-1]; shares[ds] = 100 * float(srt[: len(srt) // 4].sum() / srt.sum())
        ax.text(0.03, 0.05, f"mean change {chg.mean():.2f}", transform=ax.transAxes, fontsize=5.0, color="white",
                fontweight="bold", bbox=dict(fc="#1F2A37", ec="none", alpha=0.65, pad=0.7))
        if r == 0:
            ax.set_title("true change $t\\to t{+}10$", fontsize=6.2, pad=1.5)
    fig.text(0.075, 0.965, "same episode, same instants, same 35-D action blocks $\\mathbf{a}_{0:9}$; only the viewpoint differs",
             fontsize=6.2, color=mf.INK, ha="left", va="center")
    fig.savefig(mf.FIG / "droid_cameras.pdf"); fig.savefig(mf.FIG / "droid_cameras_preview.png", dpi=200)
    print("wrote droid_cameras", row["id"], t0, shares)


if __name__ == "__main__":
    main()
