"""Raster insets for the architecture figure, all from real data / a trained checkpoint.

observed_{0,1,2}.png : three consecutive observed DROID test frames with a faint 16x16 patch grid
transport.png        : learned transport field (k=10) over the last observed frame
gate.png             : learned gate map (k=10)
future.png           : the true frame 10 steps later (for the output card)
Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/arch_assets.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

OUT = Path(__file__).parent / "arch_assets"


def save_frame(img, path, grid=True, arrows=None, dpi=200):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(img, aspect="auto")
    if grid:
        for x in np.linspace(0, w, 17):
            ax.axvline(x, color="white", lw=0.35, alpha=0.45)
        for y in np.linspace(0, h, 17):
            ax.axhline(y, color="white", lw=0.35, alpha=0.45)
    if arrows is not None:
        sx, sy, u, v, c = arrows
        ax.quiver(sx, sy, u, v, c, cmap="viridis", angles="xy", scale_units="xy", scale=1,
                  width=0.012, headwidth=3.2, headlength=3.4)
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    fig.savefig(path, dpi=dpi); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    ep = mf.pick_teaser_episode()
    frames = mf.droid_frames(ep, steps=(0, 1, 2, 12))
    for i in range(3):
        save_frame(frames[i], OUT / f"observed_{i}.png")
    save_frame(frames[3], OUT / "future.png", grid=False)
    ckpts = sorted((mf.RES / "droid/dinov2s/shiftwm").glob("s*/best.pt"))
    dx, dy, gate, gates = mf.transport_arrows(ckpts[0], ep)
    img = frames[2]; h, w = img.shape[:2]; g = dx.shape[0]
    ys, xs = (np.arange(g) + 0.5) * h / g, (np.arange(g) + 0.5) * w / g
    X, Y = np.meshgrid(xs, ys)
    m = gate > np.quantile(gate, 0.7)
    sx, sy = X + dx * w / g, Y + dy * h / g
    # zoomed crop (10x7 patches) centred on the highest-gate region, bold arrows
    gy, gx = np.unravel_index(np.argmax(np.convolve(gate.ravel(), np.ones(1), "same").reshape(gate.shape)), gate.shape)
    cx0 = int(np.clip(gx - 5, 0, g - 10)); cy0 = int(np.clip(gy - 3, 0, g - 7))
    x0, x1 = cx0 * w / g, (cx0 + 10) * w / g; y0, y1 = cy0 * h / g, (cy0 + 7) * h / g
    fig = plt.figure(figsize=(3.2, 2.0), dpi=220); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(img, aspect="auto")
    for x in np.linspace(0, w, g + 1):
        ax.axvline(x, color="white", lw=0.4, alpha=0.5)
    for y in np.linspace(0, h, g + 1):
        ax.axhline(y, color="white", lw=0.4, alpha=0.5)
    mm = m & (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    ax.quiver(sx[mm], sy[mm], (X - sx)[mm], (Y - sy)[mm], color="#FFE066", angles="xy", scale_units="xy", scale=1,
              width=0.016, headwidth=3.2, headlength=3.4, edgecolor="#1F2A37", linewidth=0.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    fig.savefig(OUT / "transport.png", dpi=220); plt.close(fig)
    # gate overlaid on the observed frame
    fig = plt.figure(figsize=(3.2, 1.8), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(img, aspect="auto")
    ax.imshow(gates[10], cmap="viridis", alpha=0.62, extent=(-0.5, w - 0.5, h - 0.5, -0.5), aspect="auto",
              interpolation="nearest", vmin=0, vmax=float(gates[10].max()))
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    fig.savefig(OUT / "gate.png", dpi=200); plt.close(fig)
    print("assets from episode", ep, "checkpoint", ckpts[0])


if __name__ == "__main__":
    main()
