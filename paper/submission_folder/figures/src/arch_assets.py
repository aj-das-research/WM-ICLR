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
    save_frame(img, OUT / "transport.png", grid=True, arrows=(sx[m], sy[m], (X - sx)[m], (Y - sy)[m], gate[m]))
    fig = plt.figure(figsize=(2, 2), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(gates[10], cmap="viridis", vmin=0, vmax=float(gates[10].max()))
    fig.savefig(OUT / "gate.png"); plt.close(fig)
    print("assets from episode", ep, "checkpoint", ckpts[0])


if __name__ == "__main__":
    main()
