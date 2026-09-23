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


def save_frame(img, path, grid=True, arrows=None, dpi=400):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(img, aspect="auto", interpolation="lanczos")
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
    cx0 = int(np.clip(gx - 7, 0, g - 14)); cy0 = int(np.clip(gy - 5, 0, g - 10))
    x0, x1 = cx0 * w / g, (cx0 + 14) * w / g; y0, y1 = cy0 * h / g, (cy0 + 10) * h / g
    fig = plt.figure(figsize=(3.2, 2.0), dpi=400); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(img, aspect="auto", interpolation="lanczos")
    for x in np.linspace(0, w, g + 1):
        ax.axvline(x, color="white", lw=0.4, alpha=0.5)
    for y in np.linspace(0, h, g + 1):
        ax.axhline(y, color="white", lw=0.4, alpha=0.5)
    mm = m & (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    ax.quiver(sx[mm], sy[mm], (X - sx)[mm], (Y - sy)[mm], color="#FFE066", angles="xy", scale_units="xy", scale=1,
              width=0.016, headwidth=3.2, headlength=3.4, edgecolor="#1F2A37", linewidth=0.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    fig.savefig(OUT / "transport.png", dpi=400); plt.close(fig)
    # gate overlaid on the observed frame
    fig = plt.figure(figsize=(3.2, 1.8), dpi=400); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(img, aspect="auto", interpolation="lanczos")
    ax.imshow(gates[10], cmap="viridis", alpha=0.62, extent=(-0.5, w - 0.5, h - 0.5, -0.5), aspect="auto",
              interpolation="bicubic", vmin=0, vmax=float(gates[10].max()))
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    fig.savefig(OUT / "gate.png", dpi=400); plt.close(fig)
    print("assets from episode", ep, "checkpoint", ckpts[0])


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def head_data(out_tex=Path(__file__).parent / "arch_assets" / "head_data.tex", k=10):
    """Real per-patch tensors for the zoomed ShiftWM-head diagram, written as TikZ macros."""
    import json
    import torch
    from matplotlib import colormaps
    from shiftwm.v2.models import V2WorldModel
    ep = mf.pick_teaser_episode()
    root = mf.ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    with np.load(root / f"episodes/{ep}.npz") as z:
        f, a = z["features"].astype(np.float32), z["actions"]
    fm, fs = np.array(stats["feature_mean"]), np.array(stats["feature_std"])
    am, ast = np.array(stats["action_mean"]), np.array(stats["action_std"])
    fz = (f.reshape(len(f), -1, f.shape[-1]) - fm) / fs; an = (a - am) / ast
    ck = sorted((mf.RES / "droid/dinov2s/shiftwm").glob("s*/best.pt"))[0]
    st = torch.load(ck, map_location="cpu"); m = V2WorldModel(st["config"]).eval(); m.load_state_dict(st["model"])
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        pred, det = m(T(fz[None, :3]), T(an[None, :2]), T(an[None, 2:2 + k]), return_details=True)
    gate = det["gate"][0, k - 1, :, 0].numpy()
    i = int(np.argmax(gate)); g = float(gate[i])
    w = det["weights"][0, k - 1, i].numpy(); S, W = m.config.sources, m.config.window
    w3 = w.reshape(S, W, W)
    z0 = fz[2, i]; zt = fz[2 + k, i]; p = pred[0, k - 1, i].numpy(); r = det["correction"][0, k - 1, i].numpy()
    Tvec = (p - (1 - g) * z0 - r) / max(g, 1e-6)
    # 12-D view of each 384-D vector: projections on the top principal directions of this episode's features
    X = fz.reshape(-1, fz.shape[-1]); mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:12].T
    vecs = {"Zzero": z0, "Ttr": Tvec, "Rcor": r + mu, "Zhat": p, "Ztrue": zt}
    proj = {n: (v - mu) @ P for n, v in vecs.items()}
    proj["Rcor"] = r @ P
    scale = np.percentile(np.abs(np.concatenate(list(proj.values()))), 95)
    cmap = colormaps["RdBu_r"]
    lines = [f"% generated by arch_assets.py from {ck} (episode {ep}, patch {i}, k={k})"]
    for n, v in proj.items():
        cols = [cmap(0.5 + 0.5 * np.clip(x / scale, -1, 1)) for x in v]
        hexes = ",".join("%02X%02X%02X" % tuple(int(255 * c) for c in col[:3]) for col in cols)
        lines.append(f"\\def\\vec{n}{{{hexes}}}")
    for s_ in range(S):
        vals = ",".join(f"{x:.4f}" for x in (w3[s_] / w3.max()).ravel())
        lines.append(f"\\def\\win{'ABC'[s_]}{{{vals}}}")
        lines.append(f"\\def\\mass{'ABC'[s_]}{{{100 * w3[s_].sum():.0f}}}")
    lines.append(f"\\def\\gateval{{{g:.2f}}}")
    lines.append(f"\\def\\errhat{{{float(((p - zt) ** 2).mean()):.2f}}}")
    lines.append(f"\\def\\errstay{{{float(((z0 - zt) ** 2).mean()):.2f}}}")
    Path(out_tex).write_text("\n".join(lines) + "\n")
    print("head data: patch", i, "gate", round(g, 3), "mass", [round(float(w3[s].sum()), 3) for s in range(S)],
          "err hat/stay", float(((p - zt) ** 2).mean()), float(((z0 - zt) ** 2).mean()))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--head":
    head_data()
