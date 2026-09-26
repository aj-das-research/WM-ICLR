"""Slim one-row version of Fig. 4(a) ("anatomy of a win"), 5.5 x ~1.0 in: two selected held-out DROID windows x
[observed frame + transport arrows | true t+10 | Direct error | ShiftWM error], all on the same zoomed crop.

Same data, selection rule, crop rule, colour scale and helpers as paper/submission_folder/figures/src/make_interpret.py
panel (a) (results/v2/analysis/interpret/droid.npz; windows = the first two of the selection rule in summary.json;
transport field recomputed from the seed-0 ShiftWM checkpoint on CPU, as there). Writes only to this folder.

Usage (repo root): PYTHONPATH=src .venv/bin/python reviews/composite_results_fig/make_anatomy_strip.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper/submission_folder/figures/src"))
import make_interpret as MI  # noqa: E402  (helpers only; its main() is not called, so paper/ is not written)

mf = MI.mf
GREEN, BLUE, ROI, ERR = MI.GREEN, MI.BLUE, MI.ROI, MI.ERR
FS = 6.2
W, TILE, GAP, WGAP = 5.5, 0.655, 0.025, 0.12
HEAD, LEG = 0.15, 0.18
H = round(LEG + TILE + HEAD + 0.02, 3)


def main():
    D = MI.D
    S = json.loads((D / "summary.json").read_text()); Z = np.load(D / "droid.npz", allow_pickle=True)
    fig = plt.figure(figsize=(W, H))
    E_d, E_s = Z["err_di"][:2], Z["err_sw"][:2]
    vmax = float(np.quantile(np.concatenate([E_d.ravel(), E_s.ravel()]), 0.97))       # identical to make_interpret
    heads = ["observed $t$", "true $t{+}10$", "Direct error", "ShiftWM error"]
    x0_all = (W - (8 * TILE + 6 * GAP + WGAP)) / 2
    ledger = {"vmax": vmax, "windows": []}
    for r in range(2):
        ep, st = str(Z["episode"][r]), int(Z["start"][r])
        obs, fut = mf.droid_frames(ep, steps=(st + 2, st + 2 + 10)); h, w = fut.shape[:2]
        ed, es = E_d[r].reshape(16, 16), E_s[r].reshape(16, 16)
        gy, gx = np.unravel_index(np.argmax(ed + es), ed.shape)
        side = min(h, 7 * w / 16)
        px, py = (gx + 0.5) * w / 16, (gy + 0.5) * h / 16
        x0 = float(np.clip(px - side / 2, 0, w - side)); y0 = float(np.clip(py - side / 2, 0, h - side))
        box = (x0 - 0.5, y0 - 0.5, side, side)
        ext = (-0.5, w - 0.5, h - 0.5, -0.5)
        cx0, cx1 = int(x0 // (w / 16)), int(np.ceil((x0 + side) / (w / 16)))
        cy0, cy1 = int(y0 // (h / 16)), int(np.ceil((y0 + side) / (h / 16)))
        yy, xx = slice(cy0, cy1), slice(cx0, cx1)

        def tile(c):
            xi = x0_all + r * (4 * TILE + 3 * GAP + WGAP) + c * (TILE + GAP)
            a = fig.add_axes([xi / W, LEG / H, TILE / W, TILE / H])
            fig.text((xi + TILE / 2) / W, (LEG + TILE + 0.02) / H, heads[c], ha="center", va="bottom", fontsize=FS,
                     fontweight="bold", color={2: BLUE, 3: GREEN}.get(c, mf.INK))
            return a

        def crop(a):
            a.set_xlim(box[0], box[0] + box[2]); a.set_ylim(box[1] + box[3], box[1])

        ax = tile(0); ax.imshow(obs, aspect="equal", interpolation="lanczos"); crop(ax); MI.frame_axes(ax, ROI, 1.0)
        dx, dy, gate = MI.transport_for(ep, st)
        mag = np.hypot(dx, dy); sx, sy = w / 16, h / 16
        sel = [(i, j) for i in range(cy0, cy1) for j in range(cx0, cx1) if gate[i, j] > 0.5 and mag[i, j] > 0.35]
        sel = sorted(sel, key=lambda ij: -mag[ij])[:10]
        for i, j in sel:
            tx, ty = (j + 0.5) * sx, (i + 0.5) * sy
            a_ = ax.annotate("", xy=(tx, ty), xytext=(tx + dx[i, j] * sx, ty + dy[i, j] * sy), annotation_clip=True,
                             arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.15", color=GREEN, lw=0.9,
                                             shrinkA=0, shrinkB=0, clip_on=True))
            a_.arrow_patch.set_path_effects([pe.Stroke(linewidth=1.9, foreground="white"), pe.Normal()])
            a_.arrow_patch.set_clip_box(ax.bbox)
        ins = ax.inset_axes([0.02, 0.68, 0.46, 0.30]); ins.imshow(fut, aspect="equal", interpolation="lanczos")
        ins.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, ec=ROI, lw=0.7)); MI.frame_axes(ins, "white", 0.5)
        ax = tile(1); ax.imshow(fut, aspect="equal", interpolation="lanczos"); crop(ax); MI.frame_axes(ax, ROI, 1.0)
        errs = {}
        for c, (name, e, col) in enumerate((("Direct", ed, BLUE), ("ShiftWM", es, GREEN))):
            ax = tile(2 + c); ax.imshow(MI.gray(fut), aspect="equal", interpolation="lanczos")
            ax.imshow(np.clip(e / vmax, 0, 1), cmap=ERR, vmin=0, vmax=1, extent=ext, interpolation="bicubic", aspect="equal")
            crop(ax); MI.frame_axes(ax, col, 1.4 if name == "ShiftWM" else 1.0)
            v = float(e[yy, xx].mean()); errs[name] = v
            ax.text(0.95, 0.05, f"{v:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS, color="white",
                    fontweight="bold", bbox=dict(fc=col, ec="none", alpha=0.92, pad=0.8))
        ledger["windows"].append({"episode": ep, "start": st, "n_arrows": len(sel), "crop_err": errs})
    fig.legend(handles=[Patch(color=GREEN, label="transport (source $\\to$ target)"),
                        Patch(color=ROI, label="zoomed region (inset: full frame)"),
                        Patch(color=(0.7, 0.05, 0.1), alpha=0.85, label="forecast error, shared scale (value: crop mean)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=3, fontsize=FS, frameon=False,
               handlelength=1.0, handleheight=0.7, columnspacing=1.2, handletextpad=0.4)
    mf.qa(fig, "anatomy_strip", W)
    fig.savefig(HERE / "anatomy_strip.pdf"); fig.savefig(HERE / "anatomy_strip.png", dpi=400)
    (HERE / "anatomy_strip_ledger.json").write_text(json.dumps(ledger, indent=1))
    print(json.dumps(ledger, indent=1), "size", W, H)


if __name__ == "__main__":
    main()
