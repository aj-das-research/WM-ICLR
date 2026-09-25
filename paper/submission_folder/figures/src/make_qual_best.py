"""Per-benchmark 'largest-advantage' qualitative figures from results/v2/analysis/qual_best (scripts/v2/qual_select.py).

For each finished dataset: figures/qual_best_<ds>.pdf (+ qual_best_<ds>_preview.png), 5.5 in wide.
  top block   : 3 selected test windows x [observed t | true t+10 | Direct error | AR error | ShiftWM error]
                errors = per-patch squared error of the k=10 feature forecast (standardised DINOv2 features) as a red
                heat map on a desaturated copy of the true frame, ONE colour scale per dataset; corner badge = mean
                error of the window (the per-window MSE used by the selection rule).
  bottom block: (if a decoder exists) the decoded truth and the decoded Direct / AR / ShiftWM forecasts at k=10,
                aligned under the corresponding columns.
All images are shown at the camera's native aspect ratio (frames are resized, never stretched).
Also writes tables/generated/qual_best_figs.tex (one figure environment per finished dataset).
Selection rule: see scripts/v2/qual_select.py (the JSON next to each npz records it).

Usage (repo root): python paper/submission_folder/figures/src/make_qual_best.py [--src DIR]
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (sets rcParams: Times/STIX)

SRC = mf.RES / "analysis/qual_best"
GEN = mf.ROOT / "paper/submission_folder/tables/generated"
ORDER = ("droid", "openh_hamlyn", "bridge", "fractal", "language_table", "iws_pusht", "iws_box", "iws_rope")
NAME = {"droid": "DROID", "openh_hamlyn": "Open-H Hamlyn", "bridge": "BridgeData V2", "fractal": "RT-1 (Fractal)",
        "language_table": "Language-Table", "iws_pusht": "IWS Push-T", "iws_box": "IWS Box", "iws_rope": "IWS Rope"}
ASPECT = {"droid": 180 / 320, "openh_hamlyn": 480 / 848, "bridge": 480 / 640, "fractal": 256 / 320,
          "language_table": 360 / 640, "iws_pusht": 480 / 640, "iws_box": 480 / 640, "iws_rope": 480 / 640}
COL = {"direct": mf.METHODS["direct"][1], "ar": mf.METHODS["ar"][1], "shiftwm": mf.METHODS["shiftwm"][1]}
LABEL = {"direct": "Direct", "ar": "AR", "shiftwm": "ShiftWM"}
SHOW = ("direct", "ar", "shiftwm")                  # column order of the error / decoded panels
ERR = LinearSegmentedColormap.from_list(
    "err", [(1, 0.25, 0.1, 0.0), (1, 0.2, 0.08, 0.5), (0.85, 0.0, 0.05, 0.8), (0.55, 0.0, 0.1, 0.95)])
WIDTH = 5.5
PX_W = 640                                          # display resolution (width) of every image


def native(img, aspect):
    """Resize (Lanczos) to the camera's native aspect ratio; never stretch into a square box."""
    h = int(round(PX_W * aspect))
    return np.asarray(Image.fromarray(img).resize((PX_W, h), Image.LANCZOS))


def desat(img):
    g = img.astype(np.float32).mean(-1, keepdims=True)
    return np.clip(0.72 * g + 0.28 * img + 18, 0, 255).astype(np.uint8)


def frame(ax, color=None, lw=1.8):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if color:   # full 4-sided border drawn as one rectangle (never partial spines)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=color, lw=lw, clip_on=False,
                               zorder=10, joinstyle="miter"))
    else:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec="#C9CED6", lw=0.4,
                               clip_on=False, zorder=10))


def badge(ax, text, fc):
    ax.text(0.965, 0.05, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color="white",
            fontweight="bold", bbox=dict(fc=fc, ec="none", alpha=0.93, pad=0.8, boxstyle="round,pad=0.25"), zorder=11)


def image(ax, img):
    ax.imshow(img, aspect="auto", interpolation="lanczos", extent=(0, 1, 1, 0))
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)


def load(ds, src):
    f = src / f"{ds}.npz"
    return (np.load(f, allow_pickle=False), json.loads((src / f"{ds}.json").read_text())) if f.exists() else (None, None)


def make(ds, z, info):
    asp = ASPECT[ds]
    arms = [str(a) for a in z["arms"]]
    n = len(z["episode"])
    pp = z["perpatch"]                                        # [n, 3, G, G]
    vmin, vmax = (float(q) for q in np.quantile(pp, [0.05, 0.97]))   # ONE scale for all panels of this dataset
    has_dec = "decoded" in z.files
    ncol, left, right, gap, vgap = 5, 0.34, 0.02, 0.035, 0.035
    cw = (WIDTH - left - right - gap * (ncol - 1)) / ncol
    ch = cw * asp
    head, cbar_h, dec_head = 0.19, 0.27, 0.15
    block = n * ch + (n - 1) * vgap
    height = head + block + cbar_h + ((dec_head + block) if has_dec else 0) + 0.02
    fig = plt.figure(figsize=(WIDTH, height))

    def axes(y_top, c):   # y_top in inches from the top of the figure
        x = left + c * (cw + gap)
        return fig.add_axes([x / WIDTH, 1 - (y_top + ch) / height, cw / WIDTH, ch / height])

    def colx(c):
        return (left + c * (cw + gap) + cw / 2) / WIDTH

    def fy(y):
        return 1 - y / height

    def heads(y, items):
        for c, t, col in items:
            fig.text(colx(c), fy(y - 0.045), t, ha="center", va="bottom", fontsize=6.6, color=col, fontweight="bold")

    heads(head, [(0, "observed $t$", mf.INK), (1, "true $t{+}10$", mf.INK)]
          + [(2 + c, f"{LABEL[a]} error", COL[a]) for c, a in enumerate(SHOW)])
    fig.text(0.02 / WIDTH, fy(head - 0.045), "gain", ha="left", va="bottom", fontsize=5.8, color=mf.MUTED, style="italic")
    y = head
    for r in range(n):
        true = native(z["frame_true"][r], asp)
        ax = axes(y, 0); image(ax, native(z["frame_obs"][r], asp)); frame(ax)
        ax = axes(y, 1); image(ax, true); frame(ax)
        bg = desat(true)
        for c, arm in enumerate(SHOW):
            i = arms.index(arm)
            ax = axes(y, 2 + c); image(ax, bg)
            ax.imshow(np.clip((pp[r, i] - vmin) / (vmax - vmin), 0, 1), cmap=ERR, vmin=0, vmax=1,
                      extent=(0, 1, 1, 0), interpolation="bicubic", aspect="auto")
            ax.set_xlim(0, 1); ax.set_ylim(1, 0)
            frame(ax, COL["shiftwm"] if arm == "shiftwm" else None, 2.0)
            badge(ax, f"{float(z['err'][r, i]):.2f}", COL[arm])
        # row label: example number and the relative advantage 1 - err_S / min(err_D, err_AR) used for ranking
        fig.text(0.02 / WIDTH, fy(y + ch / 2), f"#{r + 1}\n{100 * float(z['adv'][r]):.0f}%", ha="left", va="center",
                 fontsize=6.0, color=mf.INK, linespacing=1.2)
        y += ch + vgap
    y -= vgap
    # shared colour scale under the three error columns; its legend text to the left of it
    x0, x1 = left + 2 * (cw + gap), left + 4 * (cw + gap) + cw
    cax = fig.add_axes([x0 / WIDTH, fy(y + 0.10), (x1 - x0) / WIDTH, 0.05 / height])
    cax.imshow(np.full((1, 2, 3), 0.93), aspect="auto", extent=(vmin, vmax, 0, 1))
    cax.imshow(np.linspace(0, 1, 256)[None], cmap=ERR, aspect="auto", extent=(vmin, vmax, 0, 1), vmin=0, vmax=1)
    cax.set_yticks([]); cax.tick_params(axis="x", labelsize=5.4, length=1.5, pad=1)
    for s in cax.spines.values():
        s.set_linewidth(0.4); s.set_edgecolor("#9AA1AB")
    cax.set_xlim(vmin, vmax)
    cax.set_xticks([vmin, (vmin + vmax) / 2, vmax])
    labs = cax.set_xticklabels([f"$\\leq${vmin:.2f}", f"{(vmin + vmax) / 2:.2f}", f"$\\geq${vmax:.2f}"])
    labs[0].set_ha("left"); labs[-1].set_ha("right")
    fig.text(x0 / WIDTH - 0.01, fy(y + 0.075), "per-patch error ($k{=}10$), one scale for all panels;"
             " badge = frame mean", ha="right", va="center", fontsize=5.8, color=mf.INK)
    fig.text(x0 / WIDTH - 0.01, fy(y + 0.185), "gain $= 1 - $err$_\\mathrm{ShiftWM}\\,/\\,\\min($err$_\\mathrm{Direct}$,"
             " err$_\\mathrm{AR})$", ha="right", va="center", fontsize=5.8, color=mf.MUTED)
    y += cbar_h
    if has_dec:
        order = [str(s) for s in z["decoded_order"]]
        dec = z["decoded"]
        y += dec_head
        heads(y, [(1, "decoded truth", mf.INK)] + [(2 + c, f"decoded {LABEL[a]}", COL[a]) for c, a in enumerate(SHOW)])
        fig.text((left + cw / 2) / WIDTH, fy(y + block / 2), "decoded $k{=}10$\nforecasts\n(RGB decoder,\nillustration only)", ha="center", va="center", fontsize=6.0, color=mf.MUTED, style="italic",
                 linespacing=1.25)
        for r in range(n):
            fig.text(0.02 / WIDTH, fy(y + ch / 2), f"#{r + 1}", ha="left", va="center", fontsize=6.0, color=mf.INK)
            ax = axes(y, 1); image(ax, native(dec[r, order.index("truth")], asp)); frame(ax)
            for c, arm in enumerate(SHOW):
                ax = axes(y, 2 + c); image(ax, native(dec[r, order.index(arm)], asp))
                frame(ax, COL["shiftwm"] if arm == "shiftwm" else None, 2.0)
            y += ch + vgap
    return fig


def caption(ds, info):
    return (rf"\textbf{{{NAME[ds]}: largest-advantage windows.}} Per-patch error at $k{{=}}10$ on the same window for"
            rf" each method (shared scale); selection rule in \cref{{app:qualitative}}.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--fig-dir", default=str(mf.FIG))
    p.add_argument("--tex", default=str(GEN / "qual_best_figs.tex"))
    a = p.parse_args()
    src, figdir = Path(a.src), Path(a.fig_dir)
    figs, done = [], []
    for ds in ORDER:
        z, info = load(ds, src)
        if z is None:
            continue
        fig = make(ds, z, info)
        fig.savefig(figdir / f"qual_best_{ds}.pdf")
        fig.savefig(figdir / f"qual_best_{ds}_preview.png", dpi=220)
        plt.close(fig)
        done.append(ds)
        figs.append("\n".join([r"\begin{figure}[tp]", r"  \centering",
                               rf"  \includegraphics[width=0.88\linewidth]{{qual_best_{ds}.pdf}}",
                               rf"  \caption{{{caption(ds, info)}}}",
                               rf"  \label{{fig:qual-best-{ds.replace('_', '-')}}}", r"\end{figure}"]))
        print("wrote", figdir / f"qual_best_{ds}.pdf", flush=True)
    tex = Path(a.tex); tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("% Generated by figures/src/make_qual_best.py (selection: scripts/v2/qual_select.py)\n"
                   + "\n".join(figs) + "\n")
    print("wrote", tex, "datasets:", done)


if __name__ == "__main__":
    main()
