"""Largest-advantage qualitative figures for the plug-in settings (appendix), in the style of make_qual_best.py.

Inputs (results/v2/analysis/qual_best/):
  dinowm_<env>.{npz,json}   scripts/external_dinowm_plugin/qual_best.py  (DINO-WM with / without the ShiftWM head)
  vjepa2ac_droid.{npz,json} scripts/v2/qual_best_vjepa.py               (V-JEPA 2-AC fine-tuned with / without the head)
Outputs: figures/qual_best_dinowm_<env>.pdf, figures/qual_best_vjepa2ac.pdf (+ _preview.png), 5.5 in wide, and
  tables/generated/qual_best_plugin_figs.tex (one figure environment per finished input) and
  tables/generated/qual_best_plugin_macros.tex (win rates / margins quoted in the text).
One row per selected window: observed frame t | true frame t+k | per model: forecast, per-patch error.
  DINO-WM : forecast = the model's own decoder applied to its step-k latent (RGB);
  V-JEPA  : forecast = shared-PCA RGB of the predicted tokens (one PCA per window, fit on the true tokens of t and t+k,
            applied identically to the truth and both forecasts; an extra column shows the true tokens).
  error   = per-patch squared error of the latent forecast (the benchmark's metric) as a red heat map on a desaturated
            copy of the true frame, ONE colour scale per figure; badge = the window's MSE (the selection metric).
The head's panels get a green frame. Wall (where the head raises error) shows the best AND the worst rollouts.
Usage (repo root): python paper/submission_folder/figures/src/make_qual_best_plugin.py
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402
from make_qual_best import ERR, badge, desat, frame, image  # noqa: E402

SRC = mf.RES / "analysis/qual_best"
GEN = mf.ROOT / "paper/submission_folder/tables/generated"
WIDTH = 5.5
BASE_C, HEAD_C = mf.METHODS["direct"][1], mf.METHODS["shiftwm"][1]
PX = 640


def resize(img, aspect):
    return np.asarray(Image.fromarray(img).resize((PX, int(round(PX * aspect))), Image.LANCZOS))


def pca_rgb(fit, *grids, g=16):
    """3-component PCA fit on `fit` [M,C]; every grid [N,C] mapped with the same projection and scaling."""
    mu = fit.mean(0)
    _, _, vt = np.linalg.svd(fit - mu, full_matrices=False)
    proj = [(x - mu) @ vt[:3].T for x in grids]
    lo, hi = np.percentile(np.concatenate(proj[:1] + [(fit - mu) @ vt[:3].T]), [1, 99], axis=0)
    return [(np.clip((p - lo) / (hi - lo + 1e-8), 0, 1).reshape(g, g, 3) * 255).astype(np.uint8) for p in proj]


def draw(rows, cols, aspect, vmin, vmax, k, errlabel, row_w=0.36):
    """rows: list of dicts {label, panels: [(kind, payload, color|None, badge_text|None, badge_color)]}.
    cols: list of (header, color, group) ; groups get an extra gap between them."""
    n, ncol = len(rows), len(cols)
    left, right, gap, ggap, vgap = row_w, 0.02, 0.03, 0.09, 0.035
    ngg = sum(1 for c in range(1, ncol) if cols[c][2] != cols[c - 1][2])
    cw = (WIDTH - left - right - gap * (ncol - 1 - ngg) - ggap * ngg) / ncol
    ch = cw * aspect
    head, cbar_h = 0.2, 0.34
    height = head + n * ch + (n - 1) * vgap + cbar_h
    fig = plt.figure(figsize=(WIDTH, height))
    xs, x = [], left
    for c in range(ncol):
        if c:
            x += cw + (ggap if cols[c][2] != cols[c - 1][2] else gap)
        xs.append(x)

    def fy(y):
        return 1 - y / height

    for c, (t, col, _) in enumerate(cols):
        fig.text((xs[c] + cw / 2) / WIDTH, fy(head - 0.045), t, ha="center", va="bottom", fontsize=6.4, color=col,
                 fontweight="bold")
    fig.text(0.02 / WIDTH, fy(head - 0.045), "gain", ha="left", va="bottom", fontsize=5.8, color=mf.MUTED,
             style="italic")
    y = head
    for r in rows:
        for c, (kind, img, fcol, btxt, bcol) in enumerate(r["panels"]):
            ax = fig.add_axes([xs[c] / WIDTH, fy(y + ch), cw / WIDTH, ch / height])
            if kind == "err":
                bg, pp = img
                image(ax, bg)
                ax.imshow(np.clip((pp - vmin) / (vmax - vmin), 0, 1), cmap=ERR, vmin=0, vmax=1, extent=(0, 1, 1, 0),
                          interpolation="bicubic", aspect="auto")
                ax.set_xlim(0, 1); ax.set_ylim(1, 0)
            elif kind == "grid":   # token grids: nearest, never smoothed
                ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, 1, 1, 0))
                ax.set_xlim(0, 1); ax.set_ylim(1, 0)
            else:
                image(ax, img)
            frame(ax, fcol, 2.0)
            if btxt:
                badge(ax, btxt, bcol)
        fig.text(0.02 / WIDTH, fy(y + ch / 2), r["label"], ha="left", va="center", fontsize=5.9,
                 color=r.get("label_color", mf.INK), linespacing=1.2)
        y += ch + vgap
    y -= vgap
    x0, x1 = xs[[c for c, col in enumerate(cols) if col[2] != "in"][0]], xs[-1] + cw
    cax = fig.add_axes([x0 / WIDTH, fy(y + 0.11), (x1 - x0) / WIDTH, 0.05 / height])
    cax.imshow(np.linspace(0, 1, 256)[None], cmap=ERR, aspect="auto", extent=(vmin, vmax, 0, 1), vmin=0, vmax=1)
    cax.set_yticks([]); cax.tick_params(axis="x", labelsize=5.4, length=1.5, pad=1)
    for s in cax.spines.values():
        s.set_linewidth(0.4); s.set_edgecolor("#9AA1AB")
    cax.set_xlim(vmin, vmax); cax.set_xticks([vmin, (vmin + vmax) / 2, vmax])
    labs = cax.set_xticklabels([f"$\\leq${vmin:.2f}", f"{(vmin + vmax) / 2:.2f}", f"$\\geq${vmax:.2f}"])
    labs[0].set_ha("left"); labs[-1].set_ha("right")
    fig.text(0.02 / WIDTH, fy(y + 0.11), f"per-patch {errlabel} error ($k{{=}}{k}$), one colour scale;\n"
             "badge = mean error of the frame", ha="left", va="center", fontsize=5.7, color=mf.INK, linespacing=1.15)
    fig.text(0.02 / WIDTH, fy(y + 0.245), "gain $= 1 - $err$_{+\\mathrm{head}}\\,/\\,$err$_{\\mathrm{no\\ head}}$",
             ha="left", va="center", fontsize=5.7, color=mf.MUTED)
    return fig


def scale(pp):
    return tuple(float(q) for q in np.quantile(pp, [0.05, 0.97]))


def dinowm_fig(env, z, info):
    k = int(z["k"]); kinds = [str(s) for s in z["kind"]]
    show = [i for i, s in enumerate(kinds) if s == "best"]
    if env == "wall":   # the head regresses on Wall: show its two best AND two worst rollouts
        show = show[:2] + [i for i, s in enumerate(kinds) if s == "worst"][:2]
    vmin, vmax = scale(z["perpatch"][show])
    rows, nb = [], {"best": 0, "worst": 0}
    for i in show:
        true = z["frame_true"][i]; bg = desat(true)
        nb[kinds[i]] += 1
        lab = f"#{nb[kinds[i]]}" if env != "wall" else f"{kinds[i]}\n#{nb[kinds[i]]}"
        panels = [("rgb", z["frame_obs"][i], None, None, None), ("rgb", true, None, None, None)]
        for m, col in ((0, BASE_C), (1, HEAD_C)):
            fc = HEAD_C if m == 1 else None
            panels += [("rgb", z["decoded"][i, m], fc, None, None),
                       ("err", (bg, z["perpatch"][i, m]), fc, f"{float(z['err'][i, m]):.3f}", col)]
        rows.append({"label": f"{lab}\n{100 * float(z['adv'][i]):+.0f}%".replace("-", "\u2212"), "panels": panels,
                     "label_color": mf.INK if kinds[i] == "best" else "#B03A2E"})
    cols = [("observed $t$", mf.INK, "in"), (f"true $t{{+}}{k}$", mf.INK, "in"),
            ("DINO-WM", BASE_C, "b"), ("DINO-WM error", BASE_C, "b"),
            ("+ShiftWM head", HEAD_C, "h"), ("+ShiftWM error", HEAD_C, "h")]
    return draw(rows, cols, 1.0, vmin, vmax, k, "latent")


def vjepa_fig(z, info):
    k = int(z["k"]); i0, j0, hh, ww = (int(v) for v in z["crop_box_ijhw"])
    asp = hh / ww
    vmin, vmax = scale(z["perpatch"])
    tok = z["tokens"].astype(np.float32)             # [n, 4, 256, C]: z_t, true z_t+k, no head, +head
    rows = []
    for i in range(len(z["episode"])):
        crop = lambda f: resize(f[i0:i0 + hh, j0:j0 + ww], asp)
        true = crop(z["frame_true"][i]); bg = desat(true)
        pz = pca_rgb(np.concatenate([tok[i, 0], tok[i, 1]]), tok[i, 1], tok[i, 2], tok[i, 3])
        panels = [("rgb", crop(z["frame_obs"][i]), None, None, None), ("rgb", true, None, None, None),
                  ("grid", pz[0], None, None, None)]
        for m, col in ((0, BASE_C), (1, HEAD_C)):
            fc = HEAD_C if m == 1 else None
            panels += [("grid", pz[1 + m], fc, None, None),
                       ("err", (bg, z["perpatch"][i, m]), fc, f"{float(z['err'][i, m]):.3f}", col)]
        rows.append({"label": f"#{i + 1}\n{100 * float(z['adv'][i]):+.0f}%".replace("-", "\u2212"), "panels": panels})
    cols = [("observed $t$", mf.INK, "in"), (f"true $t{{+}}{k}$", mf.INK, "in"), ("true tokens", mf.INK, "in"),
            ("V-JEPA 2-AC", BASE_C, "b"), ("error", BASE_C, "b"), ("+ShiftWM head", HEAD_C, "h"), ("error", HEAD_C, "h")]
    return draw(rows, cols, asp, vmin, vmax, k, "token")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--fig-dir", default=str(mf.FIG))
    p.add_argument("--gen", default=str(GEN))
    a = p.parse_args()
    src, figdir, gen = Path(a.src), Path(a.fig_dir), Path(a.gen)
    figs, macros = [], []
    jobs = [("vjepa2ac_droid", "qual_best_vjepa2ac", "vjepa"), ("dinowm_pusht", "qual_best_dinowm_pusht", "pusht"),
            ("dinowm_wall", "qual_best_dinowm_wall", "wall")]
    for stem, out, key in jobs:
        if not (src / f"{stem}.npz").exists():
            continue
        z = np.load(src / f"{stem}.npz"); info = json.loads((src / f"{stem}.json").read_text())
        fig = vjepa_fig(z, info) if key == "vjepa" else dinowm_fig(key, z, info)
        fig.savefig(figdir / f"{out}.pdf"); fig.savefig(figdir / f"{out}_preview.png", dpi=220); plt.close(fig)
        ctx = info["context_all"]
        win = ctx.get("frac_windows_shiftwm_best", ctx.get("frac_rollouts_shiftwm_better"))
        tag = {"vjepa": "Vjepa", "pusht": "Pusht", "wall": "Wall"}[key]
        macros.append(rf"\newcommand{{\qbWin{tag}}}{{{100 * win:.0f}}}")
        k = info["k"]
        if key == "vjepa":
            cap = (rf"\textbf{{V-JEPA~2-AC on DROID: largest-gain test windows}} (selected, not typical; the head lowers"
                   rf" error on \qbWinVjepa\% of windows). Token forecasts at $k{{=}}{k}$ in shared-PCA RGB, with"
                   rf" per-patch error; rule in \cref{{app:qualitative}}.")
        elif key == "pusht":
            cap = (rf"\textbf{{DINO-WM on PushT: largest-gain open-loop rollouts}} (selected; the head lowers error on"
                   rf" \qbWinPusht\% of rollouts). Each model's own decoder at step $k{{=}}{k}$ from the same start and"
                   rf" actions; per-patch latent error.")
        else:
            cap = (rf"\textbf{{DINO-WM on Wall: best and worst rollouts}} for the head, which raises error here (it wins"
                   rf" only \qbWinWall\% of rollouts). Layout as in \cref{{fig:qual-best-dinowm-pusht}}.")
        figs.append("\n".join([r"\begin{figure}[tp]", r"  \centering",
                               rf"  \includegraphics[width=\linewidth]{{{out}.pdf}}", rf"  \caption{{{cap}}}",
                               rf"  \label{{fig:{out.replace('_', '-')}}}", r"\end{figure}"]))
        print("wrote", figdir / f"{out}.pdf", flush=True)
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "qual_best_plugin_macros.tex").write_text("% Generated by figures/src/make_qual_best_plugin.py\n"
                                                     + "\n".join(macros) + "\n")
    (gen / "qual_best_plugin_figs.tex").write_text("% Generated by figures/src/make_qual_best_plugin.py\n"
                                                   + "\n".join(figs) + "\n")


if __name__ == "__main__":
    main()
