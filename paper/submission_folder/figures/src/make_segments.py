"""Segmentation view of the forecasts (results/v2/analysis/segments/<ds>, from scripts/v2/segments.py).

Main figure (figures/segments.pdf): one zoomed region-of-interest row for DROID and one for Hamlyn (the 90th-percentile
motion window of each; fixed rule), IoU-vs-horizon curves on moving windows, and the horizon-averaged IoU gain of
ShiftWM over Direct / AR with paired 95% CIs for every dataset evaluated so far.
Appendix figures (figures/segments_<ds>.pdf): both example windows (90th / 50th percentile motion) + curves.
Row layout: full observed frame t (reference mask at t, true future outline dashed, crop box) | crop of frame t with
the ShiftWM transport arrows | crop of the true frame t+K with its tracked SAM 2.1 mask | ShiftWM / Direct / AR: soft
foreground probability read out of the k=K forecast (heat) + its decision contour, true future mask dashed white,
IoU in the corner (best in green bold).
Also writes tables/generated/segments_numbers.tex (macros) and tables/generated/segments_rows.tex (appendix table).
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

D = mf.RES / "analysis/segments"
GEN = mf.ROOT / "paper/submission_folder/tables/generated"
SAMC = "#F2A900"          # reference-mask colour
G = 16
ORDER = ["droid", "openh_hamlyn", "bridge", "fractal", "iws_pusht", "iws_box", "iws_rope", "plan_pusht", "plan_reacher",
         "plan_tworoom"]
NAME = {"droid": "DROID", "openh_hamlyn": "Hamlyn", "bridge": "Bridge", "fractal": "RT-1", "iws_pusht": "IWS Push-T",
        "iws_box": "IWS Box", "iws_rope": "IWS Rope", "plan_pusht": "Push-T (sim)", "plan_reacher": "Reacher",
        "plan_tworoom": "Two-Room"}
MAC = {"droid": "Droid", "openh_hamlyn": "Hamlyn", "bridge": "Bridge", "fractal": "Fractal", "iws_pusht": "IwsPusht",
       "iws_box": "IwsBox", "iws_rope": "IwsRope", "plan_pusht": "PlanPusht", "plan_reacher": "PlanReacher",
       "plan_tworoom": "PlanTworoom"}
LEARN = ("shiftwm", "direct", "ar")
BRIGHT_GREEN = "#7CF2C4"


def summaries():
    out = {}
    for ds in ORDER:
        p = D / ds / "summary.json"
        out[ds] = json.loads(p.read_text()) if p.exists() else {"status": "pending"}
    return out


def smooth(score, h, w):
    """16x16 patch score -> [h,w] bicubic spline through the patch centres (the encoder squashes the frame)."""
    yc = (np.arange(G) + 0.5) * h / G - 0.5; xc = (np.arange(G) + 0.5) * w / G - 0.5
    return RectBivariateSpline(yc, xc, score.reshape(G, G), kx=3, ky=3)(np.arange(h), np.arange(w))


def overlay(ax, alpha_map, color):
    rgba = np.zeros(alpha_map.shape + (4,)); rgba[..., :3] = to_rgb(color); rgba[..., 3] = np.clip(alpha_map, 0, 1)
    ax.imshow(rgba, aspect="equal", interpolation="antialiased")


def contour(ax, field, color, lw, ls="-", level=0.5):
    if field.min() < level < field.max():
        ax.contour(field, levels=[level], colors=[color], linewidths=lw, linestyles=[ls])


def corner(ax, text, color="white", bold=False, fs=6.2):
    ax.text(0.96, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", bbox=dict(fc="#1F2A37", ec="none", alpha=0.75, pad=0.8))


def crop_box(m0, mk, h, w, margin=0.18):
    ys, xs = np.nonzero(m0 | mk)
    if len(ys) == 0:
        return 0, 0, min(h, w)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(y1 - y0, x1 - x0) * (1 + 2 * margin)
    side = int(np.clip(side, min(h, w) * 0.35, min(h, w)))
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    top = int(np.clip(cy - side / 2, 0, h - side)); left = int(np.clip(cx - side / 2, 0, w - side))
    return top, left, side


def arrows(ax, off, gate, sel, h, w, box, n=6):
    """Source (query + expected source offset) -> query, for the strongest-moving patches inside the crop."""
    top, left, side = box
    sx, sy = w / G, h / G
    mag = np.linalg.norm(off, axis=-1) * gate
    chosen = []
    for i in np.argsort(-mag):
        yi, xi = divmod(i, G)
        qx, qy = (xi + 0.5) * sx, (yi + 0.5) * sy
        if not sel[i] or mag[i] < 0.4 or not (left <= qx <= left + side and top <= qy <= top + side):
            continue
        if all(max(abs(yi - divmod(j, G)[0]), abs(xi - divmod(j, G)[1])) >= 2 for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    for i in chosen:
        yi, xi = divmod(i, G)
        qx, qy = (xi + 0.5) * sx, (yi + 0.5) * sy
        a = ax.annotate("", xy=(qx, qy), xytext=(qx + off[i, 0] * sx, qy + off[i, 1] * sy),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.18", color=mf.METHODS["shiftwm"][1],
                                        lw=1.2, shrinkA=0, shrinkB=0))
        a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])


def roi_row(fig, gs, Z, b, titles=True, label=None):
    names = list(Z["names"]); H = int(Z["history"]); K = Z["score"].shape[2]; tau = float(Z["tau"])
    frames, masks = Z["frames"][b], Z["masks"][b]
    obs, fut = frames[H - 1], frames[H - 1 + K]
    h, w = obs.shape[:2]
    m0, mk = masks[0].astype(bool), masks[K].astype(bool)
    top, left, side = crop_box(m0, mk, h, w)
    ious = {n: float(Z["iou"][b][names.index(n)][K - 1]) for n in names}
    learned = [n for n in LEARN if n in names]
    best = max(learned, key=lambda n: ious[n])
    axes = []
    # full frame thumbnail
    ax = fig.add_subplot(gs[0]); ax.imshow(obs, aspect="equal")
    overlay(ax, m0 * 0.35, SAMC); contour(ax, m0.astype(float), SAMC, 0.6)
    contour(ax, mk.astype(float), "white", 0.6, ls=(0, (2, 1.4)))
    ax.add_patch(Rectangle((left - 0.5, top - 0.5), side, side, fill=False, ec="white", lw=0.9))
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5); axes.append(ax)
    if label:
        ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="white",
                fontweight="bold", bbox=dict(fc="#1F2A37", ec="none", alpha=0.75, pad=0.8))

    def crop(a):
        a.set_xlim(left - 0.5, left + side - 0.5); a.set_ylim(top + side - 0.5, top - 0.5)
    # observed crop + transport
    ax = fig.add_subplot(gs[1]); ax.imshow(obs, aspect="equal")
    overlay(ax, m0 * 0.3, SAMC); contour(ax, m0.astype(float), SAMC, 0.7)
    sel = Z["score"][b][names.index("shiftwm")][K - 1] > 0
    arrows(ax, Z["offsets"][b][K - 1], Z["gate"][b][K - 1], sel, h, w, (top, left, side))
    crop(ax); axes.append(ax)
    # true future crop
    ax = fig.add_subplot(gs[2]); ax.imshow(fut, aspect="equal")
    overlay(ax, mk * 0.38, SAMC); contour(ax, mk.astype(float), SAMC, 0.8)
    corner(ax, f"moved {100 * float(Z['motion'][b]):.0f}%")
    crop(ax); axes.append(ax)
    for c, n in enumerate(learned):
        ax = fig.add_subplot(gs[3 + c]); ax.imshow(fut, aspect="equal")
        col = mf.METHODS[n][1]
        f = smooth(Z["score"][b][names.index(n)][K - 1], h, w)
        p = 1 / (1 + np.exp(-f / tau))
        overlay(ax, p * 0.6, col); contour(ax, f, col, 1.1, level=0.0)
        contour(ax, mk.astype(float), "white", 0.8, ls=(0, (2.2, 1.5)))
        corner(ax, f"IoU {ious[n]:.2f}", color=BRIGHT_GREEN if n == best else "white", bold=n == best)
        crop(ax); axes.append(ax)
    for c, a in enumerate(axes):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if titles:
            t = ["observed $t$", "$t$ + transport", f"true $t{{+}}{K}$ (SAM 2.1)", "ShiftWM (ours)", "Direct", "AR"][c]
            a.set_title(t, fontsize=6.9, pad=2.2, color=mf.METHODS["shiftwm"][1] if c == 3 else mf.INK,
                        fontweight="bold")
    return axes


def iou_curve(ax, R, sub, title, ylabel=True):
    K = len(R[sub]["shiftwm"]["mean"]); k = np.arange(1, K + 1)
    for n in ("persistence", "ar", "direct", "shiftwm"):
        if n not in R[sub]:
            continue
        lab, col, ls, mk = mf.METHODS[n]
        r = R[sub][n]
        ax.plot(k, r["mean"], color=col, ls=ls, marker=mk, markevery=3, ms=2.6, lw=1.7 if n == "shiftwm" else 1.1,
                label="ShiftWM" if n == "shiftwm" else lab.split(" (")[0], zorder=3 if n == "shiftwm" else 2)
        ax.fill_between(k, r["lo"], r["hi"], color=col, alpha=0.13, lw=0)
    ax.set_xticks([1, 5, K]); ax.set_xlim(0.7, K + 0.3)
    ax.set_xlabel("forecast step $k$", fontsize=6.6, labelpad=1); ax.tick_params(labelsize=6.1, pad=1.5)
    if ylabel:
        ax.set_ylabel("IoU", fontsize=6.6, labelpad=1.5)
    ax.set_title(title, fontsize=6.9, pad=2.5, loc="left")


def gain_panel(ax, S, sub="moving"):
    """Horizon-averaged IoU gain of ShiftWM over Direct and AR, paired 95% CI over episodes, per dataset."""
    done = [ds for ds in ORDER if S[ds].get("status") == "done"]
    y = np.arange(len(done))[::-1]
    for off, n in ((0.14, "direct"), (-0.14, "ar")):
        col, mk = mf.METHODS[n][1], mf.METHODS[n][3]
        for yy, ds in zip(y, done):
            r = S[ds]["results"][S[ds]["primary_labeller"]][sub]
            if n not in r:
                continue
            d = r[n]["avg"]["diff_sw_minus"]
            sig = d["lo"] > 0 or d["hi"] < 0
            ax.errorbar(100 * d["mean"], yy + off, xerr=[[100 * (d["mean"] - d["lo"])], [100 * (d["hi"] - d["mean"])]],
                        fmt=mk, ms=3.2, color=col, mfc=col if sig else "white", lw=1.0, capsize=1.5, capthick=0.8)
    ax.axvline(0, color=mf.MUTED, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([NAME[d] for d in done], fontsize=6.3)
    ax.tick_params(labelsize=6.1, pad=1.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("IoU gain of ShiftWM (points)", fontsize=6.6, labelpad=1)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=mf.METHODS[n][1], marker=mf.METHODS[n][3], ms=3.2, lw=0) for n in ("direct", "ar")]
    ax.legend(hs, ["vs Direct", "vs AR"], fontsize=5.6, loc="best", handletextpad=0.2, borderaxespad=0.2,
              labelspacing=0.2, title="filled: CI excludes 0", title_fontsize=5.4)


def load_examples(ds):
    return np.load(D / ds / "examples.npz", allow_pickle=True)


def dataset_figure(ds, S):
    Z = load_examples(ds); R = S["results"][S["primary_labeller"]]
    asp = Z["frames"][0].shape[2] / Z["frames"][0].shape[1]
    wr = [asp, 1, 1, 1, 1, 1]
    fig = plt.figure(figsize=(5.5, 2.95))
    for r in range(2):
        g = fig.add_gridspec(1, 6, left=0.005, right=0.995, top=0.93 - r * 0.285, bottom=0.93 - r * 0.285 - 0.25,
                             wspace=0.035, width_ratios=wr)
        q = int(round(100 * float(Z["quantiles"][r])))
        roi_row(fig, [g[0, i] for i in range(6)], Z, r, titles=r == 0, label=f"{q}th pct. motion")
    bot = fig.add_gridspec(1, 3, left=0.075, right=0.995, top=0.29, bottom=0.085, wspace=0.3, width_ratios=[1, 1, 0.75])
    a1 = fig.add_subplot(bot[0]); iou_curve(a1, R, "all", f"all windows ({R['all']['windows']})")
    a2 = fig.add_subplot(bot[1], sharey=a1); iou_curve(a2, R, "moving", f"moving windows ({R['moving']['windows']})", False)
    al = fig.add_subplot(bot[2]); al.set_axis_off()
    hs, ls = a1.get_legend_handles_labels()
    idx = [ls.index(x) for x in ("ShiftWM", "Direct", "AR", "Persistence") if x in ls]
    al.legend([hs[i] for i in idx], [ls[i] for i in idx], loc="center left", fontsize=6.4, borderaxespad=0)
    fig.savefig(mf.FIG / f"segments_{ds}.pdf"); fig.savefig(mf.FIG / f"segments_{ds}_preview.png", dpi=200)
    plt.close(fig)


def main_figure(S):
    rows = [ds for ds in ("droid", "openh_hamlyn") if S[ds].get("status") == "done"]
    if not rows:
        fig, ax = plt.subplots(figsize=(5.5, 2.6)); mf.pending(ax, "segmentation view"); fig.savefig(mf.FIG / "segments.pdf"); return
    Zs = {ds: load_examples(ds) for ds in rows}
    fig = plt.figure(figsize=(5.5, 1.2 + 0.86 * len(rows)))
    H_in = fig.get_figheight()
    row_h = 0.78 / H_in; top0 = 1 - 0.17 / H_in
    for r, ds in enumerate(rows):
        Z = Zs[ds]; asp = Z["frames"][0].shape[2] / Z["frames"][0].shape[1]
        t = top0 - r * (row_h + 0.05 / H_in)
        g = fig.add_gridspec(1, 6, left=0.005, right=0.995, top=t, bottom=t - row_h, wspace=0.035,
                             width_ratios=[asp, 1, 1, 1, 1, 1])
        roi_row(fig, [g[0, i] for i in range(6)], Z, 0, titles=r == 0, label=NAME[ds])
    bot_top = 0.92 / H_in
    bot = fig.add_gridspec(1, 3, left=0.07, right=0.995, top=bot_top, bottom=0.3 / H_in, wspace=0.42,
                           width_ratios=[1, 1, 1.15])
    for c, ds in enumerate(("droid", "openh_hamlyn")):
        if S[ds].get("status") != "done":
            continue
        R = S[ds]["results"][S[ds]["primary_labeller"]]
        a = fig.add_subplot(bot[c]); iou_curve(a, R, "moving", f"({'bc'[c]}) {NAME[ds]}: IoU, moving", ylabel=c == 0)
        if c == 0:
            a.legend(fontsize=5.6, loc="upper right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.2)
    ag = fig.add_subplot(bot[2]); gain_panel(ag, S)
    ag.set_title("(d) mean gain over $k$ vs Direct / AR", fontsize=6.9, pad=2.5, loc="left")
    fig.savefig(mf.FIG / "segments.pdf"); fig.savefig(mf.FIG / "segments_preview.png", dpi=220)
    plt.close(fig)


def fmt_ci(d):
    return f"{100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


def write_tex(S):
    GEN.mkdir(parents=True, exist_ok=True)
    L = ["% generated by paper/submission_folder/figures/src/make_segments.py from results/v2/analysis/segments"]
    rows = []
    tag = {"shiftwm": "SW", "direct": "Di", "ar": "AR", "persistence": "Pers", "oracle": "Oracle"}
    for ds in ORDER:
        s = S[ds]; m = "seg" + MAC[ds]
        if s.get("status") != "done":
            rows.append(f"{NAME[ds]} & \\multicolumn{{6}}{{l}}{{\\pend{{}}}} \\\\")
            continue
        R = s["results"][s["primary_labeller"]]; K = s["horizon"]
        for sub, pre in (("all", ""), ("moving", "Mov")):
            for n, t in tag.items():
                if n not in R[sub]:
                    continue
                r = R[sub][n]
                L.append(f"\\newcommand{{\\{m}{pre}{t}Avg}}{{{r['avg']['mean']:.2f}}}")
                L.append(f"\\newcommand{{\\{m}{pre}{t}Five}}{{{r['mean'][4]:.2f}}}")
                L.append(f"\\newcommand{{\\{m}{pre}{t}Last}}{{{r['mean'][K - 1]:.2f}}}")
                if n != "shiftwm":
                    d = r["avg"]["diff_sw_minus"]
                    L.append(f"\\newcommand{{\\{m}{pre}Gain{t}}}{{{100 * d['mean']:+.1f}}}")
                    L.append(f"\\newcommand{{\\{m}{pre}Gain{t}CI}}{{[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]}}")
                    for kk, nm in ((4, "Five"), (K - 1, "Last")):
                        L.append(f"\\newcommand{{\\{m}{pre}Gain{t}{nm}}}{{{100 * r['diff_sw_minus']['mean'][kk]:+.1f}}}")
                        L.append(f"\\newcommand{{\\{m}{pre}Gain{t}{nm}CI}}{{[{100 * r['diff_sw_minus']['lo'][kk]:+.1f}, "
                                 f"{100 * r['diff_sw_minus']['hi'][kk]:+.1f}]}}")
        L.append(f"\\newcommand{{\\{m}Windows}}{{{R['all']['windows']:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}MovWindows}}{{{R['moving']['windows']:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}Episodes}}{{{R['all']['episodes']}}}")
        L.append(f"\\newcommand{{\\{m}Considered}}{{{s['windows_considered']:,}}}".replace(",", "{,}"))
        rej = s["windows_rejected"]
        for k_, v in rej.items():
            L.append(f"\\newcommand{{\\{m}Rej{''.join(p.capitalize() for p in k_.split('_'))}}}{{{v:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\{m}Rejected}}{{{sum(rej.values()):,}}}".replace(",", "{,}"))
        for l, v in s["oracle_iou_k_last_by_labeller"].items():
            L.append(f"\\newcommand{{\\{m}Oracle{l.capitalize()}}}{{{v:.2f}}}")
        first = True
        for n in ("persistence", "ar", "direct", "shiftwm"):
            if n not in R["all"]:
                continue
            cells = []
            for sub, key in (("all", "avg"), ("all", 4), ("all", K - 1), ("moving", "avg"), ("moving", K - 1)):
                r = R[sub][n]
                v = r["avg"]["mean"] if key == "avg" else r["mean"][key]
                txt = f"{v:.3f}"
                if n == "shiftwm":
                    base = [b for b in ("direct", "ar") if b in R[sub]]
                    los = [R[sub][b]["avg"]["diff_sw_minus"]["lo"] if key == "avg" else R[sub][b]["diff_sw_minus"]["lo"][key]
                           for b in base]
                    if base and min(los) > 0:
                        txt = f"\\good{{{txt}}}"
                cells.append(txt)
            lab = {"persistence": "Persistence", "ar": "AR", "direct": "Direct", "shiftwm": "\\ours{}"}[n]
            name = f"\\multirow{{4}}{{*}}{{{NAME[ds]}}}" if first else ""
            rows.append(f"{name} & {lab} & " + " & ".join(cells) + " \\\\")
            first = False
        gd = R["moving"]["direct"]["avg"]["diff_sw_minus"]
        ga = R["moving"]["ar"]["avg"]["diff_sw_minus"] if "ar" in R["moving"] else None
        rows.append(f" & \\multicolumn{{6}}{{l}}{{\\scriptsize paired gain of \\ours{{}} (moving, mean over $k$): "
                    f"vs Direct {fmt_ci(gd)}" + (f"; vs AR {fmt_ci(ga)}" if ga else "") +
                    f"; {R['all']['windows']} windows / {R['all']['episodes']} episodes}} \\\\ \\midrule")
    (GEN / "segments_numbers.tex").write_text("\n".join(L) + "\n")
    if rows and rows[-1].endswith("\\midrule"):
        rows[-1] = rows[-1][: -len(" \\midrule")]
    (GEN / "segments_rows.tex").write_text("\n".join(rows) + "\n")


def main():
    S = summaries()
    write_tex(S)
    for ds in ORDER:
        if S[ds].get("status") == "done":
            dataset_figure(ds, S[ds])
    main_figure(S)
    print("wrote segments:", [ds for ds in ORDER if S[ds].get("status") == "done"])


if __name__ == "__main__":
    main()
