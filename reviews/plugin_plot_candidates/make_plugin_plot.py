"""Candidate plots to replace Table 2 (tab:plugin): ShiftWM as a plug-in head for V-JEPA 2-AC and DINO-WM.

Values come from the same generator as the table rows (scripts/v2/make_tables.py: vjepa_rows / dinowm_rows), which
read results/v2/external/vjepa2ac_plugin/*/[s*/]test_summary.json and
results/v2/external/dinowm_plugin/{pusht,wall}/{dinowm,dinowm_shiftwm}/openloop.json (teacher_forced).

Each row: relative change of the raw metric vs. the matched baseline, 100*(head/base - 1).
  V-JEPA 2-AC: baseline = fine-tuned (same data, schedule, loss, seeds 0/1); zero-shot drawn as a hollow marker
               relative to the same fine-tuned baseline. Skill (% of persistence error removed) is given as text.
  DINO-WM:     baseline = DINO-WM, official code, 1 seed.
Colour = better (green, ShiftWM colour) / worse (red, LOSS) in the metric's own direction (arrow in the label).
Values beyond the axis (Wall LPIPS, +194%) are clipped at the edge with a break mark and their number printed.

Usage (repo root): PYTHONPATH=src python reviews/plugin_plot_candidates/make_plugin_plot.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper/submission_folder/figures/src"))
import make_figures as mf  # noqa: E402  (palette, rcParams, type scale, qa)
sys.path.insert(0, str(ROOT / "scripts/v2"))
import make_tables as T  # noqa: E402

OUT = Path(__file__).parent
GOOD, BAD, ZS = mf.METHODS["shiftwm"][1], mf.LOSS, mf.BACKBONE
FS = mf.FS_NOTE
MINUS = "−"


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def fmt_pct(v):
    return f"{v:+.0f}%".replace("-", MINUS) if abs(v) >= 9.95 else f"{v:+.1f}%".replace("-", MINUS)


def load():
    _, vj = T.vjepa_rows()
    _, dw = T.dinowm_rows()
    zs, ft, hd = vj["zeroshot"], vj["finetune"], vj["finetune_shiftwm"]
    vrows = []
    for k, name in (("mse", "MSE"), ("moving", "moving"), ("static", "static")):
        vrows.append(dict(name=name + r" $\downarrow$", base=ft[k], head=hd[k], zs=zs[k], hib=False))
    groups = [dict(title="V-JEPA 2-AC, DROID", rows=vrows,
                   skill=(zs["skill"], ft["skill"], hd["skill"]))]
    for env, name in (("pusht", "PushT"),):
        b, o = dw[env]["dinowm"], dw[env]["dinowm_shiftwm"]
        rows = [dict(name=r"latent err. $\downarrow$", base=b["err"], head=o["err"], hib=False),
                dict(name=r"SSIM $\uparrow$", base=b["ssim"], head=o["ssim"], hib=True),
                dict(name=r"LPIPS $\downarrow$", base=b["lpips"], head=o["lpips"], hib=False)]
        groups.append(dict(title=f"DINO-WM, {name}", rows=rows))
    for g in groups:
        for r in g["rows"]:
            r["d"] = pct(r["head"], r["base"])
            r["better"] = (r["d"] > 0) if r["hib"] else (r["d"] < 0)
            if "zs" in r:
                r["dz"] = pct(r["zs"], r["base"])
    return groups


def draw_row(a, y, r, lo, hi, dy, show_zs=True):
    """Dot-arrow from the baseline (0) to the head; hollow marker for zero-shot."""
    c = GOOD if r["better"] else BAD
    d = r["d"]
    clipped = d > hi
    x_end = min(d, hi - 0.02 * (hi - lo))
    a.annotate("", xy=(x_end, y), xytext=(0, y),
               arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=c, lw=1.2,
                               shrinkA=0, shrinkB=0), zorder=3)
    if clipped:  # break mark: two short slashes before the axis edge
        for dx in (-0.055, -0.03):
            xb = hi + dx * (hi - lo) - 0.02 * (hi - lo)
            a.plot([xb - 0.006 * (hi - lo), xb + 0.006 * (hi - lo)], [y - dy, y + dy], color="white", lw=2.2,
                   zorder=4, solid_capstyle="butt")
            a.plot([xb - 0.006 * (hi - lo), xb + 0.006 * (hi - lo)], [y - dy, y + dy], color=c, lw=0.6, zorder=5)
    if show_zs and "dz" in r:
        a.scatter([r["dz"]], [y], s=13, facecolors="white", edgecolors=ZS, linewidths=0.8, zorder=4)
    a.scatter([0], [y], s=7, color=ZS, zorder=4, marker="|", linewidths=1.0)


# ------------------------------------------------------------------------------------------------ helpers
def pct_axis(a, lo, hi, ticks=(-20, 0, 20, 40)):
    a.patch.set_alpha(0)
    a.set_xlim(lo, hi); a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.grid(axis="y", visible=False); a.grid(axis="x", color="#EEF0F3", lw=0.5)
    a.axvline(0, color=ZS, lw=0.6, zorder=1)
    a.set_xticks(list(ticks)); a.set_xticklabels([f"{t:d}".replace("-", MINUS) for t in ticks])
    a.tick_params(labelsize=mf.FS_TICK, length=2, pad=1)


def key(cv, x, y, base_label="baseline"):
    """Legend line: hollow = zero-shot, tick = baseline, arrow = + ShiftWM head. Returns right edge (in)."""
    cv.scatter([x + 0.04], [y], s=13, facecolors="white", edgecolors=ZS, linewidths=0.8)
    cv.text(x + 0.10, y, "zero-shot", fontsize=FS, va="center", color=mf.INK)
    x2 = x + 0.60
    cv.plot([x2, x2], [y - 0.035, y + 0.035], color=ZS, lw=1.0)
    cv.text(x2 + 0.05, y, base_label, fontsize=FS, va="center", color=mf.INK)
    x3 = x2 + 0.10 + 0.047 * len(base_label)
    cv.annotate("", xy=(x3 + 0.20, y), xytext=(x3, y),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=GOOD, lw=1.2, shrinkA=0,
                                shrinkB=0))
    cv.text(x3 + 0.24, y, "+ ShiftWM head", fontsize=FS, va="center", color=GOOD, fontweight="bold")


# ------------------------------------------------------------------------------------------------ variant A
def variant_a(groups):
    """Half-width wrapfigure: one shared axis, groups stacked, base->head values and % change in a right column."""
    W = 2.7
    pitch, ghead, top, bot = 0.112, 0.13, 0.15, 0.30
    n_rows = sum(len(g["rows"]) for g in groups) + 1          # + skill text row
    H = top + n_rows * pitch + len(groups) * ghead + bot
    fig = plt.figure(figsize=(W, H))
    cv = fig.add_axes([0, 0, 1, 1]); cv.set_xlim(0, W); cv.set_ylim(0, H); cv.axis("off")
    xl, xa, wa, xt = 0.02, 0.60, 0.98, W - 0.02
    xv = xt - 0.36                                            # right edge of "base -> head" text
    lo, hi = -22.0, 48.0
    key(cv, 0.0, H - 0.065)
    y = H - top
    rows_y, skill_y = [], None
    for g in groups:
        y -= ghead
        cv.text(xl, y + 0.005, g["title"], fontsize=FS, fontweight="bold", color=mf.INK, va="center")
        cv.text(xt, y + 0.005, ("fine-tuned" if "skill" in g else "DINO-WM") + r"$\rightarrow$" + "head",
                fontsize=FS, color=mf.MUTED, va="center", ha="right")
        cv.plot([xl, xt], [y - ghead / 2 + 0.012] * 2, color=mf.GRID, lw=0.5)
        for r in g["rows"]:
            y -= pitch
            rows_y.append((y, r))
            c = GOOD if r["better"] else BAD
            cv.text(xl + 0.05, y, r["name"], fontsize=FS, va="center", color=mf.INK)
            cv.text(xv, y, f"{r['base']:.3f}" + r"$\rightarrow$" + f"{r['head']:.3f}", fontsize=FS, va="center",
                    ha="right", color=mf.INK)
            cv.text(xt, y, fmt_pct(r["d"]), fontsize=FS, va="center", ha="right", color=c, fontweight="bold")
        if "skill" in g:                                      # skill = % of persistence error removed (text row)
            y -= pitch
            s = g["skill"]
            cv.text(xl + 0.05, y, r"skill (%) $\uparrow$", fontsize=FS, va="center", color=mf.INK)
            cv.text(xa + (0 - lo) / (hi - lo) * wa + 0.05, y, f"zero-shot: {s[0]:.1f}", fontsize=FS, va="center", color=ZS, style="italic")
            cv.text(xv, y, f"{s[1]:.1f}" + r"$\rightarrow$" + f"{s[2]:.1f}", fontsize=FS, va="center", ha="right",
                    color=mf.INK)
            cv.text(xt, y, f"+{s[2] - s[1]:.1f}pp", fontsize=FS, va="center", ha="right", color=GOOD,
                    fontweight="bold")
    ylo, yhi = y - pitch / 2, H - top - ghead / 2
    a = fig.add_axes([xa / W, ylo / H, wa / W, (yhi - ylo) / H]); pct_axis(a, lo, hi); a.set_ylim(ylo, yhi)
    a.set_xlabel("change vs. baseline (%)", fontsize=FS, labelpad=1)
    for yy, r in rows_y:
        draw_row(a, yy, r, lo, hi, dy=0.03)
    return fig


# ------------------------------------------------------------------------------------------------ variant B
def variant_b(groups):
    """Full width, short: three side-by-side panels sharing one axis; % change and baseline value beside each row."""
    W, H = 5.5, 1.13
    fig = plt.figure(figsize=(W, H))
    cv = fig.add_axes([0, 0, 1, 1]); cv.set_xlim(0, W); cv.set_ylim(0, H); cv.axis("off")
    lo, hi = -22.0, 48.0
    lab_w, pw, txt_w, gap = 0.56, 0.82, 0.36, 0.05
    x = 0.02
    ybot, ytop = 0.29, H - 0.28
    for gi, g in enumerate(groups):
        n = len(g["rows"])
        cv.text(x, H - 0.03, g["title"], fontsize=mf.FS_LABEL, fontweight="bold", color=mf.INK, va="top")
        if "skill" in g:
            s = g["skill"]
            sub = f"base: fine-tuned; skill {s[0]:.1f}/{s[1]:.1f}/" + r"$\mathbf{" + f"{s[2]:.1f}" + r"}$%"
        else:
            sub = "base: DINO-WM (1 seed)"
        cv.text(x, H - 0.145, sub, fontsize=FS, color=mf.MUTED, va="top")
        xa = x + lab_w
        a = fig.add_axes([xa / W, ybot / H, pw / W, (ytop - ybot) / H]); pct_axis(a, lo, hi)
        a.set_ylim(n - 0.5, -0.5)
        for i, r in enumerate(g["rows"]):
            yin = ytop - (i + 0.5) * (ytop - ybot) / n
            c = GOOD if r["better"] else BAD
            cv.text(x, yin, r["name"], fontsize=FS, va="center", color=mf.INK)
            draw_row(a, i, r, lo, hi, dy=0.3)
            cv.text(xa + pw + 0.03, yin + 0.045, fmt_pct(r["d"]), fontsize=FS, va="center", color=c,
                    fontweight="bold")
            cv.text(xa + pw + 0.03, yin - 0.05, f"{r['base']:.3f}", fontsize=FS, va="center", color=mf.MUTED)
        x = xa + pw + txt_w + gap
    cv.text(W / 2 + 0.1, 0.045, "change vs. baseline (%)", fontsize=FS, color=mf.INK, ha="center", va="center")
    key(cv, 0.02, 0.045)
    cv.text(W - 0.02, 0.045, "grey: baseline value;  green better, red worse", fontsize=FS, color=mf.MUTED,
            ha="right", va="center")
    return fig


def save(fig, name, width):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    print(name, "size (in):", fig.get_figwidth(), "x", round(fig.get_figheight(), 3))
    mf.qa(fig, name, display_width=width)
    plt.close(fig)


if __name__ == "__main__":
    G = load()
    for g in G:
        for r in g["rows"]:
            print(f"{g['title']:<22} {r['name']:<24} base={r['base']:.4f} head={r['head']:.4f} d={r['d']:+.2f}%"
                  + (f" zs={r['dz']:+.2f}%" if "dz" in r else ""))
    save(variant_a(G), "plugin_plot_A_wrap", 2.7)
    save(variant_b(G), "plugin_plot_B_strip", 5.5)
