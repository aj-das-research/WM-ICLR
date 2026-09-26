"""Main-text qualitative figure (round 2): one full-width float, two rows, 5.5 x 1.88 in.

Row (a) "where features move": ONE held-out DROID window (the second window of the interpret selection rule in
    results/v2/analysis/interpret/summary.json = episode droid-4321..., start 10; the one with clear arm motion), full
    frame: [observed t + learned transport arrows (source -> target, gate > 0.5, 10 strongest, one per 2x2 block) |
    gate g at k=10 (amber opacity rises with g) | true t+10 | Direct error | ShiftWM error drop = Direct - ShiftWM].
    Error and drop use one scale (make_interpret's vmax, same units); the drop map is diverging (green = ShiftWM
    lower, violet = ShiftWM higher, transparent = equal), so patches where ShiftWM is worse stay visible.
    Transport and gate recomputed from the seed-0 ShiftWM checkpoint on CPU (make_interpret.transport_for).
    Badges: frame-mean error (Direct) and frame-mean ShiftWM error with relative change.
Row (b) "where the arm / instrument ends up": two decoded k=10 examples x [true t+10 | AR | Direct | ShiftWM], same
    data, display rule, crop rule and badge definition as paper/.../make_segments.py decoded_row()/examples_figure()
    (results/v2/analysis/segments/<ds>/decoded_examples.npz; each method's OWN decoded forecast segmented with
    Grounding-DINO + SAM 2.1; badge = distance between the centroids of the drawn predicted and true masks, in
    display px; "no arm"/"no instrument" = no segment). Examples: DROID example 2 and Hamlyn example 1 of
    examples_figure() (the rows where ShiftWM places the target best), i.e. selected, illustrative windows.
Method colours: Direct blue, ShiftWM green, AR orange (paper palette).

Usage (repo root): PYTHONPATH=src .venv/bin/python reviews/main_figs_round2/make_fig_qualitative.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper/submission_folder/figures/src"))
import make_interpret as MI  # noqa: E402  (helpers only; main() not called, paper/ not written)
import make_segments as SG  # noqa: E402  (helpers only; main() not called)

mf = MI.mf
GREEN, BLUE, ORANGE, ERR = mf.METHODS["shiftwm"][1], mf.METHODS["direct"][1], mf.METHODS["ar"][1], MI.ERR
FS = mf.FS_NOTE
W = 5.5
LM = 0.0                                  # left margin (in)
GAP, WGAP = 0.025, 0.13                   # gap between tiles / between the two examples of a row
TILE = round((W - 2 * LM - 6 * GAP - WGAP) / 8, 4)
HEAD, COLT, RGAP = 0.135, 0.125, 0.07     # row heading, column titles, gap between rows
FRAME = "#C9CED6"
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
_g = matplotlib.colors.to_rgb(GREEN); _v = matplotlib.colors.to_rgb("#7E57C2"); _y = matplotlib.colors.to_rgb("#E69F00")
_r = (0.80, 0.18, 0.12)
# error: transparent at 0, light-to-deep warm red with alpha rising to 0.8 (low error reads clean)
ERR_CMAP = LinearSegmentedColormap.from_list("err2", [(*_r, 0.0), (*_r, 0.45), ((0.55, 0.05, 0.08), 0.8)])
# reduction: violet (ShiftWM higher) <- transparent at 0 -> green (ShiftWM lower); same units/scale as the error
RED_CMAP = LinearSegmentedColormap.from_list("red2", [(0, (*_v, 0.8)), (0.5, (*_v, 0.0)), (0.5, (*_g, 0.0)),
                                                       (1, (0.0, 0.45, 0.33, 0.9))])
# gate: transparent at 0 -> amber at 1 (the paper's gate colour)
# (opacity rises faster above g = 0.5 so that "mostly moved" separates from "mostly kept"; hue fixed)
GATE_CMAP = LinearSegmentedColormap.from_list("gate2", [(0.0, (*_y, 0.0)), (0.5, (*_y, 0.12)), (1.0, (*_y, 0.9))])


NA = 5                                    # tiles per window in row (a)
TILE_A = round((W - 2 * LM - 2 * (NA - 1) * GAP - WGAP) / (2 * NA), 4)
COLT_A = 0.215                            # two-line column titles in row (a)


def xs(n=4, t=None):
    """Left edge (in) of each tile column: 2 examples x n tiles of width t."""
    t = t or TILE
    return [LM + e * (n * t + (n - 1) * GAP + WGAP) + c * (t + GAP) for e in range(2) for c in range(n)]


def tile_ax(fig, H, col, ybot, title=None, tcol=None, n=4, t=None, weight="bold"):
    t = t or TILE
    x = xs(n, t)[col]
    a = fig.add_axes([x / W, ybot / H, t / W, t / H])
    if title:
        fig.text((x + t / 2) / W, (ybot + t + 0.022) / H, title, ha="center", va="bottom", fontsize=FS,
                 fontweight=weight, color=tcol or mf.INK, linespacing=0.95)
    return a


def badge(ax, text, fc, bold=True, color="white"):
    ax.text(0.95, 0.05, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=FS, color=color,
            fontweight="bold" if bold else "normal", bbox=dict(fc=fc, ec="none", alpha=0.9, pad=0.8))


def tag(ax, text):
    ax.text(0.04, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=FS, color="white",
            fontweight="bold", bbox=dict(fc="#1F2A37", ec="none", alpha=0.75, pad=0.7))


WIN = 1                                   # second window of the interpret selection rule (clearest arm motion)
NA = 5
AW = round((W - 2 * LM - (NA - 1) * 0.03) / NA, 4)   # full-frame tiles (16:9) in row (a)


def row_anatomy(fig, H, ybot, led):
    """One held-out DROID window, full frame: [observed t + transport | gate g | true t+10 | Direct error |
    error drop of ShiftWM]. Error and drop share one scale (vmax of make_interpret, same units); the drop map is
    diverging: green = ShiftWM lower, violet = ShiftWM higher, transparent = equal."""
    Z = np.load(MI.D / "droid.npz", allow_pickle=True)
    E_d, E_s = Z["err_di"][:2], Z["err_sw"][:2]
    vmax = float(np.quantile(np.concatenate([E_d.ravel(), E_s.ravel()]), 0.97))     # identical to make_interpret
    ep, st = str(Z["episode"][WIN]), int(Z["start"][WIN])
    obs, fut = mf.droid_frames(ep, steps=(st + 2, st + 2 + 10)); h, w = fut.shape[:2]
    AH = round(AW * h / w, 4)
    ed, es = E_d[WIN].reshape(16, 16), E_s[WIN].reshape(16, 16)
    ext = (-0.5, w - 0.5, h - 0.5, -0.5)
    heads = ["observed $t$: content moves", "moved vs. kept (gate $g$)", "true $t{+}10$",
             "error: Direct", "error drop: ShiftWM"]
    tcols = [mf.INK, mf.INK, mf.INK, BLUE, GREEN]
    axs = []
    for c in range(NA):
        x = LM + c * (AW + 0.03)
        ax = fig.add_axes([x / W, ybot / H, AW / W, AH / H]); axs.append(ax)
        fig.text((x + AW / 2) / W, (ybot + AH + 0.022) / H, heads[c], ha="center", va="bottom", fontsize=FS,
                 fontweight="bold", color=tcols[c])
    dx, dy, gate = MI.transport_for(ep, st)
    ax = axs[0]; ax.imshow(obs, aspect="auto", interpolation="lanczos")
    mag = np.hypot(dx, dy); sx, sy = w / 16, h / 16
    sel = [(i, j) for i in range(16) for j in range(16) if gate[i, j] > 0.5 and mag[i, j] > 0.35]
    sel, cand = [], sorted(sel, key=lambda ij: -mag[ij])
    for ij in cand:                                   # strongest first, at most one arrow per 2x2 patch block
        if all(max(abs(ij[0] - q[0]), abs(ij[1] - q[1])) >= 2 for q in sel):
            sel.append(ij)
        if len(sel) == 10:
            break
    for i, j in sel:
        tx, ty = (j + 0.5) * sx, (i + 0.5) * sy
        a_ = ax.annotate("", xy=(tx, ty), xytext=(tx + dx[i, j] * sx, ty + dy[i, j] * sy), annotation_clip=True,
                         arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.15", color=GREEN, lw=0.9,
                                         shrinkA=0, shrinkB=0, clip_on=True))
        a_.arrow_patch.set_path_effects([pe.Stroke(linewidth=1.9, foreground="white"), pe.Normal()])
        a_.arrow_patch.set_clip_box(ax.bbox)
    ax = axs[1]; ax.imshow(MI.gray(obs), aspect="auto", interpolation="lanczos")
    ax.imshow(np.clip(gate, 0, 1), cmap=GATE_CMAP, vmin=0, vmax=1, extent=ext, interpolation="bicubic", aspect="auto")
    axs[2].imshow(fut, aspect="auto", interpolation="lanczos")
    ax = axs[3]; ax.imshow(MI.gray(fut), aspect="auto", interpolation="lanczos")
    ax.imshow(np.clip(ed / vmax, 0, 1), cmap=ERR_CMAP, vmin=0, vmax=1, extent=ext, interpolation="bicubic",
              aspect="auto")
    ax = axs[4]; ax.imshow(MI.gray(fut), aspect="auto", interpolation="lanczos")
    ax.imshow(np.clip((ed - es) / vmax, -1, 1), cmap=RED_CMAP, vmin=-1, vmax=1, extent=ext, interpolation="bicubic",
              aspect="auto")
    for c, ax in enumerate(axs):
        ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
        MI.frame_axes(ax, {3: BLUE, 4: GREEN}.get(c, FRAME), 1.1 if c >= 3 else 0.6)
    vd, vs = float(ed.mean()), float(es.mean())
    badge(axs[3], f"{vd:.2f}", BLUE)
    badge(axs[4], f"{vs:.2f} ({100 * (vs / vd - 1):+.0f}%)".replace("-", "\u2212"), GREEN)
    # tiny key for the two diverging/sequential scales, inside the gate tile
    led["anatomy"] = {"vmax": vmax, "episode": ep, "start": st, "n_arrows": len(sel),
                      "frame_err": {"Direct": vd, "ShiftWM": vs}, "frame_rel_change_pct": 100 * (vs / vd - 1),
                      "gate_quantiles_0_50_100": np.quantile(gate, [0, .5, 1]).tolist(),
                      "frac_patches_shiftwm_higher": float((ed - es < 0).mean())}
    return AH


def row_segments(fig, H, ybot, led, picks):
    led["segments"] = []
    order = ["ar", "direct", "shiftwm"]
    heads = {"ar": ("AR", ORANGE), "direct": ("Direct", BLUE), "shiftwm": ("ShiftWM", GREEN)}
    for e, (ds, Z, b, label) in enumerate(picks):
        names = list(Z["names"])
        fut, m0, mk = Z["fut"][b], Z["mask_t"][b].astype(bool), Z["mask_true"][b].astype(bool)
        segs = {n: Z["seg"][b][names.index(n)].astype(bool) for n in order}
        tc = SG._centroid(mk)
        errs = {}
        for n in order:
            pc = SG._centroid(segs[n])
            errs[n] = float("nan") if pc is None or tc is None else float(np.linalg.norm(pc - tc))
        h, w = fut.shape[:2]
        union = m0 | mk
        for n in order:
            union |= segs[n]
        top, left, side = SG.crop_box(union, mk, h, w, margin=0.12)        # identical to decoded_row

        def finish(ax, colr=FRAME, lw=0.6):
            ax.set_xlim(left - 0.5, left + side - 0.5); ax.set_ylim(top + side - 0.5, top - 0.5)
            MI.frame_axes(ax, colr, lw)
        col = 4 * e
        ax = tile_ax(fig, H, col, ybot, "true $t{+}10$"); ax.imshow(fut, aspect="equal")
        SG.overlay(ax, mk * 0.30, "white"); SG.contour(ax, mk.astype(float), "white", 1.0, ls=(0, (2.2, 1.5)))
        finish(ax); tag(ax, label)
        finite = [errs[n] for n in order if np.isfinite(errs[n])]
        for c, n in enumerate(order):
            t, colr = heads[n]
            ax = tile_ax(fig, H, col + 1 + c, ybot, t, colr if n != "ar" else ORANGE)
            ax.imshow(Z["decoded"][b][names.index(n)], aspect="equal")
            SG.overlay(ax, segs[n] * 0.45, colr); SG.contour(ax, segs[n].astype(float), colr, 1.1)
            SG.contour(ax, mk.astype(float), "white", 0.9, ls=(0, (2.2, 1.5)))
            pc = SG._centroid(segs[n])
            if pc is not None and tc is not None and np.linalg.norm(pc - tc) > 4:
                a = ax.annotate("", xy=tuple(tc), xytext=tuple(pc), zorder=5,
                                arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.17", color="white",
                                                lw=1.0, shrinkA=1.5, shrinkB=1.5))
                a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.3, foreground="#1F2A37"), pe.Normal()])
            finish(ax, colr, 1.2 if n != "ar" else 1.0)
            word = SG.WORD[ds]
            txt = f"{errs[n]:.0f} px" if np.isfinite(errs[n]) else f"no {word}"
            best = np.isfinite(errs[n]) and round(errs[n]) == round(min(finite))
            badge(ax, txt, colr if best else "#1F2A37", bold=best)
        led["segments"].append({"dataset": ds, "episode": str(Z["episode"][b]), "start": int(Z["start"][b]),
                                "label": label, "centroid_dist_px": errs,
                                "iou": {n: float(Z["iou"][b][names.index(n)]) for n in order}})


def picks():
    """Rows of make_segments.examples_figure() (same filters and order); keep DROID 2 and Hamlyn 1."""
    rows, cnt = {}, {}
    for ds in ("droid", "openh_hamlyn"):
        Z = SG.load_decoded(ds); names = list(Z["names"])
        for r in range(len(Z["episode"])):
            tc = SG._centroid(Z["mask_true"][r].astype(bool)); d = {}
            for a in ("shiftwm", "direct", "ar"):
                pc = SG._centroid(Z["seg"][r][names.index(a)].astype(bool))
                d[a] = np.inf if pc is None or tc is None else float(np.linalg.norm(pc - tc))
            if d["shiftwm"] < min(d["direct"], d["ar"]):
                cnt[ds] = cnt.get(ds, 0) + 1
                rows[(ds, cnt[ds])] = (ds, Z, r)
    want = [(("droid", 2), "DROID"), (("openh_hamlyn", 1), "Hamlyn")]
    return [rows[k] + (lab,) for k, lab in want]


def main():
    y_leg = 0.0
    rowa, rowh = HEAD + COLT + AW * 180 / 320, HEAD + COLT + TILE
    H = round(rowa + rowh + RGAP + 0.035, 3)
    fig = plt.figure(figsize=(W, H))
    led = {"size_in": [W, H], "tile_in": TILE}
    yb2 = y_leg + 0.005
    yb1 = yb2 + rowh + RGAP
    row_anatomy(fig, H, yb1, led)
    row_segments(fig, H, yb2, led, picks())
    heads = [(yb1, "(a) Where features move",
              "held-out DROID, $k{=}10$; error and drop on one scale; value: frame mean"),
             (yb2, "(b) Where the arm or instrument ends up",
              "fill: segment in each model's decoded $k{=}10$ forecast   dashed: true   value: centroid distance")]
    for (yb, t, sub), tt, ct in zip(heads, (AW * 180 / 320, TILE), (COLT, COLT)):
        yt = yb + tt + ct + 0.015
        fig.text(LM / W, yt / H, t, fontsize=mf.FS_TITLE, fontweight="bold", color=mf.INK, va="bottom", ha="left")
        fig.text((W - LM) / W, yt / H, sub, fontsize=FS, color=mf.MUTED, va="bottom", ha="right")
    mf.qa(fig, "fig_qualitative", display_width=W)
    fig.savefig(HERE / "fig_qualitative.pdf"); fig.savefig(HERE / "fig_qualitative.png", dpi=400)
    (HERE / "ledger_qualitative.json").write_text(json.dumps(led, indent=1))
    print(json.dumps(led, indent=1))


if __name__ == "__main__":
    main()
