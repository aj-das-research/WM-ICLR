"""Compact redesigns of Figure 6 (fig:ablations). Review candidates only; nothing in paper/ is written.

  A  fig6A_ablations_half.pdf     2.7 in wide, half-width wrapfigure: grouped Delta lollipops + seed-noise band + rank dots
  B  fig6B_composite.pdf          5.5 in wide composite: (a) = A | (b) gain over Direct vs. horizon, 4 datasets, test
                                   | (c) gain over Direct and AR per task (7 Open-H + 3 IWS), test
  C  fig6C_ablation_map.pdf       2.7 in wide: each ablation as a point in (Delta val. error, action-ranking accuracy)

Data (never hand-typed):
  ablations: scripts/v2/make_tables.py ablation_data()  (results/v2s/droid/dinov2s/.../summary.json, val split, seed 0)
  horizon / tasks: make_tables.load() (results/v2s/<ds>/dinov2s/<arm>/s*/eval_test.npz, seed mean), the same loader and
  root rule as Table 1 and fig:horizon. Gain = 100 (1 - err_ShiftWM / err_baseline), as in the text macros.
  Bands: 95% paired bootstrap (4000 resamples), DROID resamples recording sessions, others episodes; IWS macro =
  equal-weight mean of the three tasks, resampled within task (as make_tables.iws_ci_pct).
Seed noise: the full model's seed range on DROID validation is 0.2% (std 0.1%); the band is +-0.3%, the threshold the
caption uses ("gaps below about 0.3% are within seed noise"). NOISE below is the only hand-set constant.

Usage (repo root): PYTHONPATH=src python reviews/fig6_candidates/make_fig6_candidates.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper/submission_folder/figures/src"))
import make_figures as mf  # noqa: E402  (sets the shared rcParams / palette)
sys.path.insert(0, str(ROOT / "scripts/v2"))
import make_tables as T  # noqa: E402

OUT = Path(__file__).resolve().parent
M = mf.METHODS
GREEN, BLUE, ORANGE = M["shiftwm"][1], M["direct"][1], M["ar"][1]
NOISE = 0.3                     # +-% band = seed noise threshold stated in the caption / text
FS = mf.FS_NOTE                 # 6.2 pt
N_BOOT = 4000

GROUP_ORDER = ["Actions", "Transport", "Memory", "Correction"]
SHORT = {  # compact names; same runs as make_tables.ABLATIONS
    "direct": "none (= Direct)", "w1": "$w{=}1$ (cannot move)", "w5": "$w{=}5$", "global": "global (768 cand.)",
    "s1": "$S{=}1$ (last frame)", "tanh": "tanh-bounded", "nocorr": "none (pure transport)",
    "actfree": "none (action-free)", "ctr": "+ contrastive loss", "dinov2b_shiftwm": "DINOv2-B/14",
}
SHORT_MAP = {  # labels for the 2-D map (C)
    "direct": "Direct (no transport)", "w1": "$w{=}1$ (no move)", "w5": "$w{=}5$", "global": "global window",
    "s1": "$S{=}1$ (last frame)", "tanh": "tanh corr. (hidden under ShiftWM)", "nocorr": "no corr.", "ctr": "+contrastive",
}


def tag(r):
    return r["rel"].split("/")[-2]


def abl_rows():
    rows = T.ablation_data()
    full = next(r for r in rows if r["full"])
    var = [r for r in rows if not r["full"]]
    grouped = []
    for g in GROUP_ORDER + sorted({r["grp"] for r in var} - set(GROUP_ORDER)):
        grouped.append((g, sorted([r for r in var if r["grp"] == g], key=lambda r: -r["d"])))
    return full, [(g, rs) for g, rs in grouped if rs]


def dcolor(r):
    if tag(r) == "direct":
        return BLUE
    if r["d"] > 1.0:                 # clearly worse (as in the current figure)
        return mf.LOSS
    if r["d"] < -1.0:
        return mf.INK
    return mf.MUTED


def fmt(v):
    return f"{v:+.1f}".replace("-", "−")


# ------------------------------------------------------------------------------------------------ A: half width
def draw_ablation_list(fig, x0, y0, W, H, title=None, rank=True):
    """Grouped lollipop list drawn into the inch box (x0, y0, W, H) of `fig`. Returns nothing."""
    FW, FH = fig.get_figwidth(), fig.get_figheight()
    ax = fig.add_axes([x0 / FW, y0 / FH, W / FW, H / FH]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    full, groups = abl_rows()
    pitch, gap = 0.104, 0.035
    top = H - (0.2 if title else 0.03) - 0.13        # below the header row
    y = top - pitch / 2
    ent = [(None, full, y)]
    y -= pitch + gap
    for g, rs in groups:
        for i, r in enumerate(rs):
            ent.append((g if i == 0 else "", r, y)); y -= pitch
        y -= gap
    ybot = ent[-1][2] - pitch / 2
    xg, xl = 0.02, 0.45
    xa0, xa1 = (1.38, 2.22) if rank else (1.38, W - 0.02)
    xb0, xb1 = 2.32, W - 0.03
    if title:
        ax.text(0, H, title, fontsize=mf.FS_TITLE, fontweight="bold", color=mf.INK, va="top")
    hy = top + 0.015
    ax.text((xa0 + xa1) / 2, hy, "$\\Delta$ val. error (%)", fontsize=FS, color=mf.INK, ha="center", va="bottom")
    if rank:
        ax.text((xb0 + xb1) / 2, hy, "rank acc.", fontsize=FS, color=mf.INK, ha="center", va="bottom")
    # group separators
    for g, r, yy in ent:
        if g:
            ax.plot([0, W], [yy + pitch / 2 + gap / 2] * 2, color="#E3E6EA", lw=0.5, zorder=0)
    ax.add_patch(Rectangle((0, ent[0][2] - pitch / 2), W, pitch, fc="#E8F5EF", ec="none", zorder=0))
    for g, r, yy in ent:
        if r["full"]:
            ax.text(xg, yy, "ShiftWM (full model)", fontsize=FS, color=GREEN, fontweight="bold", va="center")
            continue
        if g:
            ax.text(xg, yy, g.lower(), fontsize=FS, color=mf.MUTED, va="center", style="italic")
        ax.text(xl, yy, SHORT.get(tag(r), r["label"]), fontsize=FS, color=mf.INK, va="center")
    # Delta panel
    a = fig.add_axes([(x0 + xa0) / FW, (y0 + ybot) / FH, (xa1 - xa0) / FW, (top - ybot) / FH])
    a.patch.set_alpha(0)
    ds = [r["d"] for _, rs in groups for r in rs]
    lo, hi = min(-2.2, min(ds) - 0.8), max(ds) + 3.6
    a.set_xlim(lo, hi); a.set_ylim(ybot, top); a.set_yticks([])
    a.axvspan(-NOISE, NOISE, color="#DADDE2", lw=0, zorder=0)
    a.axvline(0, color=mf.INK, lw=0.6, zorder=1)
    a.grid(axis="y", visible=False); a.grid(axis="x", color="#EEF0F3", lw=0.5)
    a.spines["left"].set_visible(False)
    a.tick_params(labelsize=mf.FS_TICK, length=2, pad=1)
    a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    a.set_xticklabels([]) if False else None
    for g, r, yy in ent:
        if r["full"]:
            a.scatter([0], [yy], s=18, color=GREEN, zorder=3, edgecolors="white", linewidths=0.5)
            a.text(NOISE + 0.5, yy, "grey band = seed noise", fontsize=FS, color=mf.MUTED, va="center", ha="left")
            continue
        c = dcolor(r)
        a.plot([0, r["d"]], [yy, yy], color=c, lw=1.1, alpha=0.55, solid_capstyle="butt", zorder=2)
        a.scatter([r["d"]], [yy], s=13 if tag(r) != "direct" else 15, color=c, zorder=3,
                  marker="D" if tag(r) == "direct" else "o", edgecolors="white", linewidths=0.3)
        if r["d"] >= 0:
            a.text(max(r["d"], NOISE) + 0.6, yy, fmt(r["d"]), fontsize=FS, va="center", ha="left", color=c)
        else:
            a.text(NOISE + 0.4, yy, fmt(r["d"]), fontsize=FS, va="center", ha="left", color=c)
    a.set_xlabel("worse $\\rightarrow$", fontsize=FS, labelpad=0.5, loc="right")
    if not rank:
        return
    b = fig.add_axes([(x0 + xb0) / FW, (y0 + ybot) / FH, (xb1 - xb0) / FW, (top - ybot) / FH])
    b.patch.set_alpha(0)
    rk = [r["rank"] for _, r, _ in ent if r["rank"] is not None]
    b.set_xlim(np.floor(min(rk)) - 0.25, np.ceil(max(rk)) + 0.25); b.set_ylim(ybot, top); b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.grid(axis="y", visible=False); b.grid(axis="x", color="#EEF0F3", lw=0.5)
    b.tick_params(labelsize=mf.FS_TICK, length=2, pad=1)
    b.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    b.axvline(full["rank"], color=GREEN, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for g, r, yy in ent:
        if r["rank"] is None:
            b.text(np.mean(b.get_xlim()), yy, "n/a", fontsize=FS, color=mf.MUTED, ha="center", va="center")
            continue
        c = GREEN if r["full"] else (BLUE if tag(r) == "direct" else mf.INK)
        b.scatter([r["rank"]], [yy], s=11, color=c, zorder=3, marker="D" if tag(r) == "direct" else "o",
                  edgecolors="white", linewidths=0.3)
    b.set_xlabel("better $\\rightarrow$", fontsize=FS, labelpad=0.5, loc="right")


def cand_A():
    W, H = 2.7, 1.62
    fig = plt.figure(figsize=(W, H))
    draw_ablation_list(fig, 0.0, 0.0, W, H)
    mf.qa(fig, "fig6A", W)
    save(fig, "fig6A_ablations_half")


# ------------------------------------------------------------------------------------------------ B: composite
def _groups(ds, episodes):
    if not ds.startswith("droid"):
        return np.arange(len(episodes))
    s = T._sessions(ds) or T._sessions("droid")
    g = [s.get(e, e) for e in episodes]; u = sorted(set(g))
    return np.array([u.index(x) for x in g])


def gain_ci(s, b, gi, rng, n=N_BOOT):
    """s, b: [E, K] (or [E]) per-episode error; gi: resampling unit per episode. Gain of s over b with 95% CI."""
    s2, b2 = np.atleast_2d(s.T).T, np.atleast_2d(b.T).T
    G = gi.max() + 1
    S = np.stack([np.bincount(gi, s2[:, k], G) for k in range(s2.shape[1])], 1)
    B = np.stack([np.bincount(gi, b2[:, k], G) for k in range(b2.shape[1])], 1)
    idx = rng.integers(0, G, (n, G))
    boots = 100 * (1 - S[idx].sum(1) / B[idx].sum(1))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return 100 * (1 - S.sum(0) / B.sum(0)), lo, hi


def horizon_curves(base="direct"):
    rng = np.random.default_rng(0); out = []
    for ds, lab in (("droid", "DROID"), ("language_table", "Language-Table"), ("openh_hamlyn", "Open-H"),):
        s, b = T.load(ds, "shiftwm"), T.load(ds, base)
        assert list(s["episodes"]) == list(b["episodes"])
        out.append((lab, s["seeds"]) + gain_ci(s["mse"], b["mse"], _groups(ds, s["episodes"]), rng))
    # IWS: equal-weight macro over tasks, bootstrap within task
    parts = [(T.load(t, "shiftwm")["mse"], T.load(t, base)["mse"]) for t in T.IWS_TASKS]
    K = parts[0][0].shape[1]
    mean = 100 * (1 - np.mean([p[0].mean(0) for p in parts], 0) / np.mean([p[1].mean(0) for p in parts], 0))
    sb, bb = np.zeros((N_BOOT, K)), np.zeros((N_BOOT, K))
    for s, b in parts:
        idx = rng.integers(0, len(s), (N_BOOT, len(s)))
        sb += s[idx].mean(1) / len(parts); bb += b[idx].mean(1) / len(parts)
    lo, hi = np.percentile(100 * (1 - sb / bb), [2.5, 97.5], axis=0)
    out.append(("IWS (3 tasks)", 1, mean, lo, hi))
    return out


HAM_TASKS = [("knot_tying", "knot tying"), ("needle_grasp_and_handover", "needle hand."), ("peg_transfer", "peg transfer"),
             ("suturing_1", "suturing 1"), ("suturing_2", "suturing 2"), ("tissue_lifting", "tissue lift"),
             ("tissue_retraction", "retraction")]
IWS_NAMES = {"iws_pusht": "PushT", "iws_box": "Box", "iws_rope": "Rope"}


def task_gains():
    rng = np.random.default_rng(0); rows = []
    ev = {a: T.load("openh_hamlyn", a) for a in ("shiftwm", "direct", "ar")}
    t = np.array(ev["shiftwm"]["tasks"])
    for key, lab in HAM_TASKS:
        m = t == key; r = {"label": lab, "grp": "Open-H"}
        for a in ("direct", "ar"):
            g, lo, hi = gain_ci(ev["shiftwm"]["mse"][m].mean(1), ev[a]["mse"][m].mean(1), np.arange(m.sum()), rng)
            r[a] = (g[0], lo[0], hi[0])
        rows.append(r)
    for ds in T.IWS_TASKS:
        e = {a: T.load(ds, a) for a in ("shiftwm", "direct", "ar")}; r = {"label": IWS_NAMES[ds], "grp": "IWS"}
        for a in ("direct", "ar"):
            g, lo, hi = gain_ci(e["shiftwm"]["mse"].mean(1), e[a]["mse"].mean(1), np.arange(len(e[a]["mse"])), rng)
            r[a] = (g[0], lo[0], hi[0])
        rows.append(r)
    return rows


# every line in (b)/(c) is a ShiftWM gain, so all are green shades (dark -> light) + distinct markers + direct labels;
# no method colour (Direct blue, AR orange, ...) is reused for a dataset
DS_STYLE = {"DROID": ("#00513B", "-", "o"), "Language-Table": ("#009E73", (0, (4, 1.5)), "^"),
            "Open-H": ("#4DB894", "-", "s"), "IWS (3 tasks)": ("#8A8F98", (0, (1.5, 1.2)), "D")}
DS_SHORT = {"Language-Table": "Lang.-Table", "IWS (3 tasks)": "IWS"}


def draw_horizon(fig, box):
    FW, FH = fig.get_figwidth(), fig.get_figheight()
    x0, y0, w, h = box
    a = fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])
    a.tick_params(labelsize=mf.FS_TICK, length=2, pad=1)
    ends = []
    for lab, seeds, g, lo, hi in horizon_curves():
        c, ls, mk = DS_STYLE[lab]; k = np.arange(1, len(g) + 1)
        a.fill_between(k, lo, hi, color=c, alpha=0.15, lw=0)
        a.plot(k, g, color=c, ls=ls, lw=1.1, marker=mk, ms=2.3, markevery=[0, len(g) - 1])
        ends.append([lab, len(g), float(g[-1]), c])
    a.axhline(0, color=mf.INK, lw=0.6)
    a.set_xlim(0.5, 12.5); a.set_xticks([1, 4, 8, 12]); a.set_ylim(0, 10)
    a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    a.set_xlabel("forecast step $k$", fontsize=FS, labelpad=0.5)
    a.set_ylabel("error below Direct (%)", fontsize=FS, labelpad=1)
    ends.sort(key=lambda e: e[2]); gap = 1.05
    for i in range(1, len(ends)):
        ends[i][2] = max(ends[i][2], ends[i - 1][2] + gap)
    for lab, kx, yv, c in ends:
        a.annotate(DS_SHORT.get(lab, lab), (kx, yv), xytext=(2.5, 0), textcoords="offset points", fontsize=FS,
                   color=c, va="center", annotation_clip=False)
    return a


def draw_tasks(fig, box):
    """Gain over Direct per task (mean over k), 95% CI; colours = dataset colours of (b)."""
    FW, FH = fig.get_figwidth(), fig.get_figheight()
    x0, y0, w, h = box
    rows = task_gains()
    lab_w = 0.5
    a = fig.add_axes([(x0 + lab_w) / FW, y0 / FH, (w - lab_w) / FW, h / FH])
    a.tick_params(labelsize=mf.FS_TICK, length=2, pad=1)
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r["grp"] != rows[i - 1]["grp"]:
            y += 0.7
        ys.append(y); y += 1
    ys = np.array(ys)
    for r, yy in zip(rows, ys):
        c, _, mk = DS_STYLE["Open-H" if r["grp"] == "Open-H" else "IWS (3 tasks)"]
        g, lo, hi = r["direct"]
        a.plot([lo, hi], [yy, yy], color=c, lw=0.9, alpha=0.8, solid_capstyle="butt")
        a.scatter([g], [yy], s=11, color=c, marker=mk, zorder=3, edgecolors="white", linewidths=0.3)
    a.axvline(0, color=mf.INK, lw=0.6)
    a.set_ylim(ys[-1] + 0.7, -0.7); a.set_yticks(ys); a.set_yticklabels([r["label"] for r in rows], fontsize=mf.FS_TICK)
    a.grid(axis="y", visible=False); a.tick_params(axis="y", length=0, pad=1.5)
    hi_all = max(r["direct"][2] for r in rows)
    a.set_xlim(-0.6, np.ceil(hi_all + 0.5)); a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    a.set_xlabel("below Direct (%)", fontsize=FS, labelpad=0.5)
    for grp, key in (("Open-H", "Open-H"), ("IWS", "IWS (3 tasks)")):
        sel = [i for i, r in enumerate(rows) if r["grp"] == grp]
        i_free = min(sel, key=lambda i: rows[i]["direct"][2])        # row whose CI ends furthest left: room for the tag
        a.text(0.97, ys[i_free], grp, transform=a.get_yaxis_transform(), fontsize=FS,
               color=DS_STYLE[key][0], ha="right", va="center", style="italic")
    return a


def cand_B():
    W, H = 5.5, 1.8
    fig = plt.figure(figsize=(W, H))
    draw_ablation_list(fig, 0.0, 0.0, 2.7, H - 0.02, title="(a) Ablations, DROID val. (seed 0)")
    ttl = (H - 0.02) / H
    top_pad, bot = 0.33, 0.27
    xb = 2.97
    mf.panel_title(fig, (xb - 0.2) / W, ttl, "(b) Across horizons", transform=fig.transFigure)
    draw_horizon(fig, (xb, bot, 0.98, H - bot - top_pad))
    xc = 4.33
    mf.panel_title(fig, (xc + 0.05) / W, ttl, "(c) Across tasks", transform=fig.transFigure)
    draw_tasks(fig, (xc, bot, W - xc - 0.04, H - bot - top_pad))
    fig.text((xb - 0.2) / W, (H - 0.155) / H, "ShiftWM vs. Direct, test, 95% CI", fontsize=FS, color=mf.MUTED, va="top")
    mf.qa(fig, "fig6B", W)
    save(fig, "fig6B_composite")



# ------------------------------------------------------------------------------------------------ C: 2-D map
MAP_OFF = {  # label offsets in points (dx, dy, ha, va)
    "direct": (-5, 3, "right", "bottom"), "w1": (-5, 0, "right", "center"), "s1": (-5, 0, "right", "center"),
    "nocorr": (4, 0, "left", "center"), "w5": (4, 0, "left", "center"), "global": (0, -5, "center", "top"),
}
MAP_LEADER = {  # crowded labels near the origin: text placed at a data position, thin leader to the point
    "tanh": (0.75, 93.25), "ctr": (0.85, 92.66),
}


def cand_C():
    W, H = 2.7, 1.75
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([0.33 / W, 0.29 / H, (W - 0.38) / W, (H - 0.44) / H])
    a.tick_params(labelsize=mf.FS_TICK, length=2, pad=1)
    full, groups = abl_rows()
    var = [r for _, rs in groups for r in rs]
    ranked = [r for r in var if r["rank"] is not None]
    a.axvspan(-NOISE, NOISE, color="#DADDE2", lw=0, zorder=0)
    a.axvline(0, color=mf.MUTED, lw=0.5, zorder=1); a.axhline(full["rank"], color=GREEN, lw=0.5, ls=(0, (2, 2)), zorder=1)
    a.scatter([0], [full["rank"]], s=30, color=GREEN, zorder=4, edgecolors="white", linewidths=0.6)
    a.annotate("ShiftWM (full)", (0, full["rank"]), xytext=(-6, 3), textcoords="offset points", fontsize=FS, color=GREEN,
               fontweight="bold", ha="right", va="bottom")
    for r in ranked:
        c = dcolor(r); mk = "D" if tag(r) == "direct" else "o"
        a.scatter([r["d"]], [r["rank"]], s=16, color=c, marker=mk, zorder=3, edgecolors="white", linewidths=0.4)
        if tag(r) in MAP_LEADER:
            a.annotate(SHORT_MAP.get(tag(r), tag(r)), (r["d"], r["rank"]), xytext=MAP_LEADER[tag(r)], fontsize=FS,
                       color=c, ha="left", va="center",
                       arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=1, shrinkB=2))
            continue
        dx, dy, ha, va = MAP_OFF.get(tag(r), (4, 0, "left", "center"))
        a.annotate(SHORT_MAP.get(tag(r), tag(r)), (r["d"], r["rank"]), xytext=(dx, dy), textcoords="offset points",
                   fontsize=FS, color=c, ha=ha, va=va)
    xs = [r["d"] for r in var]
    a.set_xlim(min(xs) - 1.0, max(r["d"] for r in ranked) + 0.7)
    rk = [r["rank"] for r in ranked] + [full["rank"]]
    a.set_ylim(min(rk) - 0.35, max(rk) + 0.35)
    a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    a.set_ylabel("action-ranking acc. (%)", fontsize=FS, labelpad=1)
    a.set_xlabel("$\\Delta$ val. error vs. ShiftWM (%)  worse $\\rightarrow$", fontsize=FS, labelpad=0.5)
    # action-free: no ranking defined -> strip below the axis
    af = [r for r in var if r["rank"] is None]
    for r in af:   # no actions -> no ranking accuracy; stated, not plotted
        a.text(0.98, 0.97, f"off the map: action-free,\n{fmt(r['d'])}%, cannot rank actions", transform=a.transAxes,
               fontsize=FS, color=dcolor(r), ha="right", va="top", linespacing=1.1)
    a.text(0.02, 0.03, "grey band: seed noise", transform=a.transAxes, fontsize=FS, color=mf.MUTED, ha="left", va="bottom")
    fig.text(0.02 / W, (H - 0.02) / H, "Ablations, DROID validation (seed 0)", fontsize=mf.FS_TITLE, fontweight="bold",
             color=mf.INK, va="top")
    mf.qa(fig, "fig6C", W)
    save(fig, "fig6C_ablation_map")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    only = sys.argv[1:] or ["A", "B", "C"]
    for k in only:
        {"A": cand_A, "B": cand_B, "C": cand_C}[k]()
