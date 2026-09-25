"""Generate all data-driven paper figures from completed runs in results/v2.

Every panel is computed from logged evaluation files or real dataset frames. Panels whose
inputs do not exist yet are drawn as explicit "pending" boxes -- nothing is invented.

Usage (repo root):  PYTHONPATH=src python paper/submission_folder/figures/src/make_figures.py [--device cpu]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FIG = ROOT / "paper/submission_folder/figures"
RES = ROOT / "results/v2"

# Palette validated with the dataviz validator (light mode, all checks pass; AR-TF/Direct protan
# all-pairs dE 6.4 -> always paired with distinct markers/linestyles + direct labels).
METHODS = {  # key: (label, colour, linestyle, marker)
    "persistence": ("Persistence", "#6B7280", (0, (3, 2)), None),
    "ar_tf": ("AR-TF (DINO-WM-style)", "#AA4499", "-", "s"),
    "ar": ("AR", "#D55E00", "-", "^"),
    "direct": ("Direct", "#0072B2", "-", "D"),
    "shiftwm": ("ShiftWM (ours)", "#009E73", "-", "o"),
}
INK, MUTED, GRID = "#243447", "#8A8F98", "#E6E8EB"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "image.interpolation": "lanczos", "savefig.dpi": 300, "image.resample": True,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4, "pdf.fonttype": 42,
})


# ---- shared visual system (all paper figures) -------------------------------------------------------------------
# Type scale in points *at print size* (figures are authored at their \includegraphics width, so source pt = print pt).
FS_TITLE, FS_LABEL, FS_TICK, FS_NOTE, MIN_PT = 7.5, 6.8, 6.2, 6.2, 6.0
GATE = "#E69F00"            # gate g (a mechanism quantity, not a method)
LOSS = "#B8433A"            # "ShiftWM worse" / error regions
BACKBONE = "#4A5568"        # a published backbone without the ShiftWM head (V-JEPA 2-AC, DINO-WM)
PANEL_BG, PANEL_EDGE = "#F6F8FA", "#E1E5EA"
SKILL_SCRIPTS = Path.home() / ".claude/skills/paper-figures/scripts"


def panel_title(ax_or_fig, x, y, text, transform=None, **kw):
    """Panel heading: '(a) Finding', bold, left-aligned, FS_TITLE."""
    tgt = ax_or_fig
    kw = dict(dict(fontsize=FS_TITLE, fontweight="bold", color=INK, ha="left", va="top"), **kw)
    return tgt.text(x, y, text, transform=transform or getattr(tgt, "transAxes", None) or tgt.transFigure, **kw)


def qa(fig, name, display_width=None, min_pt=MIN_PT):
    """Paper-figures skill layout audit (text overlap, overflow, clipping, effective size) at the inserted width.
    Prints issues; math sub/superscripts are rendered smaller by mathtext and are not flagged here."""
    try:
        import sys as _s
        _s.path.insert(0, str(SKILL_SCRIPTS))
        from layout_quality import audit_figure, issue_message
    except Exception:  # noqa: BLE001 -- skill not installed: skip silently
        return []
    iss = audit_figure(fig, min_font_pt=min_pt, display_width_inches=display_width or fig.get_figwidth())
    for i in iss:
        print(f"  [qa {name}]", issue_message(i))
    print(f"  [qa {name}] {len(iss)} issue(s)")
    return iss


def pending(ax, text):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0.01,rounding_size=0.03",
                                transform=ax.transAxes, fc="white", ec=MUTED, lw=0.8, ls=(0, (3, 2))))
    ax.text(0.5, 0.5, "pending\n" + text, ha="center", va="center", color=MUTED, fontsize=6.5,
            transform=ax.transAxes, wrap=True)


ROOTS = [ROOT / "results/v2s", ROOT / "results/v2"]
LEARNED = ("ar_tf", "ar", "direct", "shiftwm")


def root_for(dataset, encoder="dinov2s", split="test"):
    """Same rule as the tables: one recipe per dataset, the first root where every learned arm has finished."""
    for base in ROOTS:
        if all(list((base / dataset / encoder / a).glob(f"s*/eval_{split}.npz")) for a in ("shiftwm", "direct", "ar")):
            return base
    return ROOTS[-1]


def load_eval(dataset, encoder, arm, split="test"):
    """Stack per-seed per-episode arrays: returns dict metric -> [seeds, episodes, K] or None."""
    base = root_for(dataset, encoder, split)
    runs = sorted((base / dataset / encoder / arm).glob("s*/eval_%s.npz" % split))
    if not runs and arm in ("persistence", "linear"):
        runs = sorted((ROOTS[-1] / dataset / encoder / arm).glob("s*/eval_%s.npz" % split))
    if arm in ("persistence", "linear"):
        runs = runs[:1]
    if not runs:
        return None
    arrs = [np.load(r, allow_pickle=True) for r in runs]
    out = {m: np.stack([a[m] for a in arrs]) for m in ("mse", "cos", "rank_ok", "sens") if m in arrs[0].files}
    out["seeds"] = len(runs)
    out["episodes"] = arrs[0]["episodes"]
    return out


# ------------------------------------------------------------------------------------------ plots
def plot_horizon(ax, dataset, encoder="dinov2s", title=None):
    drawn = False
    labels = []
    for arm, (label, color, ls, marker) in METHODS.items():
        ev = load_eval(dataset, encoder, arm)
        if ev is None:
            continue
        per_ep = ev["mse"].mean(0)                         # seeds -> [E, K]
        mean = per_ep.mean(0)
        rng = np.random.default_rng(0)
        boots = np.stack([per_ep[rng.integers(0, len(per_ep), len(per_ep))].mean(0) for _ in range(1000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        k = np.arange(1, len(mean) + 1)
        ax.plot(k, mean, color=color, ls=ls, marker=marker, markevery=3, label=label,
                lw=2.0 if arm == "shiftwm" else 1.4, zorder=3 if arm == "shiftwm" else 2)
        ax.fill_between(k, lo, hi, color=color, alpha=0.15, lw=0)
        labels.append([label.split(" (")[0], k[-1], float(mean[-1])])
        drawn = True
    if not drawn:
        pending(ax, f"{dataset}: error vs horizon")
        return False
    # direct labels, spread vertically so they never overlap
    lo, hi = ax.get_ylim(); gap = 0.075 * (hi - lo)
    labels.sort(key=lambda t: t[2])
    orig = [t[2] for t in labels]
    clusters = [[0]]
    for i in range(1, len(labels)):
        (clusters[-1].append(i) if orig[i] - orig[clusters[-1][-1]] < gap else clusters.append([i]))
    for c in clusters:                     # spread each cluster evenly, centred on its lines' mean end value
        centre = float(np.mean([orig[i] for i in c]))
        for j, i in enumerate(c):
            labels[i][2] = centre + (j - (len(c) - 1) / 2) * gap
    for i in range(1, len(labels)):        # resolve any residual overlap between clusters
        labels[i][2] = max(labels[i][2], labels[i - 1][2] + gap)
    for name, x, y in labels:
        ax.annotate(name, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=6, color=INK, va="center")
    ax.set_xlabel("forecast step $k$")
    ax.set_ylabel("feature MSE")
    ax.set_title(title or dataset)
    ax.margins(x=0.18)
    return True


def fig_error_vs_horizon(show_pending=False):
    """Absolute error vs horizon (appendix, fig:horizon-abs). Benchmarks without ShiftWM results are left out
    unless show_pending (then drawn as a pending box)."""
    panels = [("droid", "DROID (test)"), ("openh_hamlyn", "Open-H surgical (test)"), ("bridge", "BridgeData V2 (test)"),
              ("language_table", "Language-Table (test)")]
    if not show_pending:
        panels = [p for p in panels if load_eval(p[0], "dinov2s", "shiftwm") is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.85 * len(panels), 1.8), constrained_layout=True)
    for ax, (ds, t) in zip(axes, panels):
        if plot_horizon(ax, ds, title=t):
            ax.set_xticks([1, 4, 7, 10])
    fig.savefig(FIG / "error_vs_horizon.pdf")
    plt.close(fig)


def relative_to_persistence(dataset, encoder="dinov2s"):
    base = load_eval(dataset, encoder, "persistence")
    if base is None:
        return None
    b = base["mse"].mean((0, 2))                          # per episode
    out = {}
    for arm in ("ar_tf", "ar", "direct", "shiftwm"):
        ev = load_eval(dataset, encoder, arm)
        if ev is not None:
            out[arm] = 100 * (1 - ev["mse"].mean((0, 2)).mean() / b.mean())
    return out


# ------------------------------------------------------------------------------------------ teaser
def droid_frames(episode_id, camera="exterior_image_1_left", steps=(0, 10)):
    root = ROOT / "data/real_video/droid_selected/processed"
    man = json.loads((root / "manifest.json").read_text())
    row = next(r for r in man["episodes"] if r["episode_id"] == episode_id)
    with np.load(root / row["cameras"][camera]["file"]) as z:
        return [z["images"][s] for s in steps]


def pick_teaser_episode():
    """Illustration episode for the teaser strip, Fig. 2 (bottom) and the mechanism tiles. Rule (stated in captions):
    among DROID test episodes whose window t0=2 is in the top 30% of true feature change (visible motion), the one with
    the largest relative k=10 advantage of ShiftWM over the better of Direct and AR, 1 - err_S / min(err_D, err_AR).
    Cached in results/v2/analysis/qualitative/teaser_pick.json (keyed by checkpoint paths)."""
    import torch
    from shiftwm.v2.models import V2WorldModel
    root = ROOT / "data/v2/features/droid/dinov2s"
    cache = RES / "analysis/qualitative/teaser_pick.json"
    cks = {}
    for arm in ("shiftwm", "direct", "ar"):
        ck = sorted((ROOT / "results/v2s/droid/dinov2s" / arm).glob("s*/best.pt")) or sorted((RES / "droid/dinov2s" / arm).glob("s*/best.pt"))
        cks[arm] = str(ck[0]) if ck else None
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("checkpoints") == cks:
            return c["episode"]
    man = json.loads((root / "manifest.json").read_text()); stats = json.loads((root / "stats.json").read_text())
    fm, fs = np.array(stats["feature_mean"], np.float32), np.array(stats["feature_std"], np.float32)
    am, ast = np.array(stats["action_mean"], np.float32), np.array(stats["action_std"], np.float32)
    rows, H_, A_, F_, Y_, chg = [], [], [], [], [], []
    for r in man["episodes"]:
        if r["split"] != "test" or r["T"] < 14:
            continue
        with np.load(root / r["file"]) as z:
            f = (z["features"][:13].astype(np.float32).reshape(13, -1, len(fm)) - fm) / fs
            a = (z["actions"][:12].astype(np.float32) - am) / ast
        rows.append(r["id"]); H_.append(f[:3]); A_.append(a[:2]); F_.append(a[2:12]); Y_.append(f[12])
        chg.append(float(np.mean((f[12] - f[2]) ** 2)))
    T = lambda x: torch.tensor(np.stack(x))
    err = {}
    for arm, ck in cks.items():
        if ck is None:
            return rows[int(np.argmax(chg))]
        st = torch.load(ck, map_location="cpu"); m = V2WorldModel(st["config"]).eval(); m.load_state_dict(st["model"])
        with torch.no_grad():
            p = torch.cat([m(T(H_[i:i + 16]), T(A_[i:i + 16]), T(F_[i:i + 16]))[:, 9] for i in range(0, len(rows), 16)])
        err[arm] = ((p.numpy() - np.stack(Y_)) ** 2).mean((1, 2))
    chg = np.array(chg); ok = chg >= np.quantile(chg, 0.7)
    adv = 1 - err["shiftwm"] / np.minimum(err["direct"], err["ar"])
    i = int(np.argmax(np.where(ok, adv, -np.inf)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"episode": rows[i], "advantage": float(adv[i]), "err": {a: float(e[i]) for a, e in err.items()},
                                 "checkpoints": cks, "rule": pick_teaser_episode.__doc__}, indent=1))
    return rows[i]


def transport_arrows(ckpt, episode_id, k=10, device="cpu"):
    """Expected source displacement (in patches) per query patch from a trained ShiftWM checkpoint."""
    import torch
    from shiftwm.v2.models import V2WorldModel
    root = ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    with np.load(root / f"episodes/{episode_id}.npz") as z:
        f, a = z["features"].astype(np.float32), z["actions"]
    st = torch.load(ckpt, map_location=device)
    model = V2WorldModel(st["config"]).to(device).eval()
    model.load_state_dict(st["model"])
    fm, fs = np.array(stats["feature_mean"]), np.array(stats["feature_std"])
    am, ast = np.array(stats["action_mean"]), np.array(stats["action_std"])
    x = torch.tensor(((f[:3] - fm) / fs).reshape(1, 3, -1, f.shape[-1]), dtype=torch.float32)
    an = (a - am) / ast
    past = torch.tensor(an[None, :2], dtype=torch.float32)
    fut = torch.tensor(an[None, 2:2 + k], dtype=torch.float32)
    with torch.no_grad():
        _, det = model(x, past, fut, return_details=True)
    w = det["weights"][0, k - 1].numpy()                 # [N, S*w*w]
    gate = det["gate"][0, k - 1, :, 0].numpy()
    gates = {kk: det["gate"][0, kk - 1, :, 0].numpy().reshape(model.config.grid, -1) for kk in (1, 5, k)}
    cfg = model.config
    g, win, s = cfg.grid, cfg.window, cfg.sources
    r = win // 2
    oy, ox = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    oy, ox = np.tile(oy.ravel(), s), np.tile(ox.ravel(), s)
    dy, dx = w @ oy, w @ ox                                # expected source offset
    return dx.reshape(g, g), dy.reshape(g, g), gate.reshape(g, g), gates


def _vjepa_skill(arm):
    base = RES / "external/vjepa2ac_plugin" / arm
    fs = sorted(base.glob("s*/test_summary.json"))
    v = [json.loads(f.read_text())["relative_gain_vs_persistence"]["mse"]["mean_over_horizons"] for f in fs]
    return 100 * float(np.mean(v)) if v else None


def _dinowm_errred(env):
    base = RES / "external/dinowm_plugin" / env
    e = {}
    for arm in ("dinowm", "dinowm_shiftwm"):
        f = base / arm / "openloop.json"
        if f.exists():
            e[arm] = json.loads(f.read_text()).get("teacher_forced", {}).get("z_visual_err_pred")
    return e if len(e) == 2 and None not in e.values() else None


def _vjepa_mse(arm):
    fs = sorted((RES / "external/vjepa2ac_plugin" / arm).glob("s*/test_summary.json"))
    v = [json.loads(f.read_text())["mean_over_horizons"]["model_mse"] for f in fs]
    return float(np.mean(v)) if v else None


def _numbers():
    """Headline numbers as the paper prints them: tables/generated/numbers.tex (scripts/v2/make_tables.py)."""
    import re
    f = ROOT / "paper/submission_folder/tables/generated/numbers.tex"
    return {k: float(v) for k, v in re.findall(r"\\renewcommand\{\\(\w+)\}\{(-?[0-9.]+)\}", f.read_text())} if f.exists() else {}


def _teaser_ckpt(arm):
    ck = sorted((ROOT / "results/v2s/droid/dinov2s" / arm).glob("s*/best.pt")) or sorted((RES / "droid/dinov2s" / arm).glob("s*/best.pt"))
    return ck[0] if ck else None


TEASER_ARMS = ("ar_tf", "ar", "direct", "shiftwm")


def teaser_forecasts(device="cpu"):
    """Real forecasts of every arm on the teaser window (pick_teaser_episode, t0 = frame 2, k = 1..10), their k=10
    features decoded to RGB by the trained feature->RGB decoder, and ShiftWM's k=10 transport field.
    Cached in results/v2/analysis/qualitative/teaser_forecasts.npz, keyed by episode + checkpoint mtimes."""
    import torch
    from PIL import Image
    from shiftwm.v2 import analysis as A
    from shiftwm.v2.models import V2WorldModel
    ep = pick_teaser_episode()
    cks = {a: _teaser_ckpt(a) for a in TEASER_ARMS}
    dp = A.decoder_path("droid")
    key = json.dumps({"v": 2, "episode": ep, "decoder": [str(dp), dp.stat().st_mtime_ns if dp.exists() else None],
                      **{a: [str(c), c.stat().st_mtime_ns] for a, c in cks.items() if c is not None}}, sort_keys=True)
    cache = RES / "analysis/qualitative/teaser_forecasts.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files}
    root = ROOT / "data/v2/features/droid/dinov2s"; stats = json.loads((root / "stats.json").read_text())
    with np.load(root / f"episodes/{ep}.npz") as z:
        f, a = z["features"][:13].astype(np.float32), z["actions"][:12]
    fz = (f.reshape(len(f), -1, f.shape[-1]) - np.array(stats["feature_mean"], np.float32)) / np.array(stats["feature_std"], np.float32)
    an = (a - np.array(stats["action_mean"], np.float32)) / np.array(stats["action_std"], np.float32)
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    out = {"key": np.array(key), "episode": np.array(ep), "arms": np.array(TEASER_ARMS)}
    truth = fz[3:13]                                                    # frames t0+1..t0+10
    pk10 = {}
    for arm in TEASER_ARMS:
        st = torch.load(cks[arm], map_location="cpu"); m = V2WorldModel(st["config"]).eval(); m.load_state_dict(st["model"])
        with torch.no_grad():
            p = m(T(fz[None, :3]), T(an[None, :2]), T(an[None, 2:12]))[0].numpy()
        out[f"err_{arm}"] = ((p - truth) ** 2).mean((1, 2))            # [10] per-step feature MSE
        out[f"perr_{arm}"] = ((p - truth) ** 2).mean(-1)               # [10, N] per-step, per-patch feature MSE
        pk10[arm] = p[9]
        del m, st
    out["err_persistence"] = ((fz[2][None] - truth) ** 2).mean((1, 2))
    out["perr_persistence"] = ((fz[2][None] - truth) ** 2).mean(-1)   # = true change per patch (moving-mask source)
    dec = A.load_decoder(dp)
    names = ["observed", "truth"] + list(TEASER_ARMS)
    img = A.decode(dec, T(np.stack([fz[2], fz[12]] + [pk10[a] for a in TEASER_ARMS]))).permute(0, 2, 3, 1).numpy()
    frames = droid_frames(ep, steps=(2, 12))
    h, w = frames[0].shape[:2]
    for n, im in zip(names, img):   # decoder output is the 224x224 full-frame encoder view -> back to the camera's aspect
        out[f"dec_{n}"] = np.asarray(Image.fromarray((im * 255).round().astype(np.uint8)).resize((2 * w, 2 * h), Image.LANCZOS))
    out["frame_t"], out["frame_t10"] = frames
    dx, dy, gate, _ = transport_arrows(str(cks["shiftwm"]), ep, k=10, device=device)
    out["dx"], out["dy"], out["gate"] = dx, dy, gate
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


RED, ORANGE = "#B03A2E", "#C0582B"


def _box(ax, x0, y0, w, h, color, lw=0.9, ls="-", z=5):
    """Full four-sided border (never axis spines)."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=color, lw=lw, ls=ls, zorder=z, joinstyle="miter"))


def _img(ax, im, x0, y0, w):
    """Draw an image at (x0, y0) with width w (inches) and its own aspect ratio; returns its height."""
    h = w * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(x0, x0 + w, y0, y0 + h), interpolation="lanczos", zorder=2)
    return h


def _mini_grid(ax, x0, y0, s, fill, n=4):
    from matplotlib.patches import Rectangle
    c = s / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + i * c, y0 + j * c), c * 0.88, c * 0.88, fc=fill(i, j), ec="none", zorder=3))


TEASER_CROP = (11, 11)          # zoom used in (a) and (b): the top 11 patch rows x left 11 patch columns (the arm's region)


def _crop(im, grid=16):
    """Top-left TEASER_CROP patches of a full-frame image (a crop, never a resize)."""
    r, c = TEASER_CROP
    return im[: int(round(im.shape[0] * r / grid)), : int(round(im.shape[1] * c / grid))]


def _text_w(ax, s, **kw):
    """Rendered width of a string in canvas inches (canvas axes use inches as data units)."""
    t = ax.text(0, 0, s, **kw); r = ax.figure.canvas.get_renderer(); bb = t.get_window_extent(r); t.remove()
    return bb.width / ax.figure.dpi


def _teaser_points(regions=True):
    """(label, best-competitor error, ShiftWM error, group, indent) relative to persistence; plus persistence refs.
    Every row: % lower error of ShiftWM than the best LEARNED competitor (matched predictors; for the plug-ins, the same
    published model without the head). Persistence is marked separately (hollow marker) where it is closer."""
    Nm = _numbers()
    reg = RES / "analysis/regions/droid_dinov2s_K10.json"
    R = json.loads(reg.read_text()) if reg.exists() else {}
    def g(arm, key):
        v = [np.mean(x[key]) for k_, x in R.items() if k_.split("/")[0] == arm]
        return float(np.mean(v)) if v else None
    pts, pers = [], {}
    def own(ds, lab, indent=0):
        sk = relative_to_persistence(ds)                        # unrounded seed means (match the text macros)
        if sk and all(a_ in sk for a_ in ("ar", "direct", "shiftwm")):
            best = max(sk[a_] for a_ in ("ar_tf", "ar", "direct") if a_ in sk)
            pts.append((lab, 1 - best / 100, 1 - sk["shiftwm"] / 100, "own", indent))
    own("droid", "DROID")
    for key, lab in ((("moving", "moving"), ("static", "static")) if regions else ()):
        if all(g(a_, key) is not None for a_ in ("persistence", "ar", "direct", "shiftwm")):
            p_ = g("persistence", key)
            pts.append((lab, min(g("direct", key), g("ar", key)) / p_, g("shiftwm", key) / p_, "own", 1))
            if p_ < min(g("direct", key), g("ar", key)):        # copying beats every learned competitor here
                pers[len(pts) - 1] = 100 * (1 - g("shiftwm", key) / p_)
    own("openh_hamlyn", "surgical")
    for ds, lab in (("bridge", "Bridge"), ("fractal", "RT-1"), ("language_table", "Lang.-Table")):
        own(ds, lab)
    if "vjepaMSERed" in Nm:                                     # fine-tuned V-JEPA 2-AC vs. + head (same budget)
        pts.append(("V-JEPA 2-AC", 1.0, 1 - Nm["vjepaMSERed"] / 100, "plug", 0))
    r_ = _dinowm_rel("pusht")                                   # Wall (+18.4% error) is reported in the text/table
    if r_:
        pts.append(("DINO-WM PushT", min(1.0, r_[0]), r_[1], "plug", 0))
    return pts, pers


def _dinowm_rel(env):
    """DINO-WM open-loop latent error of (base, +head), each divided by the persistence error (official val slices)."""
    out = {}
    for arm in ("dinowm", "dinowm_shiftwm"):
        f = RES / "external/dinowm_plugin" / env / arm / "openloop.json"
        if not f.exists():
            return None
        t = json.loads(f.read_text())["teacher_forced"]
        out[arm] = t["z_visual_err_pred"] / t["z_visual_err_persistence"]
    return out["dinowm"], out["dinowm_shiftwm"]


ERR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "err", [(1, 0.35, 0.1, 0.0), (0.95, 0.2, 0.1, 0.6), (0.45, 0.0, 0.08, 0.95)])   # darkens with error (grayscale-safe)


def _grey(im, keep=0.35):
    im = im.astype(np.float32)
    return (im.mean(-1, keepdims=True) * (1 - keep) + im * keep).astype(np.uint8)


def _outline(ax, mask, x0, y0, w, h, **kw):
    """Outline of a boolean patch mask (grid rows top->bottom) drawn over a tile at (x0, y0, w, h)."""
    g = mask.shape[0]
    cw, ch = w / mask.shape[1], h / g
    for r in range(g):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                continue
            X, Y = x0 + c * cw, y0 + h - (r + 1) * ch
            for (dr, dc), seg in (((-1, 0), ((X, Y + ch), (X + cw, Y + ch))), ((1, 0), ((X, Y), (X + cw, Y))),
                                  ((0, -1), ((X, Y), (X, Y + ch))), ((0, 1), ((X + cw, Y), (X + cw, Y + ch)))):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < g and 0 <= cc < mask.shape[1] and mask[rr, cc]):
                    ax.plot(*zip(*seg), **kw)


def _teaser_lanes(ax, D, x0, x1, top):
    """(a) The same held-out window under two output rules: Direct re-generates every patch, ShiftWM moves the arm's
    patches. Columns: frame t with the head's operation, decoded step-10 forecast, per-patch step-10 error (shared scale)."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    import matplotlib.patheffects as pe
    GREEN, BLUE = METHODS["shiftwm"][1], METHODS["direct"][1]
    gap = 0.09
    w = (x1 - x0 - 2 * gap) / 3
    fr = D["frame_t"]; h = w * fr.shape[0] / fr.shape[1]
    G = D["gate"].shape[0]
    cx = [x0 + i * (w + gap) for i in range(3)]
    for c, t in zip(cx, ("frame $t$", "forecast $t{+}10$ (decoded)", "error at $t{+}10$")):
        ax.text(c + w / 2, top, t, fontsize=FS_NOTE, color=MUTED, ha="center", va="top")
    true_chg = D["perr_persistence"][9].reshape(G, G)
    moving = true_chg >= np.quantile(true_chg, 0.75)            # same rule as the moving/static analysis (top 25%)
    E = {a: D[f"perr_{a}"][9].reshape(G, G) for a in ("direct", "shiftwm")}
    vmax = float(np.quantile(np.concatenate([E["direct"].ravel(), E["shiftwm"].ravel()]), 0.97))
    lanes = [("direct", BLUE, "Direct: re-generates all 256 patches", "dec_direct"),
             ("shiftwm", GREEN, "ShiftWM: moves the patches it has seen", "dec_shiftwm")]
    lane_h = h + 0.33
    for li, (arm, col, title, dk) in enumerate(lanes):
        yt = top - 0.16 - li * (lane_h + 0.06)                  # lane title baseline region
        y = yt - 0.13 - h                                       # tile bottom
        ax.text(x0, yt, title, fontsize=FS_LABEL, fontweight="bold", color=col, va="top")
        for c in cx[:2]:
            ax.annotate("", (c + w + gap - 0.015, y + h / 2), (c + w + 0.015, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0))
        # 1) frame t + what the head does
        _img(ax, fr, cx[0], y, w)
        cw, ch = w / G, h / G
        if arm == "direct":                                     # every patch is written anew
            ax.add_patch(Rectangle((cx[0], y), w, h, fc=BLUE, alpha=0.30, ec="none", zorder=3))
            for t in np.linspace(0, 1, G + 1)[1:-1]:
                ax.plot([cx[0] + t * w] * 2, [y, y + h], color="white", lw=0.3, alpha=0.9, zorder=3)
                ax.plot([cx[0], cx[0] + w], [y + t * h] * 2, color="white", lw=0.3, alpha=0.9, zorder=3)
        else:                                                   # learned transport: expected source -> target
            g, dx, dy = D["gate"], D["dx"], D["dy"]
            mag = np.hypot(dx, dy)
            moved = (g >= 0.5) & (mag >= 0.5)                   # gate open and content comes from >= half a patch away
            for r, c in zip(*np.nonzero(moved)):
                ax.add_patch(Rectangle((cx[0] + c * cw, y + h - (r + 1) * ch), cw, ch, fc=GREEN, alpha=0.5, ec="white",
                                       lw=0.3, zorder=3))
            for r, c in zip(*np.nonzero(moved)):
                qx, qy = cx[0] + (c + 0.5) * cw, y + h - (r + 0.5) * ch
                sx, sy = qx + dx[r, c] * cw, qy - dy[r, c] * ch
                a_ = ax.add_patch(FancyArrowPatch((sx, sy), (qx, qy), arrowstyle="-|>,head_length=0.5,head_width=0.22",
                                                  mutation_scale=2.2, lw=0.5, color="#002E21", shrinkA=0, shrinkB=0,
                                                  zorder=5))
                a_.set_path_effects([pe.Stroke(linewidth=1.1, foreground="white", alpha=0.6), pe.Normal()])
        _box(ax, cx[0], y, w, h, col, lw=0.9)
        ax.text(cx[0] + w / 2, y - 0.03, "every patch rewritten" if arm == "direct" else "learned transport",
                fontsize=FS_NOTE, color=col, ha="center", va="top")
        # 2) that model's own decoded forecast
        _img(ax, D[dk], cx[1], y, w)
        _box(ax, cx[1], y, w, h, col, lw=0.9)
        e_all = float(D[f"err_{arm}"][9])
        ax.text(cx[1] + w - 0.025, y + 0.025, f"{e_all:.2f}", fontsize=FS_NOTE, color="white", fontweight="bold", ha="right",
                va="bottom", zorder=6, bbox=dict(fc=col, ec="none", alpha=0.93, pad=0.9))
        # 3) per-patch error over the true future (shared scale), moving patches outlined
        _img(ax, _grey(D["frame_t10"]), cx[2], y, w)
        ax.imshow(np.clip(E[arm] / vmax, 0, 1), cmap=ERR_CMAP, vmin=0, vmax=1, extent=(cx[2], cx[2] + w, y, y + h),
                  interpolation="bicubic", zorder=3)
        _outline(ax, moving, cx[2], y, w, h, color="white", lw=0.5, ls=(0, (1.5, 1)), zorder=4)
        _box(ax, cx[2], y, w, h, col, lw=0.9)
        es, em = float(E[arm][~moving].mean()), float(E[arm][moving].mean())
        ax.text(cx[2] + w / 2, y - 0.03, f"static {es:.2f}   moving {em:.2f}", fontsize=FS_NOTE, color=INK, ha="center",
                va="top")
    # key for the error maps
    ky = y - 0.22
    kx = cx[2]
    ax.imshow(np.linspace(0, 1, 64)[None], cmap=ERR_CMAP, extent=(kx, kx + 0.35, ky, ky + 0.06), aspect="auto", zorder=3)
    _box(ax, kx, ky, 0.35, 0.06, "#C9CED6", lw=0.4)
    ax.text(kx - 0.04, ky + 0.03, "feature error, shared scale:  low", fontsize=FS_NOTE, color=MUTED, ha="right", va="center")
    ax.text(kx + 0.39, ky + 0.03, "high", fontsize=FS_NOTE, color=MUTED, ha="left", va="center")
    _outline(ax, np.ones((1, 1), bool), kx + 0.62, ky - 0.005, 0.07, 0.07, color=MUTED, lw=0.6, ls=(0, (1.5, 1)))
    ax.text(kx + 0.73, ky + 0.03, "moving", fontsize=FS_NOTE, color=MUTED, ha="left", va="center")
    return y


def _teaser_regions():
    """DROID test error on static / moving patches (K=10 windows; moving = top 25% true change per window), each learned
    arm relative to copying the last frame (persistence), in %. Source: results/v2/analysis/regions (Table 9)."""
    f = RES / "analysis/regions/droid_dinov2s_K10.json"
    if not f.exists():
        return None
    R = json.loads(f.read_text())
    def g(arm, key):
        v = [np.mean(x[key]) for k_, x in R.items() if k_.split("/")[0] == arm]
        return float(np.mean(v)) if v else None
    out = {}
    for key in ("static", "moving"):
        p_ = g("persistence", key)
        out[key] = {a: 100 * (g(a, key) / p_ - 1) for a in ("ar_tf", "ar", "direct", "shiftwm") if g(a, key) is not None}
    return out


def _teaser_evidence_regions(fig, ax, x0, x1, top, bottom, W, H):
    """(b) Two aligned dot plots (static | moving patches): % error vs. copying the last frame, per method."""
    Rg = _teaser_regions()
    if Rg is None:
        pending(fig.add_axes([x0 / W, bottom / H, (x1 - x0) / W, (top - bottom) / H]), "regions"); return
    arms = ["shiftwm", "direct", "ar", "ar_tf"]
    names = {"shiftwm": "ShiftWM", "direct": "Direct", "ar": "AR", "ar_tf": "AR-TF"}
    lab_w, mid = 0.47, 0.16
    pw = (x1 - x0 - lab_w - mid) / 2
    ys = np.arange(len(arms))[::-1]
    for j, (key, lim, head) in enumerate((("static", (-6, 24), "static patches"), ("moving", (-45, 3), "moving patches"))):
        xa = x0 + lab_w + j * (pw + mid)
        sax = fig.add_axes([xa / W, (bottom + 0.27) / H, pw / W, (top - 0.15 - bottom - 0.27) / H])
        sax.set_xlim(*lim); sax.set_ylim(-0.6, len(arms) - 0.4)
        sax.axvline(0, color=INK, lw=0.7, zorder=1)
        for a, yy in zip(arms, ys):
            v = Rg[key][a]; _, c, _, mk = METHODS[a]
            vd = min(v, lim[1] - 1.5)
            sax.plot([0, vd], [yy, yy], color=c, lw=1.2, alpha=0.45, solid_capstyle="butt", zorder=2)
            if v > lim[1] - 1.5:                               # off-scale: arrow head at the edge + exact value
                sax.scatter([vd], [yy], marker=">", s=22, color=c, zorder=3, clip_on=False)
            else:
                sax.scatter([vd], [yy], marker=mk, s=26 if a == "shiftwm" else 16, color=c, edgecolors="white",
                            linewidths=0.5, zorder=3, clip_on=False)
            txt = f"{v:+.1f}".replace("-", "−") if abs(v) < 1 else f"{v:+.0f}".replace("-", "−")
            if key == "static":
                sax.text(vd + 1.4, yy, txt, fontsize=FS_NOTE, color=c, va="center", ha="left",
                         fontweight="bold" if a == "shiftwm" else "normal")
            else:
                sax.text(vd - 2.2, yy, txt, fontsize=FS_NOTE, color=c, va="center", ha="right",
                         fontweight="bold" if a == "shiftwm" else "normal")
        sax.set_yticks([]); sax.spines["left"].set_visible(False)
        sax.grid(axis="y", visible=False); sax.grid(axis="x", color="#EEF0F3", lw=0.5)
        sax.tick_params(axis="x", labelsize=FS_TICK, length=2, pad=1)
        sax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20 if key == "moving" else 10))
        sax.set_title(head, fontsize=FS_NOTE, color=INK, pad=2)
        if j == 0:
            for a, yy in zip(arms, ys):
                sax.text(lim[0] - 0.12 * (lim[1] - lim[0]), yy, names[a], fontsize=FS_NOTE, ha="right", va="center",
                         color=METHODS[a][1], fontweight="bold" if a == "shiftwm" else "normal")
    ax.text((x0 + lab_w + x1) / 2, bottom, "% error vs. copying the last frame (0 = copy)", fontsize=FS_NOTE, color=INK,
            ha="center", va="bottom")


def _teaser_evidence_gains(fig, ax, x0, x1, top, bottom, W, H):
    """(c) % lower error than the best learned competitor on each dataset; plug-in rows vs. the same model w/o head."""
    pts, _ = _teaser_points(regions=False)
    if not pts:
        pending(fig.add_axes([x0 / W, bottom / H, (x1 - x0) / W, (top - bottom) / H]), "held-out results"); return
    GREEN = METHODS["shiftwm"][1]
    red = [100 * (1 - y / x) for _, x, y, *_ in pts]
    n = len(pts); n_own = sum(1 for p_ in pts if p_[3] == "own")
    ypos = [-(i + (1.1 if i >= n_own else 0)) for i in range(n)]
    lab_w = 0.86
    sax = fig.add_axes([(x0 + lab_w) / W, (bottom + 0.25) / H, (x1 - x0 - lab_w - 0.02) / W, (top - bottom - 0.25) / H])
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, sax.transData)
    hi_ = max(red) * 1.3
    sax.set_xlim(0, hi_); sax.set_ylim(min(ypos) - 0.55, 0.55)
    sax.axvline(0, color=INK, lw=0.6, zorder=1)
    for (lab, x, y, grp, ind), r, yy in zip(pts, red, ypos):
        sax.plot([0, r], [yy, yy], color=GREEN, lw=1.2, alpha=0.45, solid_capstyle="butt", zorder=2)
        sax.scatter([r], [yy], s=20 if grp == "own" else 17, color=GREEN, marker="o" if grp == "own" else "D",
                    edgecolors="white", linewidths=0.5, zorder=3)
        sax.text(r + 0.04 * hi_, yy, f"{r:.1f}", fontsize=FS_NOTE, ha="left", va="center", color=GREEN, fontweight="bold")
        ax.text(x0 + lab_w - 0.04, yy, lab, fontsize=FS_NOTE, ha="right", va="center", color=INK, transform=tr)
    sax.set_yticks([]); sax.spines["left"].set_visible(False)
    sax.grid(axis="y", visible=False); sax.grid(axis="x", color="#EEF0F3", lw=0.5)
    sax.tick_params(axis="x", labelsize=FS_TICK, length=2, pad=1)
    sax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    sax.patch.set_alpha(0)
    ax.text(x0 + 0.12, (ypos[n_own - 1] + ypos[n_own]) / 2, "plug-in head (vs. the backbone alone)", transform=tr,
            fontsize=FS_NOTE, color=MUTED, style="italic", va="center", ha="left")
    ax.text((x0 + lab_w + x1) / 2, bottom, "% lower error than best competitor", fontsize=FS_NOTE, color=INK,
            ha="center", va="bottom")


def fig_teaser(device="cpu"):
    """Figure 1. (a) One held-out DROID window under two output rules (Direct re-generates, ShiftWM moves): frame t with
    the operation, decoded step-10 forecast, per-patch error. (b) The consequence on all DROID test windows: static and
    moving-patch error vs. copying. (c) Gains on the other datasets and as a plug-in head (positive results only; the
    DINO-WM Wall regression is reported in the text and Table 2)."""
    W, H = 5.5, 2.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    ca, cr = (0.03, 3.02), (3.17, 5.47)
    try:
        D = teaser_forecasts(device)
    except Exception as e:  # noqa: BLE001  -- missing checkpoints/decoder: explicit pending boxes, nothing invented
        print("teaser forecasts unavailable:", e)
        D = None
    ax.text(ca[0], H - 0.03, "(a) One held-out DROID window, two ways to forecast", fontsize=FS_TITLE, fontweight="bold",
            color=INK, va="top")
    if D is not None:
        _teaser_lanes(ax, D, *ca, H - 0.2)
    else:
        pending(fig.add_axes([ca[0] / W, 0.05, (ca[1] - ca[0]) / W, 0.8]), "teaser window")
    ax.plot([ca[1] + 0.075] * 2, [0.06, H - 0.06], color=PANEL_EDGE, lw=0.6, zorder=0)
    ax.text(cr[0], H - 0.03, "(b) All 130 DROID test episodes", fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
    _teaser_evidence_regions(fig, ax, *cr, H - 0.2, 1.08, W, H)
    ax.text(cr[0], 1.0, "(c) Other data and plug-in heads", fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
    _teaser_evidence_gains(fig, ax, *cr, 0.83, 0.02, W, H)
    qa(fig, "teaser", 5.5)
    fig.savefig(FIG / "teaser.pdf"); fig.savefig(FIG / "teaser_preview.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    def gain():                                                   # main-text Fig. 4 (make_gain_horizon.py)
        import make_gain_horizon; make_gain_horizon.main()
    jobs = {"teaser": lambda: fig_teaser(a.device), "horizon": fig_error_vs_horizon, "gain": gain}
    for name, fn in jobs.items():
        if not a.only or name in a.only:
            fn(); print("wrote", name)


if __name__ == "__main__":
    main()
