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
        if all(list((base / dataset / encoder / a).glob(f"s*/eval_{split}.npz")) for a in LEARNED):
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


def fig_error_vs_horizon():
    panels = [("droid", "DROID (test)"), ("openh_hamlyn", "Open-H surgical (test)"), ("bridge", "BridgeData V2 (test)")]
    if load_eval("language_table", "dinov2s", "shiftwm") is not None:   # 4th panel once Language-Table results exist
        panels.append(("language_table", "Language-Table (test)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.1), constrained_layout=True)
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
    key = json.dumps({"episode": ep, "decoder": [str(dp), dp.stat().st_mtime_ns if dp.exists() else None],
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
        pk10[arm] = p[9]
        del m, st
    out["err_persistence"] = ((fz[2][None] - truth) ** 2).mean((1, 2))
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


def _teaser_story(ax, D, x0, x1, top):
    """(a) The task, what existing latent world models do, and what ShiftWM does instead."""
    from matplotlib.patches import FancyArrowPatch
    GREEN = METHODS["shiftwm"][1]
    arrow = lambda p, q, col: ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5.5, lw=0.8, color=col,
                                                           shrinkA=0, shrinkB=0, zorder=4))
    # --- task: observed DINOv2 patch grid + future actions -> future grids
    fr = D["frame_t"]; w = 0.56
    h = w * fr.shape[0] / fr.shape[1]
    y = top - h
    _img(ax, fr, x0, y, w)
    for t in np.linspace(0, 1, 17)[1:-1]:                       # the 16x16 DINOv2 patch grid (full-frame, no crop)
        ax.plot([x0 + t * w] * 2, [y, y + h], color="white", lw=0.25, alpha=0.75, zorder=3)
        ax.plot([x0, x0 + w], [y + t * h] * 2, color="white", lw=0.25, alpha=0.75, zorder=3)
    _box(ax, x0, y, w, h, "#9AA3AE", lw=0.5)
    tx = x0 + w + 0.07
    ax.text(tx, y + h / 2 + 0.06, "DINOv2 patches $\\mathbf{Z}_0$", fontsize=6.0, color=INK, va="center", ha="left")
    ax.text(tx, y + h / 2 - 0.07, "+ actions $\\rightarrow \\mathbf{Z}_{1:10}$", fontsize=6.0, color=INK, va="center", ha="left")
    # --- existing: regenerate every patch, recursively
    yb = y - 0.11
    ax.text(x0, yb, "Existing WMs regenerate every patch", fontsize=6.3, fontweight="bold", color=ORANGE, va="center")
    s, gy = 0.17, yb - 0.27
    rng = np.random.default_rng(3)
    obj = {(1, 1), (1, 2), (2, 2)}
    BLUE, LIGHT = "#2B6CB0", "#9CC3E4"
    xs = [x0 + 0.02, x0 + 0.34, x0 + 0.66, x0 + 1.0]
    labels = ["$\\mathbf{Z}_0$", "$\\hat{\\mathbf{Z}}_1$", "$\\hat{\\mathbf{Z}}_2$", "$\\hat{\\mathbf{Z}}_{10}$"]
    for i, (xx, lab) in enumerate(zip(xs, labels)):
        if i == 0:
            _mini_grid(ax, xx, gy, s, lambda a, b: BLUE if (a, b) in obj else LIGHT)
        else:                                                    # schematic: every patch re-generated, drifting more with k
            noise = rng.uniform(-1, 1, (4, 4)) * (0.06 + 0.07 * i)
            base = np.array(matplotlib.colors.to_rgb("#E8834A"))
            _mini_grid(ax, xx, gy, s, lambda a, b, nz=noise: tuple(np.clip(base + nz[a, b], 0, 1)))
        ax.text(xx + s / 2, gy - 0.025, lab, fontsize=5.7, ha="center", va="top", color=INK)
    for i in range(2):
        arrow((xs[i] + s + 0.04, gy + s / 2), (xs[i + 1] - 0.04, gy + s / 2), ORANGE)
        ax.text((xs[i] + s + xs[i + 1]) / 2, gy + s / 2 + 0.03, "$f$", fontsize=5.8, ha="center", va="bottom", color=ORANGE)
    ax.text((xs[2] + s + xs[3]) / 2, gy + s / 2, "$\\cdots$", fontsize=7, ha="center", va="center", color=ORANGE)
    fy = gy - 0.205
    prev = None
    for t in ("drift", "blur", "compounding"):
        kw = dict(fontsize=5.6, color=RED, va="bottom", ha="left")
        prev = (ax.text(x0, fy, "$\\times$ " + t, **kw) if prev is None else
                ax.annotate("$\\times$ " + t, xy=(1, 0), xycoords=prev, xytext=(3, 0), textcoords="offset points", **kw))
    # --- ShiftWM: keep, move (transport of observed features), correct
    ys = fy - 0.115
    ax.text(x0, ys, "ShiftWM moves what it has seen", fontsize=6.3, fontweight="bold", color=GREEN, va="center")
    fr2 = _crop(D["frame_t"]); w2 = 0.98; h2 = w2 * fr2.shape[0] / fr2.shape[1]
    yi = ys - 0.075 - h2
    rr, cc = TEASER_CROP
    g = D["gate"][:rr, :cc]; dx, dy = D["dx"][:rr, :cc], D["dy"][:rr, :cc]
    _img(ax, (fr2.astype(np.float32) * 0.6 + 255 * 0.4).astype(np.uint8), x0, yi, w2)
    rgba = np.zeros(g.shape + (4,)); rgba[..., :3] = matplotlib.colors.to_rgb(GREEN)
    rgba[..., 3] = 0.6 * np.clip((g - 0.5) / 0.5, 0, 1)
    ax.imshow(rgba, extent=(x0, x0 + w2, yi, yi + h2), interpolation="bicubic", zorder=3)
    cw, ch = w2 / cc, h2 / rr
    mag = np.hypot(dx, dy)
    for r in range(rr):
        for c in range(cc):
            if g[r, c] < 0.5 or mag[r, c] < 0.5:
                continue
            qx, qy = x0 + (c + 0.5) * cw, yi + h2 - (r + 0.5) * ch        # query patch (where the content lands)
            sx, sy = qx + dx[r, c] * cw, qy - dy[r, c] * ch               # expected source location in the observed grid
            ax.add_patch(FancyArrowPatch((sx, sy), (qx, qy), arrowstyle="-|>", mutation_scale=3.6, lw=0.55,
                                         color="#073B2C", shrinkA=0, shrinkB=0, zorder=5))
    _box(ax, x0, yi, w2, h2, GREEN, lw=0.8)
    ex = x0 + w2 + 0.08
    ax.text(ex, yi + h2 - 0.01, "$\\hat{\\mathbf{Z}}_k =$", fontsize=6.6, color=INK, va="top")
    step = (h2 - 0.14) / 3
    for i, (term, t, col) in enumerate((("$(1{-}g)\\,\\mathbf{Z}_0$", "keep", "#5B6270"),
                                        ("$+\;g\\,\\mathbf{T}_k$", "move", GREEN), ("$+\;\\mathbf{r}_k$", "correct", "#B7791F"))):
        yy = yi + h2 - 0.2 - i * step
        ax.text(ex + 0.4, yy, term, fontsize=6.4, color=INK, va="center", ha="right")
        ax.text(ex + 0.46, yy, t, fontsize=5.6, color=col, fontweight="bold", va="center")
    ax.text(x0, yi - 0.05, "$\\checkmark$ parallel from measured $\\mathbf{Z}_0$   $\\checkmark$ plug-in head", fontsize=5.6,
            color="#00785A", va="top")


def _teaser_window(ax, D, x0, x1, top):
    """(b) The same held-out window: every arm's own k=10 forecast, decoded by one feature->RGB decoder (arm zoom)."""
    tiles = [("true $t{+}10$", "(decoded)", "truth", INK, None), ("ShiftWM", " (ours)", "shiftwm", METHODS["shiftwm"][1], "shiftwm"),
             ("Direct", "", "direct", METHODS["direct"][1], "direct"), ("AR", " (recursive)", "ar", METHODS["ar"][1], "ar"),
             ("AR-TF", " (DINO-WM-style)", "ar_tf", METHODS["ar_tf"][1], "ar_tf"),
             ("Persistence", " (copy $t$)", "observed", METHODS["persistence"][1], "persistence")]
    gap = 0.06
    w = (x1 - x0 - gap) / 2
    ims = {k: _crop(D[f"dec_{k}"]) for _, _, k, _, _ in tiles}
    h = w * ims["truth"].shape[0] / ims["truth"].shape[1]
    pitch = h + 0.125
    for i, (lab, suf, key, col, arm) in enumerate(tiles):
        r, c = divmod(i, 2)
        xx, yy = x0 + c * (w + gap), top - 0.11 - r * pitch - h
        _img(ax, ims[key], xx, yy, w)
        ours = arm == "shiftwm"
        _box(ax, xx, yy, w, h, col if arm in ("shiftwm", None) else "#C9CED6", lw=1.4 if ours else (0.8 if arm is None else 0.5))
        t1 = ax.text(xx, yy + h + 0.03, lab, fontsize=5.9, color=col, fontweight="bold", va="baseline", ha="left")
        if suf:
            ax.annotate(suf.strip(), xy=(1, 0), xycoords=t1, xytext=(2, 0), textcoords="offset points", fontsize=5.3, color=col,
                        va="bottom", ha="left")
        if arm is not None:
            e = float(D[f"err_{arm}"][9])
            ax.text(xx + w - 0.03, yy + 0.03, f"{e:.2f}", fontsize=5.8, color="white", fontweight="bold", ha="right", va="bottom",
                    zorder=6, bbox=dict(fc=col, ec="none", alpha=0.93, pad=0.9))
    yb = top - 0.11 - 2 * pitch - h - 0.05
    ax.text(x0, yb, "Each tile: that model's own forecast, one shared decoder.\nCorner: feature error at $t{+}10$ (lower is better).", fontsize=5.2, color=MUTED, va="top", linespacing=1.15)


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


def _teaser_results(fig, ax, x0, x1, top, W, H):
    """(c) One scatter: error of the best competitor (x) vs. ShiftWM (y), both relative to persistence, log-log.
    Below the diagonal = ShiftWM better; each point carries its relative error change."""
    Nm = _numbers()
    reg = RES / "analysis/regions/droid_dinov2s_K10.json"
    R = json.loads(reg.read_text()) if reg.exists() else {}
    def g(arm, key):
        v = [np.mean(x[key]) for k_, x in R.items() if k_.split("/")[0] == arm]
        return float(np.mean(v)) if v else None
    BLUE, PINK, AMBER = "#2B6CB0", "#C2185B", "#D98E00"
    pts = []                                                    # (label, x = best competitor, y = ours, colour, dx, dy)
    if {"droidSkill", "droidSkillDirect", "droidSkillAR"} <= Nm.keys():
        pts.append(("DROID", 1 - max(Nm["droidSkillDirect"], Nm["droidSkillAR"]) / 100, 1 - Nm["droidSkill"] / 100, BLUE, 1.18, 0.9))
    for key, lab, dxy in (("moving", "moving parts", (0.66, 0.84)), ("static", "static scene", (0.62, 1.13))):
        if all(g(a_, key) is not None for a_ in ("persistence", "ar", "direct", "shiftwm")):
            p_ = g("persistence", key)
            pts.append((lab, min(g("direct", key), g("ar", key)) / p_, g("shiftwm", key) / p_, BLUE, *dxy))
    if {"hamlynSkill", "hamlynSkillDirect", "hamlynSkillAR"} <= Nm.keys():
        pts.append(("surgical", 1 - max(Nm["hamlynSkillDirect"], Nm["hamlynSkillAR"]) / 100, 1 - Nm["hamlynSkill"] / 100, PINK, 0.62, 1.12))
    for ds, lab in (("bridge", "Bridge"), ("fractal", "RT-1"), ("language_table", "Lang.-Table")):
        sk = relative_to_persistence(ds)
        if sk and all(a_ in sk for a_ in ("ar", "direct", "shiftwm")):
            pts.append((lab, 1 - max(sk["ar"], sk["direct"]) / 100, 1 - sk["shiftwm"] / 100, BLUE, 1.15, 0.9))
    if {"vjepaSkillFT", "vjepaSkillOurs"} <= Nm.keys():
        pts.append(("V-JEPA 2-AC\n+ head", 1 - Nm["vjepaSkillFT"] / 100, 1 - Nm["vjepaSkillOurs"] / 100, AMBER, 1.16, 0.78))
    for env, lab, dxy in (("pusht", "DINO-WM PushT\n+ head", (1.15, 0.8)),):   # Wall (+18.4% error) is reported in the text/table
        r_ = _dinowm_rel(env)
        if r_:
            pts.append((lab, r_[0], r_[1], AMBER, *dxy))
    if not pts:
        pending(fig.add_axes([x0 / W, 0.1, (x1 - x0) / W, 0.7]), "held-out results"); return
    short = {"DROID": "DROID", "moving parts": "moving", "static scene": "static", "surgical": "surgical",
             "V-JEPA 2-AC\n+ head": "V-JEPA", "DINO-WM PushT\n+ head": "PushT", "DINO-WM Wall\n+ head": "Wall"}
    pad_l, pad_b = 0.3, 0.36
    sax = fig.add_axes([(x0 + pad_l) / W, pad_b / H, (x1 - x0 - pad_l - 0.03) / W, (top - 0.22 - pad_b) / H])
    n = len(pts); xs = np.arange(n)
    red = [100 * (1 - y / x) for _, x, y, *_ in pts]                  # % lower error than the best competitor
    lo_, hi_ = min(0, min(red)) * 1.35, max(red) * 1.32
    sax.set_xlim(-0.6, n - 0.4); sax.set_ylim(lo_, hi_)
    n_own = sum(1 for p_ in pts if p_[3] != "#D98E00")
    if n_own < n:
        sax.axvspan(n_own - 0.5, n - 0.4, color="#FFF6E5", lw=0, zorder=0)
        sax.text((n_own - 0.5 + n - 0.4) / 2, hi_ * 0.99, "plug-in\nhead", fontsize=4.9, color="#B97800",
                 ha="center", va="top", style="italic", linespacing=0.95)
        sax.text((-0.6 + n_own - 0.5) / 2, hi_ * 0.99, "vs. matched predictors", fontsize=4.9, color="#2B6CB0",
                 ha="center", va="top", style="italic")
    sax.axhline(0, color=INK, lw=0.6, zorder=1)
    for i, ((lab, x, y, col, _, _), r) in enumerate(zip(pts, red)):
        c = col if r > 0 else "#B03A2E"
        sax.plot([i, i], [0, r], color=c, lw=2.2, solid_capstyle="round", zorder=2, alpha=0.85)
        sax.scatter([i], [r], s=34, color=c, edgecolors="white", linewidths=0.7, zorder=3)
        sax.text(i, r + (0.045 if r > 0 else -0.045) * (hi_ - lo_), f"{r:.1f}".replace("-", "\u2212"), fontsize=5.3,
                 ha="center", va="bottom" if r > 0 else "top", color=c, fontweight="bold")
    sax.set_xticks(xs); sax.set_xticklabels([short.get(p_[0], p_[0]) for p_ in pts], fontsize=4.8, rotation=30, ha="right", rotation_mode="anchor")
    sax.tick_params(axis="y", labelsize=5.3, length=2, pad=1); sax.tick_params(axis="x", length=0, pad=2)
    sax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    sax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    sax.grid(axis="x", visible=False); sax.grid(axis="y", color="#EEF0F3", lw=0.5)
    sax.set_ylabel("lower error than best competitor (%)", fontsize=5.4, labelpad=1)
    fa = RES / "analysis/anatomy/summary.json"
    if fa.exists():
        m = np.asarray(json.loads(fa.read_text())["gain_map"]["direct"])
        ax.text(x0 + 0.02, top - 0.03, f"beats Direct in {int((m > 0).sum())}/{m.size} DROID horizon-motion bins",
                fontsize=5.4, color=INK, va="top")


def _check_layout(fig, cols, W):
    """Print overlapping text boxes and text leaving its panel column (a render-time guard for the teaser)."""
    r = fig.canvas.get_renderer()
    txt = [t for a in fig.axes[1:] for t in a.get_xticklabels() if t.get_visible() and t.get_text().strip()]
    txt += [t for a in fig.axes for t in a.texts + [a.xaxis.label] if t.get_visible() and t.get_text().strip()]
    bb = [(t, t.get_window_extent(r)) for t in txt]
    dpi = fig.dpi
    for i in range(len(bb)):
        for j in range(i + 1, len(bb)):
            if bb[i][1].overlaps(bb[j][1]) and bb[j][1].x0 < bb[i][1].x1 - 1:
                print("overlap:", repr(bb[i][0].get_text()[:30]), "|", repr(bb[j][0].get_text()[:30]))
    for t, b in bb:
        x0, x1 = b.x0 / dpi, b.x1 / dpi
        if not any(a - 0.02 <= x0 and x1 <= c + 0.02 for a, c in cols):
            print("outside column:", repr(t.get_text()[:40]), round(x0, 2), round(x1, 2))


def fig_teaser(device="cpu"):
    """Figure 1. (a) task + prior WMs vs. ShiftWM, (b) one held-out window, every arm decoded, (c) held-out numbers."""
    W, H = 5.5, 2.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    cols = [(0.03, 1.83), (1.95, 3.63), (3.75, 5.47)]
    top = H - 0.21
    titles = ("(a) Move, don't regenerate", "(b) One held-out DROID window", "(c) Held-out results")
    for (a, b), t in zip(cols, titles):
        ax.text(a, H - 0.06, t, fontsize=7.2, fontweight="bold", color=INK, va="top")
    try:
        D = teaser_forecasts(device)
    except Exception as e:  # noqa: BLE001  -- missing checkpoints/decoder: explicit pending boxes, nothing invented
        print("teaser forecasts unavailable:", e)
        D = None
    if D is not None:
        _teaser_story(ax, D, *cols[0], top)
        _teaser_window(ax, D, *cols[1], top)
    else:
        for c in cols[:2]:
            pending(fig.add_axes([c[0] / W, 0.05, (c[1] - c[0]) / W, 0.8]), "teaser window")
    _teaser_results(fig, ax, *cols[2], top, W, H)
    _check_layout(fig, cols, W)
    fig.savefig(FIG / "teaser.pdf"); fig.savefig(FIG / "teaser_preview.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    jobs = {"teaser": lambda: fig_teaser(a.device), "horizon": fig_error_vs_horizon}
    for name, fn in jobs.items():
        if not a.only or name in a.only:
            fn(); print("wrote", name)


if __name__ == "__main__":
    main()
