"""Analysis and qualitative figures (Sec. 5 + appendix) from completed runs and real test frames.

Figures (paper/submission_folder/figures/):
  flow_agreement.pdf        transport field vs RAFT flow (scripts/v2/flow_agreement.py outputs)
  tradeoff.pdf              skill vs action-ranking error per benchmark + gain over Direct by episode motion
                            (make_tradeoff.py)
  qualitative_transport.pdf 3 DROID + 2 Hamlyn test windows x [observed, true k=10, decoded ShiftWM,
                            decoded AR, transport arrows, gate, RAFT flow]
  gallery_droid.pdf         6 pseudo-random DROID test episodes, k in {1,3,5,10}
  gallery_surgical.pdf      1 pseudo-random Hamlyn test episode per task (7)
  gallery_language_table.pdf 6 pseudo-random Language-Table test episodes (only once its runs exist);
                            qualitative_transport.pdf also gains 2 Language-Table rows (50th/90th pct) then
  failures.pdf              largest-error ShiftWM test windows

Deterministic selection rules (no manual picking):
  qualitative : for every test episode take the window t0 in [H-1, T-1-K] with the largest true
                standardised feature change mean((z_{t0+10} - z_{t0})^2); rank episodes by that value and
                take those at the 50th/75th/90th percentile (DROID) and 50th/90th (Hamlyn).
  galleries   : test episodes ordered by sha256("gallery:" + id); DROID: first 6; Hamlyn: first per task.
                Window t0 = clip(T // 2, H-1, T-1-K).
  failures    : the 3 DROID and 2 Hamlyn test episodes with the largest ShiftWM (first seed) k=10 MSE in
                eval_test.npz; within each, the stride-2 window with the largest k=10 MSE.
Models: first available seed of each arm; decoder: results/v2/analysis/decoder/<ds>/dinov2s/best.pt.
Every panel whose inputs are missing is drawn as a "pending" box (make_figures.pending) -- nothing is
invented. Choices are written to results/v2/analysis/qualitative/provenance.json.

Usage (repo root): PYTHONPATH=src python paper/submission_folder/figures/src/make_analysis_figures.py \
                     [--device cuda] [--only flow tradeoff qualitative gallery_droid gallery_surgical failures]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import FIG, INK, METHODS, MUTED, RES, ROOT, load_eval, pending  # noqa: E402  (sets rcParams)
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
AN = RES / "analysis"
DS_LABEL = {"droid": "DROID", "openh_hamlyn": "Hamlyn"}
DS_STYLE = {"droid": ("-", "o"), "openh_hamlyn": ((0, (4, 1.5)), "s")}   # linestyle, marker per dataset
ASPECT = {"droid": 180 / 320, "openh_hamlyn": 480 / 848, "language_table": 360 / 640}  # display h/w (native aspect)
# qualitative-only datasets (not in the flow / gain analyses); shown only once their runs exist
QUAL_EXTRA = {"language_table": "Lang.-Table"}
NAME = {**DS_LABEL, **QUAL_EXTRA}
PROV = {}


def ci(v, n=2000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ============================================================================ flow agreement
def load_flow(ds):
    files = sorted((AN / "flow" / ds).glob("shiftwm_s*.npz"))
    if not files:
        return None
    zs = [np.load(f) for f in files]
    out = {"ks": zs[0]["ks"], "seeds": len(zs)}
    for m in ("cos", "epe", "epe_zero"):
        per = [z[f"{m}_bwd"] / np.where(z["npatch_bwd"] > 0, z["npatch_bwd"], np.nan) for z in zs]
        out[m] = np.nanmean(np.stack(per), 0)                          # [E, nk] averaged over seeds
    return out


def fig_flow_agreement():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), constrained_layout=True)
    data = {ds: load_flow(ds) for ds in DS_LABEL}
    if not any(v is not None for v in data.values()):
        for ax, t in zip(axes, ("cosine vs horizon", "end-point error vs horizon", "per-episode cosine, k=10")):
            pending(ax, t)
        fig.savefig(FIG / "flow_agreement.pdf"); plt.close(fig); return
    col = METHODS["shiftwm"][1]
    pcol = METHODS["persistence"][1]
    for ds, d in data.items():
        if d is None:
            continue
        ls, mk = DS_STYLE[ds]
        k = d["ks"]
        for ax, m, color, lab in ((axes[0], "cos", col, "ShiftWM"), (axes[1], "epe", col, "ShiftWM"),
                                  (axes[1], "epe_zero", pcol, "zero motion")):
            mean = np.nanmean(d[m], 0)
            lo, hi = np.array([ci(d[m][:, c]) for c in range(len(k))]).T
            ax.errorbar(k, mean, yerr=[mean - lo, hi - mean], color=color, ls=ls if m != "epe_zero" else (0, (3, 2)),
                        marker=mk, capsize=2, lw=1.4, ms=3.5)
            ax.annotate(f"{lab}, {DS_LABEL[ds]}", (k[-1], mean[-1]), xytext=(3, 0), textcoords="offset points",
                        fontsize=5.5, color=INK, va="center")
    axes[0].axhline(0, color=pcol, lw=0.8, ls=(0, (3, 2)))
    axes[0].text(1, 0.01, "zero motion / chance", color=MUTED, fontsize=5.5, va="bottom")
    axes[0].set(xlabel="forecast step $k$", ylabel="cosine(transport, RAFT)", title="direction agreement")
    axes[1].set(xlabel="forecast step $k$", ylabel="EPE (patches)", title="end-point error")
    for ax in axes[:2]:
        ax.set_xticks(list(data[next(d for d in data if data[d] is not None)]["ks"]))
        ax.margins(x=0.35)
    # per-episode distribution at the largest horizon
    ax = axes[2]
    vals, labels = [], []
    for ds, d in data.items():
        if d is not None:
            v = d["cos"][:, -1]; vals.append(v[np.isfinite(v)]); labels.append(DS_LABEL[ds])
    parts = ax.violinplot(vals, showmedians=True, widths=0.7)
    for b in parts["bodies"]:
        b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor(col)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(col); parts[key].set_linewidth(0.8)
    ax.axhline(0, color=pcol, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
    ax.set(ylabel="cosine per test episode", title=f"distribution, $k={int(data[next(d for d in data if data[d] is not None)]['ks'][-1])}$")
    fig.savefig(FIG / "flow_agreement.pdf"); plt.close(fig)


# ============================================================================ trade-off (+ gain by motion)
def fig_tradeoff():
    """Skill vs. action-ranking error on every benchmark and the gain over Direct by motion: make_tradeoff.py."""
    import make_tradeoff
    make_tradeoff.main()


# ============================================================================ qualitative machinery
class Ctx:
    """Lazy access to caches, frames, models, decoders and RAFT for one dataset."""

    def __init__(self, ds, device):
        from shiftwm.v2 import analysis as A
        self.A, self.ds, self.device = A, ds, device
        self.root = A.cache_root(ds)
        self.manifest, self.stats = A.read_cache(self.root)
        self.frames = A.FrameSource(self.root)
        self.rows = {r["id"]: r for r in self.manifest["episodes"]}
        self._models, self._dec = {}, None
        self.H, self.K = 3, 10

    def test_rows(self):
        return [r for r in self.manifest["episodes"] if r["split"] == "test" and r["T"] >= self.H + self.K]

    def model(self, arm):
        if arm not in self._models:
            runs = self.A.run_dirs(self.ds, arm)
            if arm not in ("persistence", "linear") and not runs:
                self._models[arm] = None
            else:
                run = runs[0] if runs else None
                self._models[arm] = self.A.load_model(arm, run, self.manifest, self.H, self.K, self.device)
                PROV.setdefault("checkpoints", {})[f"{self.ds}/{arm}"] = str(run / "best.pt") if run else "param-free"
        return self._models[arm]

    def decoder(self):
        if self._dec is None:
            p = self.A.decoder_path(self.ds)
            self._dec = self.A.load_decoder(p, self.device) if p.exists() else False
        return self._dec or None

    def features(self, ep):
        if getattr(self, "_fcache", (None,))[0] == ep:
            return self._fcache[1]
        self._fcache = (ep, self._load_features(ep))
        return self._fcache[1]

    def _load_features(self, ep):
        with np.load(self.root / self.rows[ep]["file"]) as z:
            f, a = z["features"].astype(np.float32), z["actions"].astype(np.float32)
        fm, fs = np.array(self.stats["feature_mean"], np.float32), np.array(self.stats["feature_std"], np.float32)
        am, ast = np.array(self.stats["action_mean"], np.float32), np.array(self.stats["action_std"], np.float32)
        return ((f.reshape(len(f), -1, f.shape[-1]) - fm) / fs), (a - am) / ast

    def window(self, ep, t0s):
        import torch
        f, a = self.features(ep)
        t0s = np.atleast_1d(t0s)
        hist = np.stack([f[t - self.H + 1:t + 1] for t in t0s])
        past = np.stack([a[t - self.H + 1:t] for t in t0s])
        fut = np.stack([a[t:t + self.K] for t in t0s])
        tgt = np.stack([f[t + 1:t + 1 + self.K] for t in t0s])
        T = lambda x: torch.tensor(x, device=self.device)  # noqa: E731
        return T(hist), T(past), T(fut), T(tgt)

    def forecast(self, arm, ep, t0s, details=False):
        m = self.model(arm)
        if m is None:
            return None
        hist, past, fut, _ = self.window(ep, t0s)
        return self.A.predict(m, hist, past, fut, details=details)

    def decode(self, z):
        dec = self.decoder()
        if dec is None or z is None:
            return None
        return (self.A.decode(dec, z).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def frame(self, ep, t):
        if self.frames.kind == "droid":
            return self.frames.load_native(ep, [t])[0]
        return self.frames.load(ep, [t])[0]

    def frame224(self, ep, ts):
        return self.frames.load(ep, ts)

    def transport(self, ep, t0, k):
        out = self.forecast("shiftwm", ep, t0, details=True)
        if out is None:
            return None
        _, det = out
        m = self.model("shiftwm")
        dx, dy, _ = self.A.expected_offsets(det["weights"][:, k - 1], m.config)
        g = m.config.grid
        gate = det["gate"][0, k - 1, :, 0].float().cpu().numpy().reshape(g, g)
        return dx[0].cpu().numpy().reshape(g, g), dy[0].cpu().numpy().reshape(g, g), gate

    def raft(self, ep, t0, k):
        import torch
        if not hasattr(self, "_raft"):
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
            w = Raft_Large_Weights.C_T_SKHT_V2
            self._raft = (raft_large(weights=w).to(self.device).eval(), w.transforms())
        net, tf = self._raft
        im = torch.from_numpy(self.frame224(ep, [t0, t0 + k])).to(self.device).permute(0, 3, 1, 2).float() / 255
        x, y = tf(im[:1], im[1:])
        with torch.no_grad():
            fl = net(x.contiguous(), y.contiguous(), num_flow_updates=12)[-1]
        p = self.A.pooled_flow(fl, self.manifest["grid"])[0].cpu().numpy()
        return p[0], p[1]


def tag(ax, text):
    ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=4.8, color="white", va="top", ha="left",
            bbox=dict(fc=INK, alpha=0.6, lw=0, pad=0.8))


def show(ax, img, title=None, label=None):
    """Image panel; pending box (text = label or title) when img is None."""
    if img is None:
        pending(ax, label or title or ""); return
    ax.imshow(img, aspect="auto", interpolation="lanczos" if img.ndim == 3 else "nearest")
    blank(ax)
    if title:
        ax.set_title(title, fontsize=5.5, fontweight="normal", pad=1.5)


def arrows(ax, img, tail_dx, tail_dy, udx, udy, mask, color, title=None, label=None):
    """Content-motion arrows on img. Tail = patch centre + tail offset; vector (udx, udy) in patches."""
    if img is None:
        pending(ax, label or title or ""); return
    h, w = img.shape[:2]
    g = udx.shape[0]
    X, Y = np.meshgrid((np.arange(g) + 0.5) * w / g, (np.arange(g) + 0.5) * h / g)
    sx, sy = X + tail_dx * w / g, Y + tail_dy * h / g
    ax.imshow(img, aspect="auto", alpha=0.85)
    if mask.any():
        ax.quiver(sx[mask], sy[mask], (udx * w / g)[mask], (udy * h / g)[mask], color=color, angles="xy",
                  scale_units="xy", scale=1, width=0.008, headwidth=3.5, headlength=3.5, headaxislength=3.0,
                  minlength=0.1)
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    blank(ax)
    if title:
        ax.set_title(title, fontsize=5.5, fontweight="normal", pad=1.5)


def transport_panel(ax, ctx, ep, t0, k, obs, title=None, label="transport"):
    """Arrows from the expected source (p + o) to each patch p (content motion -o), top-30% gate patches."""
    tr = safe(lambda: ctx.transport(ep, t0, k))
    if tr is None or obs is None:
        pending(ax, label); return None
    dx, dy, gate = tr
    mask = gate > np.quantile(gate, 0.7)
    arrows(ax, obs, dx, dy, -dx, -dy, mask, METHODS["shiftwm"][1], title, label)
    return gate


def flow_panel(ax, ctx, ep, t0, k, obs, title=None, label="RAFT flow"):
    """Patch-pooled RAFT flow observed -> future, arrows where |flow| > 0.5 patch."""
    fl = safe(lambda: ctx.raft(ep, t0, k))
    if fl is None or obs is None:
        pending(ax, label); return
    fx, fy = fl
    arrows(ax, obs, np.zeros_like(fx), np.zeros_like(fy), fx, fy, np.hypot(fx, fy) > 0.5, "#E69F00", title, label)


def gate_panel(ax, gate, vmax, title=None, label="gate"):
    if gate is None:
        pending(ax, label); return None
    im = ax.imshow(gate, cmap="viridis", vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    blank(ax)
    if title:
        ax.set_title(title, fontsize=5.5, fontweight="normal", pad=1.5)
    return im


def safe(fn):
    try:
        return fn()
    except Exception as e:  # missing checkpoints/frames/decoder -> pending panel
        PROV.setdefault("errors", []).append(repr(e)[:200])
        return None


def contexts(device):
    out = {}
    for ds in list(DS_LABEL) + list(QUAL_EXTRA):
        c = safe(lambda ds=ds: Ctx(ds, device))
        if c is not None:
            out[ds] = c
    return out


def gallery_order(ids):
    return sorted(ids, key=lambda i: hashlib.sha256(f"gallery:{i}".encode()).hexdigest())


def mid_t0(ctx, ep):
    T = ctx.rows[ep]["T"]
    return int(np.clip(T // 2, ctx.H - 1, T - 1 - ctx.K))


# ---------------------------------------------------------------------------- selection
def select_qualitative(ctx, quantiles):
    cache = AN / "qualitative" / f"selection_{ctx.ds}.json"
    if cache.exists():
        sel = json.loads(cache.read_text())
        if sel.get("quantiles") == list(quantiles):
            return [tuple(x) for x in sel["picked"]]
    scores = []
    for r in ctx.test_rows():
        f, _ = ctx._load_features(r["id"])
        t = np.arange(ctx.H - 1, r["T"] - ctx.K)
        ch = np.concatenate([((f[u + ctx.K] - f[u]) ** 2).mean((1, 2)) for u in np.array_split(t, max(1, len(t) // 32))])
        del f
        j = int(np.argmax(ch))
        scores.append((float(ch[j]), r["id"], int(t[j])))
    scores.sort()
    picked = [scores[min(len(scores) - 1, int(round(q * (len(scores) - 1))))] for q in quantiles]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"rule": "max-over-t0 true k=10 change; episodes at quantiles", "quantiles": list(quantiles),
                                 "picked": [(e, t, s) for s, e, t in picked]}, indent=1))
    return [(e, t, s) for s, e, t in picked]


# ---------------------------------------------------------------------------- qualitative_transport
def fig_qualitative(ctxs, k=10):
    rows = []
    extra = [("language_table", (0.5, 0.9))] if has_runs(ctxs.get("language_table")) else []
    for ds, qs in [("droid", (0.5, 0.75, 0.9)), ("openh_hamlyn", (0.5, 0.9))] + extra:
        ctx = ctxs.get(ds)
        sel = safe(lambda: select_qualitative(ctx, qs)) if ctx else None
        rows += [(ds, ctx, e, t) for e, t, _ in sel] if sel else [(ds, None, None, None)] * len(qs)
    PROV["qualitative"] = [(ds, e, t) for ds, _, e, t in rows]
    cols = ["observed $t$", f"true $t{{+}}{k}$", "decoded ShiftWM", "decoded AR", "transport (ShiftWM)",
            f"gate $g_{{{k}}}$", "RAFT flow"]
    fig = plt.figure(figsize=(7.0, 0.6 * len(rows) + 0.35))
    gs = fig.add_gridspec(len(rows), len(cols), wspace=0.04, hspace=0.12, left=0.055, right=0.995, top=0.94, bottom=0.02)
    gates = []
    for i, (ds, ctx, ep, t0) in enumerate(rows):
        axs = [fig.add_subplot(gs[i, j]) for j in range(len(cols))]
        ttl = cols if i == 0 else [None] * len(cols)
        if ctx is None or ep is None:
            for ax, c in zip(axs, cols):
                pending(ax, c)
            continue
        obs = safe(lambda: ctx.frame(ep, t0)); fut = safe(lambda: ctx.frame(ep, t0 + k))
        show(axs[0], obs, ttl[0], cols[0]); show(axs[1], fut, ttl[1], cols[1])
        for j, arm in ((2, "shiftwm"), (3, "ar")):
            z = safe(lambda arm=arm: ctx.forecast(arm, ep, t0))
            img = safe(lambda z=z: ctx.decode(z[:, k - 1])) if z is not None else None
            show(axs[j], img[0] if img is not None else None, ttl[j], cols[j])
        gate = transport_panel(axs[4], ctx, ep, t0, k, obs, ttl[4], cols[4])
        gates.append((axs[5], gate))
        flow_panel(axs[6], ctx, ep, t0, k, obs, ttl[6], cols[6])
        axs[0].set_ylabel(f"{NAME[ds]}\n{short(ep)}", fontsize=5.5, color=INK, rotation=90, labelpad=2)
        axs[0].yaxis.set_visible(True); axs[0].set_yticks([])
    vmax = max([float(g.max()) for _, g in gates if g is not None] or [1.0])
    for i, (ax, g) in enumerate(gates):
        gate_panel(ax, g, vmax, f"gate $g_{{{k}}}$ (0 to {vmax:.2f})" if i == 0 else None, cols[5])
    fig.savefig(FIG / "qualitative_transport.pdf"); plt.close(fig)


def short(ep):
    if not ep:
        return ""
    return ep.replace("droid-", "")[:10] if ep.startswith("droid-") else "ep " + ep.split("__")[-1]


# ---------------------------------------------------------------------------- galleries
def gallery_block(fig, gs_block, ctx, ep, t0, ks=(1, 3, 5, 10), header=True):
    """Label column + 4 rows (true, AR, Direct, ShiftWM) x (context + len(ks)) columns. Context column:
    observed frame (row 1), ShiftWM gate (row 3) and transport arrows (row 4) at the largest k."""
    sub = gs_block.subgridspec(4, 2 + len(ks), wspace=0.03, hspace=0.06, width_ratios=[0.22] + [1] * (1 + len(ks)))
    labels = ["true", "AR", "Direct", "ShiftWM"]
    dec = {}
    for arm in ("ar", "direct", "shiftwm"):
        z = safe(lambda arm=arm: ctx.forecast(arm, ep, t0))
        dec[arm] = safe(lambda z=z: ctx.decode(z[0, [k - 1 for k in ks]])) if z is not None else None
    obs = safe(lambda: ctx.frame(ep, t0))
    kmax = ks[-1]
    for r, lab in enumerate(labels):
        lax = fig.add_subplot(sub[r, 0]); lax.set_axis_off()
        lax.text(0.5, 0.5, lab, rotation=90, ha="center", va="center", fontsize=5.5, color=INK, transform=lax.transAxes)
        for c in range(1 + len(ks)):
            ax = fig.add_subplot(sub[r, c + 1])
            title = (["observed"] + [f"$k{{=}}{k}$" for k in ks])[c] if (r == 0 and header) else None
            if c == 0:
                if r == 0:
                    show(ax, obs, title, "observed")
                elif r == 1:
                    ax.set_axis_off()
                elif r == 2:
                    tr = safe(lambda: ctx.transport(ep, t0, kmax))
                    if gate_panel(ax, tr[2] if tr else None, float(tr[2].max()) if tr else 1.0,
                                  label=f"gate $k{{=}}{kmax}$") is not None:
                        tag(ax, f"gate, max {tr[2].max():.2f}")
                elif transport_panel(ax, ctx, ep, t0, kmax, obs, label=f"transport $k{{=}}{kmax}$") is not None:
                    tag(ax, f"transport k={kmax}")
            elif r == 0:
                show(ax, safe(lambda c=c: ctx.frame(ep, t0 + ks[c - 1])), title, f"frame $k{{=}}{ks[c - 1]}$")
            else:
                arm = {1: "ar", 2: "direct", 3: "shiftwm"}[r]
                show(ax, dec[arm][c - 1] if dec[arm] is not None else None, None, f"decoded {lab}")


def gallery(ctx, picks, name, ncols=2):
    nblocks = len(picks)
    nrows = int(np.ceil(nblocks / ncols))
    height = 1.8 * nrows + 0.2
    fig = plt.figure(figsize=(7.0, height))
    gs = fig.add_gridspec(nrows, ncols, wspace=0.06, hspace=0.42, left=0.01, right=0.995,
                          top=1 - 0.28 / height, bottom=0.01)
    for b in range(nrows * ncols):
        r, c = divmod(b, ncols)
        if b >= nblocks:
            fig.add_subplot(gs[r, c]).set_axis_off(); continue
        label, ep, t0 = picks[b]
        pos = gs[r, c].get_position(fig)
        if ctx is None or ep is None:
            pending(fig.add_subplot(gs[r, c]), f"{label}\nrollout block")
        else:
            gallery_block(fig, gs[r, c], ctx, ep, t0)
        head = f"{label}" + (f"  ({short(ep)}, $t{{=}}{t0}$)" if ep else "")
        fig.text(pos.x0, pos.y1 + 0.17 / height, head, fontsize=6, color=INK, fontweight="bold", va="bottom")
    fig.savefig(FIG / name); plt.close(fig)


def fig_gallery_droid(ctxs):
    ctx = ctxs.get("droid")
    picks = []
    if ctx:
        ids = gallery_order([r["id"] for r in ctx.test_rows()])[:6]
        picks = [(f"DROID #{i + 1}", e, mid_t0(ctx, e)) for i, e in enumerate(ids)]
    picks = picks or [(f"DROID #{i + 1}", None, None) for i in range(6)]
    PROV["gallery_droid"] = picks
    gallery(ctx, picks, "gallery_droid.pdf")


def fig_gallery_surgical(ctxs):
    ctx = ctxs.get("openh_hamlyn")
    tasks = ["knot_tying", "needle_grasp_and_handover", "peg_transfer", "suturing_1", "suturing_2",
             "tissue_lifting", "tissue_retraction"]
    picks = []
    for t in tasks:
        ids = gallery_order([r["id"] for r in ctx.test_rows() if r["task"] == t]) if ctx else []
        picks.append((t.replace("_", " "), ids[0], mid_t0(ctx, ids[0])) if ids else (t.replace("_", " "), None, None))
    PROV["gallery_surgical"] = picks
    gallery(ctx if ctx else None, picks, "gallery_surgical.pdf")


def has_runs(ctx):
    """True when the dataset's cache exists and ShiftWM and AR have finished runs."""
    return ctx is not None and all(ctx.A.run_dirs(ctx.ds, a) for a in ("shiftwm", "ar", "direct"))


def fig_gallery_language_table(ctxs):
    """6 pseudo-random Language-Table test episodes (sha256 order), window at mid-episode; skipped (no file) until
    the ShiftWM/AR/Direct runs exist."""
    ctx = ctxs.get("language_table")
    if not has_runs(ctx):
        print("language_table gallery: runs missing, skipped", flush=True)
        return
    ids = gallery_order([r["id"] for r in ctx.test_rows()])[:6]
    picks = [(f"Language-Table #{i + 1}", e, mid_t0(ctx, e)) for i, e in enumerate(ids)]
    PROV["gallery_language_table"] = picks
    gallery(ctx, picks, "gallery_language_table.pdf")


# ---------------------------------------------------------------------------- failures
def pick_failures(ctx, n):
    runs = ctx.A.run_dirs(ctx.ds, "shiftwm")
    if not runs or not (runs[0] / "eval_test.npz").exists():
        return []
    z = np.load(runs[0] / "eval_test.npz", allow_pickle=True)
    order = np.argsort(-z["mse"][:, -1])[:n]
    out = []
    for i in order:
        ep = str(z["episodes"][i])
        T = ctx.rows[ep]["T"]
        t0s = np.arange(ctx.H - 1, T - ctx.K, 2)
        errs = []
        bs = 16 if str(ctx.device).startswith("cuda") else 2
        for j in range(0, len(t0s), bs):
            p = ctx.forecast("shiftwm", ep, t0s[j:j + bs])
            _, _, _, tgt = ctx.window(ep, t0s[j:j + bs])
            errs.append(((p[:, -1] - tgt[:, -1]) ** 2).mean((1, 2)).cpu().numpy())
        errs = np.concatenate(errs)
        out.append((ep, int(t0s[int(np.argmax(errs))]), float(errs.max())))
    return out


def fig_failures(ctxs, k=10):
    rows = []
    for ds, n in (("droid", 3), ("openh_hamlyn", 2)):
        ctx = ctxs.get(ds)
        sel = safe(lambda: pick_failures(ctx, n)) if ctx else None
        rows += [(ds, ctx, *s) for s in sel] if sel else [(ds, None, None, None, None)] * n
    PROV["failures"] = [(ds, e, t, m) for ds, _, e, t, m in rows]
    cols = ["observed $t$", f"true $t{{+}}{k}$", "decoded ShiftWM", "per-patch error", "transport", "gate"]
    fig = plt.figure(figsize=(7.0, 0.72 * len(rows) + 0.35))
    gs = fig.add_gridspec(len(rows), len(cols), wspace=0.04, hspace=0.14, left=0.06, right=0.995, top=0.94, bottom=0.02)
    for i, (ds, ctx, ep, t0, err) in enumerate(rows):
        axs = [fig.add_subplot(gs[i, j]) for j in range(len(cols))]
        ttl = cols if i == 0 else [None] * len(cols)
        if ctx is None or ep is None:
            for ax, c in zip(axs, cols):
                pending(ax, c)
            continue
        obs = safe(lambda: ctx.frame(ep, t0))
        show(axs[0], obs, ttl[0], cols[0]); show(axs[1], safe(lambda: ctx.frame(ep, t0 + k)), ttl[1], cols[1])
        z = safe(lambda: ctx.forecast("shiftwm", ep, t0))
        img = safe(lambda: ctx.decode(z[:, k - 1])) if z is not None else None
        show(axs[2], img[0] if img is not None else None, ttl[2], cols[2])
        tgt = safe(lambda: ctx.window(ep, t0)[3])
        if z is not None and tgt is not None:
            g = ctx.manifest["grid"]
            e = ((z[0, k - 1] - tgt[0, k - 1]) ** 2).mean(-1).cpu().numpy().reshape(g, g)
            axs[3].imshow(e, cmap="magma", aspect="auto", interpolation="nearest"); blank(axs[3])
            if ttl[3]:
                axs[3].set_title(ttl[3], fontsize=5.5, fontweight="normal", pad=1.5)
        else:
            pending(axs[3], "error map")
        gate = transport_panel(axs[4], ctx, ep, t0, k, obs, ttl[4], cols[4])
        gate_panel(axs[5], gate, float(gate.max()) if gate is not None else 1.0, ttl[5], cols[5])
        if gate is not None:
            tag(axs[5], f"max {gate.max():.2f}")
        axs[0].set_ylabel(f"{DS_LABEL[ds]}\n{short(ep)}\nMSE {err:.2f}", fontsize=5.2, color=INK, labelpad=2)
        axs[0].yaxis.set_visible(True); axs[0].set_yticks([])
    fig.savefig(FIG / "failures.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------- rollouts (replaces galleries + failures)
def window_scores(ctx, k=10, stride=2):
    """k-step whole-frame feature MSE of ShiftWM, Direct and AR and the true change (= persistence error) for every
    stride-2 test window. Cached in results/v2/analysis/qualitative/window_scores_<ds>.json."""
    cache = AN / "qualitative" / f"window_scores_{ctx.ds}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = []
    bs = 16 if str(ctx.device).startswith("cuda") else 2          # CPU: memory-bound, as in pick_failures
    for r in ctx.test_rows():
        ep = r["id"]; t0s = np.arange(ctx.H - 1, r["T"] - ctx.K, stride)
        for j in range(0, len(t0s), bs):
            tt = t0s[j:j + bs]
            hist, _, _, tgt = ctx.window(ep, tt)
            e = {"change": ((tgt[:, k - 1] - hist[:, -1]) ** 2).mean((1, 2)).cpu().numpy()}
            for arm in ("shiftwm", "direct", "ar"):
                z = ctx.forecast(arm, ep, tt)
                e[arm] = ((z[:, k - 1].float() - tgt[:, k - 1]) ** 2).mean((1, 2)).cpu().numpy()
            out += [dict(ep=ep, t0=int(t), **{a_: float(v[i]) for a_, v in e.items()}) for i, t in enumerate(tt)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def largest_margin(ctx, n):
    """Among test windows in the top half of true motion, the n windows (distinct episodes) with the largest relative
    k=10 gain 1 - err_ShiftWM / min(err_Direct, err_AR)."""
    W_ = window_scores(ctx)
    thr = float(np.median([w["change"] for w in W_]))
    cand = sorted((w for w in W_ if w["change"] >= thr), key=lambda w: -(1 - w["shiftwm"] / min(w["direct"], w["ar"])))
    out, seen = [], set()
    for w in cand:
        if w["ep"] not in seen:
            seen.add(w["ep"]); out.append(w)
        if len(out) == n:
            break
    return out


def rollout_windows(ctxs):
    """Rows of the rollout figure (disclosed rule, no manual picking): the 3 DROID and the 1 Hamlyn test windows with
    the largest margin (largest_margin), then the worst DROID window for ShiftWM (largest ShiftWM k=10 MSE; the
    failure rule). Cached in results/v2/analysis/qualitative/rollout_windows.json."""
    cache = AN / "qualitative" / "rollout_windows.json"
    if cache.exists():
        rows = [tuple(r) for r in json.loads(cache.read_text())]
        if rows and rows[0][1].startswith("largest margin"):
            return rows
    rows = []
    for ds, n in (("droid", 3), ("openh_hamlyn", 1)):
        for i, w in enumerate(largest_margin(ctxs[ds], n)):
            rows.append((ds, f"largest margin #{i + 1}", w["ep"], w["t0"]))
    sel = pick_failures(ctxs["droid"], 1)
    rows.append(("droid", "largest error", sel[0][0], sel[0][1]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=1))
    return rows


def fig_rollouts(ctxs, k=10):
    """Matched rollouts: per window, frame t with ShiftWM's learned transport; AR / Direct / ShiftWM per-patch error at
    step k on one shared scale over the true frame t+k, moving patches (top 25% true change, as Table regions) outlined,
    with static / moving means; and the whole-frame error of every method over k = 1..10 on this window."""
    import matplotlib as mpl
    import make_figures as mf
    from matplotlib.patches import FancyArrowPatch, Rectangle
    rows = [r for r in rollout_windows(ctxs) if r[1] != "largest error"]     # largest-margin windows only
    PROV["rollouts"] = rows
    arms = [("ar", "AR"), ("direct", "Direct"), ("shiftwm", "ShiftWM")]
    data = []
    for ds, lab, ep, t0 in rows:
        ctx = ctxs[ds]
        _, _, _, tgt = ctx.window(ep, t0)
        tgt = tgt[0].cpu().numpy()                                      # [K, N, C]
        hist = ctx.window(ep, t0)[0][0, -1].cpu().numpy()                # frame t features
        pe = ((hist[None] - tgt) ** 2).mean(-1)                          # persistence error = true change, [K, N]
        E = {"persistence": pe}
        for arm, _ in arms:
            z = ctx.forecast(arm, ep, t0)
            E[arm] = ((z[0].float().cpu().numpy() - tgt) ** 2).mean(-1)
        dx, dy, gate = ctx.transport(ep, t0, k)
        data.append(dict(ds=ds, lab=lab, ep=ep, t0=t0, E=E, dx=dx, dy=dy, gate=gate,
                         obs=ctx.frame(ep, t0), fut=ctx.frame(ep, t0 + k), g=ctx.manifest["grid"]))
    n = len(data)
    W, lab_w, tw, gap, pw = 5.5, 0.27, 0.86, 0.05, 0.95
    th = tw * 180 / 320
    rh = th + 0.17
    H = 0.3 + n * rh + 0.06 + 0.26
    fig = plt.figure(figsize=(W, H))
    ax0 = fig.add_axes([0, 0, 1, 1]); ax0.set_xlim(0, W); ax0.set_ylim(0, H); ax0.axis("off")
    xs = [lab_w + i * (tw + gap) for i in range(4)]
    heads = ["frame $t$ + ShiftWM transport", "AR error", "Direct error", "ShiftWM error"]
    cols = [mf.INK, METHODS["ar"][1], METHODS["direct"][1], METHODS["shiftwm"][1]]
    for x, t, c in zip(xs, heads, cols):
        ax0.text(x + tw / 2, H - 0.05, t, ha="center", va="top", fontsize=mf.FS_LABEL, fontweight="bold", color=c)
    px = xs[3] + tw + 0.42
    ax0.text(px + (W - 0.05 - px) / 2, H - 0.05, "error over $k$", ha="center", va="top", fontsize=mf.FS_LABEL,
             fontweight="bold", color=mf.INK)
    vmax = float(np.quantile(np.concatenate([d["E"][a][k - 1] for d in data for a, _ in arms]), 0.97))
    GREEN = METHODS["shiftwm"][1]
    for i, d in enumerate(data):
        y = H - 0.3 - (i + 1) * rh + 0.15 - (0.06 if d["lab"] == "largest error" else 0)
        put = lambda im, x: ax0.imshow(np.asarray(im), extent=(x, x + tw, y, y + th), aspect="auto",   # native aspect
                                       interpolation="lanczos", zorder=2)
        g = d["g"]
        chg = d["E"]["persistence"][k - 1].reshape(g, g)
        moving = chg >= np.quantile(chg, 0.75)
        name = ("DROID" if d["ds"] == "droid" else "Hamlyn") + "\n" + d["lab"].replace("largest error", "worst").replace(
            "largest margin", "margin")
        ax0.text(lab_w - 0.1, y + th / 2, name, rotation=90, ha="center", va="center", fontsize=mf.FS_NOTE,
                 color=mf.LOSS if d["lab"] == "largest error" else mf.INK, linespacing=1.0)
        # frame t + transport (gate >= 0.5, moves >= half a patch), as in Fig. 1
        put(d["obs"], xs[0])
        cw, ch = tw / g, th / g
        mag = np.hypot(d["dx"], d["dy"]); mv = (d["gate"] >= 0.5) & (mag >= 0.5)
        for r, c in zip(*np.nonzero(mv)):
            ax0.add_patch(Rectangle((xs[0] + c * cw, y + th - (r + 1) * ch), cw, ch, fc=GREEN, alpha=0.45, ec="none", zorder=3))
        import matplotlib.patheffects as pe_
        for r, c in zip(*np.nonzero(mv)):
            qx, qy = xs[0] + (c + 0.5) * cw, y + th - (r + 0.5) * ch
            a_ = ax0.add_patch(FancyArrowPatch((qx + d["dx"][r, c] * cw, qy - d["dy"][r, c] * ch), (qx, qy),
                                               arrowstyle="-|>,head_length=0.5,head_width=0.22", mutation_scale=2.2,
                                               lw=0.5, color="#002E21", shrinkA=0, shrinkB=0, zorder=5))
            a_.set_path_effects([pe_.Stroke(linewidth=1.1, foreground="white", alpha=0.6), pe_.Normal()])
        mf._box(ax0, xs[0], y, tw, th, "#9AA3AE", lw=0.5)
        # per-patch error maps, shared scale
        for j, (arm, _) in enumerate(arms, start=1):
            e = d["E"][arm][k - 1].reshape(g, g)
            put(mf._grey(np.asarray(d["fut"])), xs[j])
            ax0.imshow(np.clip(e / vmax, 0, 1), cmap=mf.ERR_CMAP, vmin=0, vmax=1, extent=(xs[j], xs[j] + tw, y, y + th),
                       interpolation="bicubic", zorder=3)
            mf._outline(ax0, moving, xs[j], y, tw, th, color="white", lw=0.45, ls=(0, (1.5, 1)), zorder=4)
            mf._box(ax0, xs[j], y, tw, th, cols[j], lw=0.9 if arm == "shiftwm" else 0.6)
            ax0.text(xs[j] + tw / 2, y - 0.025, f"static {e[~moving].mean():.2f}  moving {e[moving].mean():.2f}",
                     ha="center", va="top", fontsize=mf.FS_NOTE, color=mf.INK)
        # whole-frame error over k on this window
        sax = fig.add_axes([px / W, y / H, (W - 0.05 - px) / W, th / H])
        kk = np.arange(1, d["E"]["shiftwm"].shape[0] + 1)
        sax.plot(kk, d["E"]["persistence"].mean(1), color=METHODS["persistence"][1], ls=(0, (3, 2)), lw=0.9)
        for arm, _ in arms:
            _, c, _, mk = METHODS[arm]
            sax.plot(kk, d["E"][arm].mean(1), color=c, lw=1.4 if arm == "shiftwm" else 1.0, zorder=3 if arm == "shiftwm" else 2)
        sax.set_xticks([1, 5, 10]); sax.tick_params(labelsize=mf.FS_TICK, length=1.5, pad=1)
        sax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3)); sax.grid(axis="x", visible=False)
        sax.set_xlim(1, 10)
        if i < n - 1:
            sax.set_xticklabels([])
    # separator before the failure rows + key
    nf = sum(1 for d in data if d["lab"] == "largest error")
    ysep = H - 0.3 - (n - nf) * rh + 0.02
    ax0.plot([0.05, W - 0.05], [ysep, ysep], color=mf.PANEL_EDGE, lw=0.6)
    ky, kx = 0.07, xs[1] + 0.02
    t_ = ax0.text(xs[0], ky + 0.03, f"feature error at $k{{=}}{k}$, shared scale:", fontsize=mf.FS_NOTE, color=MUTED,
                  ha="left", va="center")
    kx = xs[0] + 1.45
    ax0.text(kx - 0.04, ky + 0.03, "low", fontsize=mf.FS_NOTE, color=MUTED, ha="right", va="center")
    ax0.imshow(np.linspace(0, 1, 64)[None], cmap=mf.ERR_CMAP, extent=(kx, kx + 0.35, ky, ky + 0.06), aspect="auto")
    mf._box(ax0, kx, ky, 0.35, 0.06, "#C9CED6", lw=0.4)
    ax0.text(kx + 0.39, ky + 0.03, "high", fontsize=mf.FS_NOTE, color=MUTED, ha="left", va="center")
    mf._outline(ax0, np.ones((1, 1), bool), kx + 0.7, ky - 0.005, 0.07, 0.07, color=MUTED, lw=0.6, ls=(0, (1.5, 1)))
    ax0.text(kx + 0.81, ky + 0.03, "moving patches (top 25% true change)", fontsize=mf.FS_NOTE, color=MUTED, ha="left",
             va="center")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=METHODS[a][1], lw=1.2) for a in ("ar", "direct", "shiftwm")] + \
         [Line2D([], [], color=METHODS["persistence"][1], lw=0.9, ls=(0, (3, 2)))]
    fig.legend(hs, ["AR", "Direct", "ShiftWM", "copy"], loc="lower right", bbox_to_anchor=(0.995, 0.0), ncol=2,
               fontsize=mf.FS_NOTE, frameon=False, handlelength=1.3, columnspacing=0.8, borderaxespad=0.2)
    mf.qa(fig, "rollouts", 5.5)
    fig.savefig(FIG / "rollouts.pdf"); fig.savefig(FIG / "rollouts_preview.png", dpi=200); plt.close(fig)


# ============================================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    lazy = {}

    def ctxs():
        if "c" not in lazy:
            lazy["c"] = contexts(a.device)
        return lazy["c"]

    jobs = {"flow": fig_flow_agreement, "tradeoff": fig_tradeoff,
            "qualitative": lambda: fig_qualitative(ctxs()), "gallery_droid": lambda: fig_gallery_droid(ctxs()),
            "gallery_surgical": lambda: fig_gallery_surgical(ctxs()),
            "gallery_language_table": lambda: fig_gallery_language_table(ctxs()), "failures": lambda: fig_failures(ctxs()),
            "rollouts": lambda: fig_rollouts(ctxs())}
    for name, fn in jobs.items():
        if not a.only or name in a.only:
            fn(); print("wrote", name, flush=True)
    out = AN / "qualitative" / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(PROV, indent=1, default=str))


if __name__ == "__main__":
    main()
