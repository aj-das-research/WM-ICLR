"""Anatomy of ShiftWM's gains on held-out DROID (all test windows, stride 2; final checkpoints as in Table 1).

  gain map     : relative error reduction of ShiftWM vs Direct and vs AR, per horizon k (1..10) x decile of true
                 per-patch change at that horizon (deciles pooled over all windows and horizons' own changes).
  composition  : the forecast's departure from the observed grid splits exactly as  Z_hat - Z0 = g (T - Z0) + r.
                 Per horizon and region (moving = top 25% true change at k, static = rest): mean gate, and the share
                 of squared departure carried by the moved term, ||g(T-Z0)||^2 / (||g(T-Z0)||^2 + ||r||^2).
  horizon      : per-horizon moving-patch error of stay / oracle move / AR / Direct / ShiftWM with episode-bootstrap CIs.
Writes results/v2/analysis/anatomy/summary.json. Run on a GPU.
"""
import json
from pathlib import Path

import numpy as np
import torch

from shiftwm.v2.train import FeatureSplit

import sys
sys.path.insert(0, str(Path(__file__).parent))
from geometry import candidates, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2/analysis/anatomy"
H, K = 3, 10


@torch.no_grad()
def main():
    dev = "cuda"
    root = ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    data = FeatureSplit(root, "test", H, K, dev, stats, stride=2)
    models, ckpts = {}, {}
    for a in ("shiftwm", "direct", "ar"):
        models[a], ckpts[a] = load(a, dev)
    cfg = models["shiftwm"].config
    arms = ("persistence", "oracle", "shiftwm", "direct", "ar")
    err = {a: [] for a in arms}; chg_all, gate_all, mv_all, rs_all = [], [], [], []
    for i in range(0, len(data), 48):
        idx = torch.arange(i, min(i + 48, len(data)), device=dev)
        hist, past, fut, tgt = data.batch(idx); hist, tgt = hist.float(), tgt.float()
        p_sw, det = models["shiftwm"](hist, past, fut, return_details=True)
        preds = {"shiftwm": p_sw.float(), "direct": models["direct"](hist, past, fut).float(),
                 "ar": models["ar"](hist, past, fut).float()}
        z0 = hist[:, -1]
        g = det["gate"].float()                                            # [B,K,N,1]
        r = det["correction"].float()                                      # [B,K,N,C]
        moved = p_sw.float() - z0[:, None] - r                             # = g (T - Z0)
        rs_all.append(torch.stack(((moved ** 2).sum(-1), (r ** 2).sum(-1)), -1).cpu())   # [B,K,N,2]
        gate_all.append(g[..., 0].cpu())
        cand, valid = candidates(hist, cfg)
        o = []
        for k in range(K):
            d = ((cand - tgt[:, k, :, None]) ** 2).mean(-1).masked_fill(~valid[None], float("inf"))
            o.append(d.min(-1).values)
        err["oracle"].append(torch.stack(o, 1).cpu())
        err["persistence"].append(((z0[:, None] - tgt) ** 2).mean(-1).cpu())
        for a, p in preds.items():
            err[a].append(((p - tgt) ** 2).mean(-1).cpu())
        c = ((tgt - z0[:, None]) ** 2).mean(-1)                            # [B,K,N]
        chg_all.append(c.cpu()); mv_all.append((c >= c.quantile(0.75, dim=-1, keepdim=True)).cpu())
    E = {a: torch.cat(v).numpy() for a, v in err.items()}                   # [W,K,N]
    C = torch.cat(chg_all).numpy(); M = torch.cat(mv_all).numpy(); G = torch.cat(gate_all).numpy()
    RS = torch.cat(rs_all).numpy()
    ep = data.episode_of.cpu().numpy(); eps = np.unique(ep)
    # gain map
    qs = np.quantile(C, np.linspace(0, 1, 11)); dec = np.clip(np.searchsorted(qs, C, side="right") - 1, 0, 9)
    gain = {}
    for base in ("direct", "ar"):
        m = np.zeros((10, K))
        for k in range(K):
            for d in range(10):
                sel = dec[:, k] == d
                m[d, k] = 100 * (1 - E["shiftwm"][:, k][sel].mean() / E[base][:, k][sel].mean())
        gain[base] = m.tolist()
    # composition
    comp = {}
    for reg, mask in (("moving", M), ("static", ~M)):
        comp[reg] = {"gate": [float(G[:, k][mask[:, k]].mean()) for k in range(K)],
                     "move_share": [float(RS[:, k, :, 0][mask[:, k]].sum() / RS[:, k][mask[:, k]].sum()) for k in range(K)]}
    # horizon curves with episode bootstrap
    rng = np.random.default_rng(0); boot = rng.integers(0, len(eps), (2000, len(eps)))
    curves = {}
    for a in arms:
        per_w = (E[a] * M).sum(-1) / M.sum(-1)                               # [W,K]
        per_ep = np.stack([per_w[ep == u].mean(0) for u in eps])            # [U,K]
        bs = per_ep[boot].mean(1)
        curves[a] = {"mean": per_ep.mean(0).tolist(), "lo": np.quantile(bs, 0.025, 0).tolist(),
                     "hi": np.quantile(bs, 0.975, 0).tolist()}
    summ = {"windows": int(len(M)), "episodes": int(len(eps)), "checkpoints": ckpts, "decile_edges": qs.tolist(),
            "gain_map": gain, "composition": comp, "moving_curves": curves}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("gain vs direct: min %.1f max %.1f; share of cells >0: %.2f" % (np.min(gain["direct"]), np.max(gain["direct"]),
          (np.array(gain["direct"]) > 0).mean()))
    print("gain vs ar: min %.1f max %.1f; share >0: %.2f" % (np.min(gain["ar"]), np.max(gain["ar"]), (np.array(gain["ar"]) > 0).mean()))
    print("composition", {r: {k: [round(x, 2) for x in v[::3]] for k, v in d.items()} for r, d in comp.items()})


if __name__ == "__main__":
    main()
