"""Feature-space geometry of forecasting on held-out DROID (the data counterpart of Fig. 3 / Propositions 1-2).

For every test window (stride 2) and every patch, at horizon k:
  persistence  ||z_{0,i} - z_{k,i}||^2                    (stay)
  oracle move  min_j ||z_{0,j} - z_{k,i}||^2 over the S*w*w observed candidates of the transport window
               (the best single observed feature, i.e. the best a pure transport could do; model-free)
  ShiftWM / Direct / AR forecast errors (final checkpoints: results/v2s, else results/v2 -- as in Table 1)
All per-patch errors are channel means of squared standardised features. Patches are split into moving (top 25%
true change within the window) and static. Also stored: 3 example patches for a 2-D view, chosen by a fixed rule
(moving, gate > 0.5 patches of the windows at the 25/50/75th percentile of mean true change; within each, the patch
with the median true change among those), with their candidates, transport weights and all forecasts.
Writes results/v2/analysis/geometry/{summary.json, examples.npz}. Run on a GPU.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from shiftwm.v2.models import V2WorldModel
from shiftwm.v2.train import FeatureSplit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2/analysis/geometry"
H, K = 3, 10


def load(arm, dev):
    for base in ("results/v2s", "results/v2"):
        ck = sorted((ROOT / base / "droid/dinov2s" / arm).glob("s*/best.pt"))
        if ck:
            st = torch.load(ck[0], map_location=dev)
            m = V2WorldModel(st["config"]).to(dev).eval(); m.load_state_dict(st["model"])
            return m, str(ck[0].relative_to(ROOT))
    raise FileNotFoundError(arm)


def candidates(hist, cfg):
    """[B,N,S*w*w,C] observed candidate features in each patch's window, and [N,S*w*w] validity."""
    b, h, n, c = hist.shape
    g, w, s = cfg.grid, cfg.window, cfg.sources
    t = hist[:, -s:].permute(0, 1, 3, 2).reshape(b * s, c, g, g)
    u = F.unfold(t, w, padding=w // 2).reshape(b, s, c, w * w, n).permute(0, 4, 1, 3, 2).reshape(b, n, s * w * w, c)
    valid = F.unfold(torch.ones(1, 1, g, g, device=hist.device), w, padding=w // 2)[0].T.bool().repeat(1, s)
    return u, valid


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
    keys = ("persistence", "oracle", "shiftwm", "direct", "ar")
    err = {k: [] for k in keys}; mov = []; wchg = []; gates = []
    for i in range(0, len(data), 48):
        idx = torch.arange(i, min(i + 48, len(data)), device=dev)
        hist, past, fut, tgt = data.batch(idx)
        hist, tgt = hist.float(), tgt.float()
        p_sw, det = models["shiftwm"](hist, past, fut, return_details=True)
        preds = {"shiftwm": p_sw.float(), "direct": models["direct"](hist, past, fut).float(),
                 "ar": models["ar"](hist, past, fut).float()}
        cand, valid = candidates(hist, cfg)
        per_k = {k: [] for k in keys}
        for k in range(K):
            y = tgt[:, k]
            d = ((cand - y[:, :, None]) ** 2).mean(-1).masked_fill(~valid[None], float("inf"))
            per_k["oracle"].append(d.min(-1).values)
            per_k["persistence"].append(((hist[:, -1] - y) ** 2).mean(-1))
            for a, p in preds.items():
                per_k[a].append(((p[:, k] - y) ** 2).mean(-1))
        for k in keys:
            err[k].append(torch.stack(per_k[k], 1).cpu())                  # [B,K,N]
        chg = ((tgt[:, K - 1] - hist[:, -1]) ** 2).mean(-1)
        mov.append((chg >= chg.quantile(0.75, dim=-1, keepdim=True)).cpu()); wchg.append(chg.mean(-1).cpu())
        gates.append(det["gate"][:, K - 1, :, 0].float().cpu())
    E = {k: torch.cat(v).numpy() for k, v in err.items()}                   # [W,K,N]
    M = torch.cat(mov).numpy(); G = torch.cat(gates).numpy(); WC = torch.cat(wchg).numpy()
    ep = data.episode_of.cpu().numpy() if torch.is_tensor(data.episode_of) else np.asarray(data.episode_of)

    def region_mean(e, mask):                                              # episode-level means (episodes = unit)
        per_w = (e * mask).sum(-1) / np.maximum(mask.sum(-1), 1)
        return np.array([per_w[ep == u].mean() for u in np.unique(ep)])

    rng = np.random.default_rng(0); U = len(np.unique(ep))
    boot = rng.integers(0, U, (5000, U))
    summ = {"windows": int(len(M)), "episodes": int(U), "checkpoints": ckpts, "horizon_k": K, "regions": {}}
    for reg, mask in (("moving", M), ("static", ~M), ("all", np.ones_like(M))):
        r = {}
        for k in keys:
            per_ep = region_mean(E[k][:, K - 1], mask)
            per_ep_avg = np.mean([region_mean(E[k][:, j], mask) for j in range(K)], 0)
            r[k] = {"k10": float(per_ep.mean()), "avg": float(per_ep_avg.mean()),
                    "by_k": [float(region_mean(E[k][:, j], mask).mean()) for j in range(K)]}
        for base in ("direct", "ar"):
            d = region_mean(E[base][:, K - 1], mask) - region_mean(E["shiftwm"][:, K - 1], mask)
            bs = d[boot].mean(1)
            r[f"shiftwm_minus_{base}_k10"] = {"mean": float(-d.mean()), "ci": [float(-np.quantile(bs, 0.975)), float(-np.quantile(bs, 0.025))]}
        # share of the oracle-transport gain over persistence that each model recovers
        gp = r["persistence"]["k10"] - r["oracle"]["k10"]
        r["oracle_gain_share"] = {a: float((r["persistence"]["k10"] - r[a]["k10"]) / gp) for a in ("shiftwm", "direct", "ar")}
        summ["regions"][reg] = r
    # per-patch win rate on moving patches at k=10
    dm = (E["direct"][:, K - 1] - E["shiftwm"][:, K - 1])[M]; am = (E["ar"][:, K - 1] - E["shiftwm"][:, K - 1])[M]
    summ["moving_patch_winrate"] = {"vs_direct": float((dm > 0).mean()), "vs_ar": float((am > 0).mean())}
    hist_bins = np.linspace(-1.5, 1.5, 61)
    # examples (fixed rule)
    # examples (rule stated in the caption): moving, high-gate (g > 0.5) patches with the largest k=10 advantage of
    # ShiftWM over the better baseline, 1 - err_S / min(err_D, err_AR), among patches whose true change is above the
    # median of moving patches; one patch per window, 3 windows from different episodes
    eS, eB = E["shiftwm"][:, K - 1], np.minimum(E["direct"][:, K - 1], E["ar"][:, K - 1])
    chgK = E["persistence"][:, K - 1]
    elig = M & (G > 0.5) & (chgK > np.median(chgK[M]))
    score = np.where(elig, 1 - eS / np.maximum(eB, 1e-6), -np.inf)
    ex, seen = [], set()
    for flat in np.argsort(-score, axis=None):
        w, i = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[w, i]) or int(ep[w]) in seen:
            continue
        ex.append((int(w), int(i))); seen.add(int(ep[w]))
        if len(ex) == 3:
            break
    summ["example_rule"] = "moving high-gate patches with the largest advantage over the better baseline, distinct episodes"
    summ["example_advantage"] = [float(score[w, i]) for w, i in ex]
    (OUT / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    idx = torch.tensor([w for w, _ in ex], device=dev)
    hist, past, fut, tgt = data.batch(idx); hist, tgt = hist.float(), tgt.float()
    p_sw, det = models["shiftwm"](hist, past, fut, return_details=True)
    p_di = models["direct"](hist, past, fut).float(); p_ar = models["ar"](hist, past, fut).float()
    cand, valid = candidates(hist, cfg)
    X = {"cand": [], "valid": [], "weights": [], "obs": [], "true": [], "shiftwm": [], "direct": [], "ar": [], "gate": []}
    for b, (w, i) in enumerate(ex):
        X["cand"].append(cand[b, i].cpu().numpy()); X["valid"].append(valid[i].cpu().numpy())
        X["weights"].append(det["weights"][b, K - 1, i].float().cpu().numpy())
        X["obs"].append(hist[b, -1, i].cpu().numpy()); X["true"].append(tgt[b, K - 1, i].cpu().numpy())
        X["shiftwm"].append(p_sw[b, K - 1, i].float().cpu().numpy()); X["direct"].append(p_di[b, K - 1, i].cpu().numpy())
        X["ar"].append(p_ar[b, K - 1, i].cpu().numpy()); X["gate"].append(float(det["gate"][b, K - 1, i, 0]))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "examples.npz", **{k: np.stack(v) for k, v in X.items()}, windows=np.array([w for w, _ in ex]),
             patches=np.array([i for _, i in ex]), diff_direct=np.histogram(np.clip(dm, -1.5, 1.5), hist_bins)[0],
             diff_ar=np.histogram(np.clip(am, -1.5, 1.5), hist_bins)[0], bins=hist_bins)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    s = summ["regions"]["moving"]
    print(json.dumps({k: round(s[k]["k10"], 4) for k in keys}), s["oracle_gain_share"], summ["moving_patch_winrate"])


if __name__ == "__main__":
    main()
