"""Non-learned feature-warping baselines (F2M / F2F-style "forecast by warping deep features") on held-out DROID.

Protocol = the matched predictors (Table 1 / geometry.py): H=3 observed frames, K=10 steps, standardised DINOv2-S
16x16 features, FeatureSplit(root, "test", 3, 10, dev, stats, stride=2); per-horizon MSE (mean over patches and
channels) averaged over the windows of each episode, then over episodes (as train.evaluate).

Every forecast backward-warps the LAST OBSERVED feature grid Z_t (t = t0, the last history frame) with bilinear
grid_sample (align_corners=False, border padding). RAFT-large (torchvision C_T_SKHT_V2, 12 iterations; same
wrapper as flow_agreement.py) runs on the 224x224 encoder-input frames and its flow is average-pooled to the 16x16
patch grid in patch units (1 patch = 14 px; analysis.pooled_flow). One model step = one cached frame (the cache is
already temporally subsampled), so the flow between consecutive observed frames is a per-step velocity.

  flow_extrap   (causal, F2M-style; the fair baseline) f = RAFT(I_{t-1} -> I_t); constant velocity, so the
                displacement after k steps is k*f and the forecast is Z_t sampled at  x - k*f(x).
  flow_extrap_bwd (causal variant) b = RAFT(I_t -> I_{t-1}), which lives on the frame-t grid; forecast = Z_t sampled
                at x + k*b(x). Same information as flow_extrap, different first-order approximation of the inverse.
  oracle_flow   (ORACLE, uses future frames; not a fair baseline) o_k = RAFT(I_{t+k} -> I_t), the exact backward
                flow at the target grid; forecast = Z_t sampled at x + o_k(x). Upper-bounds a pure warp of Z_t.
  persistence   Z_t (sanity check: must match the persistence row of Table 1 / geometry.py).

Moving/static at k=10: moving = top 25% true change ||z_{t+10,i} - z_{t,i}||^2 within each window (geometry.py);
region means per window -> per episode -> mean over episodes.
Writes results/v2/analysis/flowwarp/{summary.json, per_episode.npz}.
Usage (GPU): PYTHONPATH=src python scripts/v2/flowwarp.py [--max-episodes N --out DIR]  (smoke/timing)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_agreement import Raft  # noqa: E402  (torchvision RAFT-large wrapper already used by the paper)
from shiftwm.v2 import analysis as A  # noqa: E402
from shiftwm.v2.train import FeatureSplit  # noqa: E402

H, K = 3, 10
METHODS = ("persistence", "flow_extrap", "flow_extrap_bwd", "oracle_flow")


def warp(z, disp, g):
    """z [B,N,C] features on a g x g grid, disp [B,2,g,g] (x, y) in patch units ->
    [B,N,C] = z sampled at (patch centre + disp), bilinear, border padding."""
    b, n, c = z.shape
    x = z.transpose(1, 2).reshape(b, c, g, g)
    base = (torch.arange(g, device=z.device, dtype=torch.float32) + 0.5) * (2.0 / g) - 1.0
    gy, gx = torch.meshgrid(base, base, indexing="ij")
    grid = torch.stack((gx + disp[:, 0] * (2.0 / g), gy + disp[:, 1] * (2.0 / g)), -1)   # [B,g,g,2]
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.reshape(b, c, n).transpose(1, 2)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--raft-iters", type=int, default=12)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--out", default=str(A.ANALYSIS / "flowwarp"))
    a = p.parse_args()
    dev, out = a.device, Path(a.out)
    t_all = time.time()
    root = A.cache_root("droid")
    manifest, stats = A.read_cache(root)
    g = manifest["grid"]
    data = FeatureSplit(root, "test", H, K, dev, stats, stride=a.stride, max_episodes=a.max_episodes)
    src = A.FrameSource(root)
    frames = []
    for e in data.episodes:
        fr = torch.from_numpy(src.load(e["id"])).to(dev)
        frames.append(fr)
    t0s = (A.local_starts(data) + H - 1).tolist()
    epi = data.episode_of.tolist()
    for j in range(len(data)):                       # RGB cache is aligned 1:1 with the feature cache
        assert t0s[j] + K < len(frames[epi[j]])
    raft = Raft(dev, a.raft_iters)
    print(json.dumps({"windows": len(data), "episodes": len(data.episodes), "setup_sec": round(time.time() - t_all, 1)}),
          flush=True)

    n_ep = len(data.episodes)
    z64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)  # noqa: E731
    mse = {m: z64(n_ep, K) for m in METHODS}                      # per-episode sums of per-window MSE
    reg = {m: {r: z64(n_ep, K) for r in ("moving", "static")} for m in METHODS}
    cnt = z64(n_ep)
    fmag = z64(n_ep, 2)                                           # mean |f| (patches): all patches, moving patches
    ks = torch.arange(1, K + 1, device=dev, dtype=torch.float32)
    t_loop = time.time()
    for i in range(0, len(data), a.batch):
        idx = torch.arange(i, min(i + a.batch, len(data)), device=dev)
        sl = range(i, min(i + a.batch, len(data)))
        b = len(idx)
        hist, _, _, tgt = data.batch(idx)
        zt, tgt = hist[:, -1].float(), tgt.float()                # [B,N,C], [B,K,N,C]
        cur = torch.stack([frames[epi[j]][t0s[j]] for j in sl])
        prev = torch.stack([frames[epi[j]][t0s[j] - 1] for j in sl])
        fut = torch.stack([frames[epi[j]][t0s[j] + k] for j in sl for k in range(1, K + 1)])  # [B*K,...]
        f = A.pooled_flow(raft(prev, cur), g)                         # t-1 -> t, per-step velocity
        bw = A.pooled_flow(raft(cur, prev), g)                        # t -> t-1
        orc = A.pooled_flow(torch.cat([raft(fut[s:s + a.batch * 2], cur.repeat_interleave(K, 0)[s:s + a.batch * 2])
                                       for s in range(0, b * K, a.batch * 2)]), g).reshape(b, K, 2, g, g)
        rep = lambda x: x.repeat_interleave(K, 0)                     # noqa: E731
        zk = rep(zt)                                                  # [B*K,N,C]
        kk = ks.repeat(b)[:, None, None, None]
        preds = {"persistence": zt[:, None].expand(-1, K, -1, -1),
                 "flow_extrap": warp(zk, -kk * rep(f), g).reshape(b, K, -1, zt.shape[-1]),
                 "flow_extrap_bwd": warp(zk, kk * rep(bw), g).reshape(b, K, -1, zt.shape[-1]),
                 "oracle_flow": warp(zk, orc.reshape(b * K, 2, g, g), g).reshape(b, K, -1, zt.shape[-1])}
        chg = ((tgt[:, K - 1] - zt) ** 2).mean(-1)                    # [B,N]
        mov = (chg >= chg.quantile(0.75, dim=-1, keepdim=True)).double()
        ep = data.episode_of[idx]
        for m, pr in preds.items():
            e = ((pr - tgt) ** 2).mean(-1).double()                   # [B,K,N]
            mse[m].index_add_(0, ep, e.mean(-1))
            for r, msk in (("moving", mov), ("static", 1 - mov)):
                reg[m][r].index_add_(0, ep, (e * msk[:, None]).sum(-1) / msk.sum(-1)[:, None])
        mag = f.norm(dim=1).flatten(1).double()
        fmag.index_add_(0, ep, torch.stack((mag.mean(-1), (mag * mov).sum(-1) / mov.sum(-1)), 1))
        cnt.index_add_(0, ep, torch.ones(b, dtype=torch.float64, device=dev))
        if i == 0 or (i // a.batch) % 20 == 0:
            done = i + b
            el = time.time() - t_loop
            print(json.dumps({"done": done, "of": len(data), "sec": round(el, 1),
                              "eta_sec": round(el / done * (len(data) - done), 1)}), flush=True)

    keep = cnt > 0
    c = cnt[keep, None]
    per_ep = {m: (mse[m][keep] / c).cpu().numpy() for m in METHODS}
    per_reg = {m: {r: (reg[m][r][keep] / c).cpu().numpy() for r in ("moving", "static")} for m in METHODS}
    episodes = [e["id"] for e, k_ in zip(data.episodes, keep.tolist()) if k_]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "per_episode.npz", episodes=np.array(episodes), windows=cnt[keep].cpu().numpy(),
             **{f"mse_{m}": per_ep[m] for m in METHODS},
             **{f"{r}_{m}": per_reg[m][r] for m in METHODS for r in ("moving", "static")},
             flow_mag=(fmag[keep] / c).cpu().numpy(),
             note="[episodes, K] per-episode means of per-window MSE (standardised DINOv2-S); k index 0 = step 1")

    def ci(v):
        return [round(x, 5) for x in A.bootstrap_ci(v)]

    summ = {"dataset": "droid", "encoder": "dinov2s", "split": "test", "history": H, "horizon": K,
            "stride": a.stride, "episodes": len(episodes), "windows": int(cnt.sum().item()),
            "raft": f"torchvision raft_large C_T_SKHT_V2, {a.raft_iters} iters, 224x224 frames, avg-pooled to "
                    f"{g}x{g} patches (patch units)",
            "warp": "backward warp of last observed Z_t, grid_sample bilinear, border padding, align_corners=False",
            "metric": "per-horizon MSE (mean over patches+channels, standardised), windows->episode mean->mean "
                      "over episodes; CIs = 95% episode bootstrap",
            "regions": "moving = top 25% ||z_{t+10}-z_t||^2 per window (geometry.py); reported at k=10 (and avg)",
            "labels": {"flow_extrap": "causal F2M-style constant-velocity warp, flow RAFT(t-1 -> t), sample x - k f(x)",
                       "flow_extrap_bwd": "causal variant, flow RAFT(t -> t-1), sample x + k b(x)",
                       "oracle_flow": "ORACLE (uses future frames): RAFT(t+k -> t) backward warp; not a fair baseline",
                       "persistence": "Z_t copied (sanity check)"},
            "mean_flow_mag_patches": {"all": float(fmag[keep, 0].sum() / cnt.sum()),
                                      "moving": float(fmag[keep, 1].sum() / cnt.sum())},
            "methods": {}}
    for m in METHODS:
        e = per_ep[m]
        summ["methods"][m] = {
            "by_k": [round(float(x), 5) for x in e.mean(0)],
            "mean_h": round(float(e.mean()), 5), "mean_h_ci": ci(e.mean(1)),
            "k10": round(float(e[:, -1].mean()), 5), "k10_ci": ci(e[:, -1]),
            **{f"{r}_k10": round(float(per_reg[m][r][:, -1].mean()), 5) for r in ("moving", "static")},
            **{f"{r}_k10_ci": ci(per_reg[m][r][:, -1]) for r in ("moving", "static")},
            **{f"{r}_mean_h": round(float(per_reg[m][r].mean()), 5) for r in ("moving", "static")},
        }
    for m in ("flow_extrap", "flow_extrap_bwd", "oracle_flow"):          # paired vs persistence, per episode
        d = per_ep[m][:, -1] - per_ep["persistence"][:, -1]
        summ["methods"][m]["k10_minus_persistence"] = [round(float(d.mean()), 5), *ci(d)]
    summ["sec"] = round(time.time() - t_all, 1)
    A.write_json(out / "summary.json", summ)
    print(json.dumps({m: {k: v for k, v in s.items() if k in ("mean_h", "k10", "moving_k10", "static_k10")}
                      for m, s in summ["methods"].items()}), flush=True)
    print(json.dumps({"total_sec": summ["sec"]}), flush=True)


if __name__ == "__main__":
    main()
