"""Does ShiftWM's transport field track real motion? And where do its gains come from?

(1) Transport vs optical flow. For every test window (stride --stride) and k in {1,5,10}:
  * RAFT-large (torchvision, C_T_SKHT_V2 weights, 12 refinement iterations) on the 224x224 encoder-input
    frames (observed t0, true future t0+k), average-pooled to the 16x16 patch grid and expressed in
    patch units (1 patch = 14 px).
  * ShiftWM expected source offset per query patch o = sum_j pi_kj * offset_j (return_details weights;
    same computation as make_figures.transport_arrows(); offsets tiled over the S source frames).
    The implied content motion is m = -o.
  Primary comparison ("bwd"): backward flow future->observed, which lives at the query (future) patch
  exactly like o: agreement between -o and -flow_bwd. Secondary ("fwd", as specified in the plan):
  forward flow observed->future at the same patch index vs -o.
  Metrics on patches whose RAFT flow magnitude > --min-flow (0.5 patch): cosine similarity, end-point
  error (EPE, patches), EPE of the gate-weighted displacement -g*o, and a trivial zero-motion baseline
  (EPE = |flow|). Patch-pooled within each episode; episodes are the bootstrap unit.
(2) Gain vs motion. Per-patch squared error (mean over channels, standardised space) of ShiftWM and Direct
  (same seed) binned by the true per-patch feature change |z_{t0+k} - z_{t0}|^2 (= persistence error),
  fixed log-spaced bin edges; per-episode, per-bin sums are saved for bootstrap CIs.

Outputs (per dataset <ds>):
  results/v2/analysis/flow/<ds>/raft_pooled.npz               pooled RAFT flows (reused by figures)
  results/v2/analysis/flow/<ds>/shiftwm_s<seed>.npz           [E, len(ks)] metric arrays
  results/v2/analysis/gain_vs_motion/<ds>/s<seed>.npz         [E, len(ks)+1, bins] sums ("all" = k 1..K)
  results/v2/analysis/flow/summary.json
Usage (GPU): python scripts/v2/flow_agreement.py [--datasets droid openh_hamlyn] [--seeds 0]
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from shiftwm.v2.analysis import (ANALYSIS, ROOT, FrameSource, bootstrap_ci, expected_offsets, load_model,
                                 local_starts, pooled_flow, predict, read_cache, run_config, run_dirs, write_json)
from shiftwm.v2.train import FeatureSplit

DATASETS = ("droid", "openh_hamlyn")
EDGES = np.concatenate(([0.0], 0.02 * 2.0 ** np.arange(9), [np.inf]))  # 0, .02 ... 5.12, inf
OUT = ANALYSIS


class Raft:
    def __init__(self, device, iters=12):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        w = Raft_Large_Weights.C_T_SKHT_V2
        self.model = raft_large(weights=w).to(device).eval()
        self.tf = w.transforms()
        self.iters = iters

    @torch.no_grad()
    def __call__(self, a, b):
        """uint8 [B,H,W,3] pairs -> flow a->b in pixels [B,2,H,W] (x, y)."""
        x = a.permute(0, 3, 1, 2).float() / 255
        y = b.permute(0, 3, 1, 2).float() / 255
        x, y = self.tf(x, y)
        return self.model(x.contiguous(), y.contiguous(), num_flow_updates=self.iters)[-1].float()


def run_dataset(ds, a, raft):
    root = ROOT / "data/v2/features" / ds / a.encoder
    manifest, stats = read_cache(root)
    runs = run_dirs(ds, "shiftwm", a.encoder, a.seeds)
    if not runs:
        print(json.dumps({"skip": ds, "reason": "no shiftwm checkpoint"}), flush=True)
        return
    H, K, _ = run_config(runs[0])
    ks = [k for k in a.ks if k <= K]
    kk = torch.tensor(ks, device=a.device) - 1
    data = FeatureSplit(root, "test", H, K, a.device, stats, a.stride, None, a.max_episodes)
    frames = FrameSource(root)
    ep_frames = [torch.from_numpy(frames.load(e["id"])).to(a.device) for e in data.episodes]
    t0s = (local_starts(data) + H - 1).tolist()
    epi = data.episode_of.tolist()
    n_ep, n_k, g = len(data.episodes), len(ks), manifest["grid"]
    episodes = np.array([e["id"] for e in data.episodes])
    out = OUT / "flow" / ds
    out.mkdir(parents=True, exist_ok=True)

    # ---- RAFT once per window and horizon (shared by all seeds) ------------------------------------------
    rpath = out / "raft_pooled.npz"
    if rpath.exists() and not a.overwrite and np.load(rpath)["n_windows"] == len(data) \
            and list(np.load(rpath)["ks"]) == ks:
        z = np.load(rpath)
        fwd, bwd = (torch.from_numpy(z[n].astype(np.float32)).to(a.device) for n in ("fwd", "bwd"))
    else:
        t_start = time.time()
        fwd = torch.zeros(len(data), n_k, 2, g, g, device=a.device)
        bwd = torch.zeros_like(fwd)
        for i in range(0, len(data), a.raft_batch):
            sl = range(i, min(i + a.raft_batch, len(data)))
            obs = torch.stack([ep_frames[epi[j]][t0s[j]] for j in sl])
            for c, k in enumerate(ks):
                fut = torch.stack([ep_frames[epi[j]][t0s[j] + k] for j in sl])
                fwd[i:i + len(sl), c] = pooled_flow(raft(obs, fut), g)
                bwd[i:i + len(sl), c] = pooled_flow(raft(fut, obs), g)
        np.savez(rpath, fwd=fwd.cpu().numpy().astype(np.float16), bwd=bwd.cpu().numpy().astype(np.float16), ks=ks,
                 n_windows=len(data), episode_of=data.episode_of.cpu().numpy(), t0=np.array(t0s), episodes=episodes,
                 stride=a.stride, note="patch units; channel 0 = x (right), 1 = y (down); RAFT on 224x224 frames")
        print(json.dumps({"raft": ds, "windows": len(data), "sec": round(time.time() - t_start, 1)}), flush=True)

    for run in runs:
        seed = run.name
        model = load_model("shiftwm", run, manifest, H, K, a.device)
        cfg = model.config
        druns = run_dirs(ds, "direct", a.encoder, [int(seed[1:])])
        direct = load_model("direct", druns[0], manifest, H, K, a.device) if druns else None
        names = ("cos", "epe", "epe_gated", "epe_zero", "npatch")
        acc = {f"{m}_{v}": torch.zeros(n_ep, n_k, dtype=torch.float64, device=a.device)
               for m in names for v in ("bwd", "fwd")}
        last_mass = torch.zeros(n_ep, n_k, dtype=torch.float64, device=a.device)
        gate_mean = torch.zeros(n_ep, n_k, dtype=torch.float64, device=a.device)
        nwin = torch.zeros(n_ep, dtype=torch.float64, device=a.device)
        nb = len(EDGES) - 1
        edges = torch.tensor(EDGES[1:-1], device=a.device, dtype=torch.float32)
        gvm = {m: torch.zeros(n_ep, n_k + 1, nb, dtype=torch.float64, device=a.device) for m in ("se", "de", "pe", "cnt")}
        t_start = time.time()
        for i in range(0, len(data), a.batch_size):
            idx = torch.arange(i, min(i + a.batch_size, len(data)), device=data.starts.device)
            ep = data.episode_of[idx]
            hist, past, fut, target = data.batch(idx)
            pred, det = predict(model, hist, past, fut, details=True)
            dx, dy, lm = expected_offsets(det["weights"][:, kk], cfg)          # [B,nk,N]
            gate = det["gate"][:, kk, :, 0].float()
            o = torch.stack((dx, dy), 2)                                       # [B,nk,2,N] source offset
            m_pred = -o
            for v, fl in (("bwd", -bwd[idx]), ("fwd", fwd[idx])):             # both as content motion
                f = fl.reshape(len(idx), n_k, 2, -1)
                mag = f.norm(dim=2)
                mask = (mag > a.min_flow).double()
                cos = torch.nn.functional.cosine_similarity(m_pred, f, dim=2, eps=1e-8)
                epe = (m_pred - f).norm(dim=2)
                epe_g = (gate[:, :, None] * m_pred - f).norm(dim=2)
                for m, val in (("cos", cos), ("epe", epe), ("epe_gated", epe_g), ("epe_zero", mag),
                               ("npatch", torch.ones_like(mag))):
                    acc[f"{m}_{v}"].index_add_(0, ep, (val.double() * mask).sum(-1))
            last_mass.index_add_(0, ep, lm.double().mean(-1))
            gate_mean.index_add_(0, ep, gate.double().mean(-1))
            nwin.index_add_(0, ep, torch.ones_like(ep, dtype=torch.float64))
            if direct is not None:
                dpred = predict(direct, hist, past, fut)
                change = ((target - hist[:, -1:]) ** 2).mean(-1)                 # [B,K,N]
                se = ((pred - target) ** 2).mean(-1); de = ((dpred - target) ** 2).mean(-1)
                for c in range(n_k + 1):
                    if c < n_k:
                        ch, s_, d_ = change[:, kk[c]], se[:, kk[c]], de[:, kk[c]]
                    else:
                        ch, s_, d_ = change.flatten(1), se.flatten(1), de.flatten(1)
                    b = torch.bucketize(ch.float().contiguous(), edges)                         # [B, M]
                    flat = ep[:, None] * nb + b
                    for m, val in (("se", s_), ("de", d_), ("pe", ch), ("cnt", torch.ones_like(ch))):
                        tmp = torch.zeros(n_ep * nb, dtype=torch.float64, device=a.device)
                        tmp.index_add_(0, flat.flatten(), val.double().flatten())
                        gvm[m][:, c] += tmp.reshape(n_ep, nb)
        res = {k: v.cpu().numpy() for k, v in acc.items()}
        np.savez(out / f"shiftwm_{seed}.npz", **res, last_source_mass=(last_mass / nwin[:, None]).cpu().numpy(),
                 gate_mean=(gate_mean / nwin[:, None]).cpu().numpy(), windows=nwin.cpu().numpy(), ks=ks,
                 episodes=episodes, min_flow=a.min_flow, stride=a.stride, checkpoint=str(run / "best.pt"))
        rec = {"flow": ds, "seed": seed, "sec": round(time.time() - t_start, 1)}
        for v in ("bwd", "fwd"):
            n = res[f"npatch_{v}"].sum(0)
            rec[v] = {m: (res[f"{m}_{v}"].sum(0) / np.maximum(n, 1)).round(4).tolist()
                      for m in ("cos", "epe", "epe_gated", "epe_zero")}
            rec[v]["frac_moving"] = (n / (nwin.sum().item() * g * g)).round(4).tolist()
        print(json.dumps(rec), flush=True)
        if direct is not None:
            gdir = OUT / "gain_vs_motion" / ds
            gdir.mkdir(parents=True, exist_ok=True)
            np.savez(gdir / f"{seed}.npz", **{k: v.cpu().numpy() for k, v in gvm.items()}, edges=EDGES,
                     ks=np.array(ks + [0]), episodes=episodes, note="ks entry 0 = all horizons 1..K pooled")
            se_, de_ = gvm["se"].sum(0)[-1], gvm["de"].sum(0)[-1]
            print(json.dumps({"gain_vs_motion": ds, "seed": seed,
                              "rel_gain_pct_per_bin": (100 * (1 - se_ / de_.clamp_min(1e-12))).round(decimals=2).tolist()}),
                  flush=True)
        del model, direct


def summarise():
    summary = {}
    for f in sorted((OUT / "flow").glob("*/shiftwm_s*.npz")):
        z = np.load(f)
        ent = {"ks": z["ks"].tolist()}
        for v in ("bwd", "fwd"):
            n = np.maximum(z[f"npatch_{v}"], 1)
            for m in ("cos", "epe", "epe_gated", "epe_zero"):
                per_ep = z[f"{m}_{v}"] / n
                ok = z[f"npatch_{v}"] > 0
                ent[f"{m}_{v}"] = [[float(per_ep[ok[:, c], c].mean()), *bootstrap_ci(per_ep[ok[:, c], c])]
                                   for c in range(len(z["ks"]))]
        summary[f"{f.parent.name}/{f.stem}"] = ent
    write_json(OUT / "flow" / "summary.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--encoder", default="dinov2s")
    p.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--min-flow", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--raft-batch", type=int, default=32)
    p.add_argument("--raft-iters", type=int, default=12)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out", help="override results/v2/analysis (smoke tests)")
    a = p.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    raft = Raft(a.device, a.raft_iters)
    for ds in a.datasets:
        run_dataset(ds, a, raft)
    summarise()


if __name__ == "__main__":
    main()
