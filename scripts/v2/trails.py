"""Transport trails on held-out DROID: where does ShiftWM fetch each patch from, k = 1..10, versus optical flow?

For every test window (stride 2, as geometry.py) and horizon k = 1..10:
  * ShiftWM expected source offset of query patch i, o_k(i) = sum_j pi_kj offset_j over the S*w*w candidates of its
    window and source frames (analysis.expected_offsets; same computation as flow_agreement.py and Fig. 2), in patches.
    The trail of patch i is the sequence i + o_k(i), k = 1..10, drawn on the observed frame.
  * True trail: backward RAFT flow b_k(i) = RAFT(frame t0+k -> frame t0) average-pooled to the 16x16 grid (patch
    units), i.e. where the content seen at i at time t0+k was at time t0. Same RAFT wrapper, weights, 12 iterations,
    224x224 frames and pooling as flow_agreement.py ("bwd" comparison). No pooled flow cache existed for DROID
    (results/v2/analysis/flow/ is empty), so it is computed here and cached in raft_bwd.npz.
Moving patches: |b_10(i)| > 0.5 patch (flow_agreement.py threshold), fixed across k for a window.
Trail endpoint error per k on moving patches: EPE(o_k) = |o_k - b_k|, gate-weighted EPE |g_k o_k - b_k|, the zero-motion
baseline |b_k| and the reach limit |b_k - clip(b_k, -r, r)| (r = 3: the transport window cannot fetch from farther).
Patch-pooled within each episode; episodes are the bootstrap unit. Also split by |b_k| <= r (reachable) or not.
Examples (fixed rule): windows with >= 12 moving patches, ranked by their mean k=10 EPE on moving patches; the windows
at the 25th, 50th and 75th percentile of that ranking (nearest in rank with a not-yet-used episode) -- typical cases,
not the best ones.
Writes results/v2/analysis/trails/{raft_bwd.npz, per_episode.npz, examples.npz, summary.json}. Run on a GPU.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from flow_agreement import Raft  # noqa: E402
from geometry import load  # noqa: E402
from shiftwm.v2.analysis import FrameSource, bootstrap_ci, expected_offsets, local_starts, pooled_flow  # noqa: E402
from shiftwm.v2.train import FeatureSplit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2/analysis/trails"
H, K, B, RW, MIN_FLOW = 3, 10, 32, 8, 0.5


@torch.no_grad()
def main():
    dev = "cuda"
    t_all = time.time()
    root = ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    data = FeatureSplit(root, "test", H, K, dev, stats, stride=2)
    model, ck = load("shiftwm", dev)
    cfg = model.config; g = cfg.grid; r = cfg.window // 2
    W = len(data); U = len(data.episodes)
    t0s = (local_starts(data) + H - 1).tolist(); epi = data.episode_of.tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    rpath = OUT / "raft_bwd.npz"
    if rpath.exists() and int(np.load(rpath)["n_windows"]) == W:
        bwd = torch.from_numpy(np.load(rpath)["bwd"].astype(np.float32)).to(dev)
    else:
        src = FrameSource(root)
        frames = [torch.from_numpy(src.load(e["id"])).to(dev) for e in data.episodes]
        raft = Raft(dev)
        bwd = torch.zeros(W, K, 2, g, g, device=dev)
        for i in range(0, W, RW):
            sl = range(i, min(i + RW, W))
            obs = torch.stack([frames[epi[j]][t0s[j]] for j in sl])
            fut = torch.stack([frames[epi[j]][t0s[j] + k] for j in sl for k in range(1, K + 1)])
            f = pooled_flow(raft(fut, obs.repeat_interleave(K, 0)), g)
            bwd[i:i + len(sl)] = f.reshape(len(sl), K, 2, g, g)
            if (i // (RW)) % 25 == 0:
                print(json.dumps({"raft_done": i + len(sl), "of": W, "sec": round(time.time() - t_all, 1)}), flush=True)
        np.savez(rpath, bwd=bwd.cpu().numpy().astype(np.float16), n_windows=W, t0=np.array(t0s),
                 episode_of=np.array(epi), episodes=np.array([e["id"] for e in data.episodes]), stride=2,
                 note="RAFT(t0+k -> t0) pooled to 16x16, patch units; [W,K,2,g,g]; ch0 = x (right), ch1 = y (down)")
        del frames
    names = ("epe", "epe_gated", "epe_zero", "epe_reach", "cos", "n", "epe_in", "n_in", "epe_out", "n_out")
    acc = {m: torch.zeros(U, K, dtype=torch.float64, device=dev) for m in names}
    win_epe = torch.zeros(W, device=dev); win_n = torch.zeros(W, device=dev)
    for i in range(0, W, B):
        idx = torch.arange(i, min(i + B, W), device=dev)
        hist, past, fut, _ = data.batch(idx)
        _, det = model(hist.float(), past, fut, return_details=True)
        dx, dy, _ = expected_offsets(det["weights"], cfg)                        # [B,K,N]
        o = torch.stack((dx, dy), 2)                                             # [B,K,2,N]
        gate = det["gate"][..., 0].float()[:, :, None]                          # [B,K,1,N]
        f = bwd[idx].reshape(len(idx), K, 2, -1)
        mov = (f[:, -1].norm(dim=1) > MIN_FLOW)[:, None].double()               # [B,1,N]
        mag = f.norm(dim=2)
        epe = (o - f).norm(dim=2)
        inn = (mag <= r).double()
        vals = {"epe": epe, "epe_gated": (gate * o - f).norm(dim=2), "epe_zero": mag,
                "epe_reach": (f - f.clamp(-r, r)).norm(dim=2),
                "cos": torch.nn.functional.cosine_similarity(o, f, dim=2, eps=1e-8), "n": torch.ones_like(mag),
                "epe_in": epe * inn, "n_in": inn, "epe_out": epe * (1 - inn), "n_out": 1 - inn}
        e = data.episode_of[idx]
        for m, v in vals.items():
            acc[m].index_add_(0, e, (v.double() * mov).sum(-1))
        win_epe[idx] = (epe[:, -1] * mov[:, 0]).sum(-1).float(); win_n[idx] = mov[:, 0].sum(-1).float()
    A = {m: v.cpu().numpy() for m, v in acc.items()}
    np.savez(OUT / "per_episode.npz", **A, episodes=np.array([e["id"] for e in data.episodes]),
             note="[episodes,K] sums over moving patches (|b_10|>0.5); divide by n (or n_in / n_out)")
    # examples
    we = (win_epe / win_n.clamp_min(1)).cpu().numpy(); wn = win_n.cpu().numpy()
    cand = np.where(wn >= 12)[0]; order = cand[np.argsort(we[cand], kind="stable")]
    ex, seen = [], set()
    for q in (25, 50, 75):
        pos = int(round(q / 100 * (len(order) - 1)))
        for d in sorted(range(len(order)), key=lambda j: (abs(j - pos), j)):
            w = int(order[d])
            if epi[w] not in seen:
                ex.append(w); seen.add(epi[w]); break
    idx = torch.tensor(ex, device=dev)
    hist, past, fut, _ = data.batch(idx)
    _, det = model(hist.float(), past, fut, return_details=True)
    dx, dy, _ = expected_offsets(det["weights"], cfg)
    src = FrameSource(root)
    fr = [src.load(data.episodes[epi[w]]["id"], [t0s[w], t0s[w] + K]) for w in ex]
    np.savez(OUT / "examples.npz", windows=np.array(ex), episodes=np.array([data.episodes[epi[w]]["id"] for w in ex]),
             t0=np.array([t0s[w] for w in ex]), dx=dx.cpu().numpy(), dy=dy.cpu().numpy(),
             gate=det["gate"][..., 0].float().cpu().numpy(), flow=bwd[idx].reshape(len(ex), K, 2, -1).cpu().numpy(),
             frame_obs=np.stack([f[0] for f in fr]), frame_true=np.stack([f[1] for f in fr]),
             win_epe=we[ex], win_moving=wn[ex], percentiles=np.array([25, 50, 75]))
    ok = A["n"][:, -1] > 0
    summ = {"windows": W, "episodes": int(ok.sum()), "checkpoint": ck, "min_flow": MIN_FLOW, "reach": r,
            "moving_patches": int(A["n"][:, 0].sum()), "rule": __doc__, "examples": ex,
            "example_epe_k10": we[ex].tolist(), "candidate_windows": int(len(cand))}
    for m in ("epe", "epe_gated", "epe_zero", "epe_reach", "cos"):
        per = A[m][ok] / A["n"][ok]
        summ[m] = {"mean": per.mean(0).tolist(), "ci": [list(bootstrap_ci(per[:, c])) for c in range(K)]}
    for part in ("in", "out"):
        okp = A[f"n_{part}"] > 0
        summ[f"epe_{part}"] = [float((A[f"epe_{part}"][okp[:, c], c] / A[f"n_{part}"][okp[:, c], c]).mean()) for c in range(K)]
        summ[f"share_{part}"] = (A[f"n_{part}"].sum(0) / A["n"].sum(0)).tolist()
    summ["sec"] = round(time.time() - t_all, 1)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({m: np.round(summ[m]["mean"], 3).tolist() for m in ("epe", "epe_gated", "epe_zero", "epe_reach", "cos")}))
    print(json.dumps({k: summ[k] for k in ("examples", "example_epe_k10", "share_out", "sec")}))


if __name__ == "__main__":
    main()
