"""Null control for the oracle-move analysis (scripts/v2/geometry.py), CPU only.

Real oracle: min over S*w^2 = 3*7*7 candidates from the last S observed frames of the SAME window.
Null oracle: identical selection, but candidates come from the last S frames of a history window in an UNRELATED
test episode (fixed-point-free random permutation of episodes, seed 0), same patch-location 7x7 window. The partner
window is at the same relative position in the partner episode. Target, persistence, moving mask (top 25% true change
at k=10, per window), standardisation (stats.json, fp16 storage as in FeatureSplit), channel-mean squared error and
episode-level averaging are exactly as in geometry.py.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

t0 = time.time()
ROOT = Path("/home/Test/abhijit.das/projects/WM-ICLR")
H, K, G, W, S, STRIDE = 3, 10, 16, 7, 3, 2
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.set_num_threads(16)
root = ROOT / "data/v2/features/droid/dinov2s"
stats = json.loads((root / "stats.json").read_text())
fm = torch.tensor(stats["feature_mean"], dtype=torch.float32); fs = torch.tensor(stats["feature_std"], dtype=torch.float32)
man = json.loads((root / "manifest.json").read_text())
rows = [r for r in man["episodes"] if r["split"] == "test" and r["T"] >= H + K]   # as FeatureSplit (need = H+K)
nE = len(rows)

def load_ep(e):   # standardised, stored fp16 then used as float (exactly as FeatureSplit + geometry.py)
    with np.load(root / rows[e]["file"]) as z:
        f = torch.from_numpy(z["features"].astype(np.float32))
    f = ((f.reshape(f.shape[0], -1, f.shape[-1]) - fm) / fs).half().float()
    return f.reshape(f.shape[0], G, G, -1)                                          # [T,G,G,C]

MODE = sys.argv[2] if len(sys.argv) > 2 else "any"   # any | samelab (partner from the same DROID lab when possible)
rng = np.random.default_rng(SEED)
if MODE == "any":
    while True:
        perm = rng.permutation(nE)
        if not np.any(perm == np.arange(nE)):
            break
else:
    lab = np.array([rw["session"].split("/")[0] for rw in rows])
    perm = np.empty(nE, dtype=int); n_same = 0
    for e in range(nE):
        pool = np.flatnonzero((lab == lab[e]) & (np.arange(nE) != e))
        if len(pool) == 0:
            pool = np.flatnonzero(np.arange(nE) != e)
        else:
            n_same += 1
        perm[e] = rng.choice(pool)
    print("same-lab partners", n_same, "/", nE, flush=True)

r = W // 2
def oracle(histS, y):
    """histS [B,S,G,G,C] source frames, y [B,G,G,C] target -> [B,G*G] min over S*W*W window candidates of
    channel-mean squared error; out-of-image candidates excluded (== zero-pad + validity mask in geometry.py)."""
    B = y.shape[0]
    best = torch.full((B, G, G), float("inf"))
    for s in range(S):
        src = histS[:, s]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # candidate for patch (i,j) is src[i+dy, j+dx]
                i0, i1 = max(0, -dy), min(G, G - dy); j0, j1 = max(0, -dx), min(G, G - dx)
                d = ((src[:, i0 + dy:i1 + dy, j0 + dx:j1 + dx] - y[:, i0:i1, j0:j1]) ** 2).mean(-1)
                best[:, i0:i1, j0:j1] = torch.minimum(best[:, i0:i1, j0:j1], d)
    return best.reshape(B, G * G)

E = {"persistence": [], "oracle": [], "null": []}; M = []; ep_list = []
with torch.no_grad():
    for e in range(nE):
        f = load_ep(e); T = f.shape[0]
        p = perm[e]; fp = load_ep(p); Tp = fp.shape[0]
        st = np.arange(0, T - (H + K) + 1, STRIDE)
        hist = torch.stack([f[s:s + H] for s in st])                                 # [B,H,G,G,C]
        y = torch.stack([f[s + H + K - 1] for s in st])                              # target at k=10
        rel = st / max(T - (H + K), 1)
        nst = np.rint(rel * (Tp - H)).astype(int)                                    # partner: same relative position
        nhist = torch.stack([fp[s:s + H] for s in nst])
        E["persistence"].append(((hist[:, -1] - y) ** 2).mean(-1).reshape(len(st), -1))
        E["oracle"].append(oracle(hist[:, -S:], y))
        E["null"].append(oracle(nhist[:, -S:], y))
        chg = E["persistence"][-1]
        M.append(chg >= chg.quantile(0.75, dim=-1, keepdim=True))
        ep_list.append(np.full(len(st), e))
        del f, fp, hist, nhist
        if e % 10 == 0:
            print(e, f"{time.time() - t0:.0f}s", flush=True)
E = {k: torch.cat(v).numpy() for k, v in E.items()}; M = torch.cat(M).numpy(); ep = np.concatenate(ep_list)
U = np.unique(ep)

def region_mean(e, mask):
    per_w = (e * mask).sum(-1) / np.maximum(mask.sum(-1), 1)
    return np.array([per_w[ep == u].mean() for u in U])

out = {"mode": MODE, "seed": SEED, "windows": int(len(M)), "episodes": int(nE), "pairing": "fixed-point-free permutation of test episodes "
       "(np.random.default_rng(seed)); partner history window at same relative position", "k": K}
for reg, mask in (("moving", M), ("static", ~M), ("all", np.ones_like(M))):
    r = {k: float(region_mean(E[k], mask).mean()) for k in E}
    r["oracle_red_pct"] = 100 * (1 - r["oracle"] / r["persistence"])
    r["null_red_pct"] = 100 * (1 - r["null"] / r["persistence"])
    out[reg] = r
# bootstrap CI over episodes for the null moving reduction
pe = {k: region_mean(E[k], M) for k in E}
b = np.random.default_rng(0).integers(0, nE, (5000, nE))
red = 100 * (1 - pe["null"][b].mean(1) / pe["persistence"][b].mean(1))
out["moving"]["null_red_pct_ci95"] = [float(np.quantile(red, .025)), float(np.quantile(red, .975))]
out["runtime_s"] = time.time() - t0
p = Path(__file__).parent / f"summary_{MODE}_seed{SEED}.json"
p.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
