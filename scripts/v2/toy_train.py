"""Train the REAL V2WorldModel (tiny config) on the toy world (scripts/v2/toy_world.py), CPU-friendly.

Same model class and loss as the paper runs (shiftwm.v2.models / shiftwm.v2.train.loss_fn); only the
data pipeline differs (raw 4x4 patch pixels as features). Writes results/toy/<arm>/{best.pt,log.jsonl,eval.json}.

Usage: PYTHONPATH=src python scripts/v2/toy_train.py --arm shiftwm --steps 3000 --threads 8
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_world import patchify  # noqa: E402
from shiftwm.v2.models import V2WorldModel  # noqa: E402
from shiftwm.v2.train import loss_fn  # noqa: E402

H, K = 3, 5                      # toy: horizon 5 so the largest displacement (square, 20 px) stays in reach
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TINY = dict(dim=64, heads=4, enc_depth=2, dec_depth=2, window=11, sources=3, key_dim=32, grid=16, channels=48,
            action_dim=2, history=H, horizon=K)


class ToySplit:
    def __init__(self, root, split, stats):
        z = np.load(Path(root) / f"{split}.npz")
        mm = Path(root) / f"{split}_frames.npy"                             # uncompressed copy -> memory-mapped
        self.frames = np.load(mm, mmap_mode="r") if mm.exists() else z["frames"]   # uint8 [E,T,64,64,3]
        if DEV.type == "cuda":                                            # compute node: keep everything on the GPU
            self.frames = torch.from_numpy(np.ascontiguousarray(self.frames)).to(DEV)
        self.actions = torch.from_numpy(z["actions"])                     # [E,T-1,2] (already unit scale)
        self.npz = z                                                      # labels/pos loaded lazily
        self.mean = torch.tensor(stats["feature_mean"], device=DEV); self.std = torch.tensor(stats["feature_std"], device=DEV)
        self.actions = self.actions.to(DEV)
        E, T = self.frames.shape[:2]
        self.windows = [(e, s) for e in range(E) for s in range(T - H - K + 1)]

    def feats(self, frames_u8):
        return (patchify(frames_u8.float() / 255) - self.mean) / self.std

    def batch(self, idx):
        e = torch.tensor([self.windows[i][0] for i in idx]); s = torch.tensor([self.windows[i][1] for i in idx])
        t = s[:, None] + torch.arange(H + K)[None]
        if torch.is_tensor(self.frames):
            f = self.feats(self.frames[e[:, None].to(DEV), t.to(DEV)])
        else:
            f = self.feats(torch.from_numpy(np.ascontiguousarray(self.frames[e[:, None].numpy(), t.numpy()])).to(DEV))
        e, t = e.to(DEV), t.to(DEV)
        a = self.actions[e[:, None], t[:, :-1]]
        return f[:, :H], a[:, :H - 1], a[:, H - 1:H - 1 + K], f[:, H:]


@torch.no_grad()
def evaluate(model, data, bs=None):
    bs = bs or (64 if DEV.type == "cuda" else 16)
    model.eval()
    se_std, se_pix, n = torch.zeros(K), torch.zeros(K), 0
    for i in range(0, len(data.windows), bs):
        idx = list(range(i, min(i + bs, len(data.windows))))
        hist, past, fut, tgt = (t.to(DEV) for t in data.batch(idx))
        pred = model(hist, past, fut).float()
        se_std += ((pred - tgt) ** 2).mean((2, 3)).sum(0).cpu()
        se_pix += (((pred - tgt) * data.std) ** 2).mean((2, 3)).sum(0).cpu()   # [0,1] pixel units
        n += len(idx)
    model.train()
    return (se_std / n).tolist(), (se_pix / n).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="shiftwm")
    p.add_argument("--data", default="data/toy")
    p.add_argument("--out", default="results/toy")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=500)
    a = p.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    stats = json.loads((Path(a.data) / "stats.json").read_text())
    train, val = ToySplit(a.data, "train", stats), ToySplit(a.data, "val", stats)
    model = V2WorldModel(dict(TINY, arm=a.arm)).to(DEV)
    out = Path(a.out) / a.arm; out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w")
    print(json.dumps({"arm": a.arm, "params": model.num_params(), "windows": len(train.windows)}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    g = torch.Generator().manual_seed(a.seed)
    best, t0 = float("inf"), time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(train.windows), (a.batch,), generator=g).tolist()
        loss, logs = loss_fn(model, tuple(t.to(DEV) for t in train.batch(idx)), {})
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            rec = {"step": step, "loss": float(loss), "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v, _ = evaluate(model, val)
            rec = {"event": "val", "step": step, "val_mse": float(np.mean(v)), "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if rec["val_mse"] < best:
                best = rec["val_mse"]
                torch.save({"model": model.state_dict(), "config": model.package_config, "step": step}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV)["model"])
    test = ToySplit(a.data, "test", stats)
    s, px = evaluate(model, test)
    res = {"arm": a.arm, "params": model.num_params(), "steps": a.steps, "batch": a.batch, "threads": a.threads, "device": str(DEV),
           "train_sec": round(time.time() - t0, 1), "test_mse_std": s, "test_mse_pix": px}
    (out / "eval.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
